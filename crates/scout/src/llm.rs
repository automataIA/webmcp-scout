//! OpenAI-compatible chat client for strict structured outputs (Groq by
//! default), with a record/replay response cache so evals run offline.

use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use reqwest::StatusCode;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cache;
pub use crate::cache::CacheMode;
use crate::schema::canonical_sha256;

pub const DEFAULT_BASE_URL: &str = "https://api.groq.com/openai/v1";
pub const DEFAULT_MODEL: &str = "qwen/qwen3.8-27b";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ATTEMPTS: u32 = 5;
const MAX_RETRY_WAIT: Duration = Duration::from_secs(60);
const BACKOFF_BASE: Duration = Duration::from_millis(500);

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("no API key: set FALLBACK_API_KEY")]
    MissingApiKey,
    #[error("HTTP {status}: {message}")]
    Api { status: u16, message: String },
    #[error("still rate limited after {0} attempts")]
    RateLimited(u32),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("no cached response for request {0} (replay mode)")]
    CacheMiss(String),
    #[error("cache error: {0}")]
    Cache(String),
}

#[derive(Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// Groq `reasoning_effort`; reasoning text itself is always hidden.
    pub reasoning_effort: Option<String>,
}

impl fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .finish()
    }
}

impl LlmConfig {
    /// Reads `FALLBACK_BASE_URL`, `FALLBACK_API_KEY` and `FALLBACK_MODEL`.
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        Self {
            base_url: var("FALLBACK_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            api_key: var("FALLBACK_API_KEY"),
            model: var("FALLBACK_MODEL").unwrap_or_else(|| DEFAULT_MODEL.into()),
            reasoning_effort: Some("low".into()),
        }
    }
}

/// A chat request whose answer must match `schema` (strict mode).
#[derive(Debug, Clone)]
pub struct StructuredRequest {
    pub schema_name: String,
    pub schema: Value,
    pub system: String,
    pub user: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Completion {
    pub content: Value,
    pub usage: Usage,
    /// Latency of the successful HTTP attempt (retry waits excluded). For a
    /// cache hit, the recorded latency.
    pub latency_ms: u64,
    #[serde(skip)]
    pub attempts: u32,
    #[serde(skip)]
    pub cached: bool,
}

#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    config: LlmConfig,
    cache: Option<(PathBuf, CacheMode)>,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| LlmError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            config,
            cache: None,
        })
    }

    pub fn with_cache(mut self, dir: impl Into<PathBuf>, mode: CacheMode) -> Self {
        self.cache = (mode != CacheMode::Off).then(|| (dir.into(), mode));
        self
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub async fn complete(&self, request: &StructuredRequest) -> Result<Completion, LlmError> {
        let body = self.body(request);
        let key = canonical_sha256(&body);
        if let Some((dir, mode)) = &self.cache {
            if let Some(hit) = cache::read::<Completion>(dir, &key).map_err(LlmError::Cache)? {
                return Ok(Completion {
                    cached: true,
                    ..hit
                });
            }
            if *mode == CacheMode::Replay {
                return Err(LlmError::CacheMiss(key));
            }
        }
        let completion = self.send(&body).await?;
        if let Some((dir, _)) = &self.cache {
            cache::write(dir, &key, &body, &completion).map_err(LlmError::Cache)?;
        }
        Ok(completion)
    }

    fn body(&self, request: &StructuredRequest) -> Value {
        let mut body = json!({
            "model": self.config.model,
            "messages": [
                { "role": "system", "content": request.system },
                { "role": "user", "content": request.user },
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": true,
                    "schema": request.schema,
                },
            },
        });
        if let Some(effort) = &self.config.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
            body["reasoning_format"] = json!("hidden");
        }
        body
    }

    async fn send(&self, body: &Value) -> Result<Completion, LlmError> {
        let key = self
            .config
            .api_key
            .as_deref()
            .ok_or(LlmError::MissingApiKey)?;
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let mut attempt = 0;
        loop {
            attempt += 1;
            let started = Instant::now();
            let result = self
                .http
                .post(&url)
                .bearer_auth(key)
                .json(body)
                .send()
                .await;
            let wait = match result {
                Ok(response) if response.status().is_success() => {
                    let latency_ms = elapsed_ms(started);
                    let payload: Value = response
                        .json()
                        .await
                        .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
                    return parse_completion(&payload, latency_ms, attempt);
                }
                Ok(response) => {
                    let status = response.status();
                    let retry_after = retry_after(response.headers());
                    let message = error_message(response).await;
                    if status == StatusCode::TOO_MANY_REQUESTS {
                        if attempt >= MAX_ATTEMPTS {
                            return Err(LlmError::RateLimited(attempt));
                        }
                        retry_after.unwrap_or_else(|| backoff(attempt))
                    } else if status.is_server_error() && attempt < MAX_ATTEMPTS {
                        backoff(attempt)
                    } else {
                        return Err(LlmError::Api {
                            status: status.as_u16(),
                            message,
                        });
                    }
                }
                Err(e) if (e.is_timeout() || e.is_connect()) && attempt < MAX_ATTEMPTS => {
                    backoff(attempt)
                }
                Err(e) => return Err(LlmError::Transport(e.to_string())),
            };
            tracing::debug!(attempt, ?wait, "retrying LLM request");
            tokio::time::sleep(wait.min(MAX_RETRY_WAIT)).await;
        }
    }
}

pub(crate) fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub(crate) fn backoff(attempt: u32) -> Duration {
    BACKOFF_BASE * 2u32.pow(attempt.saturating_sub(1).min(6))
}

/// `retry-after` in seconds (Groq sends integers; fractions are accepted).
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let secs: f64 = headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs.min(3600.0)))
}

async fn error_message(response: reqwest::Response) -> String {
    let text = response.text().await.unwrap_or_default();
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| text.chars().take(500).collect())
}

fn parse_completion(
    payload: &Value,
    latency_ms: u64,
    attempts: u32,
) -> Result<Completion, LlmError> {
    let choice = &payload["choices"][0];
    let finish = choice["finish_reason"].as_str().unwrap_or("unknown");
    let text = choice["message"]["content"]
        .as_str()
        .ok_or_else(|| LlmError::InvalidResponse(format!("no content (finish_reason {finish})")))?;
    let content = serde_json::from_str(text).map_err(|e| {
        LlmError::InvalidResponse(format!("content is not JSON (finish_reason {finish}): {e}"))
    })?;
    let usage = &payload["usage"];
    Ok(Completion {
        content,
        usage: Usage {
            input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
            reasoning_tokens: usage["completion_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .unwrap_or(0),
        },
        latency_ms,
        attempts,
        cached: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(server: &MockServer, key: Option<&str>) -> LlmConfig {
        LlmConfig {
            base_url: server.uri(),
            api_key: key.map(str::to_owned),
            model: "test-model".into(),
            reasoning_effort: Some("low".into()),
        }
    }

    fn request() -> StructuredRequest {
        StructuredRequest {
            schema_name: "route".into(),
            schema: json!({
                "type": "object",
                "properties": { "tool": { "type": "string" } },
                "required": ["tool"],
                "additionalProperties": false
            }),
            system: "sys".into(),
            user: "find shoes".into(),
        }
    }

    fn ok_body() -> Value {
        json!({
            "choices": [{
                "finish_reason": "stop",
                "message": { "role": "assistant", "content": "{\"tool\":\"search\"}" }
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "completion_tokens_details": { "reasoning_tokens": 20 }
            }
        })
    }

    #[tokio::test]
    async fn sends_strict_request_and_parses_completion() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer secret"))
            .and(body_partial_json(json!({
                "model": "test-model",
                "temperature": 0,
                "reasoning_format": "hidden",
                "reasoning_effort": "low",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": { "name": "route", "strict": true }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .expect(1)
            .mount(&server)
            .await;

        let client = LlmClient::new(config(&server, Some("secret"))).unwrap();
        let completion = client.complete(&request()).await.unwrap();
        assert_eq!(completion.content, json!({ "tool": "search" }));
        assert_eq!(
            completion.usage,
            Usage {
                input_tokens: 120,
                output_tokens: 30,
                reasoning_tokens: 20
            }
        );
        assert_eq!(completion.attempts, 1);
        assert!(!completion.cached);
    }

    #[tokio::test]
    async fn retries_rate_limits_honoring_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "0")
                    .set_body_json(json!({ "error": { "message": "slow down" } })),
            )
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .mount(&server)
            .await;

        let client = LlmClient::new(config(&server, Some("k"))).unwrap();
        let completion = client.complete(&request()).await.unwrap();
        assert_eq!(completion.attempts, 3);
    }

    #[tokio::test]
    async fn client_errors_surface_the_api_message_without_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(json!({ "error": { "message": "json_schema is invalid" } })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = LlmClient::new(config(&server, Some("k"))).unwrap();
        match client.complete(&request()).await {
            Err(LlmError::Api { status, message }) => {
                assert_eq!(status, 400);
                assert_eq!(message, "json_schema is invalid");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_key_fails_before_any_request() {
        let server = MockServer::start().await;
        let client = LlmClient::new(config(&server, None)).unwrap();
        assert!(matches!(
            client.complete(&request()).await,
            Err(LlmError::MissingApiKey)
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn record_then_replay_without_key_or_network() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .expect(1)
            .mount(&server)
            .await;
        let dir = tempfile::tempdir().unwrap();

        let recorder = LlmClient::new(config(&server, Some("k")))
            .unwrap()
            .with_cache(dir.path(), CacheMode::Record);
        let recorded = recorder.complete(&request()).await.unwrap();
        let again = recorder.complete(&request()).await.unwrap();
        assert!(again.cached);

        let replayer = LlmClient::new(config(&server, None))
            .unwrap()
            .with_cache(dir.path(), CacheMode::Replay);
        let replayed = replayer.complete(&request()).await.unwrap();
        assert!(replayed.cached);
        assert_eq!(replayed.content, recorded.content);
        assert_eq!(replayed.latency_ms, recorded.latency_ms);

        let mut other = request();
        other.user = "find boots".into();
        assert!(matches!(
            replayer.complete(&other).await,
            Err(LlmError::CacheMiss(_))
        ));
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let config = LlmConfig {
            base_url: DEFAULT_BASE_URL.into(),
            api_key: Some("gsk_supersecret".into()),
            model: DEFAULT_MODEL.into(),
            reasoning_effort: None,
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("supersecret"));
        assert!(debug.contains("<redacted>"));
    }
}
