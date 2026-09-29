//! TypeSafe System One client (Jev) for closed decisions: typed questions and
//! answers, local validation, a fail-closed token budget, retries on 429/529
//! and the record/replay cache. API: `POST {base}/v1/systemone`
//! (docs.typesafe.ai/api, checked 2026-09-26).

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cache::{self, CacheMode};
use crate::llm::{backoff, elapsed_ms, retry_after};
use crate::schema::canonical_sha256;

pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
/// Pinned model: recorded evals stay comparable.
pub const PINNED_MODEL: &str = "jev-1.13.0";
/// Documented request limit.
pub const MAX_REQUEST_TOKENS: u64 = 64_000;
/// Documented limit for the state plus the longest question.
pub const MAX_STATE_QUESTION_TOKENS: u64 = 32_000;
pub const MAX_CHOICE_OPTIONS: usize = 255;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
/// The SDK default: two retries after the first attempt.
const MAX_ATTEMPTS: u32 = 3;
const MAX_RETRY_WAIT: Duration = Duration::from_secs(30);
/// TypeSafe "overloaded".
const OVERLOADED: u16 = 529;

#[derive(Debug, thiserror::Error)]
pub enum JevError {
    #[error("no API key: set TYPESAFE_API_KEY")]
    MissingApiKey,
    #[error("question `{id}`: {message}")]
    InvalidQuestion { id: String, message: String },
    /// Fail closed: the decision is escalated, never split.
    #[error("request needs ~{estimated} tokens, over the {limit} limit")]
    OverBudget { estimated: u64, limit: u64 },
    #[error("HTTP {status}: {message}")]
    Api { status: u16, message: String },
    #[error("still rate limited or overloaded after {0} attempts")]
    Unavailable(u32),
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
pub struct JevConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl fmt::Debug for JevConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JevConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("model", &self.model)
            .finish()
    }
}

impl JevConfig {
    /// Reads `TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY` and
    /// `TYPESAFE_DEFAULT_MODEL` (default: the pinned model).
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        Self {
            base_url: var("TYPESAFE_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            api_key: var("TYPESAFE_API_KEY"),
            model: var("TYPESAFE_DEFAULT_MODEL").unwrap_or_else(|| PINNED_MODEL.into()),
        }
    }
}

/// A typed question. `instructions` and criteria values may be strings or
/// structured JSON whose fields the question refers to in backticks.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// Yes/no; the answer is the probability of yes.
    Noul {
        instructions: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One option out of 1..=255. Option keys and descriptions reach the model.
    Choice {
        instructions: Value,
        criteria: BTreeMap<String, Value>,
    },
    /// A position along 2..=10 ordered levels.
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub yes: Value,
    #[serde(rename = "false")]
    pub no: Value,
}

impl Question {
    pub fn noul(instructions: impl Into<Value>) -> Self {
        Self::Noul {
            instructions: instructions.into(),
            criteria: None,
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Noul { .. } => Ok(()),
            Self::Choice { criteria, .. } => {
                if (1..=MAX_CHOICE_OPTIONS).contains(&criteria.len()) {
                    Ok(())
                } else {
                    Err(format!(
                        "a choice needs 1..={MAX_CHOICE_OPTIONS} options, got {}",
                        criteria.len()
                    ))
                }
            }
            Self::Score { criteria, .. } => {
                if (2..=10).contains(&criteria.len()) {
                    Ok(())
                } else {
                    Err(format!(
                        "a score needs 2..=10 levels, got {}",
                        criteria.len()
                    ))
                }
            }
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        /// How peaked `probabilities` is; not a calibrated probability.
        confidence: f64,
    },
    Score {
        score: f64,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

impl Answer {
    fn kind(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }
}

/// One state evaluated against a set of questions. Question ids are local:
/// they never reach the model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuestionSet {
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

impl QuestionSet {
    /// Checks the documented limits with a deliberately generous estimate
    /// (one token per three bytes of JSON).
    pub fn check_budget(&self) -> Result<(), JevError> {
        let state = estimate_tokens(&self.state);
        let longest = self
            .questions
            .values()
            .map(estimate_tokens)
            .max()
            .unwrap_or(0);
        let total = state + self.questions.values().map(estimate_tokens).sum::<u64>();
        if state + longest > MAX_STATE_QUESTION_TOKENS {
            return Err(JevError::OverBudget {
                estimated: state + longest,
                limit: MAX_STATE_QUESTION_TOKENS,
            });
        }
        if total > MAX_REQUEST_TOKENS {
            return Err(JevError::OverBudget {
                estimated: total,
                limit: MAX_REQUEST_TOKENS,
            });
        }
        Ok(())
    }
}

pub fn estimate_tokens(value: &impl Serialize) -> u64 {
    let bytes = serde_json::to_vec(value).map_or(0, |v| v.len());
    u64::try_from(bytes).unwrap_or(u64::MAX).div_ceil(3)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JevUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl std::ops::AddAssign for JevUsage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    /// The model that answered, as reported by the API.
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: JevUsage,
    /// Latency of the successful HTTP attempt; for a cache hit, the recorded one.
    pub latency_ms: u64,
    #[serde(skip)]
    pub attempts: u32,
    #[serde(skip)]
    pub cached: bool,
}

impl Evaluation {
    /// Probability of yes.
    pub fn noul(&self, id: &str) -> Option<f64> {
        match self.answers.get(id)? {
            Answer::Noul { noul } => Some(*noul),
            _ => None,
        }
    }

    pub fn choice(&self, id: &str) -> Option<ChoiceAnswer<'_>> {
        match self.answers.get(id)? {
            Answer::Choice {
                choice,
                probabilities,
                confidence,
            } => Some(ChoiceAnswer {
                choice,
                probability: probabilities.get(choice).copied().unwrap_or(0.0),
                confidence: *confidence,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChoiceAnswer<'a> {
    pub choice: &'a str,
    pub probability: f64,
    pub confidence: f64,
}

/// Clones share the request pacing.
#[derive(Clone)]
pub struct JevClient {
    http: reqwest::Client,
    config: JevConfig,
    cache: Option<(PathBuf, CacheMode)>,
    /// Least time between two HTTP requests, retries included.
    min_interval: Duration,
    /// Wait before retrying a request refused with HTTP 402; `None` fails.
    billing_retry: Option<Duration>,
    last_request: Arc<tokio::sync::Mutex<Option<Instant>>>,
}

impl JevClient {
    pub fn new(config: JevConfig) -> Result<Self, JevError> {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| JevError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            config,
            cache: None,
            min_interval: Duration::ZERO,
            billing_retry: None,
            last_request: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    pub fn with_cache(mut self, dir: impl Into<PathBuf>, mode: CacheMode) -> Self {
        self.cache = (mode != CacheMode::Off).then(|| (dir.into(), mode));
        self
    }

    /// Spaces HTTP requests at least `interval` apart, so that a long eval
    /// run does not hammer the API. Cache hits never wait.
    pub fn with_min_interval(mut self, interval: Duration) -> Self {
        self.min_interval = interval;
        self
    }

    /// Retries a request refused with HTTP 402 (billing) every `interval`,
    /// until the account accepts it again; the run waits instead of failing.
    pub fn with_billing_retry(mut self, interval: Duration) -> Self {
        self.billing_retry = Some(interval);
        self
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Validates the questions, checks the token budget, then answers from
    /// the cache or the API. Every question gets an answer of its type, or
    /// the response is rejected.
    pub async fn evaluate(&self, set: &QuestionSet) -> Result<Evaluation, JevError> {
        for (id, question) in &set.questions {
            question
                .validate()
                .map_err(|message| JevError::InvalidQuestion {
                    id: id.clone(),
                    message,
                })?;
        }
        set.check_budget()?;
        let body = json!({
            "state": set.state,
            "model": self.config.model,
            "questions": set.questions,
        });
        let key = canonical_sha256(&body);
        if let Some((dir, mode)) = &self.cache {
            if let Some(hit) = cache::read::<Evaluation>(dir, &key).map_err(JevError::Cache)? {
                return Ok(Evaluation {
                    cached: true,
                    ..hit
                });
            }
            if *mode == CacheMode::Replay {
                return Err(JevError::CacheMiss(key));
            }
        }
        let evaluation = self.send(&body).await?;
        check_answers(set, &evaluation)?;
        if let Some((dir, _)) = &self.cache {
            cache::write(dir, &key, &body, &evaluation).map_err(JevError::Cache)?;
        }
        Ok(evaluation)
    }

    async fn send(&self, body: &Value) -> Result<Evaluation, JevError> {
        let key = self
            .config
            .api_key
            .as_deref()
            .ok_or(JevError::MissingApiKey)?;
        let url = format!(
            "{}/v1/systemone",
            self.config.base_url.trim_end_matches('/')
        );
        let mut attempt = 0;
        loop {
            attempt += 1;
            self.pace().await;
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
                    let wire: Wire = response
                        .json()
                        .await
                        .map_err(|e| JevError::InvalidResponse(e.to_string()))?;
                    return Ok(Evaluation {
                        model: wire.model,
                        answers: wire.answers,
                        usage: wire.usage,
                        latency_ms,
                        attempts: attempt,
                        cached: false,
                    });
                }
                Ok(response) => {
                    let status = response.status();
                    if status == StatusCode::PAYMENT_REQUIRED
                        && let Some(wait) = self.billing_retry
                    {
                        let message = error_message(response).await;
                        tracing::warn!(?wait, %message, "Jev returned HTTP 402; retrying");
                        // Rate-limit retries count again from the next answer.
                        attempt = 0;
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                    let retryable =
                        status == StatusCode::TOO_MANY_REQUESTS || status.as_u16() == OVERLOADED;
                    if retryable && attempt >= MAX_ATTEMPTS {
                        return Err(JevError::Unavailable(attempt));
                    }
                    if !retryable {
                        return Err(JevError::Api {
                            status: status.as_u16(),
                            message: error_message(response).await,
                        });
                    }
                    retry_after(response.headers()).unwrap_or_else(|| backoff(attempt))
                }
                Err(e) if (e.is_timeout() || e.is_connect()) && attempt < MAX_ATTEMPTS => {
                    backoff(attempt)
                }
                Err(e) => return Err(JevError::Transport(e.to_string())),
            };
            tracing::debug!(attempt, ?wait, "retrying Jev request");
            tokio::time::sleep(wait.min(MAX_RETRY_WAIT)).await;
        }
    }

    async fn pace(&self) {
        let mut last = self.last_request.lock().await;
        if let Some(at) = *last {
            let wait = self.min_interval.saturating_sub(at.elapsed());
            if !wait.is_zero() {
                tracing::debug!(?wait, "pacing Jev request");
                tokio::time::sleep(wait).await;
            }
        }
        *last = Some(Instant::now());
    }
}

#[derive(Deserialize)]
struct Wire {
    model: String,
    answers: BTreeMap<String, Answer>,
    usage: JevUsage,
}

fn check_answers(set: &QuestionSet, evaluation: &Evaluation) -> Result<(), JevError> {
    for (id, question) in &set.questions {
        let answer = evaluation
            .answers
            .get(id)
            .ok_or_else(|| JevError::InvalidResponse(format!("no answer for `{id}`")))?;
        if answer.kind() != question.kind() {
            return Err(JevError::InvalidResponse(format!(
                "`{id}`: expected a {} answer, got {}",
                question.kind(),
                answer.kind()
            )));
        }
        if let (Question::Choice { criteria, .. }, Answer::Choice { choice, .. }) =
            (question, answer)
            && !criteria.contains_key(choice)
        {
            return Err(JevError::InvalidResponse(format!(
                "`{id}`: unknown option `{choice}`"
            )));
        }
    }
    Ok(())
}

/// `detail` is a message object, a string, or a list of validation errors.
async fn error_message(response: reqwest::Response) -> String {
    let text = response.text().await.unwrap_or_default();
    let parsed: Option<Value> = serde_json::from_str(&text).ok();
    let detail = parsed.as_ref().map(|v| &v["detail"]);
    let message = match detail {
        Some(Value::Object(d)) => d.get("message").and_then(Value::as_str).map(str::to_owned),
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(items)) => Some(
            items
                .iter()
                .map(|i| format!("{}: {}", i["loc"], i["msg"].as_str().unwrap_or("?")))
                .collect::<Vec<_>>()
                .join("; "),
        ),
        _ => None,
    };
    message.unwrap_or_else(|| text.chars().take(500).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(server: &MockServer, key: Option<&str>) -> JevConfig {
        JevConfig {
            base_url: server.uri(),
            api_key: key.map(str::to_owned),
            model: PINNED_MODEL.into(),
        }
    }

    fn set() -> QuestionSet {
        QuestionSet {
            state: json!({ "user_request": "soccer boots" }),
            questions: BTreeMap::from([
                (
                    "__act__".to_owned(),
                    Question::noul("Does the user ask for a product search?"),
                ),
                (
                    "__tool__".to_owned(),
                    Question::Choice {
                        instructions: json!("What does the user ask the shop to do?"),
                        criteria: BTreeMap::from([
                            ("t0".to_owned(), json!("search the catalog")),
                            ("t1".to_owned(), Value::Null),
                        ]),
                    },
                ),
            ]),
        }
    }

    /// The documented response shape (docs.typesafe.ai/api).
    fn ok_body() -> Value {
        json!({
            "model": "jev-1.13.0",
            "answers": {
                "__act__": { "type": "noul", "noul": 0.95 },
                "__tool__": {
                    "type": "choice",
                    "choice": "t0",
                    "probabilities": { "t0": 0.88, "t1": 0.12 },
                    "confidence": 0.81
                }
            },
            "usage": { "input_tokens": 318, "output_tokens": 34 }
        })
    }

    #[tokio::test]
    async fn sends_typed_questions_and_parses_answers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(header("authorization", "Bearer secret"))
            .and(body_partial_json(json!({
                "model": "jev-1.13.0",
                "state": { "user_request": "soccer boots" },
                "questions": {
                    "__act__": { "type": "noul", "instructions": "Does the user ask for a product search?" },
                    "__tool__": {
                        "type": "choice",
                        "criteria": { "t0": "search the catalog", "t1": null }
                    }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .expect(1)
            .mount(&server)
            .await;

        let client = JevClient::new(config(&server, Some("secret"))).unwrap();
        let evaluation = client.evaluate(&set()).await.unwrap();
        assert_eq!(evaluation.noul("__act__"), Some(0.95));
        let tool = evaluation.choice("__tool__").unwrap();
        assert_eq!(
            (tool.choice, tool.probability, tool.confidence),
            ("t0", 0.88, 0.81)
        );
        assert_eq!(
            evaluation.usage,
            JevUsage {
                input_tokens: 318,
                output_tokens: 34
            }
        );
        assert_eq!(evaluation.noul("__tool__"), None);
    }

    #[tokio::test]
    async fn retries_429_and_529_honoring_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(529).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .mount(&server)
            .await;

        let client = JevClient::new(config(&server, Some("k"))).unwrap();
        assert_eq!(client.evaluate(&set()).await.unwrap().attempts, 3);
    }

    #[tokio::test]
    async fn requests_are_spaced_by_the_min_interval() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .mount(&server)
            .await;

        let interval = Duration::from_millis(300);
        let client = JevClient::new(config(&server, Some("k")))
            .unwrap()
            .with_min_interval(interval);
        let started = Instant::now();
        client.evaluate(&set()).await.unwrap();
        // A clone shares the pacing.
        client.clone().evaluate(&set()).await.unwrap();
        assert!(started.elapsed() >= interval);
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn client_errors_are_not_retried() {
        for (status, body, expected) in [
            (
                401,
                json!({ "detail": "Invalid API key" }),
                "Invalid API key",
            ),
            (
                402,
                json!({ "detail": { "error_type": "billing_error", "message": "no credits" } }),
                "no credits",
            ),
            (
                422,
                json!({ "detail": [{ "loc": ["body", "questions"], "msg": "field required" }] }),
                "[\"body\",\"questions\"]: field required",
            ),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(status).set_body_json(body))
                .expect(1)
                .mount(&server)
                .await;
            let client = JevClient::new(config(&server, Some("k"))).unwrap();
            match client.evaluate(&set()).await {
                Err(JevError::Api { status: s, message }) => {
                    assert_eq!((s, message.as_str()), (status, expected));
                }
                other => panic!("{status}: unexpected {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn billing_errors_are_retried_when_enabled() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(402).set_body_json(
                json!({ "detail": { "error_type": "billing_error", "message": "no credits" } }),
            ))
            .up_to_n_times(2)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
            .mount(&server)
            .await;
        let client = JevClient::new(config(&server, Some("k")))
            .unwrap()
            .with_billing_retry(Duration::from_millis(10));
        let evaluation = client.evaluate(&set()).await.unwrap();
        assert_eq!(evaluation.attempts, 1);
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn answers_must_match_the_questions() {
        let mut body = ok_body();
        body["answers"]["__tool__"]["choice"] = json!("t9");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let client = JevClient::new(config(&server, Some("k"))).unwrap();
        let err = client.evaluate(&set()).await.unwrap_err();
        assert!(err.to_string().contains("unknown option `t9`"), "{err}");
    }

    #[tokio::test]
    async fn local_validation_and_budget_fail_before_any_request() {
        let server = MockServer::start().await;
        let client = JevClient::new(config(&server, Some("k"))).unwrap();

        let mut empty = set();
        empty.questions.insert(
            "x".into(),
            Question::Choice {
                instructions: json!("?"),
                criteria: BTreeMap::new(),
            },
        );
        assert!(matches!(
            client.evaluate(&empty).await,
            Err(JevError::InvalidQuestion { id, .. }) if id == "x"
        ));

        let mut score = set();
        score.questions.insert(
            "s".into(),
            Question::Score {
                instructions: json!("?"),
                criteria: vec![json!("only one level")],
            },
        );
        assert!(matches!(
            client.evaluate(&score).await,
            Err(JevError::InvalidQuestion { id, .. }) if id == "s"
        ));

        let mut huge = set();
        huge.state = json!({ "user_request": "x".repeat(100_000) });
        assert!(matches!(
            client.evaluate(&huge).await,
            Err(JevError::OverBudget {
                limit: MAX_STATE_QUESTION_TOKENS,
                ..
            })
        ));

        let mut many = set();
        for i in 0..40 {
            many.questions
                .insert(format!("q{i}"), Question::noul("y".repeat(5_000)));
        }
        assert!(matches!(
            client.evaluate(&many).await,
            Err(JevError::OverBudget {
                limit: MAX_REQUEST_TOKENS,
                ..
            })
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

        let recorder = JevClient::new(config(&server, Some("k")))
            .unwrap()
            .with_cache(dir.path(), CacheMode::Record);
        let recorded = recorder.evaluate(&set()).await.unwrap();
        assert!(recorder.evaluate(&set()).await.unwrap().cached);

        let replayer = JevClient::new(config(&server, None))
            .unwrap()
            .with_cache(dir.path(), CacheMode::Replay);
        let replayed = replayer.evaluate(&set()).await.unwrap();
        assert!(replayed.cached);
        assert_eq!(replayed.answers, recorded.answers);

        let mut other = set();
        other.state = json!({ "user_request": "running shoes" });
        assert!(matches!(
            replayer.evaluate(&other).await,
            Err(JevError::CacheMiss(_))
        ));
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let config = JevConfig {
            base_url: DEFAULT_BASE_URL.into(),
            api_key: Some("ts_supersecret".into()),
            model: PINNED_MODEL.into(),
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("supersecret") && debug.contains("<redacted>"));
    }
}
