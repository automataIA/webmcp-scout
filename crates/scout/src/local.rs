//! Client for the local decision model (`local_router/serve.py`): the
//! fan-out question set goes to `POST {base}/decide` on localhost, one
//! calibrated probability per option comes back. Our own protocol; answers
//! are read into the same [`Evaluation`] the hybrid router consumes, with a
//! choice's confidence set to its (calibrated) top probability.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::cache::{self, CacheMode};
use crate::jev::{Answer, Evaluation, JevUsage, Question, QuestionSet};
use crate::llm::elapsed_ms;
use crate::schema::canonical_sha256;

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8765";
/// Options are read out on the letters A..Z.
pub const MAX_OPTIONS: usize = 26;
/// Loading the model on the first request can take a while.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, thiserror::Error)]
pub enum LocalError {
    #[error("no local model name: set SCOUT_LOCAL_MODEL to the served run")]
    MissingModel,
    #[error("question `{0}` has more than {MAX_OPTIONS} options")]
    TooManyOptions(String),
    #[error("HTTP {status}: {message}")]
    Api { status: u16, message: String },
    #[error("transport error: {0}")]
    Transport(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("no cached response for request {0} (replay mode)")]
    CacheMiss(String),
    #[error("cache error: {0}")]
    Cache(String),
}

#[derive(Debug, Clone)]
pub struct LocalConfig {
    pub base_url: String,
    /// The served run's name; it keys the cache, and the server must report it.
    pub model: Option<String>,
}

impl LocalConfig {
    /// Reads `SCOUT_LOCAL_URL` (default: localhost:8765) and `SCOUT_LOCAL_MODEL`.
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        Self {
            base_url: var("SCOUT_LOCAL_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            model: var("SCOUT_LOCAL_MODEL"),
        }
    }
}

#[derive(Clone)]
pub struct LocalClient {
    http: reqwest::Client,
    config: LocalConfig,
    cache: Option<(PathBuf, CacheMode)>,
}

impl LocalClient {
    pub fn new(config: LocalConfig) -> Result<Self, LocalError> {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| LocalError::Transport(e.to_string()))?;
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
        self.config.model.as_deref().unwrap_or("unset")
    }

    /// Answers every question from the cache or the server; every question
    /// gets an answer of its type, or the response is rejected.
    pub async fn evaluate(&self, set: &QuestionSet) -> Result<Evaluation, LocalError> {
        for (id, question) in &set.questions {
            if let Question::Choice { criteria, .. } = question
                && criteria.len() > MAX_OPTIONS
            {
                return Err(LocalError::TooManyOptions(id.clone()));
            }
        }
        let model = self
            .config
            .model
            .as_deref()
            .ok_or(LocalError::MissingModel)?;
        let body = json!({ "state": set.state, "questions": set.questions });
        let key = canonical_sha256(&json!({ "model": model, "request": body }));
        if let Some((dir, mode)) = &self.cache {
            if let Some(hit) = cache::read::<Evaluation>(dir, &key).map_err(LocalError::Cache)? {
                return Ok(Evaluation {
                    cached: true,
                    ..hit
                });
            }
            if *mode == CacheMode::Replay {
                return Err(LocalError::CacheMiss(key));
            }
        }
        let evaluation = self.send(set, &body, model).await?;
        if let Some((dir, _)) = &self.cache {
            cache::write(dir, &key, &body, &evaluation).map_err(LocalError::Cache)?;
        }
        Ok(evaluation)
    }

    async fn send(
        &self,
        set: &QuestionSet,
        body: &Value,
        model: &str,
    ) -> Result<Evaluation, LocalError> {
        let url = format!("{}/decide", self.config.base_url.trim_end_matches('/'));
        let started = Instant::now();
        let response = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| LocalError::Transport(e.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LocalError::Api {
                status: status.as_u16(),
                message: text.chars().take(500).collect(),
            });
        }
        let latency_ms = elapsed_ms(started);
        let wire: Wire = response
            .json()
            .await
            .map_err(|e| LocalError::InvalidResponse(e.to_string()))?;
        if wire.model != model {
            return Err(LocalError::InvalidResponse(format!(
                "the server runs `{}`, SCOUT_LOCAL_MODEL is `{model}`",
                wire.model
            )));
        }
        Ok(Evaluation {
            model: wire.model,
            answers: answers(set, &wire.answers)?,
            usage: JevUsage::default(),
            latency_ms,
            attempts: 1,
            cached: false,
        })
    }
}

#[derive(Deserialize)]
struct Wire {
    model: String,
    answers: BTreeMap<String, BTreeMap<String, f64>>,
}

/// Option probabilities as typed answers: a yes/no is P(yes); a choice is
/// its most probable option, with that probability as the confidence.
fn answers(
    set: &QuestionSet,
    wire: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Result<BTreeMap<String, Answer>, LocalError> {
    let invalid = |id: &str, what: &str| LocalError::InvalidResponse(format!("`{id}`: {what}"));
    let mut out = BTreeMap::new();
    for (id, question) in &set.questions {
        let probs = wire.get(id).ok_or_else(|| invalid(id, "no answer"))?;
        let answer = match question {
            Question::Noul { .. } => {
                if probs.len() != 2 || !probs.contains_key("no") {
                    return Err(invalid(id, "expected `yes` and `no`"));
                }
                let noul = *probs.get("yes").ok_or_else(|| invalid(id, "no `yes`"))?;
                Answer::Noul { noul }
            }
            Question::Choice { criteria, .. } => {
                if probs.len() != criteria.len() || probs.keys().any(|k| !criteria.contains_key(k))
                {
                    return Err(invalid(id, "options differ from the question's"));
                }
                let (choice, confidence) = probs
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(k, p)| (k.clone(), *p))
                    .ok_or_else(|| invalid(id, "no options"))?;
                Answer::Choice {
                    choice,
                    probabilities: probs.clone(),
                    confidence,
                }
            }
            Question::Score { .. } => return Err(invalid(id, "score questions are unsupported")),
        };
        out.insert(id.clone(), answer);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn set() -> QuestionSet {
        QuestionSet {
            state: json!({ "user_request": "soccer boots" }),
            questions: BTreeMap::from([
                ("__act__".to_owned(), Question::noul("Act?")),
                (
                    "__tool__".to_owned(),
                    Question::Choice {
                        instructions: json!("Which tool?"),
                        criteria: BTreeMap::from([
                            ("t0".to_owned(), json!("search")),
                            ("t1".to_owned(), json!("cart")),
                        ]),
                    },
                ),
            ]),
        }
    }

    fn client(server: &MockServer, model: Option<&str>) -> LocalClient {
        LocalClient::new(LocalConfig {
            base_url: server.uri(),
            model: model.map(str::to_owned),
        })
        .unwrap()
    }

    async fn serve(body: Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/decide"))
            .and(body_partial_json(
                json!({ "state": { "user_request": "soccer boots" } }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn probabilities_become_typed_answers() {
        let server = serve(json!({
            "model": "e1",
            "latency_ms": 40,
            "answers": {
                "__act__": { "yes": 0.9, "no": 0.1 },
                "__tool__": { "t0": 0.2, "t1": 0.8 }
            }
        }))
        .await;
        let eval = client(&server, Some("e1")).evaluate(&set()).await.unwrap();
        assert_eq!(eval.noul("__act__"), Some(0.9));
        let choice = eval.choice("__tool__").unwrap();
        assert_eq!((choice.choice, choice.confidence), ("t1", 0.8));
        assert_eq!(eval.usage, JevUsage::default());
    }

    #[tokio::test]
    async fn wrong_model_and_missing_answers_are_rejected() {
        let server = serve(json!({
            "model": "e2",
            "answers": { "__act__": { "yes": 0.9, "no": 0.1 } }
        }))
        .await;
        let err = client(&server, Some("e1"))
            .evaluate(&set())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("the server runs `e2`"), "{err}");
        let err = client(&server, Some("e2"))
            .evaluate(&set())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("`__tool__`: no answer"), "{err}");
        let err = client(&server, None).evaluate(&set()).await.unwrap_err();
        assert!(matches!(err, LocalError::MissingModel));
    }

    #[tokio::test]
    async fn replay_serves_recorded_answers_without_a_server() {
        let server = serve(json!({
            "model": "e1",
            "answers": {
                "__act__": { "yes": 0.3, "no": 0.7 },
                "__tool__": { "t0": 0.6, "t1": 0.4 }
            }
        }))
        .await;
        let dir = tempfile::tempdir().unwrap();
        let recorded = client(&server, Some("e1"))
            .with_cache(dir.path(), CacheMode::Record)
            .evaluate(&set())
            .await
            .unwrap();
        let offline = LocalClient::new(LocalConfig {
            base_url: "http://127.0.0.1:9".into(),
            model: Some("e1".into()),
        })
        .unwrap()
        .with_cache(dir.path(), CacheMode::Replay);
        let replayed = offline.evaluate(&set()).await.unwrap();
        assert!(replayed.cached);
        assert_eq!(replayed.answers, recorded.answers);
        let other = LocalClient::new(LocalConfig {
            base_url: "http://127.0.0.1:9".into(),
            model: Some("e2".into()),
        })
        .unwrap()
        .with_cache(dir.path(), CacheMode::Replay);
        assert!(matches!(
            other.evaluate(&set()).await,
            Err(LocalError::CacheMiss(_))
        ));
    }
}
