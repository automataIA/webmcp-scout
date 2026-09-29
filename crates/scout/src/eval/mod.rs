//! Offline routing evaluation: labeled scenarios, scoring,
//! bootstrap intervals and a reproducible report.

pub mod compare;
pub mod dataset;
pub mod report;
pub mod score;
pub mod stats;

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::jev::{JevError, QuestionSet};
use crate::llm::LlmError;
use crate::router::hybrid::fanout_question_set;
use crate::router::{Router, RouterError};
use dataset::{Dataset, Scenario};
use score::ScenarioResult;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("{path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("{path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("invalid dataset: {0}")]
    Invalid(String),
    /// Stops the run: every later scenario would fail the same way.
    #[error("{scenario}: {source}")]
    Fatal {
        scenario: String,
        source: RouterError,
    },
}

impl EvalError {
    fn io(path: &Path, e: std::io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            message: e.to_string(),
        }
    }

    fn parse(path: &Path, e: impl std::fmt::Display) -> Self {
        Self::Parse {
            path: path.to_owned(),
            message: e.to_string(),
        }
    }
}

/// Routes every scenario sequentially (free-tier rate limits make
/// concurrency pointless) and scores it. `on_result` sees each result as it
/// lands. Per-scenario model or API errors are scored as misses; a missing
/// key, a replay cache miss or a rejected account (401/402) aborts the run.
pub async fn run_routing(
    dataset: &Dataset,
    router: &Router,
    mut on_result: impl FnMut(&ScenarioResult),
) -> Result<Vec<ScenarioResult>, EvalError> {
    let mut results = Vec::with_capacity(dataset.scenarios.len());
    for scenario in &dataset.scenarios {
        let result = match router
            .route(&scenario.request, dataset.tools(scenario))
            .await
        {
            Ok(decision) => score::score(scenario, &decision),
            Err(e) if is_fatal(&e) => {
                return Err(EvalError::Fatal {
                    scenario: scenario.id.clone(),
                    source: e,
                });
            }
            Err(e) => score::score_error(scenario, e.to_string()),
        };
        on_result(&result);
        results.push(result);
    }
    Ok(results)
}

/// A scenario with the question set the fan-out router would send for it
/// (`None` when no tool is routable or there are too many).
#[derive(Debug, Serialize)]
pub struct QuestionRecord<'a> {
    #[serde(flatten)]
    pub scenario: &'a Scenario,
    pub question_set: Option<QuestionSet>,
}

/// Question sets of every scenario, in dataset order: the input for
/// training and scoring a local decision model on the same questions.
pub fn question_records(dataset: &Dataset) -> Vec<QuestionRecord<'_>> {
    dataset
        .scenarios
        .iter()
        .map(|scenario| QuestionRecord {
            scenario,
            question_set: fanout_question_set(&scenario.request, dataset.tools(scenario)),
        })
        .collect()
}

fn is_fatal(e: &RouterError) -> bool {
    matches!(
        e,
        RouterError::Llm(LlmError::MissingApiKey | LlmError::CacheMiss(_))
            | RouterError::Jev(
                JevError::MissingApiKey
                    | JevError::CacheMiss(_)
                    | JevError::Api {
                        status: 401 | 402,
                        ..
                    }
            )
    )
}
