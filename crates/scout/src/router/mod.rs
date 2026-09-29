//! Request → tool invocation routing.

pub mod baseline;
pub mod hybrid;
pub mod numbers;
pub mod screen;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::jev::{JevError, JevUsage};
use crate::llm::{LlmError, Usage};
use crate::local::LocalError;
use crate::schema::{Support, ToolSpec};
use baseline::BaselineRouter;
use hybrid::{HybridRouter, HybridTrace};

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error(transparent)]
    Jev(#[from] JevError),
    #[error(transparent)]
    Local(#[from] LocalError),
    #[error("model output has an unexpected shape: {0}")]
    InvalidOutput(String),
}

/// A tool left out of routing because its schema is outside the subset.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExcludedTool {
    pub name: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RouteDecision {
    /// Which component decided: `llm-baseline`, `jev-hybrid`, `jev-fanout`, `local-fanout`,
    /// with `+escalated` when the hybrid router handed over to the baseline.
    pub source: String,
    /// `None`: no tool fits, or a required value is missing.
    pub tool: Option<String>,
    /// Arguments to execute with; unspecified optionals are omitted.
    pub args: Map<String, Value>,
    /// Problems against the site schema. Non-empty means not executable.
    pub validation_errors: Vec<String>,
    pub excluded: Vec<ExcludedTool>,
    pub latency_ms: u64,
    pub usage: Usage,
    pub llm_calls: u32,
    pub jev_usage: JevUsage,
    pub jev_calls: u32,
    /// Requests to the local decision model.
    pub local_calls: u32,
    /// Every request was served from the record/replay cache.
    pub cached: bool,
    /// Per-stage judgements of the hybrid router.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<HybridTrace>,
}

impl RouteDecision {
    pub fn is_valid(&self) -> bool {
        self.validation_errors.is_empty()
    }
}

/// The routers under comparison. An enum rather than a trait:
/// `async fn` in public traits is not object safe and warns on `Send`.
pub enum Router {
    Baseline(BaselineRouter),
    Hybrid(HybridRouter),
}

impl Router {
    pub async fn route(
        &self,
        request: &str,
        tools: &[ToolSpec],
    ) -> Result<RouteDecision, RouterError> {
        match self {
            Self::Baseline(router) => router.route(request, tools).await,
            Self::Hybrid(router) => router.route(request, tools).await,
        }
    }

    /// The `source` of this router's own (non-escalated) decisions.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Baseline(_) => baseline::SOURCE,
            Self::Hybrid(router) => router.name(),
        }
    }
}

/// Splits tools into routable ones and exclusions with their reasons.
pub fn partition(tools: &[ToolSpec]) -> (Vec<&ToolSpec>, Vec<ExcludedTool>) {
    let mut routable = Vec::new();
    let mut excluded = Vec::new();
    for tool in tools {
        match &tool.support {
            Support::Unsupported(reasons) => excluded.push(ExcludedTool {
                name: tool.name.clone(),
                reasons: reasons.clone(),
            }),
            _ => routable.push(tool),
        }
    }
    (routable, excluded)
}

/// Turns model-produced arguments into executable ones: drops `null`
/// optionals, rejects parameters the tool does not declare (or that are
/// outside the subset), then validates against the site schema.
pub fn finalize_args(tool: &ToolSpec, raw: &Value) -> (Map<String, Value>, Vec<String>) {
    let mut errors = Vec::new();
    let mut args = Map::new();
    match raw {
        Value::Null => {}
        Value::Object(map) => {
            for (name, value) in map {
                let Some(param) = tool.params.iter().find(|p| &p.name == name) else {
                    errors.push(format!("unexpected argument `{name}`"));
                    continue;
                };
                if value.is_null() && !param.required {
                    continue;
                }
                args.insert(name.clone(), value.clone());
            }
        }
        other => errors.push(format!("arguments are not an object: {other}")),
    }
    if let Err(schema_errors) = tool.validate_args(&Value::Object(args.clone())) {
        errors.extend(schema_errors);
    }
    (args, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webmcp::DiscoveredTool;
    use serde_json::json;

    fn search() -> ToolSpec {
        ToolSpec::from_tool(&DiscoveredTool {
            name: "search".into(),
            title: None,
            description: "Search products".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_price": { "type": "number", "minimum": 0 }
                },
                "required": ["query"]
            }),
            origin: "https://example.test".into(),
        })
    }

    #[test]
    fn null_optionals_are_dropped() {
        let (args, errors) =
            finalize_args(&search(), &json!({ "query": "boots", "max_price": null }));
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(Value::Object(args), json!({ "query": "boots" }));
    }

    #[test]
    fn violations_and_foreign_arguments_are_reported() {
        let (_, errors) = finalize_args(
            &search(),
            &json!({ "query": null, "max_price": -5, "sku": "A-1" }),
        );
        assert_eq!(errors.len(), 3, "{errors:?}");
        assert!(
            errors
                .iter()
                .any(|e| e.contains("unexpected argument `sku`"))
        );
    }

    #[test]
    fn unsupported_tools_are_excluded_with_reasons() {
        let bad = ToolSpec::from_tool(&DiscoveredTool {
            name: "book".into(),
            title: None,
            description: String::new(),
            input_schema: json!({
                "properties": { "trip": { "type": "object" } },
                "required": ["trip"]
            }),
            origin: "https://example.test".into(),
        });
        let tools = [search(), bad];
        let (routable, excluded) = partition(&tools);
        assert_eq!(routable.len(), 1);
        assert_eq!(excluded[0].name, "book");
        assert_eq!(excluded[0].reasons, ["parameter `trip`: nested object"]);
    }
}
