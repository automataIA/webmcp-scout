//! Per-scenario scoring of a routing decision against its labels.

use serde::Serialize;
use serde_json::{Map, Value};

use super::dataset::{Expect, ExpectedTool, Scenario, Split};
use crate::jev::JevUsage;
use crate::llm::Usage;
use crate::router::RouteDecision;
use crate::router::hybrid::HybridTrace;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldResult {
    pub field: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub got: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScenarioResult {
    pub id: String,
    /// Which component decided; empty when routing failed.
    pub source: String,
    pub toolset: String,
    pub split: Split,
    pub tags: Vec<String>,
    pub request: String,
    pub expected_tool: ExpectedTool,
    pub predicted_tool: Option<String>,
    pub args: Map<String, Value>,
    pub validation_errors: Vec<String>,
    /// Checked arguments; empty unless the tool is correct.
    pub fields: Vec<FieldResult>,
    pub act_ok: bool,
    pub tool_ok: bool,
    /// Whether the arguments pass the site schema; `None` when no tool was picked.
    pub schema_valid: Option<bool>,
    /// Correct tool and, if any, valid arguments with every checked field right.
    pub exact: bool,
    pub latency_ms: u64,
    pub usage: Usage,
    pub llm_calls: u32,
    pub jev_usage: JevUsage,
    pub jev_calls: u32,
    /// Left out when zero, so reports of the other routers stay unchanged.
    #[serde(skip_serializing_if = "is_zero")]
    pub local_calls: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<HybridTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl ScenarioResult {
    pub fn fields_ok(&self) -> usize {
        self.fields.iter().filter(|f| f.ok).count()
    }
}

pub fn score(scenario: &Scenario, decision: &RouteDecision) -> ScenarioResult {
    let predicted = decision.tool.as_deref();
    let tool_ok = scenario.tool.accepts(predicted);
    let fields = if tool_ok && predicted.is_some() {
        check_fields(scenario, &decision.args)
    } else {
        Vec::new()
    };
    let schema_valid = predicted.map(|_| decision.is_valid());
    let exact = tool_ok && schema_valid.unwrap_or(true) && fields.iter().all(|f| f.ok);
    ScenarioResult {
        id: scenario.id.clone(),
        source: decision.source.clone(),
        toolset: scenario.toolset.clone(),
        split: scenario.split,
        tags: scenario.tags.clone(),
        request: scenario.request.clone(),
        expected_tool: scenario.tool.clone(),
        predicted_tool: decision.tool.clone(),
        args: decision.args.clone(),
        validation_errors: decision.validation_errors.clone(),
        fields,
        act_ok: scenario.tool.expects_call() == predicted.is_some(),
        tool_ok,
        schema_valid,
        exact,
        latency_ms: decision.latency_ms,
        usage: decision.usage,
        llm_calls: decision.llm_calls,
        jev_usage: decision.jev_usage,
        jev_calls: decision.jev_calls,
        local_calls: decision.local_calls,
        hybrid: decision.hybrid.clone(),
        error: None,
    }
}

/// A scenario whose routing failed counts as wrong on every metric.
pub fn score_error(scenario: &Scenario, error: String) -> ScenarioResult {
    ScenarioResult {
        id: scenario.id.clone(),
        source: String::new(),
        toolset: scenario.toolset.clone(),
        split: scenario.split,
        tags: scenario.tags.clone(),
        request: scenario.request.clone(),
        expected_tool: scenario.tool.clone(),
        predicted_tool: None,
        args: Map::new(),
        validation_errors: Vec::new(),
        fields: Vec::new(),
        act_ok: false,
        tool_ok: false,
        schema_valid: None,
        exact: false,
        latency_ms: 0,
        usage: Usage::default(),
        llm_calls: 0,
        jev_usage: JevUsage::default(),
        jev_calls: 0,
        local_calls: 0,
        hybrid: None,
        error: Some(error),
    }
}

fn check_fields(scenario: &Scenario, args: &Map<String, Value>) -> Vec<FieldResult> {
    let expected = scenario.args.iter().map(|(field, expect)| {
        let got = args.get(field);
        let ok = matches(expect, got);
        (field, ok, got)
    });
    let absent = scenario.absent.iter().map(|field| {
        let got = args.get(field);
        (field, got.is_none(), got)
    });
    expected
        .chain(absent)
        .map(|(field, ok, got)| FieldResult {
            field: field.clone(),
            ok,
            got: got.cloned(),
        })
        .collect()
}

fn matches(expect: &Expect, got: Option<&Value>) -> bool {
    match (expect, got) {
        (Expect::Exact(want), Some(got)) => same(want, got),
        (Expect::Exact(_), None) => false,
        (Expect::Rule(rule), None) => rule.optional,
        (Expect::Rule(rule), Some(got)) => {
            let text = got.as_str().map(str::to_lowercase);
            let contains = |needle: &String| {
                text.as_deref()
                    .is_some_and(|t| t.contains(&needle.to_lowercase()))
            };
            rule.one_of
                .as_ref()
                .is_none_or(|values| values.iter().any(|v| same(v, got)))
                && rule
                    .contains_any
                    .as_ref()
                    .is_none_or(|needles| needles.iter().any(contains))
                && rule
                    .contains_all
                    .as_ref()
                    .is_none_or(|needles| needles.iter().all(contains))
        }
    }
}

/// Strings compare trimmed and case-insensitively, numbers numerically.
fn same(want: &Value, got: &Value) -> bool {
    match (want, got) {
        (Value::String(a), Value::String(b)) => a.trim().eq_ignore_ascii_case(b.trim()),
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() < 1e-9,
            _ => false,
        },
        _ => want == got,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scenario(toml_args: &str, absent: &[&str]) -> Scenario {
        let mut s: Scenario = toml::from_str(&format!(
            "id = \"x\"\nsplit = \"test\"\ntags = []\nrequest = \"r\"\ntool = \"search\"\nargs = {toml_args}\n"
        ))
        .unwrap();
        s.absent = absent.iter().map(|a| (*a).to_owned()).collect();
        s
    }

    fn decision(tool: Option<&str>, args: Value, errors: &[&str]) -> RouteDecision {
        RouteDecision {
            tool: tool.map(str::to_owned),
            args: args.as_object().cloned().unwrap_or_default(),
            validation_errors: errors.iter().map(|e| (*e).to_owned()).collect(),
            excluded: Vec::new(),
            latency_ms: 5,
            usage: Usage::default(),
            llm_calls: 1,
            ..RouteDecision::default()
        }
    }

    #[test]
    fn matchers() {
        let s = scenario(
            r#"{ query = { contains_any = ["soccer boot", "football boot"] }, category = { one_of = ["SOCCER"], optional = true }, max_price = 80 }"#,
            &["size"],
        );
        let r = score(
            &s,
            &decision(
                Some("search"),
                json!({ "query": "Soccer Boots", "max_price": 80.0 }),
                &[],
            ),
        );
        assert!(r.exact, "{r:?}");
        assert_eq!(r.fields.len(), 4);

        let r = score(
            &s,
            &decision(
                Some("search"),
                json!({ "query": "cleats", "category": "RUNNING", "max_price": 80, "size": "ALL" }),
                &[],
            ),
        );
        assert!(r.tool_ok && !r.exact);
        assert_eq!(r.fields_ok(), 1);
    }

    #[test]
    fn schema_errors_break_exact_match_but_not_tool_accuracy() {
        let s = scenario("{ max_price = 80 }", &[]);
        let r = score(
            &s,
            &decision(
                Some("search"),
                json!({ "max_price": 80 }),
                &["missing query"],
            ),
        );
        assert!(r.tool_ok && r.act_ok);
        assert_eq!(r.schema_valid, Some(false));
        assert!(!r.exact);
    }

    #[test]
    fn wrong_or_missing_tool() {
        let s = scenario("{ max_price = 80 }", &[]);
        let r = score(&s, &decision(None, json!({}), &[]));
        assert!(!r.act_ok && !r.tool_ok && !r.exact);
        assert_eq!(r.schema_valid, None);
        assert!(r.fields.is_empty());

        let mut none = s.clone();
        none.tool = ExpectedTool::One("none".into());
        none.args.clear();
        assert!(score(&none, &decision(None, json!({}), &[])).exact);
    }
}
