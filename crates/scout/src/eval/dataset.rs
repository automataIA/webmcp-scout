//! Labeled routing scenarios (`evals/scenarios/*.toml`) and the tool
//! snapshots they run against (`evals/toolsets/<name>.json`).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::EvalError;
use crate::schema::{ParamKind, ToolSpec};
use crate::webmcp::Discovery;

pub const NO_TOOL: &str = "none";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Calibration,
    Test,
    /// Training data for a local decision model; never scored.
    Train,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Test => "test",
            Self::Train => "train",
        }
    }
}

/// Expected tool: one name (`"none"` for no call) or a set of acceptable ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExpectedTool {
    One(String),
    AnyOf(Vec<String>),
}

impl ExpectedTool {
    pub fn names(&self) -> Vec<&str> {
        match self {
            Self::One(name) => vec![name.as_str()],
            Self::AnyOf(names) => names.iter().map(String::as_str).collect(),
        }
    }

    pub fn expects_call(&self) -> bool {
        self.names() != [NO_TOOL]
    }

    pub fn accepts(&self, predicted: Option<&str>) -> bool {
        self.names().contains(&predicted.unwrap_or(NO_TOOL))
    }
}

/// Expectation on one argument. A plain value must match exactly (strings
/// case-insensitively, numbers numerically).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Expect {
    Rule(Rule),
    Exact(Value),
}

/// Every set condition must hold. `optional` also accepts an absent argument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains_any: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains_all: Option<Vec<String>>,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub id: String,
    pub split: Split,
    pub tags: Vec<String>,
    pub request: String,
    pub tool: ExpectedTool,
    /// Checked arguments; others are not scored.
    #[serde(default)]
    pub args: BTreeMap<String, Expect>,
    /// Arguments that must not be set.
    #[serde(default)]
    pub absent: Vec<String>,
    /// The tool a `missing-required` request asks for; it cannot be called
    /// without the missing value, so `tool` stays `"none"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_tool: Option<String>,
    #[serde(default)]
    pub toolset: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioFile {
    toolset: String,
    #[serde(rename = "scenario")]
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub toolsets: BTreeMap<String, Vec<ToolSpec>>,
    pub scenarios: Vec<Scenario>,
}

impl Dataset {
    /// Loads every `*.toml` in `scenarios_dir` (sorted by file name) and the
    /// toolsets they reference, then checks the labels against the tools.
    pub fn load(scenarios_dir: &Path, toolsets_dir: &Path) -> Result<Self, EvalError> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(scenarios_dir)
            .map_err(|e| EvalError::io(scenarios_dir, e))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "toml"))
            .collect();
        files.sort();

        let mut toolsets = BTreeMap::new();
        let mut scenarios = Vec::new();
        for path in files {
            let text = std::fs::read_to_string(&path).map_err(|e| EvalError::io(&path, e))?;
            let file: ScenarioFile =
                toml::from_str(&text).map_err(|e| EvalError::parse(&path, e))?;
            if !toolsets.contains_key(&file.toolset) {
                let tools = load_toolset(&toolsets_dir.join(format!("{}.json", file.toolset)))?;
                toolsets.insert(file.toolset.clone(), tools);
            }
            scenarios.extend(file.scenarios.into_iter().map(|s| Scenario {
                toolset: file.toolset.clone(),
                ..s
            }));
        }
        let dataset = Self {
            toolsets,
            scenarios,
        };
        dataset.check()?;
        Ok(dataset)
    }

    pub fn tools(&self, scenario: &Scenario) -> &[ToolSpec] {
        &self.toolsets[&scenario.toolset]
    }

    fn check(&self) -> Result<(), EvalError> {
        let mut problems = Vec::new();
        let mut ids = HashSet::new();
        for s in &self.scenarios {
            if !ids.insert(&s.id) {
                problems.push(format!("{}: duplicate id", s.id));
            }
            let tools = self.tools(s);
            for name in s.tool.names() {
                if name == NO_TOOL {
                    continue;
                }
                let Some(tool) = tools.iter().find(|t| t.name == name) else {
                    problems.push(format!("{}: unknown tool `{name}`", s.id));
                    continue;
                };
                for field in s.args.keys().chain(&s.absent) {
                    if !tool.params.iter().any(|p| &p.name == field) {
                        problems.push(format!("{}: `{name}` has no parameter `{field}`", s.id));
                    }
                }
                for (field, expect) in &s.args {
                    if let Some(param) = tool.params.iter().find(|p| &p.name == field)
                        && let Err(e) = check_expect(&param.kind, expect)
                    {
                        problems.push(format!("{}: `{field}` {e}", s.id));
                    }
                }
            }
            if !s.tool.expects_call() && !(s.args.is_empty() && s.absent.is_empty()) {
                problems.push(format!("{}: arguments on a no-tool scenario", s.id));
            }
            if let Some(name) = &s.intended_tool {
                if s.tool.expects_call() || !s.tags.iter().any(|t| t == "missing-required") {
                    problems.push(format!(
                        "{}: intended_tool without a missing-required no-tool label",
                        s.id
                    ));
                }
                if !tools.iter().any(|t| &t.name == name) {
                    problems.push(format!("{}: unknown intended tool `{name}`", s.id));
                }
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(EvalError::Invalid(problems.join("; ")))
        }
    }
}

fn load_toolset(path: &Path) -> Result<Vec<ToolSpec>, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|e| EvalError::io(path, e))?;
    let discovery: Discovery =
        serde_json::from_str(&text).map_err(|e| EvalError::parse(path, e))?;
    Ok(discovery.tools.iter().map(ToolSpec::from_tool).collect())
}

/// Expected values of closed parameters must be values the tool accepts.
fn check_expect(kind: &ParamKind, expect: &Expect) -> Result<(), String> {
    let values: Vec<&Value> = match expect {
        Expect::Exact(v) => vec![v],
        Expect::Rule(rule) => rule.one_of.iter().flatten().collect(),
    };
    for value in values {
        let ok = match kind {
            ParamKind::Enum { values } => values.contains(value),
            ParamKind::Number { .. } => value.is_number(),
            ParamKind::Bool => value.is_boolean(),
            ParamKind::FreeString { .. } => value.is_string(),
            ParamKind::EnumSet { .. } | ParamKind::Unsupported { .. } => true,
        };
        if !ok {
            return Err(format!("expects {value}, not accepted by the schema"));
        }
    }
    Ok(())
}
