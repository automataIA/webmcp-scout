//! The run trace written by `scout ask` (one JSON object per line), as plain
//! serde types that also build for `wasm32-unknown-unknown`. The desktop
//! viewer reads traces with these; `scout` checks in its tests that every
//! event it writes parses back to the same JSON.
//!
//! Traces carry text from web pages and models (tool descriptions, outputs,
//! reasons): it is untrusted and must be displayed as text, never as markup.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One line of a trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Milliseconds since the run started.
    pub t_ms: u64,
    /// Site the record belongs to; absent on the final `ranked` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    #[serde(flatten)]
    pub event: Event,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    RunStarted {
        url: String,
        request: String,
    },
    ToolsDiscovered {
        url: String,
        origin: String,
        tools: Vec<ToolRef>,
        rejected: Vec<RejectedTool>,
    },
    ToolExcluded {
        tool: String,
        reason: String,
    },
    Routed {
        source: String,
        tool: Option<String>,
        args: Map<String, Value>,
        validation_errors: Vec<String>,
        latency_ms: u64,
        usage: Usage,
        llm_calls: u32,
        jev_usage: JevUsage,
        jev_calls: u32,
        cached: bool,
        /// Per-stage signals of the hybrid router, kept as JSON.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hybrid: Option<Value>,
    },
    PolicyChecked {
        tool: String,
        schema_hash: String,
    },
    Blocked {
        tool: Option<String>,
        reason: String,
    },
    Executed {
        tool: String,
        args: Map<String, Value>,
        elapsed_ms: u64,
        navigated: bool,
        url_after: String,
        output_bytes: usize,
        output_preview: Option<String>,
    },
    Adapted {
        adapter: Option<String>,
        kind: String,
        products: usize,
    },
    Ranked {
        constraints: Constraints,
        keywords: Vec<String>,
        kept: Vec<RankedRef>,
        excluded: Vec<Excluded>,
    },
    RunFinished {
        status: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

impl Event {
    /// The `event` tag, e.g. `run_started`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RunStarted { .. } => "run_started",
            Self::ToolsDiscovered { .. } => "tools_discovered",
            Self::ToolExcluded { .. } => "tool_excluded",
            Self::Routed { .. } => "routed",
            Self::PolicyChecked { .. } => "policy_checked",
            Self::Blocked { .. } => "blocked",
            Self::Executed { .. } => "executed",
            Self::Adapted { .. } => "adapted",
            Self::Ranked { .. } => "ranked",
            Self::RunFinished { .. } => "run_finished",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolRef {
    pub name: String,
    pub schema_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectedTool {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JevUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    pub max_price: Option<f64>,
    pub currency: Option<String>,
    pub size: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Money {
    pub amount: f64,
    pub currency: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedRef {
    pub source: String,
    pub id: String,
    pub name: String,
    pub price: Option<Money>,
    pub url: Option<String>,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Excluded {
    pub source: String,
    pub id: String,
    pub reason: String,
}

/// A trace line that is not a record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseError {
    /// 1-based.
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parses a JSONL trace; blank lines are skipped.
pub fn parse_jsonl(text: &str) -> Result<Vec<Record>, ParseError> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|e| ParseError {
                line: i + 1,
                message: e.to_string(),
            })
        })
        .collect()
}

/// A trace file offered by the viewer backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceFile {
    /// File name inside the traces directory.
    pub name: String,
    pub bytes: u64,
    /// Seconds since the Unix epoch.
    pub modified: u64,
}

/// Payload of the viewer's `run-finished` event: a `scout ask` it started
/// has exited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunFinished {
    /// Trace file name inside the traces directory.
    pub trace: String,
    pub success: bool,
    /// Exit code; `None` when the run was ended by a signal.
    pub code: Option<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_multi_site_trace() {
        let text = concat!(
            r#"{"t_ms":0,"site":"alpha","event":"run_started","url":"http://x/","request":"boots"}"#,
            "\n\n",
            r#"{"t_ms":9,"site":"alpha","event":"run_finished","status":"done"}"#,
            "\n",
            r#"{"t_ms":9,"event":"ranked","constraints":{"max_price":150.0,"currency":"USD","size":null},"keywords":["boots"],"kept":[{"source":"alpha","id":"1","name":"Boots","price":{"amount":99.0,"currency":"USD"},"url":null,"score":1.0}],"excluded":[]}"#,
            "\n",
        );
        let records = parse_jsonl(text).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].site.as_deref(), Some("alpha"));
        assert_eq!(records[1].event.name(), "run_finished");
        assert!(records[2].site.is_none());
        let Event::Ranked { kept, .. } = &records[2].event else {
            panic!("{:?}", records[2]);
        };
        assert_eq!(kept[0].price.as_ref().unwrap().amount, 99.0);
    }

    #[test]
    fn reports_the_line_of_a_bad_record() {
        let text = concat!(
            r#"{"t_ms":0,"event":"run_finished","status":"done"}"#,
            "\n",
            r#"{"t_ms":1,"event":"teleported"}"#,
        );
        let err = parse_jsonl(text).unwrap_err();
        assert_eq!(err.line, 2);
        assert!(err.message.contains("teleported"), "{err}");
    }

    #[test]
    fn a_site_given_twice_is_an_error() {
        // Traces written before the site moved out of `run_started`.
        let text =
            r#"{"t_ms":0,"site":"a","event":"run_started","site":"a","url":"u","request":"r"}"#;
        assert!(parse_jsonl(text).is_err());
    }
}
