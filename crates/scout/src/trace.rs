//! Structured run trace: one JSON object per step, with the
//! time elapsed since the run started. Written as JSONL.

use std::time::Instant;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::adapters::Money;
use crate::jev::JevUsage;
use crate::llm::Usage;
use crate::rank::{Constraints, Excluded};
use crate::router::hybrid::HybridTrace;
use crate::webmcp::RejectedTool;

/// Longest tool output kept verbatim in the trace.
pub const PREVIEW_CHARS: usize = 2000;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolRef {
    pub name: String,
    pub schema_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankedRef {
    pub source: String,
    pub id: String,
    pub name: String,
    pub price: Option<Money>,
    pub url: Option<String>,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The site is the record's `site`.
    RunStarted { url: String, request: String },
    ToolsDiscovered {
        url: String,
        origin: String,
        tools: Vec<ToolRef>,
        rejected: Vec<RejectedTool>,
    },
    /// Left out before routing: not allowlisted, or outside the schema subset.
    ToolExcluded { tool: String, reason: String },
    Routed {
        /// Which component decided: `llm-baseline`, `jev-hybrid`, `jev-fanout`,
        /// with `+escalated` when the hybrid router handed over to the baseline.
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
        #[serde(skip_serializing_if = "Option::is_none")]
        hybrid: Option<HybridTrace>,
    },
    /// Re-discovery right before execution confirmed the capability.
    PolicyChecked { tool: String, schema_hash: String },
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
        /// First [`PREVIEW_CHARS`] characters of the output.
        output_preview: Option<String>,
    },
    Adapted {
        adapter: Option<String>,
        kind: String,
        products: usize,
    },
    /// Products of every site after the hard filters and the score, best first.
    Ranked {
        constraints: Constraints,
        keywords: Vec<String>,
        kept: Vec<RankedRef>,
        excluded: Vec<Excluded>,
    },
    RunFinished {
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Record {
    pub t_ms: u64,
    /// Site the record belongs to; absent on the final `ranked` event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    #[serde(flatten)]
    pub event: Event,
}

#[derive(Debug)]
pub struct Trace {
    started: Instant,
    site: Option<String>,
    records: Vec<Record>,
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

impl Trace {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            site: None,
            records: Vec::new(),
        }
    }

    /// An empty trace on the same clock whose records are tagged with `site`,
    /// to be merged back with [`Trace::absorb`].
    pub fn child(&self, site: &str) -> Self {
        Self {
            started: self.started,
            site: Some(site.to_owned()),
            records: Vec::new(),
        }
    }

    /// Tags the records of an untagged trace with `site` from now on.
    pub(crate) fn tag(&mut self, site: &str) {
        self.site.get_or_insert_with(|| site.to_owned());
    }

    /// Moves the records of `children` into this trace, in time order.
    pub fn absorb(&mut self, children: impl IntoIterator<Item = Trace>) {
        for child in children {
            self.records.extend(child.records);
        }
        self.records.sort_by_key(|r| r.t_ms);
    }

    pub fn record(&mut self, event: Event) {
        let t_ms = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.records.push(Record {
            t_ms,
            site: self.site.clone(),
            event,
        });
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn to_jsonl(&self) -> String {
        self.records
            .iter()
            .map(|r| serde_json::to_string(r).expect("trace records serialize") + "\n")
            .collect()
    }
}

/// `text` cut to at most `max` characters.
pub fn preview(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_serialize_as_tagged_lines() {
        let mut trace = Trace::new();
        trace.record(Event::Blocked {
            tool: Some("buy".into()),
            reason: "not allowlisted".into(),
        });
        trace.record(Event::RunFinished {
            status: "blocked".into(),
            error: None,
        });
        let lines: Vec<Value> = trace
            .to_jsonl()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "blocked");
        assert_eq!(lines[0]["tool"], "buy");
        assert!(lines[0]["t_ms"].is_u64());
        assert!(lines[1].get("error").is_none());
    }

    /// Contract with the viewer: every event parses with `scout-types` and
    /// serializes back to the same JSON.
    #[test]
    fn every_event_round_trips_through_scout_types() {
        use crate::router::hybrid::{FieldDecision, FieldSource, Mode};

        let args: Map<String, Value> = serde_json::from_str(r#"{"query":"boots"}"#).unwrap();
        let hybrid = HybridTrace {
            mode: Mode::Staged,
            signals: None,
            tool: Some("search".into()),
            tool_probabilities: [("search".to_owned(), 0.9)].into(),
            fields: vec![FieldDecision {
                field: "query".into(),
                source: FieldSource::Llm,
                value: Some("boots".into()),
                confidence: None,
                candidates: vec![],
            }],
            verdict: None,
            escalation: None,
        };
        let events = vec![
            Event::RunStarted {
                url: "http://shop/".into(),
                request: "boots".into(),
            },
            Event::ToolsDiscovered {
                url: "http://shop/".into(),
                origin: "http://shop".into(),
                tools: vec![ToolRef {
                    name: "search".into(),
                    schema_hash: "sha256:00".into(),
                }],
                rejected: vec![RejectedTool {
                    name: "x".into(),
                    reason: "foreign origin".into(),
                }],
            },
            Event::ToolExcluded {
                tool: "buy".into(),
                reason: "not in the allowlist".into(),
            },
            Event::Routed {
                source: "jev-hybrid".into(),
                tool: Some("search".into()),
                args: args.clone(),
                validation_errors: vec![],
                latency_ms: 12,
                usage: Usage::default(),
                llm_calls: 1,
                jev_usage: JevUsage::default(),
                jev_calls: 2,
                cached: false,
                hybrid: Some(hybrid),
            },
            Event::Routed {
                source: "llm-baseline".into(),
                tool: None,
                args: Map::new(),
                validation_errors: vec!["missing query".into()],
                latency_ms: 3,
                usage: Usage::default(),
                llm_calls: 1,
                jev_usage: JevUsage::default(),
                jev_calls: 0,
                cached: true,
                hybrid: None,
            },
            Event::PolicyChecked {
                tool: "search".into(),
                schema_hash: "sha256:00".into(),
            },
            Event::Blocked {
                tool: None,
                reason: "no tool".into(),
            },
            Event::Executed {
                tool: "search".into(),
                args,
                elapsed_ms: 5,
                navigated: false,
                url_after: "http://shop/".into(),
                output_bytes: 2,
                output_preview: None,
            },
            Event::Adapted {
                adapter: Some("alpha.search_products.v1".into()),
                kind: "products".into(),
                products: 1,
            },
            Event::Ranked {
                constraints: Constraints {
                    max_price: Some(150.0),
                    currency: Some("USD".into()),
                    size: None,
                },
                keywords: vec!["boots".into()],
                kept: vec![RankedRef {
                    source: "alpha".into(),
                    id: "1".into(),
                    name: "Boots".into(),
                    price: Some(Money {
                        amount: 99.5,
                        currency: "USD".into(),
                    }),
                    url: None,
                    score: 0.5,
                }],
                excluded: vec![Excluded {
                    source: "beta".into(),
                    id: "2".into(),
                    reason: "out of stock".into(),
                }],
            },
            Event::RunFinished {
                status: "error".into(),
                error: Some("timed out".into()),
            },
        ];
        let mut trace = Trace::new();
        let mut child = trace.child("alpha");
        for event in events {
            child.record(event);
        }
        trace.absorb([child]);
        trace.record(Event::RunFinished {
            status: "done".into(),
            error: None,
        });

        let jsonl = trace.to_jsonl();
        let parsed = scout_types::parse_jsonl(&jsonl).expect("viewer parses the trace");
        assert_eq!(parsed.len(), 12);
        for (record, line) in parsed.iter().zip(jsonl.lines()) {
            let written: Value = serde_json::from_str(line).unwrap();
            assert_eq!(serde_json::to_value(record).unwrap(), written);
        }
    }

    #[test]
    fn preview_cuts_on_char_boundaries() {
        assert_eq!(preview("héllo", 2), "hé…");
        assert_eq!(preview("hi", 2), "hi");
    }
}
