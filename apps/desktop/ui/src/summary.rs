//! What the viewer shows about a trace, kept free of UI code so it is tested
//! natively.

use scout_types::{Event, Record};
use serde_json::{Map, Value};

/// One site of a run, from its records.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SiteRun {
    pub site: String,
    pub url: String,
    pub status: Option<String>,
    pub error: Option<String>,
    pub router: Option<String>,
    pub tool: Option<String>,
    pub routing_ms: Option<u64>,
    pub tokens: u64,
    pub finished_ms: Option<u64>,
}

/// The request of the run.
pub fn request(records: &[Record]) -> Option<&str> {
    records.iter().find_map(|r| match &r.event {
        Event::RunStarted { request, .. } => Some(request.as_str()),
        _ => None,
    })
}

/// Time of the last record.
pub fn duration_ms(records: &[Record]) -> u64 {
    records.iter().map(|r| r.t_ms).max().unwrap_or(0)
}

/// Sites in order of appearance.
pub fn sites(records: &[Record]) -> Vec<SiteRun> {
    let mut sites: Vec<SiteRun> = Vec::new();
    for record in records {
        let Some(name) = &record.site else { continue };
        let index = match sites.iter().position(|s| &s.site == name) {
            Some(i) => i,
            None => {
                sites.push(SiteRun {
                    site: name.clone(),
                    ..SiteRun::default()
                });
                sites.len() - 1
            }
        };
        let site = &mut sites[index];
        match &record.event {
            Event::RunStarted { url, .. } => site.url = url.clone(),
            Event::Routed {
                source,
                tool,
                latency_ms,
                usage,
                jev_usage,
                ..
            } => {
                site.router = Some(source.clone());
                site.tool = tool.clone();
                site.routing_ms = Some(*latency_ms);
                site.tokens = usage.input_tokens
                    + usage.output_tokens
                    + jev_usage.input_tokens
                    + jev_usage.output_tokens;
            }
            Event::RunFinished { status, error } => {
                site.status = Some(status.clone());
                site.error = error.clone();
                site.finished_ms = Some(record.t_ms);
            }
            _ => {}
        }
    }
    sites
}

/// One line about an event.
pub fn describe(event: &Event) -> String {
    match event {
        Event::RunStarted { url, .. } => format!("open {url}"),
        Event::ToolsDiscovered {
            origin,
            tools,
            rejected,
            ..
        } => {
            let mut line = format!("{} tool(s) on {origin}", tools.len());
            if !rejected.is_empty() {
                line.push_str(&format!(", {} rejected", rejected.len()));
            }
            line
        }
        Event::ToolExcluded { tool, reason } => format!("{tool}: {reason}"),
        Event::Routed {
            source,
            tool,
            args,
            validation_errors,
            latency_ms,
            cached,
            ..
        } => {
            let decision = match tool {
                Some(tool) => format!("{tool}({})", compact(args)),
                None => "no tool".into(),
            };
            let mut line = format!("{source} → {decision} in {latency_ms} ms");
            if *cached {
                line.push_str(" (cached)");
            }
            if !validation_errors.is_empty() {
                line.push_str(&format!("; invalid: {}", validation_errors.join("; ")));
            }
            line
        }
        Event::PolicyChecked { tool, schema_hash } => {
            format!("{tool} still allowlisted ({})", short_hash(schema_hash))
        }
        Event::Blocked { tool, reason } => match tool {
            Some(tool) => format!("{tool}: {reason}"),
            None => reason.clone(),
        },
        Event::Executed {
            tool,
            elapsed_ms,
            output_bytes,
            navigated,
            url_after,
            ..
        } => {
            let mut line = format!("{tool} returned {output_bytes} bytes in {elapsed_ms} ms");
            if *navigated {
                line.push_str(&format!(", navigated to {url_after}"));
            }
            line
        }
        Event::Adapted {
            adapter,
            kind,
            products,
        } => match adapter {
            Some(adapter) => format!("{adapter} → {products} {kind}"),
            None => format!("no adapter → {kind}"),
        },
        Event::Ranked { kept, excluded, .. } => {
            format!(
                "{} product(s) kept, {} excluded",
                kept.len(),
                excluded.len()
            )
        }
        Event::RunFinished { status, error } => match error {
            Some(error) => format!("{status}: {error}"),
            None => status.clone(),
        },
    }
}

/// DaisyUI badge colour of an event.
pub fn tone(event: &Event) -> &'static str {
    match event {
        Event::Blocked { .. } => "badge-error",
        Event::ToolExcluded { .. } => "badge-warning",
        Event::PolicyChecked { .. } | Event::Executed { .. } => "badge-success",
        Event::Routed { .. } | Event::Ranked { .. } => "badge-info",
        Event::RunFinished { status, .. } => status_tone(status),
        _ => "badge-ghost",
    }
}

pub fn status_tone(status: &str) -> &'static str {
    match status {
        "done" => "badge-success",
        "no_tool" | "invalid" => "badge-warning",
        _ => "badge-error",
    }
}

fn compact(args: &Map<String, Value>) -> String {
    args.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn short_hash(hash: &str) -> &str {
    let hex = hash.strip_prefix("sha256:").unwrap_or(hash);
    hex.get(..12).unwrap_or(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Record> {
        scout_types::parse_jsonl(include_str!("../sample.jsonl")).unwrap()
    }

    #[test]
    fn summarizes_the_sample_run() {
        let records = sample();
        assert_eq!(request(&records), Some("soccer shoes under 150 dollars"));
        assert!(duration_ms(&records) > 0);
        let sites = sites(&records);
        let names: Vec<&str> = sites.iter().map(|s| s.site.as_str()).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"sport") && names.contains(&"shoes"));
        for site in &sites {
            assert_eq!(site.status.as_deref(), Some("done"), "{site:?}");
            assert_eq!(site.router.as_deref(), Some("llm-baseline"));
            assert!(site.tool.is_some() && site.tokens > 0 && !site.url.is_empty());
        }
    }

    #[test]
    fn describes_every_event_in_one_line() {
        for record in sample() {
            let line = describe(&record.event);
            assert!(!line.is_empty() && !line.contains('\n'), "{line:?}");
        }
    }

    #[test]
    fn describes_routing_and_blocks() {
        let routed: Event = serde_json::from_value(serde_json::json!({
            "event": "routed", "source": "jev-hybrid", "tool": "search",
            "args": {"query": "boots"}, "validation_errors": [], "latency_ms": 12,
            "usage": {"input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0},
            "llm_calls": 1, "jev_usage": {"input_tokens": 1, "output_tokens": 1},
            "jev_calls": 2, "cached": true
        }))
        .unwrap();
        assert_eq!(
            describe(&routed),
            r#"jev-hybrid → search(query="boots") in 12 ms (cached)"#
        );
        let blocked = Event::Blocked {
            tool: Some("search".into()),
            reason: "tool `search` is no longer registered".into(),
        };
        assert_eq!(
            describe(&blocked),
            "search: tool `search` is no longer registered"
        );
        assert_eq!(tone(&blocked), "badge-error");
        assert_eq!(short_hash("sha256:b1eaab9711f450db8ea4"), "b1eaab9711f4");
    }
}
