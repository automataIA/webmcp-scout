//! Markdown report and per-scenario JSONL. Both are functions of the results
//! only (no timestamps, no cache state), so a replay reproduces them exactly.

use std::collections::BTreeMap;
use std::fmt::Write;

use super::score::ScenarioResult;
use super::stats::{Estimate, RESAMPLES, SEED, bootstrap_ratio, percentile};

type GroupKey = fn(&ScenarioResult) -> Vec<String>;

pub struct ReportMeta {
    pub title: String,
    pub router: String,
    pub model: String,
}

pub fn baseline_meta(name: &str, model: &str) -> ReportMeta {
    ReportMeta {
        title: format!("Routing report: {name}"),
        router: "LLM-only baseline (one strict structured-output call picks the tool and every argument)".into(),
        model: model.into(),
    }
}

pub fn hybrid_meta(name: &str, llm_model: &str, jev_model: &str) -> ReportMeta {
    ReportMeta {
        title: format!("Routing report: {name}"),
        router: "Hybrid (Jev decides act/tool/closed arguments, code parses numbers, one strict LLM call fills free strings), ungated".into(),
        model: format!("{llm_model}, Jev `{jev_model}`"),
    }
}

pub fn local_meta(name: &str, llm_model: &str, local_model: &str) -> ReportMeta {
    ReportMeta {
        title: format!("Routing report: {name}"),
        router: "Local (the local decision model answers the fan-out questions: act/tool/closed arguments; code parses numbers, one strict LLM call fills free strings), ungated".into(),
        // The writer wraps the whole line in backticks.
        model: format!("{llm_model}`, local model `{local_model}"),
    }
}

pub fn render_jsonl(results: &[ScenarioResult]) -> String {
    results
        .iter()
        .map(|r| serde_json::to_string(r).expect("results serialize") + "\n")
        .collect()
}

pub fn render_markdown(meta: &ReportMeta, results: &[ScenarioResult]) -> String {
    let mut out = String::new();
    let count = |split: &str| results.iter().filter(|r| r.split.as_str() == split).count();
    let toolsets: Vec<&str> = {
        let mut t: Vec<&str> = results.iter().map(|r| r.toolset.as_str()).collect();
        t.dedup();
        t
    };
    let _ = writeln!(out, "# {}\n", meta.title);
    let _ = writeln!(out, "- Router: {}", meta.router);
    let _ = writeln!(out, "- Model: `{}`", meta.model);
    let _ = writeln!(
        out,
        "- Dataset: {} scenarios ({} calibration, {} test) over toolsets {}",
        results.len(),
        count("calibration"),
        count("test"),
        toolsets.join(", ")
    );
    let _ = writeln!(
        out,
        "- Intervals: 95% percentile bootstrap over scenarios, {RESAMPLES} resamples, seed {SEED:#x}"
    );
    let _ = writeln!(
        out,
        "- Latency: the successful API call as recorded (rate-limit waits excluded)\n"
    );

    let _ = writeln!(out, "## Overall\n");
    let _ = writeln!(out, "| Metric | Value | 95% CI | n |");
    let _ = writeln!(out, "|---|---:|---|---:|");
    for (name, estimate) in metrics(results) {
        let _ = match estimate {
            Some(e) => writeln!(
                out,
                "| {name} | {} | {} – {} | {} |",
                pct(e.value),
                pct(e.low),
                pct(e.high),
                e.n
            ),
            None => writeln!(out, "| {name} | – | – | 0 |"),
        };
    }
    let errors = results.iter().filter(|r| r.error.is_some()).count();
    let latencies: Vec<u64> = results
        .iter()
        .filter(|r| r.error.is_none())
        .map(|r| r.latency_ms)
        .collect();
    let answered = latencies.len().max(1) as f64;
    let sum = |f: fn(&ScenarioResult) -> u64| results.iter().map(f).sum::<u64>() as f64;
    let _ = writeln!(out);
    let _ = writeln!(out, "| Cost | Value |");
    let _ = writeln!(out, "|---|---:|");
    let _ = writeln!(out, "| Routing errors | {errors} |");
    let _ = writeln!(
        out,
        "| Latency p50 / p95 | {} ms / {} ms |",
        opt(percentile(&latencies, 0.5)),
        opt(percentile(&latencies, 0.95))
    );
    let _ = writeln!(
        out,
        "| LLM calls per request | {:.2} |",
        sum(|r| u64::from(r.llm_calls)) / answered
    );
    let _ = writeln!(
        out,
        "| Tokens per request: input / output / of which reasoning | {:.0} / {:.0} / {:.0} |",
        sum(|r| r.usage.input_tokens) / answered,
        sum(|r| r.usage.output_tokens) / answered,
        sum(|r| r.usage.reasoning_tokens) / answered
    );
    if results.iter().any(|r| r.jev_calls > 0) {
        let _ = writeln!(
            out,
            "| Jev calls per request | {:.2} |",
            sum(|r| u64::from(r.jev_calls)) / answered
        );
        let _ = writeln!(
            out,
            "| Jev tokens per request: input / output | {:.0} / {:.0} |",
            sum(|r| r.jev_usage.input_tokens) / answered,
            sum(|r| r.jev_usage.output_tokens) / answered
        );
    }
    if results.iter().any(|r| r.local_calls > 0) {
        let _ = writeln!(
            out,
            "| Local model calls per request | {:.2} |",
            sum(|r| u64::from(r.local_calls)) / answered
        );
    }

    let groups: [(&str, GroupKey); 3] = [
        ("split", |r| vec![r.split.as_str().to_owned()]),
        ("toolset", |r| vec![r.toolset.clone()]),
        ("tag", |r| r.tags.clone()),
    ];
    for (label, key) in groups {
        let mut by: BTreeMap<String, Vec<&ScenarioResult>> = BTreeMap::new();
        for r in results {
            for k in key(r) {
                by.entry(k).or_default().push(r);
            }
        }
        let _ = writeln!(out, "\n## By {label}\n");
        let _ = writeln!(
            out,
            "| {label} | n | Act/no-act | Tool | Fields | Schema-valid | Exact |"
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|");
        for (k, rs) in by {
            let owned: Vec<ScenarioResult> = rs.into_iter().cloned().collect();
            let values: Vec<String> = metrics(&owned)
                .into_iter()
                .map(|(_, e)| e.map_or_else(|| "–".to_owned(), |e| pct(e.value)))
                .collect();
            let _ = writeln!(out, "| {k} | {} | {} |", owned.len(), values.join(" | "));
        }
    }

    let _ = writeln!(out, "\n## Misses\n");
    let misses: Vec<&ScenarioResult> = results.iter().filter(|r| !r.exact).collect();
    if misses.is_empty() {
        let _ = writeln!(out, "None.");
    } else {
        let _ = writeln!(out, "| id | request | expected | predicted | problem |");
        let _ = writeln!(out, "|---|---|---|---|---|");
        for r in misses {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} |",
                r.id,
                cell(&r.request),
                r.expected_tool.names().join(" or "),
                r.predicted_tool.as_deref().unwrap_or("none"),
                cell(&problem(r))
            );
        }
    }
    out
}

pub(crate) fn metrics(results: &[ScenarioResult]) -> Vec<(&'static str, Option<Estimate>)> {
    let unit = |f: &dyn Fn(&ScenarioResult) -> (f64, f64)| {
        let units: Vec<(f64, f64)> = results.iter().map(f).collect();
        bootstrap_ratio(&units, RESAMPLES, SEED)
    };
    vec![
        ("Act/no-act accuracy", unit(&|r| (f64::from(r.act_ok), 1.0))),
        ("Tool accuracy", unit(&|r| (f64::from(r.tool_ok), 1.0))),
        (
            "Field accuracy (correct tool)",
            unit(&|r| (r.fields_ok() as f64, r.fields.len() as f64)),
        ),
        (
            "Schema validity (tool picked)",
            unit(&|r| match r.schema_valid {
                Some(valid) => (f64::from(valid), 1.0),
                None => (0.0, 0.0),
            }),
        ),
        ("Exact invocation", unit(&|r| (f64::from(r.exact), 1.0))),
    ]
}

fn problem(r: &ScenarioResult) -> String {
    if let Some(error) = &r.error {
        return format!("error: {error}");
    }
    if !r.tool_ok {
        return "wrong tool".to_owned();
    }
    let mut parts: Vec<String> = r
        .fields
        .iter()
        .filter(|f| !f.ok)
        .map(|f| match &f.got {
            Some(v) => format!("{} = {v}", f.field),
            None => format!("{} missing", f.field),
        })
        .collect();
    parts.extend(r.validation_errors.iter().map(|e| format!("invalid: {e}")));
    parts.join("; ")
}

pub(crate) fn pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

pub(crate) fn opt(v: Option<u64>) -> String {
    v.map_or_else(|| "–".to_owned(), |v| v.to_string())
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}
