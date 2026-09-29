//! Paired comparison of the hybrid routers against the baseline:
//! the gate is calibrated on the calibration split, applied post hoc (an
//! escalated scenario takes the baseline's answer and pays for both), and
//! every router is measured on the test split with McNemar and a paired
//! bootstrap, per-stage accuracy and risk-coverage.

use std::collections::BTreeMap;
use std::fmt::Write;

use super::dataset::Split;
use super::report::{metrics, opt, pct};
use super::score::ScenarioResult;
use super::stats::{RESAMPLES, SEED, bootstrap_ratio, mcnemar_exact, percentile};
use crate::router::hybrid::{FieldSource, Gate, Signals, Verdict};

/// Jev's documented floor, and the lowest threshold calibration returns.
const FLOOR: f64 = 0.5;
const GRID: f64 = 100.0;
const ESCALATED: &str = "+escalated";

/// One router's results over the whole dataset.
pub struct Variant {
    pub name: String,
    pub results: Vec<ScenarioResult>,
}

/// The request asks for a tool action, even if a required value is missing.
pub fn act_label(r: &ScenarioResult) -> bool {
    r.expected_tool.expects_call() || r.tags.iter().any(|t| t == "missing-required")
}

/// Signals of a raw hybrid decision the gate can judge; `None` for errors
/// and for decisions that escalated on their own.
fn signals(r: &ScenarioResult) -> Option<Signals> {
    let trace = r.hybrid.as_ref()?;
    if r.error.is_some() || trace.escalation.is_some() {
        return None;
    }
    trace.signals
}

/// The smallest grid value strictly above `x`.
fn above(x: f64) -> f64 {
    ((x * GRID).floor() + 1.0) / GRID
}

/// The largest grid value not above `x`.
fn at_most(x: f64) -> f64 {
    (x * GRID).floor() / GRID
}

/// A closed judgement (Jev or parser, including an omission) got a field wrong.
fn closed_field_error(r: &ScenarioResult) -> bool {
    let Some(trace) = &r.hybrid else {
        return false;
    };
    r.fields.iter().filter(|f| !f.ok).any(|f| {
        trace
            .fields
            .iter()
            .find(|d| d.field == f.field)
            .is_some_and(|d| d.source != FieldSource::Llm)
    })
}

/// Thresholds that make no gate-accepted mistake on the calibration split,
/// never below Jev's 0.5 floor:
/// - `act_yes` above every `act` of a request that wants no tool;
/// - `act_no` at most the lowest `act` of a request that wants one;
/// - `tool` above every confidence of a wrong tool choice;
/// - `args` above every signal of a correct tool with a wrong closed field.
pub fn calibrate(results: &[ScenarioResult]) -> Gate {
    let mut gate = Gate {
        act_yes: FLOOR,
        act_no: FLOOR,
        tool: FLOOR,
        args: FLOOR,
    };
    for r in results.iter().filter(|r| r.split == Split::Calibration) {
        let Some(s) = signals(r) else {
            continue;
        };
        if act_label(r) {
            gate.act_no = gate.act_no.min(at_most(s.act));
        } else {
            gate.act_yes = gate.act_yes.max(above(s.act));
        }
        if !r.expected_tool.expects_call() {
            continue;
        }
        let chosen = r.hybrid.as_ref().and_then(|t| t.tool.as_deref());
        if !r.expected_tool.accepts(chosen) {
            gate.tool = gate.tool.max(above(s.tool));
        } else if closed_field_error(r) {
            gate.args = gate.args.max(above(s.args));
        }
    }
    gate
}

/// A raw hybrid result as the gate would have served it.
pub fn apply_gate(
    gate: &Gate,
    hybrid: &ScenarioResult,
    baseline: &ScenarioResult,
) -> ScenarioResult {
    let Some(s) = signals(hybrid) else {
        return hybrid.clone();
    };
    let verdict = gate.decide(&s);
    let mut trace = hybrid.hybrid.clone();
    if let Some(t) = trace.as_mut() {
        t.verdict = Some(verdict);
    }
    match verdict {
        Verdict::Act => ScenarioResult {
            hybrid: trace,
            ..hybrid.clone()
        },
        Verdict::NoTool => ScenarioResult {
            predicted_tool: None,
            args: Default::default(),
            validation_errors: Vec::new(),
            fields: Vec::new(),
            act_ok: !hybrid.expected_tool.expects_call(),
            tool_ok: hybrid.expected_tool.accepts(None),
            schema_valid: None,
            exact: hybrid.expected_tool.accepts(None),
            hybrid: trace,
            ..hybrid.clone()
        },
        Verdict::Escalate => {
            let mut usage = baseline.usage;
            usage += hybrid.usage;
            ScenarioResult {
                source: format!("{}{ESCALATED}", hybrid.source),
                latency_ms: baseline.latency_ms + hybrid.latency_ms,
                usage,
                llm_calls: baseline.llm_calls + hybrid.llm_calls,
                jev_usage: hybrid.jev_usage,
                jev_calls: hybrid.jev_calls,
                local_calls: hybrid.local_calls,
                hybrid: trace.map(|t| crate::router::hybrid::HybridTrace {
                    escalation: Some("below the gate thresholds".to_owned()),
                    ..t
                }),
                ..baseline.clone()
            }
        }
    }
}

/// Gated results, paired with the baseline by scenario id.
pub fn gated(
    gate: &Gate,
    hybrid: &[ScenarioResult],
    baseline: &[ScenarioResult],
) -> Vec<ScenarioResult> {
    let by_id: BTreeMap<&str, &ScenarioResult> =
        baseline.iter().map(|r| (r.id.as_str(), r)).collect();
    hybrid
        .iter()
        .map(|h| match by_id.get(h.id.as_str()) {
            Some(b) => apply_gate(gate, h, b),
            None => h.clone(),
        })
        .collect()
}

fn test_split(results: &[ScenarioResult]) -> Vec<ScenarioResult> {
    results
        .iter()
        .filter(|r| r.split == Split::Test)
        .cloned()
        .collect()
}

/// A raw result's confidence in its own decision: in acting, the weakest
/// signal; in not acting, how sure `act` is of it.
fn confidence(s: &Signals) -> f64 {
    if s.act >= FLOOR {
        s.act.min(s.tool).min(s.args)
    } else {
        1.0 - s.act
    }
}

/// Who answered the hybrid routers' closed questions, with its model.
pub enum Decider {
    Jev(String),
    Local(String),
}

pub struct Comparison {
    pub llm_model: String,
    pub decider: Decider,
    pub baseline: Variant,
    /// Raw (ungated) hybrid variants with their calibrated gates.
    pub hybrids: Vec<(Variant, Gate)>,
}

impl Comparison {
    pub fn render_markdown(&self) -> String {
        let mut out = String::new();
        let all = &self.baseline.results;
        let count = |split: Split| all.iter().filter(|r| r.split == split).count();
        let (title, decider, decider_model) = match &self.decider {
            Decider::Jev(model) => ("hybrid vs baseline (M4)", "Jev", model),
            Decider::Local(model) => ("local router vs baseline", "local model", model),
        };
        let _ = writeln!(out, "# Routing report: {title}\n");
        let _ = writeln!(
            out,
            "- Models: LLM `{}`, {decider} `{decider_model}`",
            self.llm_model
        );
        let _ = writeln!(
            out,
            "- Dataset: {} scenarios; gates calibrated on the {} calibration scenarios, every table below on the {} test scenarios",
            all.len(),
            count(Split::Calibration),
            count(Split::Test)
        );
        let _ = writeln!(
            out,
            "- Gated: an escalated scenario takes the baseline's answer and pays for both routers (latency summed: the calls run in sequence)"
        );
        let _ = writeln!(
            out,
            "- Intervals: 95% percentile bootstrap over scenarios, {RESAMPLES} resamples, seed {SEED:#x}; McNemar exact two-sided\n"
        );

        let _ = writeln!(out, "## Calibrated gates\n");
        let _ = writeln!(out, "| Router | act_yes | act_no | tool | args |");
        let _ = writeln!(out, "|---|---:|---:|---:|---:|");
        for (v, g) in &self.hybrids {
            let _ = writeln!(
                out,
                "| {} | {:.2} | {:.2} | {:.2} | {:.2} |",
                v.name, g.act_yes, g.act_no, g.tool, g.args
            );
        }

        let baseline = test_split(&self.baseline.results);
        let mut rows: Vec<(String, Vec<ScenarioResult>)> =
            vec![(self.baseline.name.clone(), baseline.clone())];
        for (v, g) in &self.hybrids {
            rows.push((format!("{} (raw)", v.name), test_split(&v.results)));
            let gated = gated(g, &v.results, &self.baseline.results);
            rows.push((format!("{} (gated)", v.name), test_split(&gated)));
        }

        let _ = writeln!(out, "\n## Accuracy\n");
        let _ = writeln!(
            out,
            "| Router | Exact (95% CI) | Act/no-act | Tool | Fields | Schema-valid |"
        );
        let _ = writeln!(out, "|---|---|---:|---:|---:|---:|");
        for (name, rs) in &rows {
            let m = metrics(rs);
            let value = |i: usize| m[i].1.map_or_else(|| "–".to_owned(), |e| pct(e.value));
            let exact = m[4].1.map_or_else(
                || "–".to_owned(),
                |e| format!("{} ({} – {})", pct(e.value), pct(e.low), pct(e.high)),
            );
            let _ = writeln!(
                out,
                "| {name} | {exact} | {} | {} | {} | {} |",
                value(0),
                value(1),
                value(2),
                value(3)
            );
        }

        let _ = writeln!(out, "\n## Cost and latency (per request)\n");
        let jev = matches!(self.decider, Decider::Jev(_));
        let _ = if jev {
            writeln!(
                out,
                "| Router | p50 / p95 ms | LLM calls | LLM tokens in / out | Jev calls | Jev tokens in / out | Escalated | Errors |\n|---|---|---:|---|---:|---|---:|---:|"
            )
        } else {
            writeln!(
                out,
                "| Router | p50 / p95 ms | LLM calls | LLM tokens in / out | Local model calls | Escalated | Errors |\n|---|---|---:|---|---:|---:|---:|"
            )
        };
        for (name, rs) in &rows {
            let n = rs.len().max(1) as f64;
            let latencies: Vec<u64> = rs
                .iter()
                .filter(|r| r.error.is_none())
                .map(|r| r.latency_ms)
                .collect();
            let mean =
                |f: &dyn Fn(&ScenarioResult) -> u64| rs.iter().map(f).sum::<u64>() as f64 / n;
            let escalated = rs.iter().filter(|r| r.source.ends_with(ESCALATED)).count();
            let errors = rs.iter().filter(|r| r.error.is_some()).count();
            let decider_cells = if jev {
                format!(
                    "{:.2} | {:.0} / {:.0}",
                    mean(&|r| u64::from(r.jev_calls)),
                    mean(&|r| r.jev_usage.input_tokens),
                    mean(&|r| r.jev_usage.output_tokens)
                )
            } else {
                format!("{:.2}", mean(&|r| u64::from(r.local_calls)))
            };
            let _ = writeln!(
                out,
                "| {name} | {} / {} | {:.2} | {:.0} / {:.0} | {decider_cells} | {} | {errors} |",
                opt(percentile(&latencies, 0.5)),
                opt(percentile(&latencies, 0.95)),
                mean(&|r| u64::from(r.llm_calls)),
                mean(&|r| r.usage.input_tokens),
                mean(&|r| r.usage.output_tokens),
                pct(escalated as f64 / n),
            );
        }

        let _ = writeln!(out, "\n## Paired against the baseline (exact invocation)\n");
        let _ = writeln!(
            out,
            "| Router | Both right | Only baseline | Only router | Both wrong | Δ exact (95% CI) | McNemar p |"
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---|---:|");
        let base: BTreeMap<&str, bool> =
            baseline.iter().map(|r| (r.id.as_str(), r.exact)).collect();
        for (name, rs) in rows.iter().skip(1) {
            let pairs: Vec<(bool, bool)> = rs
                .iter()
                .filter_map(|r| base.get(r.id.as_str()).map(|&b| (b, r.exact)))
                .collect();
            let cell = |b: bool, v: bool| pairs.iter().filter(|p| **p == (b, v)).count();
            let units: Vec<(f64, f64)> = pairs
                .iter()
                .map(|&(b, v)| (f64::from(v) - f64::from(b), 1.0))
                .collect();
            let delta = bootstrap_ratio(&units, RESAMPLES, SEED).map_or_else(
                || "–".to_owned(),
                |e| {
                    format!(
                        "{} ({} – {})",
                        signed(e.value),
                        signed(e.low),
                        signed(e.high)
                    )
                },
            );
            let _ = writeln!(
                out,
                "| {name} | {} | {} | {} | {} | {delta} | {:.3} |",
                cell(true, true),
                cell(true, false),
                cell(false, true),
                cell(false, false),
                mcnemar_exact(cell(true, false), cell(false, true))
            );
        }

        let _ = writeln!(out, "\n## Per stage (raw hybrid)\n");
        let _ = writeln!(
            out,
            "| Router | Stage 1 act | Stage 1 tool | Closed fields ({}/parser) | LLM fields |",
            if jev { "Jev" } else { "local" }
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|");
        for (v, _) in &self.hybrids {
            let rs = test_split(&v.results);
            let _ = writeln!(out, "| {} | {} |", v.name, stages(&rs).join(" | "));
        }

        let _ = writeln!(out, "\n## Risk-coverage (raw hybrid, exploratory)\n");
        let _ = writeln!(
            out,
            "Scenarios ranked by the router's confidence in its own decision (acting: min of act, tool and args signals; not acting: 1 − act). Risk: share of non-exact decisions among the covered ones. Coverage points are rounded up to whole scenarios.\n"
        );
        let steps = [0.25, 0.5, 0.75, 1.0];
        let header: Vec<String> = steps.iter().map(|c| format!("{:.0}%", c * 100.0)).collect();
        let _ = writeln!(out, "| Router | {} |", header.join(" | "));
        let _ = writeln!(out, "|---|{}", "---:|".repeat(steps.len()));
        for (v, _) in &self.hybrids {
            let risks = risk_coverage(&test_split(&v.results), &steps);
            let cells: Vec<String> = risks
                .iter()
                .map(|r| r.map_or_else(|| "–".to_owned(), pct))
                .collect();
            let _ = writeln!(out, "| {} | {} |", v.name, cells.join(" | "));
        }
        out
    }
}

fn signed(v: f64) -> String {
    format!("{:+.1} pp", v * 100.0)
}

fn ratio(ok: usize, n: usize) -> String {
    if n == 0 {
        "–".to_owned()
    } else {
        format!("{} ({ok}/{n})", pct(ok as f64 / n as f64))
    }
}

/// Stage 1 act and tool accuracy, and field accuracy by who decided.
fn stages(results: &[ScenarioResult]) -> Vec<String> {
    let judged: Vec<(&ScenarioResult, Signals)> = results
        .iter()
        .filter_map(|r| signals(r).map(|s| (r, s)))
        .collect();
    let act_ok = judged
        .iter()
        .filter(|(r, s)| (s.act >= FLOOR) == act_label(r))
        .count();
    let calls: Vec<&ScenarioResult> = judged
        .iter()
        .map(|(r, _)| *r)
        .filter(|r| r.expected_tool.expects_call())
        .collect();
    let tool_ok = calls
        .iter()
        .filter(|r| {
            let chosen = r.hybrid.as_ref().and_then(|t| t.tool.as_deref());
            r.expected_tool.accepts(chosen)
        })
        .count();
    let (mut closed, mut llm) = ((0, 0), (0, 0));
    for (r, _) in &judged {
        let Some(trace) = &r.hybrid else { continue };
        for f in &r.fields {
            let source = trace
                .fields
                .iter()
                .find(|d| d.field == f.field)
                .map(|d| d.source);
            let bucket = match source {
                Some(FieldSource::Llm) => &mut llm,
                Some(_) => &mut closed,
                None => continue,
            };
            bucket.0 += usize::from(f.ok);
            bucket.1 += 1;
        }
    }
    vec![
        ratio(act_ok, judged.len()),
        ratio(tool_ok, calls.len()),
        ratio(closed.0, closed.1),
        ratio(llm.0, llm.1),
    ]
}

/// Risk at each coverage level, over the results the gate could judge.
fn risk_coverage(results: &[ScenarioResult], coverage: &[f64]) -> Vec<Option<f64>> {
    let mut ranked: Vec<(f64, bool)> = results
        .iter()
        .filter_map(|r| signals(r).map(|s| (confidence(&s), r.exact)))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    coverage
        .iter()
        .map(|c| {
            let k = (c * ranked.len() as f64).ceil() as usize;
            (k > 0).then(|| {
                let wrong = ranked[..k].iter().filter(|(_, exact)| !exact).count();
                wrong as f64 / k as f64
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dataset::ExpectedTool;
    use crate::eval::score::FieldResult;
    use crate::jev::JevUsage;
    use crate::llm::Usage;
    use crate::router::hybrid::{FieldDecision, HybridTrace, Mode};
    use serde_json::json;

    fn expected(tool: Option<&str>) -> ExpectedTool {
        let name = tool.unwrap_or("none");
        serde_json::from_value(json!(name)).unwrap()
    }

    /// A raw hybrid result. `chosen` is the stage-1 choice; the decision acts
    /// on it when `act >= 0.5`.
    fn hybrid(
        id: &str,
        split: Split,
        want: Option<&str>,
        chosen: &str,
        s: Signals,
        field_ok: bool,
    ) -> ScenarioResult {
        let acts = s.act >= FLOOR;
        let expected_tool = expected(want);
        let tool_ok = expected_tool.accepts(acts.then_some(chosen));
        let fields = if acts && tool_ok && want.is_some() {
            vec![FieldResult {
                field: "category".into(),
                ok: field_ok,
                got: Some(json!("SOCCER")),
            }]
        } else {
            Vec::new()
        };
        ScenarioResult {
            id: id.into(),
            source: "jev-hybrid".into(),
            toolset: "alpha".into(),
            split,
            tags: Vec::new(),
            request: "r".into(),
            act_ok: expected_tool.expects_call() == acts,
            exact: tool_ok && fields.iter().all(|f| f.ok),
            expected_tool,
            predicted_tool: acts.then(|| chosen.to_owned()),
            args: Default::default(),
            validation_errors: Vec::new(),
            fields,
            tool_ok,
            schema_valid: acts.then_some(true),
            latency_ms: 300,
            usage: Usage {
                input_tokens: 50,
                ..Usage::default()
            },
            llm_calls: 1,
            jev_usage: JevUsage {
                input_tokens: 400,
                output_tokens: 8,
            },
            jev_calls: 2,
            local_calls: 0,
            hybrid: Some(HybridTrace {
                mode: Mode::Staged,
                signals: Some(s),
                tool: Some(chosen.into()),
                tool_probabilities: Default::default(),
                fields: vec![FieldDecision {
                    field: "category".into(),
                    source: FieldSource::JevChoice,
                    value: Some(json!("SOCCER")),
                    confidence: Some(s.args),
                    candidates: Vec::new(),
                }],
                verdict: None,
                escalation: None,
            }),
            error: None,
        }
    }

    fn sig(act: f64, tool: f64, args: f64) -> Signals {
        Signals { act, tool, args }
    }

    fn calibration_set() -> Vec<ScenarioResult> {
        use Split::{Calibration as C, Test as T};
        vec![
            hybrid(
                "ok",
                C,
                Some("search"),
                "search",
                sig(0.98, 0.97, 0.96),
                true,
            ),
            // Wants no tool, act 0.713: act_yes must go above it.
            hybrid("chat", C, None, "search", sig(0.713, 0.9, 1.0), true),
            // Wants a tool, act 0.42: act_no at most 0.42.
            hybrid(
                "shy",
                C,
                Some("search"),
                "search",
                sig(0.42, 0.9, 1.0),
                true,
            ),
            // Wrong tool at 0.8, wrong closed field at 0.66.
            hybrid("tool", C, Some("search"), "get", sig(0.95, 0.8, 1.0), true),
            hybrid(
                "field",
                C,
                Some("search"),
                "search",
                sig(0.95, 0.95, 0.66),
                false,
            ),
            // The test split never moves the thresholds.
            hybrid("test", T, None, "search", sig(0.99, 0.99, 1.0), true),
        ]
    }

    #[test]
    fn calibration_puts_thresholds_just_past_every_mistake() {
        let gate = calibrate(&calibration_set());
        assert_eq!(
            gate,
            Gate {
                act_yes: 0.72,
                act_no: 0.42,
                tool: 0.81,
                args: 0.67
            }
        );
        // Nothing wrong: the floors hold.
        let gate = calibrate(&calibration_set()[..1]);
        assert_eq!(
            (gate.act_yes, gate.act_no, gate.tool, gate.args),
            (0.5, 0.5, 0.5, 0.5)
        );
    }

    #[test]
    fn gate_applied_post_hoc() {
        let set = calibration_set();
        let gate = calibrate(&set);
        let baseline: Vec<ScenarioResult> = set
            .iter()
            .map(|h| ScenarioResult {
                source: "llm-baseline".into(),
                hybrid: None,
                jev_calls: 0,
                jev_usage: JevUsage::default(),
                latency_ms: 500,
                exact: true,
                ..h.clone()
            })
            .collect();
        let out = gated(&gate, &set, &baseline);
        // Confident and right: kept.
        assert_eq!(out[0].source, "jev-hybrid");
        assert_eq!(out[0].hybrid.as_ref().unwrap().verdict, Some(Verdict::Act));
        // Low field confidence: the baseline answers, both are paid for.
        let escalated = &out[4];
        assert_eq!(escalated.source, "jev-hybrid+escalated");
        assert!(escalated.exact);
        assert_eq!(
            (
                escalated.latency_ms,
                escalated.llm_calls,
                escalated.jev_calls
            ),
            (800, 2, 2)
        );
        assert_eq!(escalated.usage.input_tokens, 100);

        let no = Gate {
            act_no: 0.8,
            ..gate
        };
        let out = gated(&no, &set[1..2], &baseline[1..2]);
        assert_eq!(out[0].predicted_tool, None);
        assert!(out[0].exact && out[0].act_ok);
    }

    #[test]
    fn stages_and_risk_coverage() {
        let set = calibration_set();
        // act: ok, chat (act yes but wants none), shy (act no but wants one),
        // tool, field, test (wants none) → 3/6 right.
        assert_eq!(stages(&set)[0], "50.0% (3/6)");
        // Stage-1 tool over the four that want a call: "tool" is wrong.
        assert_eq!(stages(&set)[1], "75.0% (3/4)");
        assert_eq!(stages(&set)[2], "50.0% (1/2)");
        let risks = risk_coverage(&set, &[0.25, 1.0]);
        // Top 2 by confidence: "test" (0.99, wrong) and "ok" (0.96); only
        // "ok" is exact overall.
        assert_eq!(risks, [Some(0.5), Some(5.0 / 6.0)]);
    }

    fn comparison(decider: Decider) -> String {
        let set = calibration_set();
        let baseline: Vec<ScenarioResult> = set
            .iter()
            .map(|h| ScenarioResult {
                source: "llm-baseline".into(),
                hybrid: None,
                ..h.clone()
            })
            .collect();
        let gate = calibrate(&set);
        Comparison {
            llm_model: "m".into(),
            decider,
            baseline: Variant {
                name: "baseline".into(),
                results: baseline,
            },
            hybrids: vec![(
                Variant {
                    name: "hybrid".into(),
                    results: set,
                },
                gate,
            )],
        }
        .render_markdown()
    }

    #[test]
    fn report_renders_every_section() {
        let report = comparison(Decider::Jev("jev-1.13.0".into()));
        for section in [
            "## Calibrated gates",
            "| hybrid | 0.72 | 0.42 | 0.81 | 0.67 |",
            "## Accuracy",
            "| hybrid (gated) |",
            "## Paired against the baseline",
            "## Per stage",
            "## Risk-coverage",
        ] {
            assert!(report.contains(section), "missing {section}:\n{report}");
        }
    }

    #[test]
    fn local_report_names_the_local_model_and_no_jev_columns() {
        let report = comparison(Decider::Local("e1".into()));
        assert!(report.starts_with("# Routing report: local router vs baseline"));
        assert!(report.contains("- Models: LLM `m`, local model `e1`"));
        assert!(report.contains("| Local model calls |"));
        assert!(report.contains("Closed fields (local/parser)"));
        assert!(!report.contains("Jev"), "{report}");
    }
}
