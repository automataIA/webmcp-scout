# Routing report: hybrid vs baseline (M4)

- Models: LLM `qwen/qwen3.8-27b`, Jev `jev-1.13.0`
- Dataset: 37 scenarios; gates calibrated on the 16 calibration scenarios, every table below on the 21 test scenarios
- Gated: an escalated scenario takes the baseline's answer and pays for both routers (latency summed: the calls run in sequence)
- Intervals: 95% percentile bootstrap over scenarios, 2000 resamples, seed 0x5c070e7a1b000002; McNemar exact two-sided

## Calibrated gates

| Router | act_yes | act_no | tool | args |
|---|---:|---:|---:|---:|
| jev-hybrid | 0.50 | 0.50 | 0.50 | 0.50 |
| jev-fanout | 0.50 | 0.50 | 0.50 | 0.50 |

## Accuracy

| Router | Exact (95% CI) | Act/no-act | Tool | Fields | Schema-valid |
|---|---|---:|---:|---:|---:|
| llm-baseline | 95.2% (85.7% – 100.0%) | 100.0% | 95.2% | 100.0% | 100.0% |
| jev-hybrid (raw) | 95.2% (85.7% – 100.0%) | 100.0% | 95.2% | 100.0% | 100.0% |
| jev-hybrid (gated) | 95.2% (85.7% – 100.0%) | 100.0% | 95.2% | 100.0% | 100.0% |
| jev-fanout (raw) | 95.2% (85.7% – 100.0%) | 100.0% | 95.2% | 100.0% | 100.0% |
| jev-fanout (gated) | 95.2% (85.7% – 100.0%) | 100.0% | 95.2% | 100.0% | 100.0% |

## Cost and latency (per request)

| Router | p50 / p95 ms | LLM calls | LLM tokens in / out | Jev calls | Jev tokens in / out | Escalated | Errors |
|---|---|---:|---|---:|---|---:|---:|
| llm-baseline | 468 / 1233 | 1.00 | 488 / 165 | 0.00 | 0 / 0 | 0.0% | 0 |
| jev-hybrid (raw) | 577 / 1272 | 0.57 | 166 / 67 | 1.38 | 1052 / 116 | 0.0% | 0 |
| jev-hybrid (gated) | 577 / 1695 | 0.62 | 192 / 82 | 1.38 | 1052 / 116 | 4.8% | 0 |
| jev-fanout (raw) | 566 / 1015 | 0.57 | 166 / 67 | 1.00 | 979 / 161 | 0.0% | 0 |
| jev-fanout (gated) | 566 / 1444 | 0.62 | 192 / 82 | 1.00 | 979 / 161 | 4.8% | 0 |

## Paired against the baseline (exact invocation)

| Router | Both right | Only baseline | Only router | Both wrong | Δ exact (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---|---:|
| jev-hybrid (raw) | 20 | 0 | 0 | 1 | +0.0 pp (+0.0 pp – +0.0 pp) | 1.000 |
| jev-hybrid (gated) | 20 | 0 | 0 | 1 | +0.0 pp (+0.0 pp – +0.0 pp) | 1.000 |
| jev-fanout (raw) | 20 | 0 | 0 | 1 | +0.0 pp (+0.0 pp – +0.0 pp) | 1.000 |
| jev-fanout (gated) | 20 | 0 | 0 | 1 | +0.0 pp (+0.0 pp – +0.0 pp) | 1.000 |

## Per stage (raw hybrid)

| Router | Stage 1 act | Stage 1 tool | Closed fields (Jev/parser) | LLM fields |
|---|---:|---:|---:|---:|
| jev-hybrid | 100.0% (21/21) | 92.3% (12/13) | 100.0% (8/8) | 100.0% (10/10) |
| jev-fanout | 100.0% (21/21) | 92.3% (12/13) | 100.0% (8/8) | 100.0% (10/10) |

## Risk-coverage (raw hybrid, exploratory)

Scenarios ranked by the router's confidence in its own decision (acting: min of act, tool and args signals; not acting: 1 − act). Risk: share of non-exact decisions among the covered ones. Coverage points are rounded up to whole scenarios.

| Router | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| jev-hybrid | 0.0% | 0.0% | 0.0% | 4.8% |
| jev-fanout | 0.0% | 0.0% | 0.0% | 4.8% |
