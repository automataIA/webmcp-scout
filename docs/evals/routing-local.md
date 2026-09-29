# Routing report: local router vs baseline

- Models: LLM `qwen/qwen3.8-27b`, local model `qwen3-4b-lora-t13`
- Dataset: 37 scenarios; gates calibrated on the 16 calibration scenarios, every table below on the 21 test scenarios
- Gated: an escalated scenario takes the baseline's answer and pays for both routers (latency summed: the calls run in sequence)
- Intervals: 95% percentile bootstrap over scenarios, 2000 resamples, seed 0x5c070e7a1b000002; McNemar exact two-sided

## Calibrated gates

| Router | act_yes | act_no | tool | args |
|---|---:|---:|---:|---:|
| local-fanout | 0.50 | 0.50 | 0.50 | 0.96 |

## Accuracy

| Router | Exact (95% CI) | Act/no-act | Tool | Fields | Schema-valid |
|---|---|---:|---:|---:|---:|
| llm-baseline | 95.2% (85.7% – 100.0%) | 100.0% | 95.2% | 100.0% | 100.0% |
| local-fanout (raw) | 100.0% (100.0% – 100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| local-fanout (gated) | 100.0% (100.0% – 100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |

## Cost and latency (per request)

| Router | p50 / p95 ms | LLM calls | LLM tokens in / out | Local model calls | Escalated | Errors |
|---|---|---:|---|---:|---:|---:|
| llm-baseline | 468 / 1233 | 1.00 | 488 / 165 | 0.00 | 0.0% | 0 |
| local-fanout (raw) | 697 / 1329 | 0.57 | 164 / 63 | 1.00 | 0.0% | 0 |
| local-fanout (gated) | 697 / 1329 | 0.62 | 181 / 68 | 1.00 | 4.8% | 0 |

## Paired against the baseline (exact invocation)

| Router | Both right | Only baseline | Only router | Both wrong | Δ exact (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---|---:|
| local-fanout (raw) | 20 | 0 | 1 | 0 | +4.8 pp (+0.0 pp – +14.3 pp) | 1.000 |
| local-fanout (gated) | 20 | 0 | 1 | 0 | +4.8 pp (+0.0 pp – +14.3 pp) | 1.000 |

## Per stage (raw hybrid)

| Router | Stage 1 act | Stage 1 tool | Closed fields (local/parser) | LLM fields |
|---|---:|---:|---:|---:|
| local-fanout | 100.0% (21/21) | 100.0% (13/13) | 100.0% (8/8) | 100.0% (11/11) |

## Risk-coverage (raw hybrid, exploratory)

Scenarios ranked by the router's confidence in its own decision (acting: min of act, tool and args signals; not acting: 1 − act). Risk: share of non-exact decisions among the covered ones. Coverage points are rounded up to whole scenarios.

| Router | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| local-fanout | 0.0% | 0.0% | 0.0% | 0.0% |
