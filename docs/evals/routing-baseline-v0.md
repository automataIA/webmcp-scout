# Routing report: routing-baseline-v0

- Router: LLM-only baseline (one strict structured-output call picks the tool and every argument)
- Model: `qwen/qwen3.8-27b`
- Dataset: 37 scenarios (16 calibration, 21 test) over toolsets alpha, gamma, shoes, sport
- Intervals: 95% percentile bootstrap over scenarios, 2000 resamples, seed 0x5c070e7a1b000002
- Latency: the successful API call as recorded (rate-limit waits excluded)

## Overall

| Metric | Value | 95% CI | n |
|---|---:|---|---:|
| Act/no-act accuracy | 100.0% | 100.0% – 100.0% | 37 |
| Tool accuracy | 97.3% | 91.9% – 100.0% | 37 |
| Field accuracy (correct tool) | 97.9% | 92.5% – 100.0% | 22 |
| Schema validity (tool picked) | 100.0% | 100.0% – 100.0% | 28 |
| Exact invocation | 94.6% | 86.5% – 100.0% | 37 |

| Cost | Value |
|---|---:|
| Routing errors | 0 |
| Latency p50 / p95 | 477 ms / 1233 ms |
| LLM calls per request | 1.00 |
| Tokens per request: input / output / of which reasoning | 496 / 159 / 130 |

## By split

| split | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 16 | 100.0% | 100.0% | 96.4% | 100.0% | 93.8% |
| test | 21 | 100.0% | 95.2% | 100.0% | 100.0% | 95.2% |

## By toolset

| toolset | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| alpha | 11 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| gamma | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| shoes | 9 | 100.0% | 100.0% | 75.0% | 100.0% | 88.9% |
| sport | 12 | 100.0% | 91.7% | 100.0% | 100.0% | 91.7% |

## By tag

| tag | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous | 1 | 100.0% | 0.0% | – | 100.0% | 0.0% |
| cart | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| detail | 4 | 100.0% | 75.0% | 100.0% | 100.0% | 75.0% |
| en | 37 | 100.0% | 97.3% | 97.9% | 100.0% | 94.6% |
| filter | 3 | 100.0% | 100.0% | 66.7% | 100.0% | 66.7% |
| info | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| list | 1 | 100.0% | 100.0% | – | 100.0% | 100.0% |
| missing-required | 3 | 100.0% | 100.0% | – | – | 100.0% |
| navigation | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| no-tool | 5 | 100.0% | 100.0% | – | – | 100.0% |
| numbers | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| search | 13 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| size | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| unsupported | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Misses

| id | request | expected | predicted | problem |
|---|---|---|---|---|
| shoes-05 | running shoes | filter_by_category | filter_by_category | category = "running shoes" |
| sport-16 | show me the Speedster running shoes | get_product_info or view_product | search_product | wrong tool |
