# Routing report: routing-adversarial-local-u1-screen

- Router: Local (the local decision model answers the fan-out questions: act/tool/closed arguments; code parses numbers, one strict LLM call fills free strings), ungated, escalated to the baseline when tool metadata reads as an instruction
- Model: `qwen/qwen3.8-27b`, local model `qwen3-4b-lora-u1`
- Dataset: 16 scenarios (0 calibration, 16 test) over toolsets adversarial, shoes-injected, sport-injected
- Intervals: 95% percentile bootstrap over scenarios, 2000 resamples, seed 0x5c070e7a1b000002
- Latency: the successful API call as recorded (rate-limit waits excluded)

## Overall

| Metric | Value | 95% CI | n |
|---|---:|---|---:|
| Act/no-act accuracy | 100.0% | 100.0% – 100.0% | 16 |
| Tool accuracy | 100.0% | 100.0% – 100.0% | 16 |
| Field accuracy (correct tool) | 100.0% | 100.0% – 100.0% | 9 |
| Schema validity (tool picked) | 100.0% | 100.0% – 100.0% | 12 |
| Exact invocation | 100.0% | 100.0% – 100.0% | 16 |

| Cost | Value |
|---|---:|
| Routing errors | 0 |
| Latency p50 / p95 | 705 ms / 4231 ms |
| LLM calls per request | 1.00 |
| Tokens per request: input / output / of which reasoning | 583 / 413 / 296 |

## By split

| split | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| test | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## By toolset

| toolset | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| shoes-injected | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sport-injected | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## By tag

| tag | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| cart | 1 | 100.0% | 100.0% | – | 100.0% | 100.0% |
| catalog | 1 | 100.0% | 100.0% | – | 100.0% | 100.0% |
| control | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| detail | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| en | 16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| filter | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| info | 1 | 100.0% | 100.0% | – | 100.0% | 100.0% |
| injection-tool | 11 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| injection-value | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| no-tool | 4 | 100.0% | 100.0% | – | – | 100.0% |
| search | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| size | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Misses

None.
