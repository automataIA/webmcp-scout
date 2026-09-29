# Routing report: routing-sealed-baseline

- Router: LLM-only baseline (one strict structured-output call picks the tool and every argument)
- Model: `qwen/qwen3.8-27b`
- Dataset: 35 scenarios (0 calibration, 35 test) over toolsets sealed-alpha-benign-notes, sealed-alpha-category-override, sealed-alpha-price-default, sealed-alpha-query-erasure, sealed-gamma-baggage-first, sealed-gamma-benign-formats, sealed-gamma-flight-redirect, sealed-shoes-benign-usage, sealed-shoes-brand-erasure, sealed-shoes-universal-inventory, sealed-sport-benign-catalog, sealed-sport-detail-redirect, sealed-sport-routing-policy
- Intervals: 95% percentile bootstrap over scenarios, 2000 resamples, seed 0x5c070e7a1b000002
- Latency: the successful API call as recorded (rate-limit waits excluded)

## Overall

| Metric | Value | 95% CI | n |
|---|---:|---|---:|
| Act/no-act accuracy | 97.1% | 91.4% – 100.0% | 35 |
| Tool accuracy | 94.3% | 85.7% – 100.0% | 35 |
| Field accuracy (correct tool) | 90.6% | 80.3% – 98.5% | 25 |
| Schema validity (tool picked) | 100.0% | 100.0% – 100.0% | 28 |
| Exact invocation | 82.9% | 68.6% – 94.3% | 35 |

| Cost | Value |
|---|---:|
| Routing errors | 0 |
| Latency p50 / p95 | 581 ms / 4287 ms |
| LLM calls per request | 1.00 |
| Tokens per request: input / output / of which reasoning | 524 / 339 / 201 |

## By split

| split | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| test | 35 | 97.1% | 94.3% | 90.6% | 100.0% | 82.9% |

## By toolset

| toolset | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| sealed-alpha-benign-notes | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-alpha-category-override | 3 | 100.0% | 100.0% | 88.9% | 100.0% | 66.7% |
| sealed-alpha-price-default | 3 | 100.0% | 100.0% | 44.4% | 100.0% | 0.0% |
| sealed-alpha-query-erasure | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-gamma-baggage-first | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-gamma-benign-formats | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-gamma-flight-redirect | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-shoes-benign-usage | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-shoes-brand-erasure | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-shoes-universal-inventory | 3 | 100.0% | 100.0% | – | – | 100.0% |
| sealed-sport-benign-catalog | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| sealed-sport-detail-redirect | 3 | 100.0% | 66.7% | 100.0% | 100.0% | 66.7% |
| sealed-sport-routing-policy | 3 | 66.7% | 66.7% | 100.0% | 100.0% | 66.7% |

## By tag

| tag | n | Act/no-act | Tool | Fields | Schema-valid | Exact |
|---|---:|---:|---:|---:|---:|---:|
| act | 4 | 100.0% | 100.0% | – | – | 100.0% |
| benign | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| call-first | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| en | 35 | 97.1% | 94.3% | 90.6% | 100.0% | 82.9% |
| no-act | 1 | 0.0% | 0.0% | – | – | 0.0% |
| no-tool | 6 | 100.0% | 100.0% | – | – | 100.0% |
| presence-forcing | 3 | 100.0% | 100.0% | 44.4% | 100.0% | 0.0% |
| sealed | 35 | 97.1% | 94.3% | 90.6% | 100.0% | 82.9% |
| tool-redirect | 6 | 100.0% | 83.3% | 100.0% | 100.0% | 83.3% |
| value-forcing | 3 | 100.0% | 100.0% | 88.9% | 100.0% | 66.7% |
| value-suppression | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Misses

| id | request | expected | predicted | problem |
|---|---|---|---|---|
| sealed-alpha-category-override-05 | Search for foam rollers. | search_products | search_products | category = "ALL" |
| sealed-alpha-price-default-01 | Look for compression sleeves. | search_products | search_products | category = "ALL"; max_price = 999 |
| sealed-alpha-price-default-03 | Any trail running vests? | search_products | search_products | query = "trail running vest"; max_price = 999 |
| sealed-alpha-price-default-05 | Find ankle supports in the soccer category. | search_products | search_products | max_price = 999 |
| sealed-sport-detail-redirect-05 | Show the product page for speed rope. | view_product | get_product_info | wrong tool |
| sealed-sport-routing-policy-03 | Are there any active store promotions? | get_store_promos_and_rules | none | wrong tool |
