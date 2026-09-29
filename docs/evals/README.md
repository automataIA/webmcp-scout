# Evaluation reports

This folder holds the routing reports behind the numbers in the main [README](../../README.md). A report scores one router on a set of labelled requests: for each request, did it pick the right tool (or correctly pick none) and fill every argument right ("exact invocation")?

- `<name>.md`: the report for people (accuracy with 95% bootstrap intervals, latency, tokens, misses).
- `<name>.jsonl`: one line per request (request, expected and predicted tool, arguments, latency, tokens).

## Reports

| Report | Request set | Router |
|---|---|---|
| [routing-baseline-v0](routing-baseline-v0.md) | dataset v0, all 37 | LLM-only baseline |
| [routing-m4](routing-m4.md) | dataset v0, 21 test | baseline against the Jev hybrid, staged and fan-out (only the baseline JSONL is published) |
| [routing-local-u1](routing-local-u1.md) | dataset v0, 21 test | baseline against the local router `qwen3-4b-lora-u1` (current) |
| [routing-local](routing-local.md) | dataset v0, 21 test | baseline against the earlier local router `qwen3-4b-lora-t13` |
| [routing-adversarial-baseline](routing-adversarial-baseline.md) | adversarial, 16 | baseline |
| [routing-adversarial-local](routing-adversarial-local.md) | adversarial, 16 | local `t13` |
| [routing-adversarial-local-u1](routing-adversarial-local-u1.md) | adversarial, 16 | local `u1` |
| [routing-adversarial-local-u1-screen](routing-adversarial-local-u1-screen.md) | adversarial, 16 | local `u1` with the metadata screen |
| [routing-sealed-baseline](routing-sealed-baseline.md) | sealed, 35 | baseline |
| [routing-sealed-local](routing-sealed-local.md) | sealed, 35 | local `u1` |
| [routing-sealed-local-screen](routing-sealed-local-screen.md) | sealed, 35 | local `u1` with the metadata screen |

## Request sets (`evals/`, not published)

The labelled requests, the tool snapshots they run against and the recorded model answers live in `evals/`, which is kept out of the repository. What it holds:

| Path | Content |
|---|---|
| `evals/scenarios/` | Dataset v0: 37 English requests on four toolsets (`alpha` fixture shop, `gamma` flight search, Sport Shop, Shoe Store), split 16 for calibration and 21 for test |
| `evals/adversarial/` | 16 held-out test requests whose tool descriptions carry prompt injection (push a fixed value, a mandatory first tool or a pre-authorized purchase) |
| `evals/sealed/` | 35 held-out test requests on 9 attacked and 4 benign variants of the toolsets, written by a model of another family and scored once |
| `evals/train/` | 1,433 synthetic training requests (never scored): a teacher LLM writes groups of requests that differ in one decisive fact, labels them, relabels each one independently, and only the requests where both labels agree are kept |
| `evals/train-twins/` | 2,866 copies of the training requests on toolsets with injected descriptions, same labels |
| `evals/toolsets/` | The tool lists (names, descriptions, schemas) each set runs against: snapshots of the sites, plus their attacked variants |
| `evals/cache/` | Recorded LLM and local-model answers, keyed by request hash, so the reports replay offline without API keys |

The recorded Jev answers of the M4 comparison are not published at all, so the Jev part of `routing-m4` cannot be replayed.

## Reproducing a report

With a local copy of `evals/`:

```sh
# re-score from the recorded answers, no API keys needed
cargo run -p scout-cli -- eval routing --replay
cargo run -p scout-cli -- eval routing --replay --scenarios evals/adversarial --name routing-adversarial-baseline

# check that every committed report replays byte for byte
cargo test -p scout --features evals --test eval_replay
```

The training, twin and sealed sets were built by earlier versions of `scripts/gen_train.py`, `gen_twins.py` and `gen_sealed.py`; the current scripts write English requests only and do not rebuild those sets from the recorded answers.

Without `--replay` the commands call the models for answers missing from the cache and record them (keys in `.env`, see `.env.example`). `scout eval questions --scenarios evals/train --out train.jsonl` turns a set into the question sets the local router is trained on (`local_router/`).
