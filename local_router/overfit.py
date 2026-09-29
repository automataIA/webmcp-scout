"""Overfitting checks for a trained run, after training: nothing here feeds back into it.

    uv run python overfit.py --run out/runs/<run> --train train.jsonl --scenarios m4.jsonl \
        --adversarial adversarial.jsonl

Scores the base model and the run's adapter on five tiers, from closest to farthest from the
training data: train (what the adapter saw), dev (held-out groups of the same generator),
calibration and test (the hand-written scenarios) and adversarial (injected tool metadata).
For each: accuracy, NLL, Brier and ECE at temperature 1 and at the temperature fitted on
calibration, split by question kind; the per-question NLL quantiles of train
and dev (a memorizing model has a much lower train loss than dev loss); and each request's
word overlap with its nearest training request, with accuracy per overlap bucket (a model
that only works on near-copies of its training data fails on the low-overlap bucket).
The act question of a scored scenario that lacks a required value is left out: its label says
no tool, while the model is trained to answer yes with the intended tool and the router then
declines for the missing value (the routing reports score that end to end).
Writes <run>/overfit.json.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import torch

from local_router.data import examples, read_records
from local_router.model import Scorer
from train import fit_temperature, load_examples, metrics, score

BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 1.01)]


def scored_examples(records):
    unlabeled = {r["id"] for r in records if "missing-required" in r["tags"] and not r.get("intended_tool")}
    return [e for e in examples(records) if not (e.kind == "act" and e.id in unlabeled)]


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def nearest(request, train_words):
    w = words(request)
    return max(len(w & t) / len(w | t) for t in train_words) if w else 0.0


def nll_quantiles(scored):
    losses = sorted(-float(torch.log_softmax(s["logits"], -1)[s["gold"]].logsumexp(0)) for s in scored)
    q = lambda p: losses[min(len(losses) - 1, int(p * len(losses)))]
    return {"median": q(0.5), "p90": q(0.9), "p99": q(0.99), "over_0.1": sum(x > 0.1 for x in losses) / len(losses)}


def by(scored, exs, field, t):
    relabeled = [{**s, "kind": getattr(e, field)} for s, e in zip(scored, exs)]
    return {k: v for k, v in metrics(relabeled, t).items() if k != "all"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--adversarial", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    run = Path(args.run)
    trained = json.loads((run / "metrics.json").read_text())["args"]
    train_recs = read_records(args.train)
    train_ex, dev_ex, _ = load_examples(args.train, args.scenarios, trained["dev_share"], trained["fraction"],
                                             trained["seed"])
    scenarios = read_records(args.scenarios)
    tiers = {"train": train_ex, "dev": dev_ex,
             "calibration": scored_examples([r for r in scenarios if r["split"] == "calibration"]),
             "test": scored_examples([r for r in scenarios if r["split"] == "test"]),
             "adversarial": scored_examples(read_records(args.adversarial))}

    # Nearest training request by word Jaccard, over the requests the adapter trained on.
    train_ids = {e.id for e in train_ex}
    train_words = [words(r["request"]) for r in train_recs if r["id"] in train_ids]
    requests = {r["id"]: r["request"] for r in train_recs + scenarios + read_records(args.adversarial)}
    overlap = {name: {e.id: nearest(requests[e.id], train_words) for e in exs}
               for name, exs in tiers.items() if name != "train"}

    report = {"run": run.name, "tiers": {k: len(v) for k, v in tiers.items()}, "models": {}}
    for model, adapter in [("base", None), ("lora", run / "adapter")]:
        scorer = Scorer(adapter=str(adapter) if adapter else None)
        scored = {name: score(scorer, exs, args.batch) for name, exs in tiers.items()}
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
        t = fit_temperature(scored["calibration"])
        out = {"temperature": t, "tiers": {}}
        for name, s in scored.items():
            exs = tiers[name]
            entry = {"t1": metrics(s, 1.0)["all"], "calibrated": metrics(s, t)["all"],
                     "kind": by(s, exs, "kind", t)}
            if name in ("train", "dev"):
                entry["nll_quantiles_t1"] = nll_quantiles(s)
            if name in overlap:
                buckets = {}
                for lo, hi in BUCKETS:
                    idx = [i for i, e in enumerate(exs) if lo <= overlap[name][e.id] < hi]
                    if idx:
                        buckets[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = metrics([s[i] for i in idx], t)["all"]
                entry["overlap"] = buckets
            out["tiers"][name] = entry
        report["models"][model] = out
    report["overlap"] = {name: sorted(v.values())[len(v) // 2] for name, v in overlap.items()}
    (run / "overfit.json").write_text(json.dumps(report, indent=2) + "\n")

    for model, out in report["models"].items():
        print(f"{model} (temperature {out['temperature']})")
        for name, entry in out["tiers"].items():
            a, c = entry["t1"], entry["calibrated"]
            print(f"  {name:12} n {a['n']:5}  acc {a['accuracy']:.3f}  nll t1 {a['nll']:.4f}  "
                  f"nll cal {c['nll']:.4f}  ece cal {c['ece']:.4f}")
    print("median nearest-train overlap: " + json.dumps({k: round(v, 3) for k, v in report["overlap"].items()}))


if __name__ == "__main__":
    main()
