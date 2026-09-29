"""Train the LoRA decision model on evals/train question sets.

    cargo run -p scout-cli -- eval questions --scenarios evals/train --out train.jsonl
    cargo run -p scout-cli -- eval questions --scenarios evals/scenarios --out m4.jsonl
    uv run python train.py --train train.jsonl --calibration m4.jsonl --out out/run

Dev = whole contrastive groups held out of evals/train. Dev is scored a few times per epoch
and the adapter with the lowest dev NLL is kept (early stopping: a second epoch overfits).
With `--twins` (question sets of evals/train-twins, exported with `--toolsets
evals/train-twins/toolsets`), a share of the training scenarios also trains on its pool `a`
injected twin, and dev also holds the pool `b` twins of dev groups (written by another
teacher): a twin is in its clean scenario's group, so it never straddles the split.
The temperature is fitted on the calibration split of the M4 scenarios; their test split is
never read here. `tune.py` searches the hyperparameters with the same `fit`.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from local_router.data import balance_act, examples, group_of, in_dev, permuted, read_records, split
from local_router.model import Scorer
from local_router.prompts import options, render

SPLIT_SEED = 20260926
TEMPERATURES = [round(0.25 * 1.05**i, 4) for i in range(90)]


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def loss_terms(logits, golds):
    """Cross-entropy of the right set plus the Brier score of that event."""
    logp = torch.log_softmax(logits, dim=-1)
    right = torch.stack([torch.logsumexp(logp[i, sorted(g)], dim=0) for i, g in enumerate(golds)])
    return -right, (1 - right.exp()) ** 2


@torch.no_grad()
def score(scorer, exs, batch_size):
    """Logits in the questions' own option order, with the right option indices."""
    scorer.model.eval()
    out = []
    for batch in batches(exs, batch_size):
        orders = [options(e.question) for e in batch]
        logits = scorer.logits(scorer.encode([(e.state, e.question, o) for e, o in zip(batch, orders)]),
                               [len(o) for o in orders])
        for e, o, row in zip(batch, orders, logits):
            out.append({"kind": e.kind, "logits": row[:len(o)].cpu(),
                        "gold": [i for i, (k, _) in enumerate(o) if k in e.gold]})
    return out


def metrics(scored, t):
    groups = {}
    for s in scored:
        for name in ("all", s["kind"]):
            groups.setdefault(name, []).append(s)
    out = {}
    for name, items in sorted(groups.items()):
        probs = [torch.softmax(s["logits"] / t, dim=-1) for s in items]
        right = [float(p[s["gold"]].sum()) for p, s in zip(probs, items)]
        hit = [int(p.argmax()) in s["gold"] for p, s in zip(probs, items)]
        conf = [float(p.max()) for p in probs]
        out[name] = {"n": len(items), "accuracy": sum(hit) / len(items),
                     "nll": -sum(math.log(max(r, 1e-12)) for r in right) / len(items),
                     "brier": sum((1 - r) ** 2 for r in right) / len(items),
                     "ece": ece(conf, hit)}
    return out


def ece(conf, hit, bins=10):
    total = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            total += abs(sum(conf[i] for i in idx) - sum(hit[i] for i in idx)) / len(conf)
    return total


def fit_temperature(scored):
    return min(TEMPERATURES, key=lambda t: metrics(scored, t)["all"]["nll"])


def load_examples(train_path, calibration_path, dev_share, fraction, seed, twins_path=None, twin_share=0.0):
    train_recs, dev_recs = split(read_records(train_path), dev_share, SPLIT_SEED)
    if fraction < 1:
        train_recs = [r for r in train_recs if in_dev(group_of(r), fraction, seed)]
    if twins_path:
        twin_train, twin_dev = split(read_records(twins_path), dev_share, SPLIT_SEED)
        rng = random.Random(seed)
        ids = {r["id"] for r in train_recs}
        train_recs += [r for r in twin_train if "twin-a" in r["tags"] and r["id"].removesuffix("-a") in ids
                       and rng.random() < twin_share]
        dev_recs += [r for r in twin_dev if "twin-b" in r["tags"]]
    train_ex = balance_act(examples(train_recs), seed)
    dev_ex = balance_act(examples(dev_recs), SPLIT_SEED)
    cal_ex = examples([r for r in read_records(calibration_path) if r["split"] == "calibration"])
    return train_ex, dev_ex, cal_ex


def fit(scorer, train_ex, dev_ex, *, epochs, batch, lr, brier, seed, evals_per_epoch, on_check=None):
    """Trains in place and restores the adapter with the lowest dev NLL (at temperature 1).

    `on_check(k, dev)` runs after the k-th dev check (k from 1) and may raise to stop early.
    Returns the dev history and the kept check's step."""
    params = [p for p in scorer.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    per_epoch = math.ceil(len(train_ex) / batch)
    steps = epochs * per_epoch
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / (steps - warmup))))
    checks = {min(per_epoch, math.ceil(per_epoch * (i + 1) / evals_per_epoch)) for i in range(evals_per_epoch)}
    rng = random.Random(seed)
    history, step, best = [], 0, None
    for epoch in range(epochs):
        scorer.model.train()
        order = train_ex[:]
        rng.shuffle(order)
        started, running = time.time(), []
        for n, chunk in enumerate(batches(order, batch), 1):
            orders = [permuted(e, rng) for e in chunk]
            logits = scorer.logits(scorer.encode([(e.state, e.question, o) for e, o in zip(chunk, orders)]),
                                   [len(o) for o in orders])
            golds = [[i for i, (k, _) in enumerate(o) if k in e.gold] for e, o in zip(chunk, orders)]
            ce, sq = loss_terms(logits, golds)
            loss = (ce + brier * sq).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running.append(float(loss))
            if step % 25 == 0:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {sum(running[-25:]) / 25:.4f} "
                      f"{(time.time() - started) / len(running):.2f}s/step", flush=True)
            if n not in checks:
                continue
            dev = metrics(score(scorer, dev_ex, batch), 1.0)
            scorer.model.train()
            history.append({"epoch": epoch + 1, "step": step, "dev": dev})
            print(f"step {step} dev nll {dev['all']['nll']:.4f}: "
                  + json.dumps({k: round(v['accuracy'], 4) for k, v in dev.items()}), flush=True)
            if best is None or dev["all"]["nll"] < best[0]:
                state = get_peft_model_state_dict(scorer.model)
                best = (dev["all"]["nll"], step, {k: v.detach().cpu().clone() for k, v in state.items()})
            if on_check:
                on_check(len(history), dev)
    set_peft_model_state_dict(scorer.model, best[2])
    print(f"kept the adapter of step {best[1]}", flush=True)
    return history, best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--evals-per-epoch", type=int, default=2, help="dev checks for early stopping")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--brier", type=float, default=1.0, help="weight of the Brier term in the loss")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dev-share", type=float, default=0.15)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of training groups, for learning curves")
    ap.add_argument("--twins", help="question sets of the injected twins")
    ap.add_argument("--twin-share", type=float, default=0.3, help="share of training scenarios with a twin")
    ap.add_argument("--no-train", action="store_true", help="score the base model only")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_ex, dev_ex, cal_ex = load_examples(args.train, args.calibration, args.dev_share, args.fraction, args.seed,
                                             args.twins, args.twin_share)
    print(f"examples: train {len(train_ex)}, dev {len(dev_ex)}, calibration {len(cal_ex)}", flush=True)

    scorer = Scorer(train=not args.no_train, rank=args.rank, dropout=args.dropout)
    lengths = [len(scorer.tokenizer(render(scorer.tokenizer, e.state, e.question, options(e.question)))["input_ids"])
               for e in train_ex + dev_ex + cal_ex]
    print(f"prompt tokens: max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f}", flush=True)

    history, best_step = [], None
    if not args.no_train:
        history, best_step = fit(scorer, train_ex, dev_ex, epochs=args.epochs, batch=args.batch, lr=args.lr,
                                 brier=args.brier, seed=args.seed, evals_per_epoch=args.evals_per_epoch)

    dev_scored, cal_scored = score(scorer, dev_ex, args.batch), score(scorer, cal_ex, args.batch)
    t = fit_temperature(cal_scored)
    result = {"args": vars(args), "examples": {"train": len(train_ex), "dev": len(dev_ex), "calibration": len(cal_ex)},
              "history": history, "best_step": best_step, "temperature": t,
              "dev": metrics(dev_scored, 1.0), "dev_calibrated": metrics(dev_scored, t),
              # Pool b twin ids end in "-b"; empty without --twins.
              "dev_clean": metrics([s for s, e in zip(dev_scored, dev_ex) if not e.id.endswith("-b")], 1.0),
              "dev_injected": metrics([s for s, e in zip(dev_scored, dev_ex) if e.id.endswith("-b")], 1.0),
              "calibration": metrics(cal_scored, 1.0), "calibration_calibrated": metrics(cal_scored, t)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_train:
        scorer.model.save_pretrained(out / "adapter")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"temperature {t}; dev " + json.dumps({k: round(v['accuracy'], 4) for k, v in result['dev'].items()}))
    print("calibration " + json.dumps({k: round(v['accuracy'], 4) for k, v in result['calibration'].items()}))


if __name__ == "__main__":
    main()
