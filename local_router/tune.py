"""Search the LoRA hyperparameters with Optuna, on dev only.

    uv run python tune.py --train train.jsonl --calibration m4.jsonl --out out/tune [--trials 12]

Objective (minimized): the lowest dev NLL at temperature 1 over the dev checks of one epoch,
i.e. the adapter `train.py` would keep. NLL is a proper scoring rule, so it rewards right
answers and penalizes the overconfidence that overfitting brings, which accuracy (~99% on dev)
cannot separate. Each dev check reports the running best; from the third check on, a trial that
is worse than the median of earlier trials at the same check is pruned (earlier checks favour fast
learners over the best finishers). The study is stored in sqlite and resumes where it stopped:
create `<out>/STOP` to stop after the current trial; a trial left running by a killed process is
marked failed on the next start. The winner is written to best.json; retrain it with train.py,
which also fits the temperature. The calibration file is loaded only to share train.py's loader: never scored here.
"""

import argparse
import gc
import json
import random
from pathlib import Path

import optuna
import torch

from local_router.model import Scorer
from train import fit, load_examples

DEFAULT = {"lr": 1e-4, "rank": 16, "dropout": 0.05, "brier": 1.0}


def objective(train_ex, dev_ex, args):
    def run(trial):
        # rank 32 does not fit next to the 4B model on 12 GB.
        params = {"lr": trial.suggest_float("lr", 3e-5, 3e-4, log=True),
                  "rank": trial.suggest_categorical("rank", [8, 16]),
                  "dropout": trial.suggest_float("dropout", 0.0, 0.2),
                  "brier": trial.suggest_float("brier", 0.0, 2.0)}
        print(f"trial {trial.number}: {json.dumps(params)}", flush=True)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        scorer = Scorer(train=True, rank=params["rank"], dropout=params["dropout"])
        best = [float("inf")]

        def on_check(k, dev):
            best[0] = min(best[0], dev["all"]["nll"])
            trial.report(best[0], k)
            if trial.should_prune():
                raise optuna.TrialPruned(f"check {k}: best dev nll {best[0]:.4f}")

        try:
            history, step = fit(scorer, train_ex, dev_ex, epochs=1, batch=args.batch, lr=params["lr"],
                                brier=params["brier"], seed=args.seed, evals_per_epoch=args.checks,
                                on_check=on_check)
        finally:
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
        kept = min(history, key=lambda h: h["dev"]["all"]["nll"])["dev"]
        trial.set_user_attr("best_step", step)
        trial.set_user_attr("dev", {k: {"accuracy": v["accuracy"], "nll": v["nll"]} for k, v in kept.items()})
        return best[0]

    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=5 * 3600, help="seconds")
    ap.add_argument("--checks", type=int, default=4, help="dev checks per epoch (pruning points)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dev-share", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_ex, dev_ex, _ = load_examples(args.train, args.calibration, args.dev_share, 1.0, args.seed)
    print(f"examples: train {len(train_ex)}, dev {len(dev_ex)}", flush=True)

    study = optuna.create_study(
        storage=f"sqlite:///{out / 'optuna.db'}", study_name="local-router", direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3))
    for stale in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
        print(f"trial {stale.number} was left running: marked failed", flush=True)
    study.enqueue_trial(DEFAULT, skip_if_exists=True)
    stop = out / "STOP"
    stop.unlink(missing_ok=True)
    done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
    study.optimize(objective(train_ex, dev_ex, args), n_trials=max(0, args.trials - done), timeout=args.timeout,
                   gc_after_trial=True, callbacks=[lambda study, _: stop.exists() and study.stop()])
    if stop.exists():
        print(f"stopped by {stop}; run again to resume", flush=True)

    best = study.best_trial
    summary = {"objective": "lowest dev nll at temperature 1 over the dev checks of one epoch",
               "best": {"number": best.number, "value": best.value, "params": best.params, **best.user_attrs},
               "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
                           **t.user_attrs} for t in study.trials]}
    (out / "best.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"best trial {best.number}: dev nll {best.value:.4f} {json.dumps(best.params)}", flush=True)


if __name__ == "__main__":
    main()
