"""Question examples from `scout eval questions` records."""

import hashlib
import json
import random
from dataclasses import dataclass

from .prompts import options
from .targets import compile_targets

# Share of no-tool requests per toolset in the training quotas (scripts/gen_train.py).
NO_TOOL_SHARE = {"alpha": 0.25, "gamma": 0.25, "shoes": 0.25, "sport": 0.30}


@dataclass
class Example:
    id: str
    key: str
    kind: str  # act, tool, presence, value
    toolset: str
    group: str
    state: dict
    question: dict
    gold: set


def read_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def kind(key):
    if key in ("__act__", "__tool__"):
        return key.strip("_")
    return "presence" if key.endswith("?") else "value"


def base_toolset(record):
    """An injected twin's clean toolset (its `base-` tag), else the record's own."""
    return next((t.removeprefix("base-") for t in record["tags"] if t.startswith("base-")), record["toolset"])


def group_of(record):
    tag = next((t for t in record["tags"] if t.startswith("group-")), record["id"])
    return f"{base_toolset(record)}/{tag}"


def examples(records):
    out = []
    for r in records:
        for key, gold in compile_targets(r).items():
            question = r["question_set"]["questions"][key]
            if not gold <= {k for k, _ in options(question)}:
                raise ValueError(f"{r['id']} {key}: target {gold} not among the options")
            out.append(Example(r["id"], key, kind(key), base_toolset(r), group_of(r),
                               r["question_set"]["state"], question, gold))
    return out


def in_dev(group, share, seed):
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def split(records, dev_share, seed):
    """Whole groups go to dev, so translations and contrastive siblings never straddle."""
    train = [r for r in records if not in_dev(group_of(r), dev_share, seed)]
    dev = [r for r in records if in_dev(group_of(r), dev_share, seed)]
    return train, dev


def balance_act(exs, seed):
    """Drop act questions of acting requests beyond each toolset's no-tool quota; the other
    questions of those requests stay."""
    rng = random.Random(seed)
    dropped = set()
    for toolset, share in NO_TOOL_SHARE.items():
        acts = [e for e in exs if e.toolset == toolset and e.kind == "act"]
        yes = [e for e in acts if e.gold == {"yes"}]
        cap = round(sum(e.gold == {"no"} for e in acts) * (1 - share) / share)
        if len(yes) > cap:
            dropped |= {id(e) for e in rng.sample(yes, len(yes) - cap)}
    return [e for e in exs if id(e) not in dropped]


def permuted(example, rng):
    """Options in a random order, so no letter carries a prior."""
    order = options(example.question)
    rng.shuffle(order)
    return order
