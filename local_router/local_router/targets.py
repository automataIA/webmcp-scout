"""Training targets for the questions of one labelled scenario.

A target is the set of option keys that count as right. Questions the label does not settle are
masked (left out): other tools' fields, the tool of a no-tool request, fields of an ambiguous
tool, optional values one labelling pass was unsure of, and values the options cannot express.
"""


def gold_tools(record):
    names = record["tool"] if isinstance(record["tool"], list) else [record["tool"]]
    if names == ["none"] and record.get("intended_tool"):
        return [record["intended_tool"]], False
    return [n for n in names if n != "none"], "none" in names


def compile_targets(record):
    """{question key: set of right option keys} for the questions the label settles."""
    qs = record["question_set"]
    if qs is None:
        return {}
    questions = qs["questions"]
    tools, maybe_none = gold_tools(record)
    targets = {}
    if not (tools and maybe_none):
        targets["__act__"] = {"yes"} if tools else {"no"}
    if tools:
        targets["__tool__"] = {k for k, v in questions["__tool__"]["criteria"].items() if v["name"] in tools}
    if len(tools) != 1:
        return targets
    prefix = tools[0] + "."
    args, absent = record.get("args") or {}, set(record.get("absent") or [])
    for key, question in questions.items():
        if not key.startswith(prefix) or "[" in key:
            continue
        name = key[len(prefix):]
        if name.endswith("?"):
            target = presence(name[:-1], args, absent)
        else:
            target = value(name, question, args, absent)
        if target:
            targets[key] = target
    return targets


def is_optional(expect):
    return isinstance(expect, dict) and expect.get("optional", False)


def presence(param, args, absent):
    if param in args:
        return None if is_optional(args[param]) else {"yes"}
    return {"no"} if param in absent else None


def value(param, question, args, absent):
    if param not in args:
        if param in absent:
            return {"no"} if question["type"] == "noul" else ({"none"} if "none" in question["criteria"] else None)
        return None
    expect = args[param]
    if isinstance(expect, dict):
        if expect.get("optional") or "one_of" not in expect:
            return None
        wanted = expect["one_of"]
    else:
        wanted = [expect]
    if question["type"] == "noul":
        return {"yes" if wanted[0] else "no"} if len(wanted) == 1 and isinstance(wanted[0], bool) else None
    right = {k for k, v in question["criteria"].items() if any(matches(v, w) for w in wanted)}
    return right or None


def matches(option, wanted):
    if isinstance(option, dict) and "number" in option:
        try:
            return float(option["number"]) == float(wanted)
        except (TypeError, ValueError):
            return False
    return isinstance(option, str) and isinstance(wanted, str) and option.lower() == wanted.lower()
