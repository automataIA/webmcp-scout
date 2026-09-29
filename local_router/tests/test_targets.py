from transformers import AutoTokenizer

from local_router.data import balance_act, examples, split
from local_router.model import BASE
from local_router.prompts import letter_ids, options, render
from local_router.targets import compile_targets

TOOLS = {"t0": {"name": "get_product", "description": "Get one product."},
         "t1": {"name": "search_products", "description": "Search."}}
QUESTIONS = {
    "__act__": {"type": "noul", "instructions": "Act?"},
    "__tool__": {"type": "choice", "instructions": "Which tool?", "criteria": TOOLS},
    "search_products.category": {"type": "choice", "instructions": "Category?",
                                 "criteria": {"o0": "ALL", "o1": "SOCCER", "o2": "RUNNING"}},
    "search_products.category?": {"type": "noul", "instructions": "Category given?"},
    "search_products.max_price": {"type": "choice", "instructions": "Which number?",
                                  "criteria": {"c0": {"number": "2", "context": "2 pairs"},
                                               "c1": {"number": "50", "context": "under 50"},
                                               "none": "None of these numbers"}},
    "search_products.max_price?": {"type": "noul", "instructions": "Price given?"},
}


def record(tool, args=None, absent=None, tags=("synthetic", "en", "group-00-0"), **extra):
    return {"id": "x-001", "split": "train", "toolset": "alpha", "tags": list(tags), "request": "r",
            "tool": tool, "args": args or {}, "absent": absent or [],
            "question_set": {"state": {"available_tools": [], "user_request": "r"}, "questions": QUESTIONS},
            **extra}


def test_acting_request_trains_only_the_gold_tools_fields():
    targets = compile_targets(record("search_products", {"query": "boots", "category": "SOCCER", "max_price": 50},
                                     tags=("synthetic", "en", "group-00-0")))
    assert targets == {"__act__": {"yes"}, "__tool__": {"t1"},
                       "search_products.category": {"o1"}, "search_products.category?": {"yes"},
                       "search_products.max_price": {"c1"}, "search_products.max_price?": {"yes"}}


def test_no_tool_request_trains_only_the_act_question():
    assert compile_targets(record("none", tags=("synthetic", "en", "no-tool", "group-01-0"))) == {"__act__": {"no"}}


def test_missing_required_request_acts_with_its_intended_tool():
    targets = compile_targets(record("none", {"category": "RUNNING"}, ["max_price"],
                                     tags=("synthetic", "en", "missing-required", "group-02-0"),
                                     intended_tool="search_products"))
    assert targets["__act__"] == {"yes"} and targets["__tool__"] == {"t1"}
    assert targets["search_products.max_price?"] == {"no"}
    assert targets["search_products.max_price"] == {"none"}


def test_uncertain_optional_values_are_masked():
    targets = compile_targets(record("search_products", {"category": {"one_of": ["SOCCER"], "optional": True}}))
    assert "search_products.category" not in targets and "search_products.category?" not in targets


def test_one_of_accepts_every_listed_value_and_ambiguous_tools_mask_fields():
    targets = compile_targets(record("search_products", {"category": {"one_of": ["SOCCER", "running"]}}))
    assert targets["search_products.category"] == {"o1", "o2"}
    targets = compile_targets(record(["get_product", "search_products"], {"category": "SOCCER"}))
    assert targets == {"__act__": {"yes"}, "__tool__": {"t0", "t1"}}
    assert "__act__" not in compile_targets(record(["none", "search_products"]))


def test_dev_split_keeps_groups_whole_and_act_balance_keeps_other_questions():
    records = [dict(record("search_products", {"category": "SOCCER"}, tags=("synthetic", "en", f"group-{g:02d}-0")),
                    id=f"x-{g}-{k}") for g in range(40) for k in range(2)]
    train, dev = split(records, 0.3, 7)
    assert {r["tags"][2] for r in train}.isdisjoint({r["tags"][2] for r in dev})
    exs = examples(records)
    balanced = balance_act(exs, 1)
    assert not any(e.kind == "act" for e in balanced)  # no no-tool request, so no act question kept
    assert sum(e.kind != "act" for e in balanced) == sum(e.kind != "act" for e in exs)


def test_letters_are_single_tokens_and_the_prompt_ends_at_the_answer():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    assert len(set(letter_ids(tokenizer))) == 26
    question = QUESTIONS["search_products.category"]
    text = render(tokenizer, {"available_tools": [], "user_request": "r"}, question, options(question))
    assert text.endswith("ANSWER:<|im_end|>\n<|im_start|>assistant\n")
    assert "<think>" not in text


def test_yes_no_criteria_are_read_under_scouts_true_and_false_keys():
    question = {"type": "noul", "instructions": "Act?", "criteria": {"true": "the user asks", "false": "chat only"}}
    assert options(question) == [("yes", "the user asks"), ("no", "chat only")]
    assert options({"type": "noul", "instructions": "Act?"}) == [("yes", "yes"), ("no", "no")]
