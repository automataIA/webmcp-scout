import os

import pytest
import torch

from local_router.model import Scorer, shared_prefix
from local_router.prompts import options


def test_shared_prefix_leaves_a_token_in_every_sequence():
    assert shared_prefix([[1, 2, 3, 4], [1, 2, 5]]) == 2
    assert shared_prefix([[1, 2, 3], [1, 2, 3]]) == 2
    assert shared_prefix([[1, 2, 3]]) == 2
    assert shared_prefix([[1, 2], [3, 4]]) == 0


@pytest.mark.skipif(os.environ.get("LOCAL_ROUTER_GPU") != "1", reason="loads the base model on the GPU")
def test_shared_prefix_scoring_matches_the_padded_batch():
    state = {"available_tools": [{"name": "search_products", "description": "Search the catalog.",
                                  "parameters": {"category": ["ALL", "SOCCER", "RUNNING"]}},
                                 {"name": "get_product", "description": "Get one product by SKU."}],
             "user_request": "running shoes under 90 euros"}
    questions = [{"type": "noul", "instructions": "Should a tool be called?"},
                 {"type": "choice", "instructions": "Which tool?",
                  "criteria": {"t0": {"name": "search_products"}, "t1": {"name": "get_product"}}},
                 {"type": "choice", "instructions": "Which category?",
                  "criteria": {"o0": "ALL", "o1": "SOCCER", "o2": "RUNNING"}}]
    scorer = Scorer()
    scorer.model.eval()
    items = [(state, q, options(q)) for q in questions]
    counts = [len(o) for _, _, o in items]
    with torch.no_grad():
        padded = torch.softmax(scorer.logits(scorer.encode(items), counts), -1)
        shared = torch.softmax(scorer.set_logits(items, counts), -1)
    assert torch.equal(torch.isfinite(padded.log()), torch.isfinite(shared.log()))
    assert (padded - shared).abs().max() < 0.05
