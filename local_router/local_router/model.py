"""Qwen3 with a LoRA adapter, read out on the logits of the option letters."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .prompts import letter_ids, render

BASE = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Scorer:
    def __init__(self, base=BASE, adapter=None, train=False, rank=16, dropout=0.05):
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        if train:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
        elif train:
            config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=dropout,
                                target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
            model = get_peft_model(model, config)
        self.model = model
        self.letters = torch.tensor(letter_ids(self.tokenizer), device="cuda")

    def encode(self, items):
        """items: (state, question, order) triples -> padded batch on the GPU."""
        texts = [render(self.tokenizer, s, q, o) for s, q, o in items]
        return self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

    def logits(self, batch, counts):
        """fp32 logits of the first `counts[i]` letters, -inf beyond; [B, 26].

        Computed from the last hidden state in fp32: bf16 logits would round close options
        to ties."""
        model = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
        hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        return self._readout(model, hidden[:, -1], counts)

    @torch.no_grad()
    def set_logits(self, items, counts):
        """`logits(encode(items), counts)` for the questions of one request, for serving.

        The prompts share everything up to the question (system prompt, tools, request), about
        85% of each: that prefix runs once and its KV cache serves every question. Results match
        the padded batch up to bf16 noise on near ties."""
        ids = [self.tokenizer(render(self.tokenizer, s, q, o), add_special_tokens=False)["input_ids"]
               for s, q, o in items]
        n = shared_prefix(ids)
        model = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
        cache = DynamicCache()
        model.model(input_ids=torch.tensor([ids[0][:n]], device="cuda"), past_key_values=cache, use_cache=True)
        cache.batch_repeat_interleave(len(ids))
        # Right padding: a question's tokens never attend to the padding after them.
        tails = [i[n:] for i in ids]
        width = max(map(len, tails))
        input_ids = torch.tensor([t + [self.tokenizer.pad_token_id] * (width - len(t)) for t in tails], device="cuda")
        mask = torch.tensor([[1] * (n + len(t)) + [0] * (width - len(t)) for t in tails], device="cuda")
        hidden = model.model(input_ids=input_ids, attention_mask=mask, past_key_values=cache,
                             use_cache=True).last_hidden_state
        last = hidden[torch.arange(len(tails), device="cuda"), torch.tensor([len(t) - 1 for t in tails], device="cuda")]
        return self._readout(model, last, counts)

    def _readout(self, model, last, counts):
        weight = model.get_output_embeddings().weight[self.letters]
        out = last.float() @ weight.float().T
        mask = torch.arange(out.shape[1], device=out.device)[None, :] >= torch.tensor(counts, device=out.device)[:, None]
        return out.masked_fill(mask, float("-inf"))


def shared_prefix(ids):
    """Length of the token prefix common to every sequence, leaving at least one token in each."""
    n = 0
    while all(len(i) > n + 1 for i in ids) and len({i[n] for i in ids}) == 1:
        n += 1
    return n
