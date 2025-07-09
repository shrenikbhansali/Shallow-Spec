from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub evaluation.eval to avoid import errors during tests.
import types

dummy_mod = types.ModuleType("evaluation.eval")
dummy_mod.run_eval = lambda *a, **k: None
dummy_mod.reorg_answer_file = lambda *a, **k: None
sys.modules.setdefault("evaluation.eval", dummy_mod)

from kangaroo.kangaroo_model import KangarooModel
from evaluation.inference_kangaroo import kangaroo_forward
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import tempfile


def test_hidden_hook():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.pad_token = tok.eos_token
    config = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = AutoModelForCausalLM.from_config(config)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=False)
        dummy = type("obj", (), {"dtype": "float16"})()
        kg = KangarooModel(tmp, None, dummy, EARLY_STOP_LAYER=1)
        prompt = tok("Hello", return_tensors="pt")
        kg.eval()
        ids, new_tok, _, _, trace = kangaroo_forward(
            prompt,
            kg,
            tok,
            max_new_tokens=4,
            do_sample=False,
            return_trace=True,
            EARLY_STOP_LAYER=1,
        )
    assert len(trace) == new_tok
    for step in trace:
        assert step.hidden.shape[-1] == kg.base_model.config.hidden_size
        assert step.logits.shape[-1] == tok.vocab_size
        assert int(step.accept.item()) in (0, 1)
