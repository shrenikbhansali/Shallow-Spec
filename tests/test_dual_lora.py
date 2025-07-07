import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig
import peft
import packaging.version

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kangaroo.sgp_lora import inject_dual_lora, split_lora_params, get_lora_param_names


def _create_tiny_llama():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model


def test_dual_lora_injection():
    model = _create_tiny_llama()
    peft_model = inject_dual_lora(model, exit_layer=3, rank_fast=4, rank_slow=4)
    fast, slow = split_lora_params(peft_model)

    assert packaging.version.parse(peft.__version__) >= packaging.version.parse("0.6.0")

    assert len(fast) > 0 and len(slow) > 0
    for p in fast:
        assert p.requires_grad is False
    for p in slow:
        assert p.requires_grad is False

    fast_names = get_lora_param_names(peft_model, "lora_S")
    slow_names = get_lora_param_names(peft_model, "lora_D")
    assert fast_names.isdisjoint(slow_names)
    assert set(peft_model.active_adapters) == {"lora_S", "lora_D"}

    peft_model.eval()
    tokens = torch.randint(0, peft_model.config.vocab_size, (1, 4))
    with torch.no_grad():
        logits_zero = peft_model(tokens).logits
        for name, param in peft_model.named_parameters():
            if ".lora_S." in name or ".lora_D." in name:
                param.fill_(1.0)
        logits_mod = peft_model(tokens).logits

    assert not torch.allclose(logits_zero, logits_mod)
