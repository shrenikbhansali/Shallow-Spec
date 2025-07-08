from __future__ import annotations

import re
from typing import List, Tuple, Set


_LAYER_RE = re.compile(r"(?:^|\.)(?:encoder\.)?layers\.(\d+)\.")


def _extract_idx(name: str) -> int | None:
    """Return layer index encoded in a parameter name."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None

import torch
from peft import LoraConfig, get_peft_model, PeftModel


def inject_lora(base_model, exit_layer: int, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Wrap layers <= exit_layer with LoRA adapters."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    # freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    # enable LoRA params only for early layers
    for name, module in model.named_modules():
        if any(key in name for key in ["lora_A", "lora_B"]):
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])
                if idx <= exit_layer:
                    for p in module.parameters():
                        p.requires_grad = True
    return model


def inject_dual_lora(
    base_model: torch.nn.Module,
    exit_layer: int,
    *,
    rank_fast: int = 8,
    rank_slow: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> PeftModel:
    """Inject two LoRA adapter groups into a model.

    Args:
        base_model: Base HuggingFace model to modify.
        exit_layer: Index of the last draft layer.
        rank_fast: LoRA rank for the shallow adapter family.
        rank_slow: LoRA rank for the deep adapter family.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout probability.

    Returns:
        The ``PeftModel`` with ``lora_S`` and ``lora_D`` adapters registered.
    """

    transformer = getattr(base_model, "model", base_model)
    num_layers = len(transformer.layers)

    cfg_s = LoraConfig(
        r=rank_fast,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(exit_layer + 1)),
        layers_pattern="layers",
    )
    peft_model = get_peft_model(base_model, cfg_s, adapter_name="lora_S")

    cfg_d = LoraConfig(
        r=rank_slow,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(exit_layer + 1, num_layers)),
        layers_pattern="layers",
    )
    peft_model.add_adapter("lora_D", cfg_d)

    # Activate BOTH adapters for every forward(). Inside the wrapped
    # ``base_model`` this ensures LoRA weights for both families are used.
    peft_model.base_model.set_adapter(["lora_S", "lora_D"])

    for param in peft_model.parameters():
        param.requires_grad = False

    setattr(peft_model, "exit_layer", exit_layer)
    return peft_model


def get_lora_param_names(model: torch.nn.Module, prefix: str) -> Set[str]:
    """Return LoRA parameter names containing ``prefix``.

    Args:
        model: The model with LoRA adapters.
        prefix: Adapter prefix (``"lora_S"`` or ``"lora_D"``).

    Returns:
        Set of parameter names that belong to the specified adapter.
    """

    return {n for n, _ in model.named_parameters() if f".{prefix}." in n}


def enable_lora_grads(model: PeftModel, family: str, enable: bool = True) -> None:
    """(Un)freeze LoRA parameters belonging to a given adapter family."""
    token = f".{family}."
    for name, param in model.named_parameters():
        if token in name and ("lora_A" in name or "lora_B" in name):
            param.requires_grad = enable


def split_lora_params(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split LoRA parameters into fast and slow groups.

    Args:
        model: PEFT model produced by :func:`inject_dual_lora`.

    Returns:
        A tuple ``(fast_params, slow_params)`` with parameters for the ``lora_S``
        and ``lora_D`` adapters respectively.
    """

    exit_layer = getattr(model, "exit_layer", None)
    if exit_layer is None:
        raise AttributeError("Model missing `exit_layer` attribute")

    fast_params: List[torch.nn.Parameter] = []
    slow_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        idx = _extract_idx(name)
        if idx is None:
            continue
        if ".lora_S." in name and idx <= exit_layer:
            fast_params.append(param)
        elif ".lora_D." in name and idx > exit_layer:
            slow_params.append(param)

    return fast_params, slow_params
