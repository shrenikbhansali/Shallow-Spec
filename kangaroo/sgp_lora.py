from peft import LoraConfig, get_peft_model


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
