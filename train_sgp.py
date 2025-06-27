import argparse
import os

import torch
from transformers import AutoTokenizer
from accelerate import Accelerator

from kangaroo.kangaroo_model import KangarooModel
from kangaroo.sgp_lora import inject_lora
from data.dolly_loader import build as build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="SGP Fine-tuning")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exit_layer", type=int, default=6)
    parser.add_argument("--beta_exit", type=float, default=0.1)
    parser.add_argument("--detach-exit", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_batch", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dummy = argparse.Namespace(dtype="bfloat16")
    model = KangarooModel(args.model_name, None, dummy, EARLY_STOP_LAYER=args.exit_layer)
    model.base_model = inject_lora(model.base_model, args.exit_layer)
    model.base_model.gradient_checkpointing_enable()

    dataloader = build_dataloader(tokenizer, args.per_device_batch, seq=tokenizer.model_max_length or 512)

    accelerator = Accelerator(mixed_precision="bf16")
    model, dataloader = accelerator.prepare(model, dataloader)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"], labels=batch["labels"], beta_exit=args.beta_exit, detach_exit=args.detach_exit)
                loss = outputs[0]
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
        if accelerator.is_main_process:
            print(f"Epoch {epoch}: loss {total_loss/len(dataloader):.4f}")

    if accelerator.is_main_process:
        os.makedirs("checkpoints/sgp", exist_ok=True)
    accelerator.save_state("checkpoints/sgp")


if __name__ == "__main__":
    main()
