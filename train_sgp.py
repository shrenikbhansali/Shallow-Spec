import argparse
import os
import json
import time

import torch
from transformers import AutoTokenizer
from peft.utils import get_peft_model_state_dict
from accelerate import Accelerator

from kangaroo.kangaroo_model import KangarooModel
from kangaroo.sgp_lora import inject_lora
from data.dolly_loader import build as build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="SGP Fine-tuning")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exit_layer", type=int, default=6)
    parser.add_argument("--beta_exit", type=float, default=0.7)
    parser.add_argument("--detach-exit", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_batch", type=int, default=1)
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dummy = argparse.Namespace(dtype="bfloat16")
    model = KangarooModel(args.model_name, None, dummy, EARLY_STOP_LAYER=args.exit_layer)
    model.base_model = inject_lora(model.base_model, args.exit_layer)
    model.base_model.gradient_checkpointing_enable()

    accelerator = Accelerator(mixed_precision="bf16")

    model = accelerator.prepare(model)
    lora_param_names = set(get_peft_model_state_dict(model.base_model).keys())

    dataloader = build_dataloader(tokenizer, args.per_device_batch, seq=1024)

    dataloader = accelerator.prepare_data_loader(dataloader)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_file = open(f"logs/sgp_train_{timestamp}.jsonl", "w")
    log_header = {
        "timestamp": timestamp,
        "model": args.model_name,
        "exit_layer": args.exit_layer,
        "beta_exit": args.beta_exit,
        "detach_exit": args.detach_exit,
        "batch": args.per_device_batch,
        "epochs": args.epochs,
    }
    log_file.write(json.dumps({"config": log_header}) + "\n")

    loss_main_start = grad_norm_start = None
    loss_main_end = grad_norm_end = loss_exit_end = None
    step_idx = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    batch["input_ids"],
                    labels=batch["labels"],
                    beta_exit=args.beta_exit,
                    detach_exit=args.detach_exit,
                )
                loss = outputs[0]
                accelerator.backward(loss)

                grads = [
                    p.grad.detach()
                    for n, p in model.named_parameters()
                    if n in lora_param_names and p.grad is not None
                ]
                if grads:
                    norms = torch.stack([g.norm(2) for g in grads])
                    grad_norm = norms.norm(2).item()
                else:
                    grad_norm = 0.0

                optimizer.step()
                optimizer.zero_grad()

                metrics = outputs[1]
                loss_main = metrics.get("loss_main")
                loss_exit = metrics.get("loss_exit")
                if loss_main is not None:
                    loss_main = loss_main.item()
                if loss_exit is not None:
                    loss_exit = loss_exit.item()

                step_idx += 1
                if loss_main_start is None:
                    loss_main_start = loss_main
                    grad_norm_start = grad_norm
                loss_main_end = loss_main
                grad_norm_end = grad_norm
                loss_exit_end = loss_exit

                log_file.write(json.dumps({
                    "step": step_idx,
                    "loss_main": loss_main,
                    "loss_exit": loss_exit,
                    "grad_norm": grad_norm
                }) + "\n")

                total_loss += loss.item()
        if accelerator.is_main_process:
            print(f"Epoch {epoch}: loss {total_loss / len(dataloader):.4f}")

    log_file.close()

    if accelerator.is_main_process:
        os.makedirs("checkpoints/sgp_adapter", exist_ok=True)
        model.base_model.save_pretrained("checkpoints/sgp_adapter")
    accelerator.save_state("checkpoints/sgp")

    return {
        "loss_main_start": loss_main_start,
        "loss_main_end": loss_main_end,
        "grad_norm_start": grad_norm_start,
        "grad_norm_end": grad_norm_end,
        "loss_exit_end": loss_exit_end,
    }


if __name__ == "__main__":
    main()
