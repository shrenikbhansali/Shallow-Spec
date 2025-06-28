import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from kangaroo.kangaroo_model import KangarooModel
from kangaroo.sgp_lora import inject_lora
from evaluation.inference_kangaroo import kangaroo_forward


def build_dataset(tokenizer, n_samples, seq=1024):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.select(range(n_samples))

    def _format(example):
        text = example["instruction"]
        if example.get("context"):
            ctx = example["context"]
            if ctx:
                text += "\n" + ctx
        text += "\n" + example["response"]
        tokens = tokenizer(text, truncation=True, max_length=seq, padding="max_length")
        input_ids = tokens["input_ids"]
        if tokenizer.bos_token_id is not None:
            input_ids[0] = tokenizer.bos_token_id
        tokens["labels"] = input_ids.copy()
        return {k: torch.tensor(v) for k, v in tokens.items()}

    ds = ds.map(_format)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exit_layer", type=int, default=6)
    parser.add_argument("--n_samples", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dummy = argparse.Namespace(dtype="bfloat16")
    model = KangarooModel(args.model_name, None, dummy, EARLY_STOP_LAYER=args.exit_layer)
    model.base_model = inject_lora(model.base_model, args.exit_layer)
    model.base_model.load_adapter(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    ds = build_dataset(tokenizer, args.n_samples)

    # Perplexity
    nll = 0.0
    count = 0
    for item in ds:
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        with torch.no_grad():
            loss, info = model(input_ids, labels=input_ids, beta_exit=0.0, detach_exit=True)
        token_count = (input_ids != tokenizer.pad_token_id).sum().item()
        nll += info["loss_main"].item() * token_count
        count += token_count
    ppl = float(torch.exp(torch.tensor(nll / count)))

    # Acceptance rate and speed
    total_accept = 0
    total_tokens = 0
    total_time = 0.0
    for item in ds:
        prompt = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        start = time.time()
        output_ids, new_token, idx, acc_list = kangaroo_forward(
            inputs,
            model,
            tokenizer,
            max_new_tokens=64,
            do_sample=False,
            EARLY_STOP_LAYER=args.exit_layer,
        )
        total_time += time.time() - start
        total_tokens += int(new_token)
        total_accept += sum(acc_list)
    accept_rate = total_accept / max(total_tokens, 1)
    tok_per_sec = total_tokens / max(total_time, 1e-6)

    res = {
        "accept_rate": accept_rate,
        "tok_per_sec": tok_per_sec,
        "ppl": ppl,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/tmp_eval.json", "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
