#!/usr/bin/env python
# Evaluate an SGP-fine-tuned Kangaroo model:
#   • verifier perplexity                (ppl)
#   • draft acceptance-rate              (accept_rate)
#   • decoding throughput (tokens / s)   (tok_per_sec)

import argparse, json, time, sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel

sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from kangaroo.kangaroo_model import KangarooModel
from evaluation.inference_kangaroo import kangaroo_forward

# --------------------------------------------------------------------- #
# Dataset helper                                                        #
# --------------------------------------------------------------------- #
def build_dataset(tokenizer, n_samples: int, seq_len: int = 1024):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.select(range(n_samples))

    def fmt(ex):
        text = ex["instruction"]
        ctx = ex.get("context") or ""
        if ctx:
            text += "\n" + ctx
        text += "\n" + ex["response"]

        enc = tokenizer(
            text, truncation=True, max_length=seq_len,
            padding="max_length", return_tensors="pt"
        )
        ids = enc.input_ids.squeeze(0)
        if tokenizer.bos_token_id is not None:
            ids[0] = tokenizer.bos_token_id
        return {"input_ids": ids, "labels": ids.clone()}

    ds = ds.map(fmt, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds

# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--exit_layer", type=int, default=6)
    p.add_argument("--n_samples", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokeniser
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

    # Model wrapper
    dummy = argparse.Namespace(dtype="bfloat16")
    kg_model = KangarooModel(
        base_model_name_or_path=args.model_name,
        adapter_model_path=None,
        args=dummy,
        EARLY_STOP_LAYER=args.exit_layer,
    )
    kg_model.base_model = PeftModel.from_pretrained(kg_model.base_model, args.ckpt_dir)
    dtype = kg_model.base_model.dtype
    kg_model.exit_proj = kg_model.exit_proj.to(device=device, dtype=dtype)
    kg_model.head_model = kg_model.head_model.to(device=device, dtype=dtype)
    kg_model.eval().to(device)

    # Dataset
    ds = build_dataset(tok, args.n_samples)

    # --------------------------------- #
    #           Perplexity              #
    # --------------------------------- #
    total_nll = total_tok = 0
    pad_id = tok.pad_token_id
    for row in ds:
        ids = row["input_ids"].unsqueeze(0).to(device)
        with torch.no_grad():
            _, info = kg_model(ids, labels=ids, beta_exit=0.0, detach_exit=True)
        n_tok = ids.numel() if pad_id is None else (ids != pad_id).sum().item()
        total_nll += info["loss_main"].item() * n_tok
        total_tok += n_tok
    ppl = float(torch.exp(torch.tensor(total_nll / max(total_tok, 1))))

    # --------------------------------- #
    #  Draft acceptance & throughput    #
    # --------------------------------- #
    total_acc, total_gen, total_time = 0, 0, 0.0

    for row in ds:
        # prompt = tok.decode(row["input_ids"].tolist(), skip_special_tokens=True)

        # Full BatchEncoding (with .input_ids) so inference util works
        # enc = tok(prompt, return_tensors="pt")
        from types import SimpleNamespace
        input_ids = row["input_ids"].unsqueeze(0).to(device)
        enc = SimpleNamespace(input_ids=input_ids)
        enc = enc.to(device)

        # 1️⃣  Prime KV-cache by running full forward once
        with torch.no_grad():
            kg_model.base_model(enc.input_ids, use_cache=True)

        # 2️⃣  spec-decoding
        t0 = time.perf_counter()
        _, new_tok, _, acc = kangaroo_forward(
            enc, kg_model, tok,
            max_new_tokens=64, do_sample=False,
            EARLY_STOP_LAYER=args.exit_layer,
        )
        total_time += time.perf_counter() - t0
        total_gen += int(new_tok)
        total_acc += int(sum(acc))

    accept_rate = total_acc / max(total_gen, 1)
    tok_per_sec = total_gen / max(total_time, 1e-6)

    # Save + print
    Path("results").mkdir(exist_ok=True)
    res = {"ppl": ppl, "accept_rate": accept_rate, "tok_per_sec": tok_per_sec}
    with open("results/tmp_eval.json", "w") as f:
        json.dump(res, f)
    print(json.dumps(res, indent=2))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
