# train_dvi.py
"""Online DVI-RL training script.
See AGENTS.md section 2 for algorithm details."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from kangaroo.kangaroo_model import KangarooModel
from kangaroo.sgp_lora import (
    inject_dual_lora,
    split_lora_params,
    enable_lora_grads,
)
from evaluation.inference_kangaroo import kangaroo_forward
from training.buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="DVI-RL training")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exit_layer", type=int, required=True)
    parser.add_argument("--fast_batch", type=int, default=256)
    parser.add_argument("--slow_every", type=int, default=1000)
    parser.add_argument("--baseline_momentum", type=float, default=0.95)
    parser.add_argument("--beta_kl", type=float, default=0.1)
    parser.add_argument(
        "--stream_dataset",
        type=str,
        default="RyokoAI/ShareGPT52K",
    )
    parser.add_argument("--max_prompts", type=int, default=None)
    return parser.parse_args()


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _get_prompt(example) -> str:
    # support multi-turn chat
    for field in ("messages", "history", "turns"):
        if field in example and isinstance(example[field], list):
            parts = []
            for msg in example[field]:
                if isinstance(msg, dict):
                    parts.append(msg.get("content") or msg.get("value") or "")
                else:
                    parts.append(str(msg))
            return "\n".join([p for p in parts if p])
    # fallback single field
    for key in ("prompt", "text"):
        if key in example and isinstance(example[key], str):
            return example[key]
    return str(example)


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = build_tokenizer(args.model_name)

    dummy = argparse.Namespace(dtype="bfloat16")
    model = KangarooModel(
        args.model_name, None, dummy, EARLY_STOP_LAYER=args.exit_layer
    )
    model.base_model = inject_dual_lora(model.base_model, args.exit_layer)
    model.base_model.parallelize()
    model.adapter_model.to("cuda:0")
    model.exit_proj.to("cuda:0")
    model.head_model.to("cuda:0")
    model.eval()
    dtype = torch.bfloat16 if hasattr(model.base_model, "dtype") else torch.float32
    dtype = getattr(model.base_model, "dtype", dtype)

    fast_params, slow_params = split_lora_params(model.base_model)
    if not fast_params or not slow_params:
        raise RuntimeError(
            "split_lora_params() returned an empty list "
            f"(fast={len(fast_params)}  slow={len(slow_params)}). "
            "Check adapter names or layer-index logic."
        )
    fast_opt = torch.optim.AdamW(fast_params, lr=2e-4)
    slow_opt = torch.optim.AdamW(slow_params, lr=5e-6)

    rl_buffer = ReplayBuffer(capacity=args.fast_batch * 2, device=device)
    verifier_buffer = ReplayBuffer(capacity=args.fast_batch * 4, device=device)
    baseline = 0.0
    step_fast = 0

    ds = load_dataset(args.stream_dataset, split="train", streaming=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"dvi_train_{timestamp}.jsonl"
    with open(log_path, "w") as log_f:
        log_f.write(json.dumps({"config": vars(args)}) + "\n")

        for idx, row in enumerate(ds):
            prompt = _get_prompt(row)
            if not prompt or not prompt.strip():
                continue
            enc = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _, _, _, accept_list, trace = kangaroo_forward(
                    enc,
                    model,
                    tok,
                    max_new_tokens=64,
                    do_sample=False,
                    EARLY_STOP_LAYER=args.exit_layer,
                    return_trace=True,
                )

            for step in trace:
                reward = float(step.accept.item())
                conf = torch.softmax(step.logits[0, -1], -1)[step.token.item()].item()
                # append to both buffers
                rl_buffer.append(
                    hidden=step.hidden[0, -1].to(device, dtype),
                    token=int(step.token.item()),
                    reward=reward,
                    conf=conf,
                )
                verifier_buffer.append(
                    hidden=step.hidden[0, -1].to(device, dtype),
                    token=int(step.token.item()),
                    reward=reward,
                    conf=conf,
                )
                # update baseline EMA on every new reward
                with torch.no_grad():
                    baseline = (
                        args.baseline_momentum * baseline
                        + (1 - args.baseline_momentum) * reward
                    )

            if rl_buffer.accepted_count() >= args.fast_batch:
                ## FAST (REINFORCE) update on *accepted* only
                fast_batch = rl_buffer.sample(args.fast_batch, accepted_only=True)
                enable_lora_grads(model.base_model, "lora_S", True)
                enable_lora_grads(model.base_model, "lora_D", False)
                fast_opt.zero_grad(set_to_none=True)

                h = fast_batch["hidden"].to(device, dtype)
                tok_b = fast_batch["token"].to(device)
                r = fast_batch["reward"].to(device, dtype)

                logits = model.exit_proj(h)
                log_pi = (
                    torch.log_softmax(logits, -1)
                    .gather(1, tok_b.unsqueeze(1))
                    .squeeze(1)
                )
                adv = (r - baseline).detach()
                loss_fast = -(adv * log_pi).mean()
                loss_fast.backward()
                torch.nn.utils.clip_grad_norm_(fast_params, 1.0)
                fast_opt.step()

                # re-freeze for safety
                enable_lora_grads(model.base_model, "lora_S", False)
                enable_lora_grads(model.base_model, "lora_D", True)

                rl_buffer.clear(accepted_only=True)
                step_fast += 1

                if (
                    step_fast % args.slow_every == 0
                    and len(verifier_buffer) >= args.fast_batch
                ):
                    slow_batch = verifier_buffer.sample(
                        args.fast_batch, accepted_only=False
                    )
                    enable_lora_grads(model.base_model, "lora_S", False)
                    enable_lora_grads(model.base_model, "lora_D", True)
                    slow_opt.zero_grad(set_to_none=True)

                    h_all = slow_batch["hidden"].to(device, dtype)
                    tok_all = slow_batch["token"].to(device)
                    with torch.no_grad():
                        logits_frozen = (
                            model.head_model(h_all.unsqueeze(1)).squeeze(1).detach()
                        )

                    logits_new = model.head_model(h_all.unsqueeze(1)).squeeze(1)
                    loss_ce = F.cross_entropy(logits_new, tok_all)
                    loss_kl = F.kl_div(
                        torch.log_softmax(logits_new, -1),
                        torch.softmax(logits_frozen, -1),
                        reduction="batchmean",
                    )
                    loss_slow = loss_ce + args.beta_kl * loss_kl
                    loss_slow.backward()
                    torch.nn.utils.clip_grad_norm_(slow_params, 1.0)
                    slow_opt.step()
                    # clear verifier buffer after training
                    verifier_buffer.clear(accepted_only=False)
                    # re-freeze everything
                    enable_lora_grads(model.base_model, "lora_S", False)
                    enable_lora_grads(model.base_model, "lora_D", False)

            if (idx + 1) % 100 == 0:
                accept_rate = float(sum(accept_list)) / max(len(accept_list), 1)
                log_dict = {
                    "idx": idx + 1,
                    "step": step_fast,
                    "baseline": baseline,
                    "accept_rate": accept_rate,
                    "time": time.time(),
                }
                if "loss_fast" in locals():
                    log_dict["loss_fast"] = float(loss_fast.item())
                if "loss_slow" in locals():
                    log_dict["loss_slow"] = float(loss_slow.item())
                log_f.write(json.dumps(log_dict) + "\n")
                log_f.flush()

            if args.max_prompts is not None and idx + 1 >= args.max_prompts:
                break

    ckpt_dir = Path("checkpoints/dvi_lora")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(str(ckpt_dir))


if __name__ == "__main__":
    main()

# -----------------------------
# Unit test stubs
# -----------------------------
if __name__ != "__main__":
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    def test_split():
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
        )
        model = AutoModelForCausalLM.from_config(config)
        peft_model = inject_dual_lora(model, exit_layer=1)
        fast, slow = split_lora_params(peft_model)
        assert len(fast) > 0 and len(slow) > 0

    def test_buffer():
        buf = ReplayBuffer(capacity=2, device=torch.device("cpu"))
        buf.append(torch.zeros(4), token=1, reward=1.0, conf=0.5)
        assert len(buf) == 1
        batch = buf.sample(1)
        assert batch["hidden"].shape == (1, 4)

    def test_advantage_variance():
        buf = ReplayBuffer(8, torch.device("cpu"))
        for r in [0, 1, 0, 1]:
            buf.append(torch.zeros(4), token=1, reward=r, conf=0.5)
        b = buf.sample(4, accepted_only=False)
        assert (b["reward"] == 1).any() and (b["reward"] == 0).any()
