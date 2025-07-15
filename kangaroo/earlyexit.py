# kangaroo/earlyexit.py
# --------------------------------------------------------------------------- #
# Early‑exit LLAMA utilities for self‑speculative decoding                    #
# --------------------------------------------------------------------------- #
from __future__ import annotations
from typing import NamedTuple, Tuple, List, Optional
import torch
from torch import Tensor
from transformers.models.llama import LlamaForCausalLM

# --------------------------------------------------------------------------- #
# Minimal KV‑cache                                                            #
# --------------------------------------------------------------------------- #
class SimpleKVCache:
    __slots__ = ("key", "value")

    def __init__(self, key: Tensor, value: Tensor):
        self.key, self.value = key, value  # (B, H, T, D)

    def update(self, new_k, new_v, *_):
        if self.key.shape[2] == 0 and self.key.device != new_k.device:
            self.key   = self.key.to(new_k.device, non_blocking=True)
            self.value = self.value.to(new_v.device, non_blocking=True)
        if self.key.device != new_k.device:
            new_k = new_k.to(self.key.device, non_blocking=True)
            new_v = new_v.to(self.value.device, non_blocking=True)
        self.key   = torch.cat([self.key,   new_k], dim=2)
        self.value = torch.cat([self.value, new_v], dim=2)
        return self.key, self.value

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.key.shape  # (B, H, T)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _causal_mask(attn: Tensor, tgt_len: int, past_len: int) -> Tensor:
    causal   = torch.tril(torch.ones((tgt_len, tgt_len + past_len),
                                     device=attn.device, dtype=torch.bool))
    causal   = causal.unsqueeze(0).unsqueeze(0)           # (1,1,T,T_total)
    pad_mask = attn.unsqueeze(1).unsqueeze(2)             # (B,1,1,T_total)
    return causal & pad_mask

class SpecStep(NamedTuple):
    hidden: Tensor
    logits: Tensor
    accept: Tensor
    token : Tensor

# --------------------------------------------------------------------------- #
# Early‑exit LLAMA                                                            #
# --------------------------------------------------------------------------- #
class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, EARLY_STOP_LAYER: int):
        super().__init__(config)
        self.early_exit_layer = EARLY_STOP_LAYER
        self.past_key_values: Optional[List[SimpleKVCache]] = None

    # -------------------------- KV‑cache init --------------------------- #
    def _init_cache(self, bsz, n_layers, n_heads, head_dim, device):
        if self.past_key_values is not None:
            for i, kv in enumerate(self.past_key_values):
                if isinstance(kv, tuple):
                    self.past_key_values[i] = SimpleKVCache(kv[0], kv[1])
            return
        zeros = torch.zeros((bsz, n_heads, 0, head_dim),
                            dtype=self.dtype, device=device)
        self.past_key_values = [SimpleKVCache(zeros, zeros) for _ in range(n_layers)]

    # -------------------------- Core runner ---------------------------- #
    @torch.no_grad()
    def _run_layers(self, hidden, attn_mask, pos_ids, pos_emb, layers, start_idx):
        for offset, layer in enumerate(layers):
            idx        = start_idx + offset
            layer_dev  = next(layer.parameters()).device

            # ① migrate cache to the layer’s shard
            kv = self.past_key_values[idx]
            if kv.key.device != layer_dev:
                kv.key   = kv.key.to(layer_dev, non_blocking=True)
                kv.value = kv.value.to(layer_dev, non_blocking=True)

            # ② migrate inputs
            if hidden.device    != layer_dev: hidden    = hidden.to(layer_dev, non_blocking=True)
            if attn_mask.device != layer_dev: attn_mask = attn_mask.to(layer_dev, non_blocking=True)
            if pos_ids.device   != layer_dev: pos_ids   = pos_ids.to(layer_dev, non_blocking=True)
            if isinstance(pos_emb, tuple):
                pos_emb = tuple(p.to(layer_dev, non_blocking=True) for p in pos_emb)
            else:
                pos_emb = pos_emb.to(layer_dev, non_blocking=True)

            out = layer(
                hidden,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_value=kv,
                use_cache=True,
                output_attentions=False,
            )

            # HF variants
            present = None
            if isinstance(out, tuple):
                if len(out) == 2:   hidden, present = out
                elif len(out) == 3: hidden, _, present = out
                else:               hidden = out[0]
            else:
                hidden = out
            if present is not None:
                kv.update(present[0], present[1])
        return hidden

    # ------------------ Draft / verifier convenience ------------------- #
    @torch.no_grad()
    def forward_draft_or_large_model(
        self, *, in_tokens_small=None, in_features_large=None, position_ids=None
    ):
        if (in_tokens_small is None) == (in_features_large is None):
            raise ValueError("Specify exactly one of in_tokens_small or in_features_large")

        draft     = in_tokens_small is not None
        base      = in_tokens_small if draft else in_features_large
        bsz, step_len = base.shape[:2]
        nh        = self.config.num_attention_heads
        hd        = self.config.hidden_size // nh
        self._init_cache(bsz, len(self.model.layers), nh, hd, base.device)

        focus     = 0 if draft else self.early_exit_layer
        past_len  = self.past_key_values[focus].shape[2]
        total_len = past_len + step_len

        if position_ids is None:
            position_ids = (
                torch.arange(past_len, total_len, device=base.device)
                .unsqueeze(0)
                .expand(bsz, step_len)
            )

        if draft:
            inp_emb = self.model.embed_tokens(in_tokens_small)
            pos_emb = self.model.rotary_emb(inp_emb, position_ids)
        else:
            inp_emb = None
            pos_emb = self.model.rotary_emb(in_features_large, position_ids)

        attn_mask = _causal_mask(
            torch.ones((bsz, total_len), dtype=torch.bool, device=base.device),
            step_len,                       # <- was undefined “T”
            past_len,
        )

        if draft:
            layers, start, hidden_in = self.model.layers[:self.early_exit_layer], 0, inp_emb
        else:
            layers, start, hidden_in = self.model.layers[self.early_exit_layer:], self.early_exit_layer, in_features_large

        hidden = self._run_layers(hidden_in, attn_mask, position_ids, pos_emb, layers, start)
        return hidden if draft else (hidden, self.model.norm(hidden))

    # -------------------- One speculative micro‑step -------------------- #
    @torch.no_grad()
    def spec_decode_step(self, in_tokens_small, position_ids, exit_layer, temperature=1.0) -> SpecStep:
        self.early_exit_layer = exit_layer

        # ---------- 1. Draft pass ----------
        exit_h = self.forward_draft_or_large_model(
            in_tokens_small=in_tokens_small,
            position_ids=position_ids,
        )

        # match device *and* dtype of exit_proj.weight
        dev = self.exit_proj.weight.device
        dtype = self.exit_proj.weight.dtype
        draft_logits = self.exit_proj(
            exit_h.to(device=dev, dtype=dtype, non_blocking=True)
        ).float()

        # sample / greedy
        draft_tok = (
            draft_logits[:, -1].argmax(dim=-1, keepdim=True)
            if temperature == 0.0
            else torch.multinomial(torch.softmax(draft_logits[:, -1] / temperature, dim=-1), 1)
        )

        # ---------- 2. Verifier continuation ----------
        _, deep_h = self.forward_draft_or_large_model(
            in_features_large=exit_h, position_ids=position_ids
        )

        w = self.head_model.weight
        if deep_h.device != w.device or deep_h.dtype != w.dtype:
            deep_h_for_head = deep_h.to(device=w.device, dtype=w.dtype, non_blocking=True)
        else:
            deep_h_for_head = deep_h
        final_logits = self.head_model(deep_h_for_head).float()

        accept = (final_logits.argmax(dim=-1, keepdim=True) == draft_tok).to(torch.uint8)

        return SpecStep(
            hidden=exit_h.detach().cpu(),
            logits=draft_logits.detach().cpu(),
            accept=accept.cpu(),
            token=draft_tok.detach().cpu(),
        )
