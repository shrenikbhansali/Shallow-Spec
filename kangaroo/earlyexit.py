# kangaroo/earlyexit.py
# --------------------------------------------------------------------------- #
# Early-exit LLAMA utilities for self-speculative decoding                    #
# Provides:
#   - SimpleKVCache: a minimal KV cache with an .update() API
#   - _causal_mask: batched causal mask without HF private helpers
#   - EarlyExitLlamaForCausalLM: draft/verifier forward & spec_decode_step
# --------------------------------------------------------------------------- #
from __future__ import annotations

from typing import NamedTuple, Tuple, List, Optional

import torch
from torch import Tensor
from transformers.models.llama import LlamaForCausalLM


# --------------------------------------------------------------------------- #
# Minimal KV-cache compatible with HF attention layers                        #
# --------------------------------------------------------------------------- #
class SimpleKVCache:
    __slots__ = ("key", "value")

    def __init__(self, key: Tensor, value: Tensor):
        # key,value: (batch, num_heads, seq_len, head_dim)
        self.key, self.value = key, value

    def update(self, new_k: Tensor, new_v: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Append along time dimension, return (key, value) so HF attention can unpack.
        Any extra HF args/kwargs are ignored.
        """
        # concat on seq_len axis (dim=2)
        self.key = torch.cat([self.key, new_k], dim=2)
        self.value = torch.cat([self.value, new_v], dim=2)
        return self.key, self.value

    @property
    def shape(self) -> Tuple[int, int, int]:
        # returns (batch, num_heads, seq_len)
        return self.key.shape


# --------------------------------------------------------------------------- #
# Build a local (B,1,T,T_total) causal mask from a (B, T_total) bool mask     #
# --------------------------------------------------------------------------- #
def _causal_mask(attn: Tensor, tgt_len: int, past_len: int) -> Tensor:
    """
    attn: (B, T_total) boolean mask (all True for no padding).
    Returns: (B, 1, tgt_len, T_total) boolean mask for autoregressive decoding.
    """
    # causal part: (1,1,tgt_len, tgt_len+past_len)
    causal = torch.tril(
        torch.ones((tgt_len, tgt_len + past_len), device=attn.device, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)
    # embed attn: (B,1,1,T_total)
    pad_mask = attn.unsqueeze(1).unsqueeze(2)
    return causal & pad_mask


# --------------------------------------------------------------------------- #
# SpecStep: data returned by one spec_decode_step                             #
# --------------------------------------------------------------------------- #
class SpecStep(NamedTuple):
    hidden: Tensor   # (B, 1, H)
    logits: Tensor   # (B, 1, V)
    accept: Tensor   # (B, 1) uint8
    token: Tensor    # (B, 1) long


# --------------------------------------------------------------------------- #
# Early-exit LLAMA with draft/verifier & spec_decode_step helpers            #
# --------------------------------------------------------------------------- #
class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, EARLY_STOP_LAYER: int):
        super().__init__(config)
        self.early_exit_layer = EARLY_STOP_LAYER
        # will hold List[SimpleKVCache] after first run
        self.past_key_values: Optional[List[SimpleKVCache]] = None

    def _init_cache(self, bsz: int, n_layers: int, n_heads: int, head_dim: int, device: torch.device):
        """
        Initialize or wrap past_key_values to SimpleKVCache.
        """
        if self.past_key_values is not None:
            # wrap any HF-preserved tuples into our SimpleKVCache
            for i, kv in enumerate(self.past_key_values):
                if isinstance(kv, tuple):
                    self.past_key_values[i] = SimpleKVCache(kv[0], kv[1])
            return

        # create empty keys/values: seq_len=0
        zeros = torch.zeros((bsz, n_heads, 0, head_dim), dtype=self.dtype, device=device)
        self.past_key_values = [SimpleKVCache(zeros, zeros) for _ in range(n_layers)]

    @torch.no_grad()
    def _run_layers(
        self,
        hidden: Tensor,
        attn_mask: Tensor,
        pos_ids: Tensor,
        pos_emb: Tensor,
        layers,
        start_idx: int,
    ) -> Tensor:
        """
        Run a slice of decoder layers [start_idx : start_idx+len(layers)],
        updating KV cache via SimpleKVCache.update().
        """
        for offset, layer in enumerate(layers):
            idx = start_idx + offset
            out = layer(
                hidden,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_value=self.past_key_values[idx],
                use_cache=True,
                output_attentions=False,
            )
            # HF layer may return hidden only, or (hidden, present), or (hidden, attn, present)
            present = None
            if isinstance(out, tuple):
                if len(out) == 2:
                    hidden, present = out
                elif len(out) == 3:
                    hidden, _, present = out
                else:
                    hidden = out[0]
            else:
                hidden = out
            if present is not None:
                # present is a (key, value) tuple
                self.past_key_values[idx].update(present[0], present[1])
        return hidden

    @torch.no_grad()
    def forward_draft_or_large_model(
        self,
        *,
        in_tokens_small: Tensor | None = None,
        in_features_large: Tensor | None = None,
        position_ids: Tensor | None = None,
    ):
        """
        Draft pass (token→hidden) if in_tokens_small provided,
        or Verifier continuation (hidden→hidden, returns (hidden, normed_hidden))
        """
        # exactly one of these must be not None
        if (in_tokens_small is None) == (in_features_large is None):
            raise ValueError("Specify exactly one of in_tokens_small or in_features_large")

        draft = in_tokens_small is not None
        base = in_tokens_small if draft else in_features_large
        bsz, step_len = base.shape[:2]

        # lazy-init KV cache
        nh = self.config.num_attention_heads
        hd = self.config.hidden_size // nh
        self._init_cache(bsz, len(self.model.layers), nh, hd, base.device)

        # how much past we have at this branch
        focus_layer = 0 if draft else self.early_exit_layer
        past_len = self.past_key_values[focus_layer].shape[2]
        total_len = past_len + step_len

        # default position_ids if not given
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, total_len, device=base.device)
                .unsqueeze(0)
                .expand(bsz, step_len)
            )

        # embeddings + rotary pos emb
        if draft:
            inp_emb = self.model.embed_tokens(in_tokens_small)
            pos_emb = self.model.rotary_emb(inp_emb, position_ids)
        else:
            inp_emb = None
            pos_emb = self.model.rotary_emb(in_features_large, position_ids)

        # causal mask
        attn = torch.ones((bsz, total_len), dtype=torch.bool, device=base.device)
        attn_mask = _causal_mask(attn, step_len, past_len)

        # select the slice of layers
        if draft:
            layers = self.model.layers[: self.early_exit_layer]
            start = 0
            hidden_in = inp_emb
        else:
            layers = self.model.layers[self.early_exit_layer :]
            start = self.early_exit_layer
            hidden_in = in_features_large

        # run and return
        hidden = self._run_layers(
            hidden=hidden_in,
            attn_mask=attn_mask,
            pos_ids=position_ids,
            pos_emb=pos_emb,
            layers=layers,
            start_idx=start,
        )
        if draft:
            return hidden         # (B,1,H)
        else:
            return hidden, self.model.norm(hidden)

    @torch.no_grad()
    def spec_decode_step(
        self,
        in_tokens_small: Tensor,
        position_ids: Tensor,
        exit_layer: int,
        temperature: float = 1.0,
    ) -> SpecStep:
        """
        One speculative decoding step:
          1) draft up to exit_layer → draft_logits & sample→draft_tok
          2) verifier on that hidden → final_logits → accept
        Returns SpecStep(hidden, logits, accept, token)
        """
        # set exit point
        self.early_exit_layer = exit_layer

        # 1. draft pass to get exit_hidden
        exit_h = self.forward_draft_or_large_model(
            in_tokens_small=in_tokens_small,
            position_ids=position_ids,
        )
        draft_logits = self.exit_proj(exit_h).float()

        # sample or greedy
        if temperature == 0.0:
            draft_tok = draft_logits[:, -1].argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(draft_logits[:, -1] / temperature, dim=-1)
            draft_tok = torch.multinomial(probs, num_samples=1)

        # 2. verifier continuation
        _, deep_h = self.forward_draft_or_large_model(
            in_features_large=exit_h, position_ids=position_ids
        )
        final_logits = self.head_model(deep_h).float()

        accept = (final_logits.argmax(dim=-1, keepdim=True) == draft_tok).to(torch.uint8)

        return SpecStep(
            hidden=exit_h.detach().cpu(),
            logits=draft_logits.detach().cpu(),
            accept=accept.cpu(),
            token=draft_tok.detach().cpu(),
        )
