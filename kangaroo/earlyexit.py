from __future__ import annotations

from typing import NamedTuple

import torch
from transformers.models.llama import LlamaForCausalLM


class SpecStep(NamedTuple):
    """Data for a single speculative decoding step."""

    hidden: torch.Tensor
    logits: torch.Tensor
    accept: torch.Tensor
    token: torch.Tensor


class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, EARLY_STOP_LAYER):
        super().__init__(config)

        self.past_key_values = None
        self.early_exit_layer = EARLY_STOP_LAYER

    @torch.no_grad()
    def forward_draft_or_large_model(self, in_tokens_small=None, in_features_large=None, position_ids=None):
        use_cache = True
        assert self.past_key_values is not None

        if in_tokens_small is not None and in_features_large is not None:
            raise ValueError("You cannot specify both in_tokens_small and in_features_large at the same time")
        elif in_tokens_small is not None:
            batch_size, seq_length = in_tokens_small.shape
        elif in_features_large is not None:
            batch_size, seq_length, _ = in_features_large.shape
        else:
            raise ValueError("You have to specify either in_tokens_small or in_features_large")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        focu_layer = 0 if in_tokens_small is not None else self.early_exit_layer
        past_key_values_length = self.past_key_values[focu_layer][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = in_tokens_small.device if in_tokens_small is not None else in_features_large.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if in_tokens_small is not None:
            inputs_embeds = self.model.embed_tokens(in_tokens_small)
        
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device if in_tokens_small is not None else in_features_large.device
        )
        attention_mask = self.model._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds if in_tokens_small is not None else in_features_large, past_key_values_length
        )

        hidden_states = inputs_embeds if in_tokens_small is not None else in_features_large
        LAYERS = self.model.layers[:self.early_exit_layer] if in_tokens_small is not None else self.model.layers[self.early_exit_layer:]

        for idx, decoder_layer in enumerate(LAYERS):
            layer_idx = idx if in_tokens_small is not None else idx + self.early_exit_layer
            past_key_value = self.past_key_values[layer_idx] # if self.past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            self.past_key_values[layer_idx] = layer_outputs[1]
        
        if in_features_large is not None:
            return hidden_states, self.model.norm(hidden_states)
        return hidden_states

    @torch.no_grad()
    def spec_decode_step(
        self,
        in_tokens_small: torch.LongTensor,
        position_ids: torch.LongTensor,
        exit_layer: int,
        temperature: float = 1.0,
    ) -> SpecStep:
        """Generate one token using speculative decoding.

        Args:
            in_tokens_small: Input tokens of shape ``[B, 1]``.
            position_ids: Position ids of shape ``[B, 1]``.
            exit_layer: Index of the last draft layer.
            temperature: Sampling temperature. ``0`` for greedy.

        Returns:
            SpecStep: hidden state, draft logits and accept flag.
        """

        self.early_exit_layer = exit_layer
        assert hasattr(self, "exit_proj") and hasattr(self, "head_model")

        exit_hidden = self.forward_draft_or_large_model(
            in_tokens_small=in_tokens_small, position_ids=position_ids
        )
        draft_logits = self.exit_proj(exit_hidden).float()

        if temperature == 0.0:
            draft_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
        else:
            probs = torch.softmax(draft_logits[:, -1, :] / temperature, dim=-1)
            draft_token = torch.multinomial(probs, num_samples=1)

        _, deep_hidden = self.forward_draft_or_large_model(
            in_features_large=exit_hidden, position_ids=position_ids
        )
        final_logits = self.head_model(deep_hidden).float()

        accept = (final_logits.argmax(dim=-1, keepdim=True) == draft_token).to(torch.uint8)

        return SpecStep(
            hidden=exit_hidden.detach().cpu(),
            logits=draft_logits.detach().cpu(),
            accept=accept.cpu(),
            token=draft_token.detach().cpu(),
        )
