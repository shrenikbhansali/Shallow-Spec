import os
import json
import torch
import torch.nn as nn

from fastchat.utils import str_to_torch_dtype
from transformers.models.llama import LlamaConfig

from kangaroo.adapter import AdapterModel
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
import torch.nn.functional as F

class KangarooModel(nn.Module):

    def __init__(
            self,
            base_model_name_or_path,
            adapter_model_path,
            args,
            EARLY_STOP_LAYER = 2,
    ):
        super().__init__()
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=str_to_torch_dtype(args.dtype), device_map="auto", EARLY_STOP_LAYER = EARLY_STOP_LAYER)
        self.base_model = self.base_model.eval()

        self.exit_layer = EARLY_STOP_LAYER

        config = LlamaConfig.from_pretrained(os.path.join(adapter_model_path, "config.json"))
        self.adapter_model = AdapterModel(config)

        self.adapter_model.load_state_dict(torch.load(os.path.join(adapter_model_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
        self.adapter_model = self.adapter_model.eval().to(self.base_model.device)

        if args.dtype == "float16":
            self.adapter_model = self.adapter_model.half()

        self.head_model = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with open(os.path.join(base_model_name_or_path, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(base_model_name_or_path, head_path), map_location='cpu')
        tensor = weights["lm_head.weight"].float()
        self.head_model.weight.data = tensor
        self.head_model = self.head_model.eval().to(self.base_model.device)

        self.exit_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.exit_proj.weight.data = tensor.clone()

        if args.dtype == "float16":
            self.head_model = self.head_model.half()
            self.exit_proj = self.exit_proj.half()

    def forward(self, input_ids, labels=None, beta_exit=0.1, detach_exit=True):
        model = self.base_model.model
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        inputs_embeds = model.embed_tokens(input_ids)
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
        cache_pos = torch.arange(seq_len, device=device)
        causal_mask = model._update_causal_mask(attention_mask, inputs_embeds, cache_pos, None, False)
        position_ids = cache_pos.unsqueeze(0)
        pos_embeds = model.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for idx, layer in enumerate(model.layers[: self.exit_layer + 1]):
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=position_ids,
                position_embeddings=pos_embeds,
            )
            hidden_states = layer_outputs[0]

        exit_hidden = hidden_states

        hidden_final_input = exit_hidden.detach() if detach_exit else exit_hidden
        with torch.no_grad():
            hidden_states = hidden_final_input
            for layer in model.layers[self.exit_layer + 1 :]:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=position_ids,
                    position_embeddings=pos_embeds,
                )
                hidden_states = layer_outputs[0]
            hidden_states = model.norm(hidden_states)

        draft_logits = self.exit_proj(exit_hidden)
        final_logits = self.head_model(hidden_states)

        loss_exit = None
        loss_main = None
        if labels is not None:
            loss_main = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss_exit = F.cross_entropy(
                draft_logits.view(-1, draft_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss_main + beta_exit * loss_exit
        else:
            loss = None

        return loss, {
            "loss_main": loss_main.detach() if loss_main is not None else None,
            "loss_exit": loss_exit.detach() if loss_exit is not None else None,
        }






