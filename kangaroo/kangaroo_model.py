import os
import json
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load

from fastchat.utils import str_to_torch_dtype
from transformers.models.llama import LlamaConfig

from kangaroo.adapter import AdapterModel
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from transformers.models.llama.modeling_llama import create_causal_mask
import torch.nn.functional as F


class KangarooModel(nn.Module):

    def __init__(
        self,
        base_model_name_or_path,
        adapter_model_path,
        args,
        EARLY_STOP_LAYER=2,
    ):
        super().__init__()
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            device_map="auto",
            EARLY_STOP_LAYER=EARLY_STOP_LAYER,
        )
        self.base_model = self.base_model.eval()

        self.exit_layer = EARLY_STOP_LAYER

        config = LlamaConfig.from_pretrained(base_model_name_or_path)
        self.adapter_model = AdapterModel(config)

        if adapter_model_path is not None:
            if os.path.isdir(adapter_model_path):
                adapter_weights = os.path.join(adapter_model_path, "pytorch_model.bin")
            else:
                adapter_weights = hf_hub_download(adapter_model_path, "pytorch_model.bin")

            self.adapter_model.load_state_dict(torch.load(adapter_weights, map_location="cpu"), strict=False)
        self.adapter_model = self.adapter_model.eval().to(self.base_model.device)

        if args.dtype == "float16":
            self.adapter_model = self.adapter_model.half()

        self.head_model = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        head_path = None
        if os.path.isdir(base_model_name_or_path):
            index_file = os.path.join(base_model_name_or_path, "pytorch_model.bin.index.json")
            if os.path.exists(index_file):
                with open(index_file, "r") as f:
                    index_json = json.loads(f.read())
                    head_path = index_json["weight_map"]["lm_head.weight"]
        else:
            try:
                index_file = hf_hub_download(base_model_name_or_path, "pytorch_model.bin.index.json")
                with open(index_file, "r") as f:
                    index_json = json.loads(f.read())
                    head_path = index_json["weight_map"]["lm_head.weight"]
            except Exception:
                head_path = None

        if head_path is None:
            # fall back to single weight file
            try:
                head_path = "pytorch_model.bin"
                if not os.path.isdir(base_model_name_or_path):
                    hf_hub_download(base_model_name_or_path, head_path)
            except Exception:
                head_path = "model.safetensors"
                if not os.path.isdir(base_model_name_or_path):
                    hf_hub_download(base_model_name_or_path, head_path)

        if os.path.isdir(base_model_name_or_path):
            head_file = os.path.join(base_model_name_or_path, head_path)
        else:
            head_file = hf_hub_download(base_model_name_or_path, head_path)

        if head_file.endswith(".safetensors"):
            weights = safe_load(head_file)
        else:
            weights = torch.load(head_file, map_location="cpu")
        tensor = weights["lm_head.weight"].float()
        self.head_model.weight.data = tensor
        self.head_model = self.head_model.eval().to(self.base_model.device)

        self.exit_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.exit_proj.weight.data = tensor.clone()

        if args.dtype == "float16":
            self.head_model = self.head_model.half()
            self.exit_proj = self.exit_proj.half()

    def forward(self, input_ids, labels=None, beta_exit=0.1, detach_exit=True):
        """Run the model with an early-exit head.

        Parameters
        ----------
        input_ids: torch.LongTensor
            Input token ids of shape ``(batch, seq_len)``.
        labels: torch.LongTensor, optional
            Target labels used to compute the loss.
        beta_exit: float
            Weight of the auxiliary exit loss.
        detach_exit: bool
            Whether to detach the hidden state before the final layers.
        """

        model = self.base_model.model
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        inputs_embeds = model.embed_tokens(input_ids)

        attention_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
        cache_pos = torch.arange(seq_len, device=device)
        causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_pos,
            past_key_values=None,
        )

        position_ids = cache_pos.unsqueeze(0)
        pos_embeds = model.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in model.layers[: self.exit_layer + 1]:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_pos,
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
                    cache_position=cache_pos,
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
