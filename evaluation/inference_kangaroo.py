"""Generate answers with models from HuggingFace.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
import os

from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.kangaroo_model import KangarooModel
from kangaroo.earlyexit import SpecStep

def kangaroo_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    do_sample=False,
    max_length=2048,
    EARLY_STOP_LAYER=2,
    SPECULATIVE_DECODING_STEPS=6,
    threshold=0.6,
    return_trace: bool = False,
):
    context_tokens = inputs.input_ids
    device = context_tokens.device
    token_eos = tokenizer.eos_token_id
    batch_size, context_length = context_tokens.shape

    # --- DYNAMIC BUFFER SIZING -------------------
    # Ensure buffer can hold context + new tokens
    needed_len = context_length + max_new_tokens
    if max_length < needed_len:
        max_length = needed_len

    # Allocate token buffer and position ids
    global_tokens = torch.full(
        (batch_size, max_length),
        token_eos,
        dtype=torch.long,
        device=device,
    )
    global_position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    accept_length_list = [1]

    start_index = context_length
    global_tokens[:, :start_index] = context_tokens

    # Init KV-cache and sample the first token
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index]
        output = model.base_model(
            context_tokens,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=True,
        )
        model.base_model.past_key_values = list(output.past_key_values)
        hidden_state = output.hidden_states[-1]
        logits = output.logits
        global_tokens[:, start_index] = torch.argmax(
            logits[:, -1, :], dim=-1
        ).item()
        hidden_state_early = output.hidden_states[EARLY_STOP_LAYER]

        # KV-cache for the adapter
        hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(
            inputs_embeds=hidden_state_early[:, :, :],
            position_ids=global_position_ids[:, :context_length],
            use_cache=True,
        )

    if return_trace:
        trace: list[SpecStep] = []
        token = global_tokens[:, start_index - 1 : start_index]
        for _ in range(max_new_tokens):
            pos_ids = global_position_ids[:, start_index - 1 : start_index]
            step = model.base_model.spec_decode_step(
                in_tokens_small=token,
                position_ids=pos_ids,
                exit_layer=EARLY_STOP_LAYER,
                temperature=0.0 if not do_sample else 1.0,
            )
            trace.append(step)
            token = step.token
            global_tokens[:, start_index] = token.squeeze(-1)
            start_index += 1
            if token.eq(token_eos).all().item() or start_index >= max_length:
                break

        output_ids = global_tokens[0, :start_index].tolist()
        new_token = start_index - context_length
        idx = len(trace) - 1
        return [output_ids], new_token, idx, [int(s.accept.item()) for s in trace], trace

    total_inference_steps = 0

    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False

        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:

            start_index_copy = start_index
            end_index = start_index + 1

            # STEP 1: Small model decoding
            for step in range(1 + SPECULATIVE_DECODING_STEPS):
                assert adapter_past_key_values[0][0].shape[2] <= end_index - 1, (
                    "{} - {}".format(adapter_past_key_values[0][0].shape, end_index - 1)
                )
                in_tokens_small = global_tokens[:, end_index - 1 : end_index]
                if adapter_past_key_values[0][0].shape[2] < end_index - 1:
                    # Once all drafted tokens are accepted, the KV-cache
                    # of the last draft token for the adapter is missing.
                    position_ids = global_position_ids[:, start_index - 1 : end_index]
                    hidden_state_early_last = exited_hidden_states[:, -1:, :]
                else:
                    position_ids = global_position_ids[:, end_index - 1 : end_index]
                    hidden_state_early_last = None

                hidden_state_early = model.base_model.forward_draft_or_large_model(
                    in_tokens_small=in_tokens_small[:, -1:], position_ids=position_ids[:, -1:]
                )

                if step == 0:
                    exited_hidden_states = None

                exited_hidden_states = (
                    hidden_state_early
                    if exited_hidden_states is None
                    else torch.cat([exited_hidden_states, hidden_state_early], dim=1)
                )

                if hidden_state_early_last is not None:
                    hidden_state_early = torch.cat(
                        [hidden_state_early_last, hidden_state_early], dim=1
                    )

                # early exiting
                if step == SPECULATIVE_DECODING_STEPS or (
                    step > 0 and predict_score < threshold
                ):
                    break

                hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(
                    inputs_embeds=hidden_state_early,
                    position_ids=position_ids,
                    past_key_values=adapter_past_key_values,
                    use_cache=True,
                )

                predict_logits = model.head_model(hidden_state[:, -1:, :]).float()
                global_tokens[:, end_index] = torch.argmax(
                    predict_logits[:, -1, :], dim=-1
                )

                end_index += 1
                predict_score = predict_logits.softmax(dim=-1).max().item()

            # STEP 2: Big model inference
            position_ids = global_position_ids[:, start_index:end_index]
            assert (
                model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape[2]
                == start_index
            ), "{} - {}".format(
                model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape,
                start_index,
            )
            assert exited_hidden_states.shape[1] == position_ids.shape[1]
            hidden_state_, hidden_state = model.base_model.forward_draft_or_large_model(
                in_features_large=exited_hidden_states, position_ids=position_ids
            )

            logits = model.head_model(hidden_state).float()
            output_tokens = torch.argmax(logits[:, :, :], dim=-1)

            # Verification for greedy decoding
            output_length = end_index - start_index
            for i in range(output_length):
                if (
                    i == output_length - 1
                    or output_tokens[0, i] == token_eos
                    or output_tokens[0, i] != global_tokens[0, start_index + 1 + i]
                ):
                    global_tokens[0, start_index + 1 + i] = output_tokens[0, i]
                    start_index = start_index + 1 + i
                    if output_tokens[0, i] == token_eos:
                        stop = True
                    break

            accept_length_list.append(start_index - start_index_copy)
            hidden_state = hidden_state[:, : output_length - (end_index - start_index), :]

            # STEP 4: Post process KV-cache
            if model.base_model.past_key_values[0][0].shape[2] > start_index:
                past_key_values_large_ = []
                for k, v in model.base_model.past_key_values:
                    past_key_values_large_.append(
                        (k[:, :, :start_index, :], v[:, :, :start_index, :])
                    )
                model.base_model.past_key_values = past_key_values_large_

            if adapter_past_key_values[0][0].shape[2] > start_index:
                adapter_past_key_values_ = []
                for k, v in adapter_past_key_values:
                    adapter_past_key_values_.append(
                        (k[:, :, :start_index, :], v[:, :, :start_index, :])
                    )
                adapter_past_key_values = tuple(adapter_past_key_values_)
                del adapter_past_key_values_

            total_inference_steps += 1

            if stop:
                break

    output_ids = global_tokens[0, : start_index + 1].tolist()
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1
    return [output_ids], new_token, idx, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--exitlayer",
        type=int,
        default=2,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="The number of GPUs per model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    question_file = f"data/question.jsonl"

    model = KangarooModel(
        args.model_path,
        args.adapter_path,
        args,
        EARLY_STOP_LAYER=args.exitlayer,
    )
    model.base_model.parallelize()
    model.adapter_model.to("cuda:0")
    model.exit_proj.to("cuda:0")
    model.head_model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    do_sample = False

    assert not args.answer_file
    os.makedirs(f"data/{args.bench_name}/{args.model_id}", exist_ok=True)

    for run in range(3):
        answer_file = f"data/{args.bench_name}/{args.model_id}/{run}.jsonl"
        print(f"Output to {answer_file}")

        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=kangaroo_forward,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            do_sample=do_sample,
            threshold=args.threshold,
            SPECULATIVE_DECODING_STEPS=args.steps,
            EARLY_STOP_LAYER=args.exitlayer,
        )

        reorg_answer_file(answer_file)
