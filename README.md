We build ontop of the following codebase: Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting</h1></div>



## This is their explanation / Kangaroo:

self-speculative decoding framework Kangaroo, which uses a fixed shallow sub-network as a self-draft model, with the remaining layers serving as the larger target model. We train a lightweight and efficient adapter module on top of the sub-network to bridge the gap between the sub-network and the full model’s representation ability. The adapter network consists of only one multi-head attention and two
normalization layers. Surprisingly, we find this simple design efficient but powerful. To further reduce the inference latency of the self-draft model, we introduce an additional early exiting mechanism for generating draft tokens, aiming to avoid
unnecessary costs on more difficult tokens.


##  We implement shallow-gradient propagation (SGP)

In normal fine-tuning, the loss is computed on the **final** decoder layer; the gradient must flow backward through the entire stack. For very deep models this gradient can become weak or noisy by the time it reaches the earliest blocks (vanishing-/sparse-gradient problem).

**SGP** inserts an **auxiliary early-exit head** after a shallow layer **L** (here `exit_layer = 6`).
During training we:

* **Freeze** every base weight (so we never disturb the pretrained representation).
* Add **LoRA adapters** only to layers 0 … L.
* Compute two losses

  * `loss_exit` on the logits produced by the early-exit head (always back-propagated).
  * `loss_main` on the final head.
* **Detach** the hidden state that feeds the later layers (`detach_exit=True`), so `loss_main` does **not** back-prop through layers 0 … L.

  ```python
  hidden_final_input = exit_hidden.detach()   # gradient stops here
  ```

  Therefore the *only* gradients reaching layers 0 … L come from `loss_exit`; the path is short (≤ L + 1 layers) → **shallow-gradient path**.

---

### How the code is organized

| Component                        | Where                                      | Key line(s)                                                                   |
| -------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------- |
| **Early-exit head**              | `kangaroo_model.py` → `self.exit_proj`     | `draft_logits = self.exit_proj(exit_hidden)`                                  |
| **LoRA injection on layers ≤ L** | `inject_lora()`                            | Checks layer index before enabling `requires_grad=True`.                      |
| **Detach flag**                  | `kangaroo_model.forward()`                 | `exit_hidden.detach() if detach_exit else exit_hidden`                        |
| **Optimizer**                    | `train_sgp.py`                             | Filters `p.requires_grad`, so *only* LoRA matrices + exit head are trainable. |
| **Combined loss**                | `loss = loss_main + beta_exit * loss_exit` | With `beta_exit = 0.7` we emphasise the shallow path.                         |

Because the verifier layers (`> L`) run under the detached hidden state, they are **frozen**: no gradients, no weight updates, no extra memory.


## Running the code

#### Running Shallow-Gradient-Path Fine-Tuning

We train a LoRA adapter on the [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
dataset using any HuggingFace model:

```
python train_sgp.py --model_name <hf_model_name> --exit_layer 6
```


#### Evaluating an SGP Run:
This runs the evaluation scrip (WIP), which checks perplexity, token acceptance rate, and throughput
At some point, we need to be evaluating using specbench, since that's the literature standard at this point 
```
python evaluation/sgp_eval.py \
  --ckpt_dir checkpoints/sgp_adapter \
  --model_name meta-llama/Llama-2-7b-hf \
  --exit_layer 6 \
  --n_samples 64
```

#### Running an end to end SGP-FT Experiment:
This first fine-tunes the model with SGP (works), saves a model checkpoint locally, and then evaluates the model using the eval script from above
```
python run_sgp_experiment.py --model_name meta-llama/Llama-2-7b-hf 
```
