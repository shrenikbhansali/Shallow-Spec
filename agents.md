# DVI-RL Self-Speculation Training Project

---

## 0. Motivation & Intention

### Background

Speculative decoding is typically used *only* for speeding up inference in large language models (LLMs). A shallow “draft” network proposes tokens, which a deeper “verifier” network then either accepts or rejects. This pipeline saves compute during generation.

Existing work (like Kangaroo) demonstrates self-speculative decoding for lossless acceleration, but *does not* use speculative decoding for learning.

### What We Are Doing

This project proposes **Draft→Verify→Improve Reinforcement Learning (DVI-RL)**:

> Use speculative decoding as a *training-time* primitive. The draft module acts as a policy proposing tokens. The verifier module acts as a learned critic that decides whether the proposal is accepted. The draft module is then updated to improve its probability of proposing tokens that the verifier accepts.

**This is novel because:**

* It leverages speculative decoding not for inference efficiency, but as a *learning signal*.
* It unifies language modeling with reinforcement learning from model-generated feedback.

**Intended Use Cases:**

* Continual or online learning from streaming conversational data (e.g., ShareGPT).
* Learning from unlabelled interaction logs without requiring external reward models.
* Training shallow adapters in resource-constrained settings (e.g., federated learning).

---

## 1. High-Level System Overview

```
┌────────────────────────┐
│        Prompt          │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│  Shallow Draft Layers  │
│ (EarlyExit up to L=k)  │
└─────────┬──────────────┘
          │ logits_draft
          ▼
┌────────────────────────┐
│      Sample token      │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│  Deep Verifier Layers  │
│ (Remaining L-L_k)      │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│     Accept / Reject    │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────────┐
│      REINFORCE Update      │
│   (Draft gets reward=1/0)  │
└────────────────────────────┘
```

---

## 2. Core Technical Concepts

### 2.1 Draft Model (Policy π)

* The *shallow* part of the transformer up to a configurable `EARLY_STOP_LAYER`.
* Produces draft logits:

  ```
  draft_logits = exit_proj(exit_hidden)
  ```
* These logits are a categorical distribution π over the vocabulary.
* Sampling is done deterministically (argmax) or with temperature.

**LoRA Parameters:**

* **`lora_S`**: The adapters attached only to the shallow draft layers.
* These are the parameters updated by REINFORCE.

---

### 2.2 Verifier Model

* The *deep* part of the transformer (from `EARLY_STOP_LAYER+1` to end).
* Computes the final logits:

  ```
  final_logits = head_model(deep_hidden)
  ```
* Accepts the draft token if it matches the verifier argmax.
* Optionally outputs a confidence score (softmax probability).

**LoRA Parameters:**

* **`lora_D`**: The adapters attached only to the deep verifier layers.
* These can be optionally fine-tuned with supervised loss (cross-entropy).

---

### 2.3 Reinforcement Learning Objective

The draft module is trained to maximize the expected reward:

```
L_draft = -E[ (r - baseline) * log π(t|h) ]
```

where:

* `r = 1` if verifier accepted the token.
* `baseline` = running exponential average of the accept rate.

This is a REINFORCE loss with a simple scalar reward and a baseline to reduce variance.

---

### 2.4 Slow Verifier Update

Periodically, we also fine-tune the verifier module to improve its consistency:

```
L_verifier = CE(verifier_logits, t) + λ * KL(verifier_logits || verifier_logits_frozen)
```

where:

* CE = cross-entropy loss on observed tokens.
* KL = conservative penalty to avoid drift.
* λ = scaling factor.

This is **optional** but recommended for stabilizing learning.

---

### 2.5 Replay Buffer

Because generation is sequential and samples are sparse, we maintain a buffer:

* Each record:

  ```
  {
    "hidden": exit_hidden,
    "token": sampled_token,
    "reward": accept/reject,
    "verifier_confidence": p
  }
  ```
* When buffer size reaches `FAST_BATCH_SIZE`, we process a batch RL update.

---


## Module Directory Layout

* `kangaroo/`

  * `kangaroo_model.py`

    * Wraps a base Llama model.
    * Defines `KangarooModel`, which exposes `forward()` producing:

      * Draft logits (`exit_proj`).
      * Final logits (`head_model`).
  * `earlyexit.py`

    * Provides `EarlyExitLlamaForCausalLM` with:

      * `forward_draft_or_large_model()` that:

        * Accepts inputs.
        * Returns hidden states at each speculative step.
        * Updates `past_key_values`.
  * `sgp_lora.py`

    * Defines `inject_lora()` to:

      * Add **LoRA adapters** to a model.
      * For DVI-RL, must support:

        * LoRA adapters named `lora_S` for shallow policy.
        * LoRA adapters named `lora_D` for slow verifier.

* `train_dvi.py`

  * Main training loop:

    * Streaming ShareGPT dataset.
    * Collects speculative generations.
    * Stores buffer of `(hidden, token, accept signal)`.
    * Runs REINFORCE on `lora_S`.
    * Periodically fine-tunes `lora_D`.
  * Must be stateless between process restarts.
  * Logs every step to JSONL.

* `evaluation/sgp_eval.py`

  * Evaluates accept-rate, throughput, perplexity.
  * **Unchanged**, can be adapted to DVI-RL.

* `run_dvi_experiment.py`

  * Orchestrates training and evaluation.

---

## Module Responsibilities

### kangaroo\_model.py

* Defines `KangarooModel`.
* Must expose:

  * `.exit_proj` → torch.nn.Linear.
  * `.head_model` → torch.nn.Linear.
* Must not hold optimizer state.
* Accepts:

  ```python
  KangarooModel(
    base_model_name_or_path: str,
    adapter_model_path: Optional[str],
    args: Namespace,
    EARLY_STOP_LAYER: int
  )
  ```
* For DVI-RL, `exit_proj` produces policy logits π.

---

### earlyexit.py

* Provides:

  * `forward_draft_or_large_model`:

    * Must return:

      * Hidden states **at every speculative step**.
      * Verifier logits.
    * Updates `past_key_values`.

---

### sgp\_lora.py

* Must define:

  ```python
  inject_lora(base_model, exit_layer: int) -> PeftModel
  ```
* For DVI-RL:

  * Accepts config for `lora_S` and `lora_D`.
  * `lora_S`: attached to layers <= exit\_layer.
  * `lora_D`: attached to layers > exit\_layer.
* Must expose:

  * `get_peft_model_state_dict()` for checkpointing.

---

### train\_dvi.py

* Implements **Draft→Verify→Improve RL training loop**.
* Requirements:

  * Maintains a buffer (ReplayBuffer) storing:

    * `exit_hidden: torch.FloatTensor`
    * `draft_token: int`
    * `reward: float`
    * `verifier_confidence: float`
  * Baseline is Exponential Moving Average.
  * Every **FAST\_BATCH\_SIZE accepted tokens**, computes:

    ```python
    loss_rl = -E[(r - baseline) * log π(x̂|h)]
    ```

    and steps `lora_S`.
  * Every **SLOW\_EVERY fast steps**, computes:

    ```python
    loss_slow = CE(verifier_logits, x̂) + λ KL(verifier_logits || frozen)
    ```

    and steps `lora_D`.
  * Logs:

    * `step`
    * `loss_rl`
    * `loss_slow`
    * `accept_rate`
    * `ppl_sample`
  * Should expose a `main(args: Namespace)` function.

---

### run\_dvi\_experiment.py

* Orchestrates:

  * Initial LoRA injection.
  * Training (`train_dvi.py`).
  * Evaluation (`evaluation/sgp_eval.py`).
* Arguments:

  * `--model_name`
  * `--exit_layer`
  * `--fast_batch`
  * `--slow_every`
  * `--beta_kl`
  * `--baseline_momentum`
* Outputs:

  * `results/` JSONs.

---

## Data & Training

* **Input Data**: ShareGPT or other conversation logs.
* **Tokenization**: `AutoTokenizer` with `bos_token_id`.
* **Sampling**:

  * Deterministic (`do_sample=False`).
  * Max tokens=64.
* **Reward Signal**:

  * `1.0` if accepted, `0.0` if rejected.
  * Verifier confidence computed as `softmax(logits)[token]`.

---

## Coding Conventions

* All logs: JSONL.
* All optimizers: AdamW.
* `lora_S` LR = `2e-4`.
* `lora_D` LR = `5e-6`.
* LoRA adapters must be checkpointed separately.
* `torch.bfloat16` by default.
* All modules must be `torch.nn.Module`.
* No global state except:

  * `baseline` EMA.
  * ReplayBuffer.

---

## Expected Flow

1. `run_dvi_experiment.py`:

   * Instantiates `KangarooModel`.
   * Injects dual LoRA (`lora_S`, `lora_D`).
   * Calls `train_dvi.py`.
2. `train_dvi.py`:

   * For each prompt:

     * Runs `kangaroo_forward`.
     * Collects `(h, token, acc, conf)`.
   * On buffer full:

     * Computes REINFORCE loss.
     * Updates `lora_S`.
   * On `slow_every`:

     * Computes slow CE+KL loss.
     * Updates `lora_D`.
3. `evaluation/sgp_eval.py`:

   * Measures accept rate and perplexity.

---

## Example Usage

```python
from kangaroo.kangaroo_model import KangarooModel
from kangaroo.sgp_lora import inject_lora
from train_dvi import main as train_dvi_main

model = KangarooModel(...)
model.base_model = inject_lora(model.base_model, exit_layer=6)

args = Namespace(...)
train_dvi_main(args)
```


