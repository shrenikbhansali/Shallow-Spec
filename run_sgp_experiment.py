import argparse
import json
import time
import subprocess
from pathlib import Path

import train_sgp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    # 1. Train
    train_args = argparse.Namespace(
        model_name=args.model_name,
        exit_layer=6,
        beta_exit=0.7,
        detach_exit=True,
        epochs=1,
        per_device_batch=2,
    )
    train_summary = train_sgp.main(train_args)

    # 2. Evaluate
    eval_cmd = [
        "python",
        "evaluation/sgp_eval.py",
        "--ckpt_dir",
        "checkpoints/sgp_adapter",
        "--model_name",
        args.model_name,
        "--exit_layer",
        "6",
        "--n_samples",
        "64",
    ]
    subprocess.run(eval_cmd, check=True)

    with open("results/tmp_eval.json") as f:
        eval_res = json.load(f)

    Path("results").mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = f"results/sgp_run_{timestamp}.json"

    loss_main_start = train_summary["loss_main_start"]
    loss_main_end = train_summary["loss_main_end"]
    grad_norm_start = train_summary["grad_norm_start"]
    grad_norm_end = train_summary["grad_norm_end"]

    accept_rate = eval_res.get("accept_rate", 0)

    status = "SUCCESS" if (loss_main_end is not None and loss_main_start is not None and loss_main_end <= 0.8 * loss_main_start and accept_rate >= 0.50) else "FAIL"

    results = {
        "loss_main_start": loss_main_start,
        "loss_main_end": loss_main_end,
        "grad_norm_start": grad_norm_start,
        "grad_norm_end": grad_norm_end,
        "accept_rate": accept_rate,
        "ppl": eval_res.get("ppl"),
        "tok_per_sec": eval_res.get("tok_per_sec"),
        "status": status,
    }

    with open(results_path, "w") as f:
        json.dump(results, f)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
