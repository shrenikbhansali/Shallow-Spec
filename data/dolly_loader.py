from datasets import load_dataset
from torch.utils.data import DataLoader
import torch


def build(tokenizer, batch: int, seq: int):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    def _format(example):
        text = example["instruction"]
        if example.get("context"):
            ctx = example["context"]
            if ctx:
                text += "\n" + ctx
        text += "\n" + example["response"]
        tokens = tokenizer(text, truncation=True, max_length=seq, padding="max_length")
        input_ids = tokens["input_ids"]
        if tokenizer.bos_token_id is not None:
            input_ids[0] = tokenizer.bos_token_id
        tokens["labels"] = input_ids.copy()
        return {k: torch.tensor(v) for k, v in tokens.items()}

    ds = ds.map(_format)
    ds.set_format(type="torch")
    return DataLoader(ds, batch_size=batch, shuffle=True)
