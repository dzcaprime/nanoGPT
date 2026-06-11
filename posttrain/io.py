import json
import os
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_token_label_split(out_dir, split, ids, labels):
    np.array(ids, dtype=np.uint16).tofile(os.path.join(out_dir, f"{split}.bin"))
    np.array(labels, dtype=np.int64).tofile(os.path.join(out_dir, f"{split}_mask.bin"))


def save_meta(out_dir, meta):
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def load_checkpoint_model(out_dir, device, compile_model=False):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    return model, checkpoint


def save_training_checkpoint(out_dir, model, optimizer, model_args, iter_num, best_val_loss, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }, os.path.join(out_dir, "ckpt.pt"))


def split_rows(rows, val_fraction):
    if not rows:
        raise ValueError("no rows found")
    val_count = max(1, int(len(rows) * val_fraction)) if len(rows) > 1 else 1
    val_count = min(val_count, len(rows))
    return rows[:-val_count] or rows[:1], rows[-val_count:]
