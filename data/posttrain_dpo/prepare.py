import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tiktoken

from posttrain.chat_format import ASSISTANT, encode_with_assistant_labels
from posttrain.io import read_jsonl, split_rows


def _messages(row, key):
    return list(row["prompt"]) + [{"role": ASSISTANT, "content": row[key]}]


def encode_record(row, enc):
    chosen_ids, chosen_labels = encode_with_assistant_labels(enc, _messages(row, "chosen"))
    rejected_ids, rejected_labels = encode_with_assistant_labels(enc, _messages(row, "rejected"))
    return {
        "chosen_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_ids": rejected_ids,
        "rejected_labels": rejected_labels,
    }


def prepare(input_path, out_dir, val_fraction):
    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    rows = read_jsonl(input_path)
    for row in rows:
        if not all(key in row for key in ("prompt", "chosen", "rejected")):
            raise ValueError("preference rows need prompt, chosen, and rejected")
    train_rows, val_rows = split_rows(rows, val_fraction)
    artifacts = {
        "train": [encode_record(row, enc) for row in train_rows],
        "val": [encode_record(row, enc) for row in val_rows],
    }
    with open(os.path.join(out_dir, "preferences.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    print(f"train has {len(artifacts['train'])} preference pairs")
    print(f"val has {len(artifacts['val'])} preference pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "sample_preferences.jsonl"))
    parser.add_argument("--out_dir", default=os.path.dirname(__file__))
    parser.add_argument("--val_fraction", type=float, default=0.5)
    args = parser.parse_args()
    prepare(args.input, args.out_dir, args.val_fraction)
