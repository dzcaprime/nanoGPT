import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tiktoken

from posttrain.chat_format import encode_with_assistant_labels
from posttrain.io import read_jsonl, save_meta, split_rows, write_token_label_split


def encode_rows(rows, enc):
    ids = []
    labels = []
    for row in rows:
        row_ids, row_labels = encode_with_assistant_labels(enc, row["messages"])
        ids.extend(row_ids)
        labels.extend(row_labels)
    return ids, labels


def prepare(input_path, out_dir, val_fraction):
    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    rows = read_jsonl(input_path)
    train_rows, val_rows = split_rows(rows, val_fraction)
    train_ids, train_labels = encode_rows(train_rows, enc)
    val_ids, val_labels = encode_rows(val_rows, enc)
    write_token_label_split(out_dir, "train", train_ids, train_labels)
    write_token_label_split(out_dir, "val", val_ids, val_labels)
    save_meta(out_dir, {"tokenizer": "gpt2", "format": "chat_sft_v1"})
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "sample.jsonl"))
    parser.add_argument("--out_dir", default=os.path.dirname(__file__))
    parser.add_argument("--val_fraction", type=float, default=0.2)
    args = parser.parse_args()
    prepare(args.input, args.out_dir, args.val_fraction)
