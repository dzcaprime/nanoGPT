"""
Supervised fine-tuning for the nanoGPT post-training teaching pipeline.
"""

import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig
from posttrain.io import load_checkpoint_model, save_training_checkpoint

out_dir = "out-sft"
source_out_dir = "out-shakespeare-gpt2-124m"
eval_interval = 50
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = "resume"
dataset = "posttrain_sft"
gradient_accumulation_steps = 1
batch_size = 1
block_size = 128
learning_rate = 3e-5
max_iters = 100
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
warmup_iters = 10
lr_decay_iters = 100
min_lr = 3e-6
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}


def apply_config():
    global config
    exec(open("configurator.py").read(), globals())
    config = {k: globals()[k] for k in config_keys}


def load_sft_split(data_dir, split):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    labels = np.memmap(os.path.join(data_dir, f"{split}_mask.bin"), dtype=np.int64, mode="r")
    if len(data) != len(labels):
        raise ValueError("token and label files must have the same length")
    return data, labels


def _slice_pad(array, start, length, pad_value):
    out = np.full(length, pad_value, dtype=np.int64)
    values = array[start:start + length]
    out[:len(values)] = values
    return out


def get_sft_batch_from_split(split_data, batch_size, block_size, device, device_type):
    data, labels = split_data
    if len(data) <= block_size:
        ix = torch.zeros((batch_size,), dtype=torch.long)
    else:
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.from_numpy(np.stack([_slice_pad(data, int(i), block_size, 0) for i in ix]))
    y = torch.from_numpy(np.stack([_slice_pad(labels, int(i) + 1, block_size, -1) for i in ix]))
    if device_type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_sft_batch(data_dir, split, batch_size, block_size, device, device_type):
    return get_sft_batch_from_split(load_sft_split(data_dir, split), batch_size, block_size, device, device_type)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    apply_config()
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    data_dir = os.path.join("data", dataset)
    data_splits = {split: load_sft_split(data_dir, split) for split in ["train", "val"]}

    if init_from == "resume":
        model, checkpoint = load_checkpoint_model(source_out_dir, device, False)
        model_args = checkpoint["model_args"]
    elif init_from.startswith("gpt2"):
        model = GPT.from_pretrained(init_from, {"dropout": 0.0})
        model_args = {k: getattr(model.config, k) for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]}
    else:
        model_args = dict(n_layer=6, n_head=6, n_embd=384, block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
        model = GPT(GPTConfig(**model_args))

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size
    model.to(device)
    if compile:
        model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    raw_model = model
    best_val_loss = 1e9
    iter_num = 0

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_sft_batch_from_split(data_splits[split], batch_size, block_size, device, device_type)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    X, Y = get_sft_batch_from_split(data_splits["train"], batch_size, block_size, device, device_type)
    t0 = time.time()
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    save_training_checkpoint(out_dir, raw_model, optimizer, model_args, iter_num, best_val_loss, config)
                    print(f"saving checkpoint to {out_dir}")
        if iter_num == 0 and eval_only:
            break

        for _ in range(gradient_accumulation_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_sft_batch_from_split(data_splits["train"], batch_size, block_size, device, device_type)
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, time {(time.time() - t0) * 1000:.2f}ms")
        t0 = time.time()
        iter_num += 1
        if iter_num > max_iters:
            if not os.path.exists(os.path.join(out_dir, "ckpt.pt")):
                save_training_checkpoint(out_dir, raw_model, optimizer, model_args, iter_num, best_val_loss, config)
            break


if __name__ == "__main__":
    main()
