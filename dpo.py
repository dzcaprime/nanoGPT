import math
import os
import pickle
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from posttrain.io import load_checkpoint_model, save_training_checkpoint

out_dir = "out-dpo"
source_out_dir = "out-sft-smoke"
dataset = "posttrain_dpo"
eval_interval = 20
eval_iters = 10
log_interval = 1
batch_size = 1
block_size = 128
beta = 0.1
learning_rate = 1e-5
max_iters = 100
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}


def apply_config():
    global config
    exec(open("configurator.py").read(), globals())
    config = {k: globals()[k] for k in config_keys}


def sequence_logprob(logits, labels):
    if logits.size(1) != labels.size(1):
        raise ValueError("logits and labels length mismatch")
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != -1
    safe_labels = labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta_value):
    logits = beta_value * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(logits).mean()


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def _pad(values, pad_value, max_len):
    values = values[:max_len]
    return values + [pad_value] * (max_len - len(values))


def _shift_pair(record, ids_key, labels_key):
    return record[ids_key][:-1], record[labels_key][1:]


def batch_from_records(records, device):
    shifted = []
    for record in records:
        chosen_ids, chosen_labels = _shift_pair(record, "chosen_ids", "chosen_labels")
        rejected_ids, rejected_labels = _shift_pair(record, "rejected_ids", "rejected_labels")
        shifted.append({
            "chosen_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        })
    max_len = min(block_size, max(max(len(r["chosen_ids"]), len(r["rejected_ids"])) for r in shifted))
    out = {}
    for key, pad in [("chosen_ids", 0), ("chosen_labels", -1), ("rejected_ids", 0), ("rejected_labels", -1)]:
        tensor = torch.tensor([_pad(r[key], pad, max_len) for r in shifted], dtype=torch.long, device=device)
        out[key] = tensor
    return out


def chosen_rejected_logprobs(model, batch):
    input_ids = torch.cat([batch["chosen_ids"], batch["rejected_ids"]], dim=0)
    labels = torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0)
    logits = model(input_ids, labels)[0]
    logprobs = sequence_logprob(logits, labels)
    return logprobs.chunk(2)


@torch.no_grad()
def precompute_reference_logprobs(reference, records, ctx, device):
    values = []
    for record in records:
        batch = batch_from_records([record], device)
        with ctx:
            ref_chosen, ref_rejected = chosen_rejected_logprobs(reference, batch)
        values.append((ref_chosen.detach().cpu(), ref_rejected.detach().cpu()))
    return values


def load_records():
    with open(os.path.join("data", dataset, "preferences.pkl"), "rb") as f:
        return pickle.load(f)


def main():
    apply_config()
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    policy, checkpoint = load_checkpoint_model(source_out_dir, device, compile)
    reference, _ = load_checkpoint_model(source_out_dir, device, False)
    freeze_model(reference)
    optimizer = policy.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    records = load_records()
    train_records = records["train"]
    val_records = records["val"]
    train_ref_logprobs = precompute_reference_logprobs(reference, train_records, ctx, device)
    val_ref_logprobs = precompute_reference_logprobs(reference, val_records, ctx, device)
    del reference
    if device_type == "cuda":
        torch.cuda.empty_cache()
    best_val_loss = math.inf

    def estimate(split_records, ref_logprobs):
        policy.eval()
        losses = []
        with torch.no_grad():
            for i in range(eval_iters):
                record_idx = i % len(split_records)
                batch = batch_from_records([split_records[record_idx]], device)
                ref_c, ref_r = ref_logprobs[record_idx]
                ref_c = ref_c.to(device)
                ref_r = ref_r.to(device)
                with ctx:
                    pi_c, pi_r = chosen_rejected_logprobs(policy, batch)
                    losses.append(dpo_loss(pi_c, pi_r, ref_c, ref_r, beta).item())
        policy.train()
        return sum(losses) / len(losses)

    t0 = time.time()
    for iter_num in range(max_iters + 1):
        if iter_num % eval_interval == 0:
            val_loss = estimate(val_records, val_ref_logprobs)
            print(f"step {iter_num}: val dpo loss {val_loss:.4f}")
            if iter_num > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_training_checkpoint(out_dir, policy, optimizer, checkpoint["model_args"], iter_num, best_val_loss, config)
                print(f"saving checkpoint to {out_dir}")

        record_idx = iter_num % len(train_records)
        batch = batch_from_records([train_records[record_idx]], device)
        ref_c, ref_r = train_ref_logprobs[record_idx]
        ref_c = ref_c.to(device)
        ref_r = ref_r.to(device)
        with ctx:
            pi_c, pi_r = chosen_rejected_logprobs(policy, batch)
            loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: dpo loss {loss.item():.4f}, time {(time.time() - t0) * 1000:.2f}ms")
        t0 = time.time()
    if not os.path.exists(os.path.join(out_dir, "ckpt.pt")):
        save_training_checkpoint(out_dir, policy, optimizer, checkpoint["model_args"], max_iters, best_val_loss, config)


if __name__ == "__main__":
    main()
