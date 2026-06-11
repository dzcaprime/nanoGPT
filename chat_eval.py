import json
import os
from contextlib import nullcontext

import torch
import tiktoken

from model import GPT
from posttrain.chat_format import ASSISTANT, EOT, format_messages
from posttrain.graders import grade_example
from posttrain.io import load_checkpoint_model, read_jsonl, write_jsonl

init_from = "resume"
out_dir = "out-sft-smoke"
eval_path = "data/posttrain_eval/sample_eval.jsonl"
eval_out_dir = "eval_runs"
run_name = "chat_eval"
max_new_tokens = 32
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]


def apply_config():
    exec(open("configurator.py").read(), globals())


def write_eval_results(run_dir, rows):
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "results.jsonl"), rows)
    num_passed = sum(1 for row in rows if row["grade"]["passed"])
    summary = {"num_examples": len(rows), "num_passed": num_passed, "accuracy": num_passed / len(rows) if rows else 0.0}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_model_for_eval():
    if init_from == "resume":
        model, _ = load_checkpoint_model(out_dir, device, compile)
    elif init_from.startswith("gpt2"):
        model = GPT.from_pretrained(init_from, {"dropout": 0.0})
        model.to(device)
        if compile:
            model = torch.compile(model)
    else:
        raise ValueError(f"unknown init_from: {init_from}")
    model.eval()
    return model


def generate_completion(model, enc, prompt, ctx, device_arg, max_tokens, temp, top_k_arg):
    ids = enc.encode(prompt, allowed_special={EOT})
    x = torch.tensor(ids, dtype=torch.long, device=device_arg)[None, ...]
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_tokens, temperature=temp, top_k=top_k_arg)
    text = enc.decode(y[0].tolist()[len(ids):])
    return text.split(EOT)[0].strip()


def main():
    apply_config()
    torch.manual_seed(seed)
    enc = tiktoken.get_encoding("gpt2")
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    model = load_model_for_eval()
    rows = []
    for example in read_jsonl(eval_path):
        prompt_messages = list(example["messages"]) + [{"role": ASSISTANT, "content": ""}]
        prompt = format_messages(prompt_messages, include_eot=False)
        output = generate_completion(model, enc, prompt, ctx, device, max_new_tokens, temperature, top_k)
        rows.append({"id": example.get("id"), "prompt": prompt, "output": output, "grade": grade_example(example, output)})
    summary = write_eval_results(os.path.join(eval_out_dir, run_name), rows)
    print(summary)


if __name__ == "__main__":
    main()
