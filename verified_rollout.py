import os
from contextlib import nullcontext

import torch
import tiktoken

from chat_eval import generate_completion
from model import GPT
from posttrain.chat_format import ASSISTANT, format_messages
from posttrain.graders import grade_example
from posttrain.io import load_checkpoint_model, read_jsonl, write_jsonl

init_from = "resume"
out_dir = "out-dpo-smoke"
eval_path = "data/posttrain_eval/sample_eval.jsonl"
verified_out_dir = "data/posttrain_verified"
run_name = "verified"
samples_per_prompt = 2
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


def split_rollouts(examples, outputs_by_id):
    accepted = []
    rejected = []
    for example in examples:
        for output in outputs_by_id.get(example.get("id"), []):
            grade = grade_example(example, output)
            row = {"id": example.get("id"), "output": output, "grade": grade}
            if grade["passed"]:
                accepted.append({
                    "messages": list(example["messages"]) + [{"role": ASSISTANT, "content": output}]
                })
            else:
                rejected.append(row)
    return accepted, rejected


def main():
    apply_config()
    torch.manual_seed(seed)
    enc = tiktoken.get_encoding("gpt2")
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if init_from == "resume":
        model, _ = load_checkpoint_model(out_dir, device, compile)
    elif init_from.startswith("gpt2"):
        model = GPT.from_pretrained(init_from, {"dropout": 0.0})
        model.to(device)
    else:
        raise ValueError(f"unknown init_from: {init_from}")
    model.eval()

    accepted = []
    rejected = []
    for example in read_jsonl(eval_path):
        prompt_messages = list(example["messages"]) + [{"role": ASSISTANT, "content": ""}]
        prompt = format_messages(prompt_messages, include_eot=False)
        for _ in range(samples_per_prompt):
            output = generate_completion(model, enc, prompt, ctx, device, max_new_tokens, temperature, top_k)
            grade = grade_example(example, output)
            if grade["passed"]:
                accepted.append({"messages": list(example["messages"]) + [{"role": ASSISTANT, "content": output}]})
            else:
                rejected.append({"id": example.get("id"), "output": output, "grade": grade})
    os.makedirs(verified_out_dir, exist_ok=True)
    write_jsonl(os.path.join(verified_out_dir, "accepted.jsonl"), accepted)
    write_jsonl(os.path.join(verified_out_dir, "rejected.jsonl"), rejected)
    print({"accepted": len(accepted), "rejected": len(rejected)})


if __name__ == "__main__":
    main()
