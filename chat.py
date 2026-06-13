"""
Minimal chat sampling entrypoint for post-training checkpoints.
"""
import argparse
from contextlib import nullcontext, redirect_stdout
import os
import warnings

import torch
import tiktoken

from posttrain.chat_format import ASSISTANT, EOT, SYSTEM, USER, format_messages
from posttrain.io import load_checkpoint_model
from rag.index import load_index, retrieve


DEFAULT_OUT_DIRS = "out-sft-full,out-dpo-full,out-verified-full"
DEFAULT_OUT_DIR = "out-verified-full"


def build_messages(system_prompt, user_prompt):
    return [
        {"role": SYSTEM, "content": system_prompt},
        {"role": USER, "content": user_prompt},
    ]


def build_prompt(messages):
    prompt_messages = list(messages) + [{"role": ASSISTANT, "content": ""}]
    return format_messages(prompt_messages, include_eot=False)


def format_retrieved_context(chunks):
    entries = []
    for i, chunk in enumerate(chunks, start=1):
        entries.append(f"[{i}] {chunk['source']}:{chunk['start']}-{chunk['end']}\n{chunk['text']}")
    return "Retrieved context:\n" + "\n\n".join(entries)


def build_rag_user_prompt(user_prompt, chunks):
    if not chunks:
        return user_prompt
    return f"{format_retrieved_context(chunks)}\n\nQuestion:\n{user_prompt}"


def decode_until_eot(enc, token_ids):
    text = enc.decode(token_ids)
    return text.split(EOT, 1)[0].strip()


@torch.no_grad()
def generate_until_eot(model, x, eot_id, max_new_tokens, temperature, top_k):
    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, idx_next), dim=1)
        if idx_next.item() == eot_id:
            break
    return x


def load_model(out_dir, device, compile_model, quiet):
    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
            model, _ = load_checkpoint_model(out_dir, device, compile_model)
    else:
        model, _ = load_checkpoint_model(out_dir, device, compile_model)
    model.eval()
    return model


def generate_response(model, prompt, enc, args):
    prompt_ids = enc.encode(prompt, allowed_special={EOT})
    x = torch.tensor(prompt_ids, dtype=torch.long, device=args.device)[None, ...]
    y = generate_until_eot(
        model,
        x,
        enc.encode(EOT, allowed_special={EOT})[0],
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    return decode_until_eot(enc, y[0].tolist()[len(prompt_ids):])


def run_chat(out_dir, prompt, enc, args):
    model = load_model(out_dir, args.device, args.compile, args.quiet)
    return generate_response(model, prompt, enc, args)


def run_interactive(args, enc, ctx):
    model = load_model(args.out_dir, args.device, args.compile, args.quiet)
    index = load_index(args.rag_index) if args.rag_index else None
    messages = [{"role": SYSTEM, "content": args.system}]
    print(f"Loaded {args.out_dir}. Type /exit to quit.")
    while True:
        try:
            user_prompt = input("User > ").strip()
        except EOFError:
            print()
            break
        if not user_prompt or user_prompt == "/exit":
            break
        messages.append({"role": USER, "content": user_prompt})
        prompt_messages = messages
        if index is not None:
            chunks = retrieve(index, user_prompt, args.rag_top_k, args.rag_max_chars)
            prompt_messages = list(messages)
            prompt_messages[-1] = {"role": USER, "content": build_rag_user_prompt(user_prompt, chunks)}
        prompt = build_prompt(prompt_messages)
        with ctx:
            output = generate_response(model, prompt, enc, args)
        print(f"Assistant > {output if output else '(empty)'}")
        messages.append({"role": ASSISTANT, "content": output})


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with post-training nanoGPT checkpoints.")
    parser.add_argument("--interactive", action="store_true", help="Run a multi-turn chat REPL with one checkpoint.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Checkpoint directory for --interactive mode.")
    parser.add_argument("--out_dirs", default=DEFAULT_OUT_DIRS, help="Comma-separated checkpoint directories.")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt.")
    parser.add_argument("--message", help="User message for one-shot comparison mode.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rag_index", help="Directory containing a local RAG index.json.")
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--rag_max_chars", type=int, default=3000)
    parser.add_argument(
        "--dtype",
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Hide model load parameter counts.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.interactive and args.message is None:
        raise ValueError("--message is required unless --interactive is set")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")
    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    enc = tiktoken.get_encoding("gpt2")
    if args.interactive:
        run_interactive(args, enc, ctx)
        return

    user_prompt = args.message
    if args.rag_index:
        index = load_index(args.rag_index)
        chunks = retrieve(index, args.message, args.rag_top_k, args.rag_max_chars)
        user_prompt = build_rag_user_prompt(args.message, chunks)
    prompt = build_prompt(build_messages(args.system, user_prompt))

    for out_dir in [value.strip() for value in args.out_dirs.split(",") if value.strip()]:
        with ctx:
            output = run_chat(out_dir, prompt, enc, args)
        print(f"[{out_dir}]")
        print(output if output else "(empty)")
        print()


if __name__ == "__main__":
    main()
