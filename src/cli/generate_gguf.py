"""Generic GGUF raw-block CUDA greedy generation smoke/perf entrypoint.

Dispatches by ``general.architecture`` through the MoE registry, so the same
CLI serves every model that implements ``MoEModelSpec.build_token_runtime``
(currently MiniMax-M2). Token-id seed-file / CSV smoke and prefill/decode TPS
only; tokenizer/text generation lives in the model-specific entrypoints.
"""

from __future__ import annotations

import argparse

import torch

from src.runtime.generation import parse_tokens_csv, read_seed_tokens, run_gguf_generation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GGUF raw-block CUDA greedy generation smoke (TP)")
    parser.add_argument("--gguf-path", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="auto", help="override general.architecture detection")
    parser.add_argument("--seed-file", type=str, default="")
    parser.add_argument("--tokens", type=str, default="", help="comma-separated token ids when --seed-file is not used")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--layers", type=int, default=0, help="debug: limit layer count; 0 means full model")
    parser.add_argument("--gpu-memory-gib", type=float, default=22.0)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if args.seed_file:
        prompt_tokens = read_seed_tokens(args.seed_file)
    elif args.tokens:
        prompt_tokens = parse_tokens_csv(args.tokens)
    else:
        raise ValueError("provide --seed-file or --tokens")

    run_gguf_generation(
        args.gguf_path,
        prompt_tokens=prompt_tokens,
        max_new_tokens=int(args.max_new_tokens),
        architecture=args.architecture,
        dtype=dtype,
        n_layers=None if int(args.layers) <= 0 else int(args.layers),
        gpu_memory_gib=float(args.gpu_memory_gib),
    )


if __name__ == "__main__":
    main()
