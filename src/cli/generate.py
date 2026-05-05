from argparse import ArgumentParser

from src.runtime.generation import main


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--routed-experts-device", type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--pd-mode", type=str, choices=["off", "scheduler"], default="off")
    parser.add_argument("--pd-prefill-chunk-tokens", type=int, default=0)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        args.max_new_tokens,
        args.temperature,
        args.routed_experts_device,
        args.pd_mode,
        args.pd_prefill_chunk_tokens,
    )
