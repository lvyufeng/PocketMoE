import time

import torch

from src.loader.gguf.tensor_reader import GGUFTensorDataReader, get_iq2xxs_signed_grid_tensor
from src.kernels.cuda_loader import load_cuda_kernel

GGUF_PATH = "/mnt/data1/dsv4_inference/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"


def _type_id(type_name: str) -> int:
    return 0 if type_name == "iq2_xxs" else 1


def main():
    cuda_mod = load_cuda_kernel()
    assert cuda_mod is not None and hasattr(cuda_mod, "gguf_quant_gemm_forward")
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    x = torch.randn(16, 4096, dtype=torch.bfloat16, device=device)
    grid = get_iq2xxs_signed_grid_tensor().to(device)
    with GGUFTensorDataReader(GGUF_PATH) as reader:
        for name, expert, row_start, rows in [
            ("blk.0.ffn_gate_exps.weight", 0, 0, 128),
            ("blk.0.ffn_up_exps.weight", 0, 0, 128),
            ("blk.0.ffn_down_exps.weight", 0, 0, 128),
        ]:
            blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, expert, row_start, rows)
            x_part = x[:, :in_dim].contiguous()
            blocks_gpu = blocks.to(device, non_blocking=True).contiguous()
            grid_arg = grid if type_name == "iq2_xxs" else torch.empty(0, dtype=torch.int8, device=device)
            weight = reader.read_routed_expert(name, expert, row_start, rows).to(device=device, dtype=torch.float32)
            ref = torch.nn.functional.linear(x_part.float(), weight, None)
            out = cuda_mod.gguf_quant_gemm_forward(x_part, blocks_gpu, int(in_dim), _type_id(type_name), grid_arg).float()
            torch.cuda.synchronize(device)
            diff = (out - ref).abs()
            loops = 20
            t0 = time.perf_counter()
            for _ in range(loops):
                out = cuda_mod.gguf_quant_gemm_forward(x_part, blocks_gpu, int(in_dim), _type_id(type_name), grid_arg)
            torch.cuda.synchronize(device)
            print(
                name,
                type_name,
                "in_dim", in_dim,
                "rows", rows,
                "max_abs", float(diff.max()),
                "mean_abs", float(diff.mean()),
                "cuda_s", (time.perf_counter() - t0) / loops,
            )


if __name__ == "__main__":
    main()
