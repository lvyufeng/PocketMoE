import time

import torch

from src.gguf.tensor_reader import GGUFTensorDataReader, get_iq2xxs_signed_grid_tensor
from src.runtime.moe.cpu_backend import _apply_native_runtime_config, _load_native_mod


GGUF_PATH = "/mnt/data1/dsv4_inference/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"


def _native_matmul(native_mod, reader, name, x, expert, row_start, rows):
    blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, expert, row_start, rows)
    out = torch.empty((x.shape[0], rows), dtype=torch.float32)
    grid = get_iq2xxs_signed_grid_tensor() if type_name == "iq2_xxs" else torch.empty(0, dtype=torch.int8)
    x_contig = x.contiguous()
    native_mod.gguf_quantized_matmul(
        x_contig.data_ptr(),
        blocks.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        in_dim,
        rows,
        blocks.shape[1],
        blocks.shape[2],
        0 if type_name == "iq2_xxs" else 1,
        grid.data_ptr(),
    )
    return out


def main():
    native_mod = _load_native_mod()
    assert native_mod is not None and hasattr(native_mod, "gguf_quantized_matmul")
    _apply_native_runtime_config(native_mod)
    torch.manual_seed(0)
    x = torch.randn(2, 4096, dtype=torch.float32)
    with GGUFTensorDataReader(GGUF_PATH) as reader:
        for name, rows in [
            ("blk.0.ffn_gate_exps.weight", 128),
            ("blk.0.ffn_up_exps.weight", 128),
            ("blk.0.ffn_down_exps.weight", 128),
        ]:
            weight = reader.read_routed_expert(name, 0, 0, rows)
            ref = torch.nn.functional.linear(x[:, : weight.shape[1]], weight, None)
            t0 = time.perf_counter()
            for _ in range(5):
                ref = torch.nn.functional.linear(x[:, : weight.shape[1]], weight, None)
            ref_s = (time.perf_counter() - t0) / 5
            t0 = time.perf_counter()
            out = _native_matmul(native_mod, reader, name, x[:, : weight.shape[1]], 0, 0, rows)
            native_first_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(5):
                out = _native_matmul(native_mod, reader, name, x[:, : weight.shape[1]], 0, 0, rows)
            native_s = (time.perf_counter() - t0) / 5
            diff = (out - ref).abs()
            print(name, "max_abs", float(diff.max()), "mean_abs", float(diff.mean()), "ref_linear_s", ref_s, "native_first_s", native_first_s, "native_s", native_s)


if __name__ == "__main__":
    main()
