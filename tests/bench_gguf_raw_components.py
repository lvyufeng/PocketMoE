import time

import torch

from src.gguf.tensor_reader import GGUFTensorDataReader, get_iq2xxs_signed_grid_tensor
from src.runtime.moe.cpu_backend import _apply_native_runtime_config, _load_native_mod


GGUF_PATH = "/mnt/data1/dsv4_inference/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"


def _matmul(native_mod, blocks, type_name, in_dim, x):
    out = torch.empty((x.shape[0], blocks.shape[0]), dtype=torch.float32)
    grid = get_iq2xxs_signed_grid_tensor() if type_name == "iq2_xxs" else torch.empty(0, dtype=torch.int8)
    native_mod.gguf_quantized_matmul(
        x.data_ptr(),
        blocks.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        in_dim,
        blocks.shape[0],
        blocks.shape[1],
        blocks.shape[2],
        0 if type_name == "iq2_xxs" else 1,
        grid.data_ptr(),
    )
    return out


def _bench(name, rows, x_cols, loops=100):
    native_mod = _load_native_mod()
    assert native_mod is not None and hasattr(native_mod, "gguf_quantized_matmul")
    _apply_native_runtime_config(native_mod)
    x = torch.randn(1, x_cols, dtype=torch.float32).contiguous()
    with GGUFTensorDataReader(GGUF_PATH) as reader:
        blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, 0, 0, rows)
        _matmul(native_mod, blocks, type_name, in_dim, x[:, :in_dim])
        t0 = time.perf_counter()
        for _ in range(loops):
            blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, 0, 0, rows)
        read_s = (time.perf_counter() - t0) / loops
        t0 = time.perf_counter()
        for _ in range(loops):
            _matmul(native_mod, blocks, type_name, in_dim, x[:, :in_dim])
        matmul_s = (time.perf_counter() - t0) / loops
        t0 = time.perf_counter()
        for _ in range(loops):
            blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, 0, 0, rows)
            _matmul(native_mod, blocks, type_name, in_dim, x[:, :in_dim])
        combined_s = (time.perf_counter() - t0) / loops
        print(name, "rows", rows, "type", type_name, "read_s", read_s, "matmul_s", matmul_s, "combined_s", combined_s)


def main():
    _bench("blk.0.ffn_gate_exps.weight", 2048, 4096)
    _bench("blk.0.ffn_up_exps.weight", 2048, 4096)
    _bench("blk.0.ffn_down_exps.weight", 4096, 2048)


if __name__ == "__main__":
    main()
