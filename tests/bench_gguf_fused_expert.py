import time

import torch

from src.gguf.tensor_reader import GGUFTensorDataReader, get_iq2xxs_signed_grid_tensor
from src.runtime.moe.cpu_backend import _apply_native_runtime_config, _load_native_mod


GGUF_PATH = "/mnt/data1/dsv4_inference/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"


def _type_id(type_name: str) -> int:
    return 0 if type_name == "iq2_xxs" else 1


def _fused(native_mod, reader, x, weight=None):
    w1, w1_type, _ = reader.read_routed_expert_blocks("blk.0.ffn_gate_exps.weight", 0, 0, 2048)
    w3, w3_type, _ = reader.read_routed_expert_blocks("blk.0.ffn_up_exps.weight", 0, 0, 2048)
    w2, w2_type, _ = reader.read_routed_expert_blocks("blk.0.ffn_down_exps.weight", 0, 0, 4096)
    out = torch.empty((x.shape[0], 4096), dtype=torch.float32)
    grid = get_iq2xxs_signed_grid_tensor()
    route_w = weight.contiguous() if weight is not None else torch.empty(0, dtype=torch.float32)
    native_mod.gguf_expert_forward(
        x.contiguous().data_ptr(),
        route_w.data_ptr() if weight is not None else 0,
        out.data_ptr(),
        w1.data_ptr(),
        w3.data_ptr(),
        w2.data_ptr(),
        x.shape[0],
        4096,
        2048,
        w1.shape[1],
        w1.shape[2],
        _type_id(w1_type),
        w3.shape[1],
        w3.shape[2],
        _type_id(w3_type),
        w2.shape[1],
        w2.shape[2],
        _type_id(w2_type),
        10.0,
        grid.data_ptr(),
    )
    return out


def _reference(reader, x, weight=None):
    w1 = reader.read_routed_expert("blk.0.ffn_gate_exps.weight", 0)
    w3 = reader.read_routed_expert("blk.0.ffn_up_exps.weight", 0)
    w2 = reader.read_routed_expert("blk.0.ffn_down_exps.weight", 0)
    gate = torch.nn.functional.linear(x, w1, None)
    up = torch.nn.functional.linear(x, w3, None)
    up = torch.clamp(up, min=-10.0, max=10.0)
    gate = torch.clamp(gate, max=10.0)
    y = torch.nn.functional.silu(gate) * up
    if weight is not None:
        y = weight * y
    return torch.nn.functional.linear(y, w2, None)


def main():
    native_mod = _load_native_mod()
    assert native_mod is not None and hasattr(native_mod, "gguf_expert_forward")
    _apply_native_runtime_config(native_mod)
    torch.manual_seed(0)
    x = torch.randn(1, 4096, dtype=torch.float32).contiguous()
    weight = torch.tensor([[0.75]], dtype=torch.float32)
    with GGUFTensorDataReader(GGUF_PATH) as reader:
        ref = _reference(reader, x, weight)
        out = _fused(native_mod, reader, x, weight)
        diff = (out - ref).abs()
        print("max_abs", float(diff.max()), "mean_abs", float(diff.mean()))
        loops = 20
        _fused(native_mod, reader, x, weight)
        t0 = time.perf_counter()
        for _ in range(loops):
            _fused(native_mod, reader, x, weight)
        print("fused_s", (time.perf_counter() - t0) / loops)


if __name__ == "__main__":
    main()
