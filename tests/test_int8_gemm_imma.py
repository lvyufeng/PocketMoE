"""Numerical correctness check for the cuBLAS IMMA INT8 GEMM path.

Compares int8_gemm_forward / int8_gemm_pair_forward with DEEPSEEK_INT8_GEMM_IMMA
disabled (dp4a baseline) vs enabled (cuBLAS IMMA) across the actual decode
shapes used by the model.

Run inside the deepseek conda env, e.g.:
  CUDA_VISIBLE_DEVICES=0 conda run -n deepseek PYTHONPATH=$PWD python tests/test_int8_gemm_imma.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(THIS_DIR), "inference"))


def quantize_w_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-row symmetric int8 quantization, matches the test path used by
    # other kernels in this repo (one fp32 scale per output row).
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=-1).clamp_min(1e-6)
    scale = amax / 127.0
    wq = (w / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return wq, scale.contiguous()


def run_single(ext, k, n, device, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.float32, device=device) * 0.05
    wq, ws = quantize_w_int8(w)
    y = ext.int8_gemm_forward(x, wq, ws)
    return y.to(torch.float32)


def run_pair(ext, k, n, device, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    w0 = torch.randn(n, k, dtype=torch.float32, device=device) * 0.05
    w1 = torch.randn(n, k, dtype=torch.float32, device=device) * 0.05
    wq0, ws0 = quantize_w_int8(w0)
    wq1, ws1 = quantize_w_int8(w1)
    y = ext.int8_gemm_pair_forward(x, wq0, ws0, wq1, ws1)
    return y[0].to(torch.float32), y[1].to(torch.float32)


def reload_ext():
    # Force a fresh load so the env var is re-read.
    import src.kernels.cuda_loader as cb
    cb._NATIVE_MOD = None
    return cb.load_cuda_kernel()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return
    device = torch.device("cuda:0")

    # Decode shapes from config_w8a8.json under TP=4.
    cases_single = [
        ("wq_a   (dim->q_lora_rank)",   4096, 1024),
        ("wq_b   (q_lora_rank->qH/4)",  1024, 8192),
        ("wkv    (dim->kv_dim)",        4096,  512),
        ("wo_b   (gO/4 -> dim)",        2048, 4096),
        ("idx_wqB(q_lora_rank->iH/4)",  1024, 2048),
        ("shared_w2 (interTP->dim)",     512, 4096),
    ]
    cases_pair = [
        ("shared_w1/w3 (dim->interTP)", 4096,  512),
    ]

    # Baseline (dp4a)
    os.environ["DEEPSEEK_INT8_GEMM_IMMA"] = "0"
    ext = reload_ext()
    base_single = {name: run_single(ext, k, n, device) for name, k, n in cases_single}
    base_pair = {name: run_pair(ext, k, n, device) for name, k, n in cases_pair}

    # IMMA
    os.environ["DEEPSEEK_INT8_GEMM_IMMA"] = "1"
    ext = reload_ext()
    imma_single = {name: run_single(ext, k, n, device) for name, k, n in cases_single}
    imma_pair = {name: run_pair(ext, k, n, device) for name, k, n in cases_pair}

    failed = 0
    for name in base_single:
        diff = (base_single[name] - imma_single[name]).abs()
        rel = diff / base_single[name].abs().clamp_min(1e-6)
        # Both paths should produce identical int32 accumulators; the bf16
        # rounding at the end is the only place divergence can happen.
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        ok = max_abs < 5e-3 and max_rel < 5e-3
        marker = "OK " if ok else "BAD"
        print(f"[{marker}] single {name:40s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        if not ok:
            failed += 1
    for name in base_pair:
        for i in (0, 1):
            diff = (base_pair[name][i] - imma_pair[name][i]).abs()
            rel = diff / base_pair[name][i].abs().clamp_min(1e-6)
            max_abs = diff.max().item()
            max_rel = rel.max().item()
            ok = max_abs < 5e-3 and max_rel < 5e-3
            marker = "OK " if ok else "BAD"
            print(f"[{marker}] pair[{i}] {name:36s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
            if not ok:
                failed += 1
    if failed:
        print(f"FAILED {failed}")
        sys.exit(2)
    print("ALL OK")


if __name__ == "__main__":
    main()
