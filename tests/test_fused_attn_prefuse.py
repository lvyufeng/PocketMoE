"""Numerical correctness check for the Plan B-小-v2 fused attention prefuse kernels.

Compares fused_q_rmsnorm_rope_inplace + fused_kv_rope_actquant_inplace against
the eager PyTorch reference (apply_rotary_emb + RMSNorm + blockfp8_act_quant).

Run inside the deepseek conda env, e.g.:
  CUDA_VISIBLE_DEVICES=0 conda run -n deepseek PYTHONPATH=$PWD/inference python tests/test_fused_attn_prefuse.py
"""

import math
import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from src.kernels.cuda_loader import load_cuda_kernel  # noqa: E402
from src.kernels.ops import act_quant  # noqa: E402
from src.models.deepseek_v4.runtime import apply_rotary_emb, precompute_freqs_cis, RMSNorm  # noqa: E402


def make_freqs(seqlen: int, rd: int, device):
    # Mirror precompute_freqs_cis with simple defaults; only shape/dtype matters for the test.
    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    t = torch.arange(seqlen, dtype=torch.float32)
    f = torch.outer(t, freqs)
    fc = torch.polar(torch.ones_like(f), f).to(device)  # complex64 [S, rd/2]
    return fc


def test_q(ext, device, seed=0):
    torch.manual_seed(seed)
    B, S, H, D, rd = 1, 1, 16, 512, 64
    eps = 1e-6
    q_ref = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    q_test = q_ref.clone()
    fc = make_freqs(S, rd, device)
    fr = fc.real.to(torch.float32).contiguous()
    fi = fc.imag.to(torch.float32).contiguous()

    # Reference path (eager).
    q_ref *= torch.rsqrt(q_ref.square().mean(-1, keepdim=True) + eps)
    apply_rotary_emb(q_ref[..., -rd:], fc)

    # Fused path.
    ext.fused_q_rmsnorm_rope_inplace(q_test, fr, fi, float(eps))

    diff = (q_ref.float() - q_test.float()).abs()
    print(f"[q] max abs diff = {diff.max().item():.5e}, mean = {diff.mean().item():.5e}")
    # Reference computes mean+rsqrt in bf16, fused kernel does it in fp32; the
    # bf16 unit-of-least-precision is ~1/128 * |x|, so we expect a few %x absolute
    # difference at the tail. 3e-2 is comfortably above bf16 noise.
    assert diff.max().item() < 3e-2, "q fused kernel mismatch beyond tolerance"


def test_kv(ext, device, seed=0):
    torch.manual_seed(seed)
    B, S, kv_dim, rd = 1, 1, 512, 64
    norm_eps = 1e-6
    kv_ref = torch.randn(B, S, kv_dim, dtype=torch.bfloat16, device=device)
    kv_test = kv_ref.clone()
    norm = RMSNorm(kv_dim, norm_eps).to(device)
    # randomize weights so the kernel multiply path is actually exercised
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.05)
    norm_w = norm.weight.detach().contiguous()  # fp32 (matches checkpoint)
    fc = make_freqs(S, rd, device)
    fr = fc.real.to(torch.float32).contiguous()
    fi = fc.imag.to(torch.float32).contiguous()

    # Reference (matches model.py:806-810).
    kv_ref = norm(kv_ref)
    apply_rotary_emb(kv_ref[..., -rd:], fc)
    act_quant(kv_ref[..., :-rd], 64, "ue8m0", torch.float32, True)

    # Fused.
    ext.fused_kv_rope_actquant_inplace(kv_test, norm_w, fr, fi, 64, float(norm_eps))

    diff = (kv_ref.float() - kv_test.float()).abs()
    print(f"[kv] max abs diff = {diff.max().item():.5e}, mean = {diff.mean().item():.5e}")
    # FP8 simulation introduces quantization noise of roughly 1/256 * scale,
    # so we tolerate a wider envelope here than for q.
    assert diff.max().item() < 5e-2, "kv fused kernel mismatch beyond tolerance"


def test_kv_with_cache(ext, device, seed=1):
    """Plan B-小-v3: kernel also writes the produced row into kv_cache_out[b, slot, :].

    Verifies both:
      - kv (the input/output buffer) holds the same result as the no-cache call
      - kv_cache[b, slot, :] holds an identical copy of that result
      - other slots in kv_cache stay untouched (no spillage)
    """
    torch.manual_seed(seed)
    B, S, kv_dim, rd = 1, 1, 512, 64
    cache_len = 1024
    norm_eps = 1e-6
    kv_ref = torch.randn(B, S, kv_dim, dtype=torch.bfloat16, device=device)
    kv_via_old = kv_ref.clone()
    kv_via_new = kv_ref.clone()
    norm = RMSNorm(kv_dim, norm_eps).to(device)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.05)
    norm_w = norm.weight.detach().contiguous()
    fc = make_freqs(S, rd, device)
    fr = fc.real.to(torch.float32).contiguous()
    fi = fc.imag.to(torch.float32).contiguous()

    # Old path (no cache): produces canonical kv result.
    ext.fused_kv_rope_actquant_inplace(kv_via_old, norm_w, fr, fi, 64, float(norm_eps))

    # New path: same call but also dumps into kv_cache[:, slot, :].
    cache = torch.zeros(B, cache_len, kv_dim, dtype=torch.bfloat16, device=device)
    # Pre-fill an unrelated slot to detect spillage.
    cache[0, 7] = torch.randn(kv_dim, dtype=torch.bfloat16, device=device)
    cache_before_other = cache.clone()
    slot = 13
    ext.fused_kv_rope_actquant_inplace(
        kv_via_new, norm_w, fr, fi, 64, float(norm_eps), cache, slot
    )

    diff_kv = (kv_via_old.float() - kv_via_new.float()).abs()
    print(f"[kv_cache] kv-vs-kv max abs diff = {diff_kv.max().item():.5e}")
    assert diff_kv.max().item() == 0.0, "kv output diverged when kv_cache_out passed"

    diff_slot = (kv_via_old[0, 0].float() - cache[0, slot].float()).abs()
    print(f"[kv_cache] cache_slot-vs-kv max abs diff = {diff_slot.max().item():.5e}")
    assert diff_slot.max().item() == 0.0, "kv_cache slot mismatch"

    # Check no other cache slots were written.
    for s in range(cache_len):
        if s == slot:
            continue
        if (cache[0, s] != cache_before_other[0, s]).any():
            raise AssertionError(f"kv_cache slot {s} clobbered")
    print("[kv_cache] no spillage to other slots")


def test_o_inv(ext, device, seed=0):
    torch.manual_seed(seed)
    B, S, H, D, rd = 1, 1, 16, 512, 64
    o_ref = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    o_test = o_ref.clone()
    fc = make_freqs(S, rd, device)
    fr = fc.real.to(torch.float32).contiguous()
    fi = fc.imag.to(torch.float32).contiguous()

    # Reference: apply_rotary_emb(..., inverse=True)
    apply_rotary_emb(o_ref[..., -rd:], fc, True)

    # Fused.
    ext.fused_o_inverse_rope_inplace(o_test, fr, fi)

    diff = (o_ref.float() - o_test.float()).abs()
    print(f"[o_inv] max abs diff = {diff.max().item():.5e}, mean = {diff.mean().item():.5e}")
    # Same fp32-vs-bf16 multiply ULP envelope as the q kernel.
    assert diff.max().item() < 3e-2, "o inverse rope mismatch beyond tolerance"


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    if not hasattr(ext, "fused_q_rmsnorm_rope_inplace"):
        print("ERROR: extension was not rebuilt with fused attn prefuse kernels.")
        sys.exit(2)
    test_q(ext, device)
    test_kv(ext, device)
    test_kv_with_cache(ext, device)
    test_o_inv(ext, device)
    print("OK")


if __name__ == "__main__":
    main()
