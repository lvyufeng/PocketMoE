"""Export a FlashMemory scorer parity fixture for the C++ kernel test.

Loads the real FlashMemory checkpoint, builds mock hidden + compressed_k,
runs the reference retriever.forward_and_score for one layer (l10), and writes
a flat binary fixture the C++ test reads back to compare its kernel output.

Fixture format (little-endian):
  magic   uint32  = 0x464D5343  ("FMSC")
  version uint32  = 1
  n_chunks int32
  n_heads  int32
  head_dim int32
  q_lora_rank int32
  hidden_dim  int32
  rope_dim    int32
  position    int64
  rms_norm_eps float32
  then raw float32 arrays (row-major):
    wq_a        [q_lora_rank, hidden_dim]
    wq_b        [n_heads*head_dim, q_lora_rank]
    q_norm      [q_lora_rank]
    weights_proj[n_heads, hidden_dim]
    inv_freqs   [rope_dim/2]          (YaRN mixed freqs)
    hidden      [hidden_dim]
  then uint8 array:
    compressed_k [n_chunks, head_dim+4]
  then float32 array:
    ref_logits  [n_chunks]    (pre-sigmoid)
    ref_scores  [n_chunks]    (sigmoid)
"""
import argparse
import struct
import sys

import torch

sys.path.insert(0, "/tmp/FlashMemory-Deepseek-V4")
from retriever import FlashMemoryRetriever, precompute_freqs_cis  # noqa: E402
from demo import make_mock_compressed_k  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default="l10")
    ap.add_argument("--n-chunks", type=int, default=48)
    ap.add_argument("--position", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-position", type=int, default=524288)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = FlashMemoryRetriever.from_checkpoint(
        args.ckpt, device="cpu", max_position=args.max_position
    )
    model.eval()
    scorer = model.scorers[args.layer]
    n_heads = scorer.n_heads
    head_dim = scorer.head_dim
    rope_dim = scorer.rope_dim
    q_lora_rank = scorer.wq_a.shape[0]
    hidden_dim = scorer.wq_a.shape[1]
    rms_eps = scorer.rms_norm_eps

    B, N = 1, args.n_chunks
    hidden = torch.randn(B, hidden_dim, dtype=torch.float32)
    compressed_k = make_mock_compressed_k(B, N, head_dim=head_dim, device="cpu", seed=args.seed)
    positions = torch.tensor([args.position], dtype=torch.int64)

    # Reference logits + scores for this single layer.
    logits = model.score_layer(args.layer, hidden, compressed_k, positions, apply_sigmoid=False)[0]
    scores = torch.sigmoid(logits)

    # The C++ kernel folds YaRN freqs into a single "mixed" inv-freq vector and
    # applies the position at scoring time. Reconstruct the same mixed vector:
    # freqs_cis = polar(1, outer(t, mixed)); angle at position p, pair i is
    # p * mixed[i]. precompute_freqs_cis row 1 gives mixed (angles at t=1).
    fc = model.freqs_cis  # [max_pos, rope_dim/2] complex
    angle_at_1 = torch.angle(fc[1])  # [rope_dim/2]  == mixed (since t=1)
    inv_freqs = angle_at_1.float()

    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", 0x464D5343, 1))
        f.write(struct.pack("<iiiiii", N, n_heads, head_dim, q_lora_rank, hidden_dim, rope_dim))
        f.write(struct.pack("<q", args.position))
        f.write(struct.pack("<f", float(rms_eps)))

        def wf32(t):
            f.write(t.detach().cpu().contiguous().float().numpy().tobytes())

        wf32(scorer.wq_a)
        wf32(scorer.wq_b)
        wf32(scorer.q_norm_weight)
        wf32(scorer.weights_proj)
        wf32(inv_freqs)
        wf32(hidden[0])
        f.write(compressed_k[0].contiguous().cpu().numpy().tobytes())
        wf32(logits)
        wf32(scores)

    print(f"[fixture] wrote {args.out}")
    print(f"[fixture] layer={args.layer} N={N} n_heads={n_heads} head_dim={head_dim} "
          f"q_lora_rank={q_lora_rank} hidden_dim={hidden_dim} rope_dim={rope_dim} pos={args.position}")
    print(f"[fixture] ref scores: min={scores.min():.4f} mean={scores.mean():.4f} max={scores.max():.4f}")
    print(f"[fixture] ref logits: min={logits.min():.4f} mean={logits.mean():.4f} max={logits.max():.4f}")


if __name__ == "__main__":
    main()
