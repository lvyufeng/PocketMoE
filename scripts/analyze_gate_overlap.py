"""Parse gate top-k dump from transformer.py and compute overlap of consecutive
decode-token top-k expert sets per layer.

Usage: python scripts/analyze_gate_overlap.py [path]
Defaults to .tmp/gate_topk.jsonl. Assumes the dump came from a 15-token prefill
+ 15-decode-token run (43 layers).
"""
import ast
import os
import sys
from statistics import mean

path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data1/dsv4_inference/.tmp/gate_topk.jsonl"
NUM_LAYERS = 43
NUM_HASH = 3
HEADER_TOKENS = 15

with open(path) as f:
    lines = [ln.strip() for ln in f if ln.strip()]

assert len(lines) % NUM_LAYERS == 0, f"line count {len(lines)} not a multiple of {NUM_LAYERS}"
calls_per_layer = len(lines) // NUM_LAYERS

# rows: [call_idx][layer_id] = (is_hash, list-of-token-topk-tuples)
rows_per_layer = [[] for _ in range(NUM_LAYERS)]
for ln_idx, ln in enumerate(lines):
    layer = ln_idx % NUM_LAYERS
    hash_str, indices_str = ln.split(" ", 1)
    is_hash = bool(int(hash_str))
    indices = ast.literal_eval(indices_str)
    rows_per_layer[layer].append((is_hash, [tuple(t) for t in indices]))

# Build per-layer flat decode-step list of top-k sets. The first call per layer
# is prefill (HEADER_TOKENS tokens); subsequent calls are 1-token decode steps.
def per_layer_decode_sets(layer_id):
    seq = []
    for call_idx, (_h, rows) in enumerate(rows_per_layer[layer_id]):
        if call_idx == 0:
            # prefill: take the last header token as "decode-equivalent" start
            if rows:
                seq.append(set(rows[-1]))
        else:
            assert len(rows) == 1, f"call {call_idx} layer {layer_id} got {len(rows)} rows"
            seq.append(set(rows[0]))
    return seq

# For each adjacent pair (i, i+1) in the decode sequence (including the bridge
# from last-prefill-token to first-decode-token), compute |A ∩ B|.
per_layer_overlap = []
for layer in range(NUM_LAYERS):
    seq = per_layer_decode_sets(layer)
    pairs = [len(seq[i] & seq[i + 1]) for i in range(len(seq) - 1)]
    per_layer_overlap.append((layer, pairs))

print(f"calls_per_layer={calls_per_layer} (1 prefill + {calls_per_layer - 1} decode)")
print(f"pairs_per_layer={len(per_layer_overlap[0][1])}")
print()
print(f"{'layer':>5} {'mode':>5} {'mean_overlap':>13} {'min':>4} {'max':>4}")
hash_means = []
nonhash_means = []
for layer, pairs in per_layer_overlap:
    mode = "hash" if layer < NUM_HASH else "score"
    m = mean(pairs)
    print(f"{layer:>5} {mode:>5} {m:>13.3f} {min(pairs):>4} {max(pairs):>4}")
    (hash_means if layer < NUM_HASH else nonhash_means).append(m)

print()
print(f"hash layers (0..{NUM_HASH-1}) mean overlap = {mean(hash_means):.3f} / 6")
print(f"score layers ({NUM_HASH}..{NUM_LAYERS-1}) mean overlap = {mean(nonhash_means):.3f} / 6")
print(f"  → score layers fractional overlap = {mean(nonhash_means)/6:.3%}")

# Length=2 H2D cost model: for layers with overlap k out of topk=6, unique
# experts loaded for the 2 tokens = 12 - k. Baseline length=1 = 6.
# Ratio = (12 - k) / 6.
score_overlap = mean(nonhash_means)
length2_moe_ratio = (12 - score_overlap) / 6.0
print()
print(f"Length=2 MoE H2D cost ratio (score layers) = {length2_moe_ratio:.3f}× length=1")
print(f"Length=2 attention cost ratio ≈ 1.1×")
print(f"  → forward cost ratio ≈ 0.3×1.1 + 0.7×{length2_moe_ratio:.2f} = {0.3*1.1 + 0.7*length2_moe_ratio:.3f}× length=1")
print(f"  → spec speedup = 1.78 / above (accept 0.78) = {1.78 / (0.3*1.1 + 0.7*length2_moe_ratio + 0.05):.3f}×")
