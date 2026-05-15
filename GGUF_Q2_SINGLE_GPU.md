# GGUF Q2 single-GPU 2080 Ti host-memory mode

This note documents the single-GPU path for running the GGUF IQ2_XXS/Q2_K checkpoint on one RTX 2080 Ti with routed experts kept in host memory.

## How to run

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
CASE=short_short \
REPEAT=1 \
MAX_MODEL_LEN=1024 \
bash scripts/run_gguf_q2_tp_resident.sh
```

When `NPROC_PER_NODE=1`, `scripts/run_gguf_q2_tp_resident.sh` automatically uses:

- `PARTITION_POLICY=legacy`
- `DEEPSEEK_GGUF_GPU_PREFILL_MOE=0`
- `DEEPSEEK_GGUF_GPU_GROUPED_MOE=0`

The script keeps active-expert decode acceleration enabled:

- `DEEPSEEK_GGUF_GPU_DECODE_ACTIVE_EXPERT=1`
- `DEEPSEEK_GGUF_GPU_DECODE_GROUPED=1`
- `DEEPSEEK_GGUF_GPU_DECODE_SINGLE_TOKEN=1`
- `DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE=1`
- `DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE_SIZE=16`

Full-layer GGUF grouped prefill staging is disabled by default on one GPU because staging all 256 local routed experts for one layer is about 1.7 GiB/layer. This mode keeps routed GGUF experts in host memory and stages only active decode experts to GPU.

## Validated hardware

- GPU: 1 x NVIDIA GeForce RTX 2080 Ti, 22 GiB.
- CPU/system memory: same dual Xeon E5-2696 v4, 1 TiB host-memory machine used for the 4-GPU README numbers.
- Runtime: `NPROC_PER_NODE=1`, `PARTITION_POLICY=legacy`, GGUF routed experts on CPU/host memory.

## Performance results

All numbers below were measured through the OpenAI-compatible resident benchmark path with no explicit warmup unless noted.

| Case | Prompt tokens | Decode tokens | Prefill | Decode TPS | Wall time | Host PSS peak | GPU peak | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Short prompt, short decode, cold | 5 | 7 | 4.50s (1.11 tok/s) | 2.11 | 8.09s | 21.33 GiB | 20.34 GiB | Smoke path. |
| Short prompt, short decode, repeat | 5 | 7 | 4.87s (1.03 tok/s) | 3.12 | 7.24s | 21.33 GiB | 20.34 GiB | Warm slot/cache path. |
| Forced 64-token decode, run 1 | 24 | 63 | 8.57s (2.80 tok/s) | 2.51 | 34.21s | 38.85 GiB | 20.29 GiB | Prompt asks the model to count 1..100. |
| Forced 64-token decode, run 2 | 24 | 63 | 13.28s (1.81 tok/s) | 2.29 | 41.27s | 38.85 GiB | 20.29 GiB | Repeat request. |
| ~128-token prompt | 149 | 7 | 26.02s (5.73 tok/s) | 1.76 | 30.17s | 50.44 GiB | 20.87 GiB | Already poor TTFT. |
| ~256-token prompt | 290 | 7 | 41.93s (6.92 tok/s) | 1.58 | 46.55s | 55.38 GiB | 20.87 GiB | Not usable for interactive serving. |
| ~512-token prompt | 557 | 7 | 78.06s (7.14 tok/s) | 1.51 | 84.28s | 61.21 GiB | 20.87 GiB | Not usable. |
| ~1024-token prompt | 1,045 | 7 | 139.98s (7.47 tok/s) | 1.54 | 144.73s | 64.49 GiB | 20.88 GiB | Not usable. |
| ~2048-token prompt | 2,101 | n/a | did not finish within a useful time window | n/a | stopped | 71.58 GiB | 21.55 GiB | Treated as unusable. |

A short 5-token prompt also completed with `MAX_MODEL_LEN=131072`, so the configuration can allocate a large max sequence setting for tiny requests. That does not make long prompts usable: practical performance is dominated by routed-expert prefill, not the nominal `MAX_MODEL_LEN` setting.

## Practical conclusion

Single-GPU GGUF Q2 mode is useful as a smoke/demo path and can produce short responses on one RTX 2080 Ti. For short inputs, decode is around 2.3-2.5 tok/s on longer forced output, with occasional short-cache runs near 3 tok/s.

It is not a practical long-prompt serving path. Once the prompt reaches even ~128-256 tokens, TTFT is already tens of seconds. At 512-1024 tokens, TTFT is over one minute. The reason is that the safe single-GPU path disables full-layer grouped GPU prefill staging to stay within the 22 GiB GPU memory budget, so long prefill falls back to the CPU/host GGUF routed-expert path and touches large parts of the routed expert mmap.

For production-like usage on this repository, prefer the 4-GPU GGUF Q2 TP resident path or the FP4 resident path. Single-GPU Q2 should be treated as functional validation and very short-prompt experimentation only.
