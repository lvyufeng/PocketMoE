# DSV4 Inference

[English](README.md) | [中文](README_CN.md)

DeepSeek-V4-Flash W8A8 standalone inference runtime with a 4-rank OpenAI-compatible service.

## Motivation

This project packages the standalone DeepSeek-V4-Flash W8A8 inference path into a reusable service-oriented runtime. The current focus is practical deployment on a 4-GPU tensor-parallel machine while preserving the optimized decode path developed in this repo:

- OpenAI-compatible `/v1/chat/completions` API.
- True token-level SSE streaming, not post-generation text chunking.
- CPU routed expert weights with GPU active-expert decode staging.
- Best-known PD scheduler settings available through scripts.
- Repository layout without machine-local absolute paths, so checkpoint and tool locations are provided by environment variables.

## Hardware and current performance

The measurements in this repository were taken on:

- GPU: 4 x NVIDIA GeForce RTX 2080 Ti, 22 GiB each.
- CPU: dual-socket Intel Xeon E5-2696 v4 @ 2.20 GHz.
- CPU topology: 88 logical CPUs, 2 sockets, 22 cores/socket, 2 threads/core.
- Runtime mode: `torchrun --nproc-per-node 4`, one rank per GPU.

Best currently validated path:

- Routed MoE expert weights live on CPU.
- Decode uses active routed-expert H2D staging and GPU MoE compute.
- Shared experts use INT8 GPU kernels.
- Logical PD scheduler is enabled.
- OpenAI service uses the same default optimized environment as the best scheduler script.

Representative benchmark results on this machine:

| Case | Prefill | Decode TPS | Notes |
| --- | ---: | ---: | --- |
| short_short | ~2.05s in latest smoke | 2.59 tok/s | `PD_CASE=short_short`, 7 decode tokens |
| short_short | ~4.04s | 2.67 tok/s | prior all-case run with active GPU MoE |
| short_long | ~4.67s | 2.71 tok/s | prior all-case run with active GPU MoE |
| long_short | ~14.00s | 2.73 tok/s | prior all-case run with active GPU MoE |
| long_long | ~12.75s | 2.64 tok/s | prior all-case run with active GPU MoE |

The short prefill number is noisy on this machine, so compare decode TPS and long-prompt cases when evaluating optimization changes.

## How to run

### 1. Install Python dependencies

Use your own Python environment with PyTorch/CUDA available:

```bash
python -m pip install -r requirements.txt
```

### 2. Build native extensions

Compile all C++/CUDA extensions through the single root `setup.py`:

```bash
python setup.py build_ext
```

Compiled `.so` files are copied to `build/extensions/` and are local build artifacts.

### 3. Convert the checkpoint

The runtime checkpoint `DeepSeek-V4-Flash-w8a8` is not the original upstream checkpoint. It is produced by converting the original Hugging Face/model checkpoint into this repo's W8A8 runtime format.

```bash
PYTHONPATH=$PWD python -m src.cli.convert_checkpoint \
  --hf-ckpt-path /path/to/original/DeepSeek-V4-Flash \
  --save-path checkpoints/DeepSeek-V4-Flash-w8a8
```

Then use the converted directory as `CKPT_PATH`:

```bash
export CKPT_PATH=$PWD/checkpoints/DeepSeek-V4-Flash-w8a8
```

### 4. Provide the checkpoint

Either place the checkpoint at the default repo-relative location:

```text
checkpoints/DeepSeek-V4-Flash-w8a8
```

or pass it explicitly:

```bash
export CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8
```

If `torchrun` or `python` are not on `PATH`, pass them explicitly:

```bash
export TORCHRUN=/path/to/torchrun
export PYTHON=/path/to/python
```

### 5. Start the OpenAI-compatible server

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
bash scripts/run_openai_server.sh
```

Common overrides:

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
HOST=127.0.0.1 \
PORT=8000 \
MASTER_PORT=29920 \
NPROC_PER_NODE=4 \
bash scripts/run_openai_server.sh
```

### 6. Call the API

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Non-streaming chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash-w8a8",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":8,
    "temperature":0,
    "stream":false
  }'
```

Token-level streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash-w8a8",
    "messages":[{"role":"user","content":"请从1数到10"}],
    "max_tokens":64,
    "temperature":0,
    "stream":true
  }'
```

### 7. Run benchmark smoke

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
DEEPSEEK_GPU_MOE_DECODE_ACTIVE=1 \
PD_CASE=short_short \
bash scripts/run_best_scheduler.sh
```

For longer runs, set `PD_CASE=all` or one of `short_short`, `short_long`, `long_short`, `long_long`.
Temporary logs default to `.tmp/`; override with `DSV4_TMP_DIR=/path/to/tmp`.

### 8. Run tests

```bash
PYTHONPATH=$PWD python tests/test_moe_single_token.py
PYTHONPATH=$PWD python tests/test_fused_attn_prefuse.py
PYTHONPATH=$PWD python tests/test_int8_gemm_imma.py
PYTHONPATH=$PWD python tests/test_encoding_dsv4.py
```

## To do list

1. Support multi-request concurrency.
2. Add hot/cold expert statistics and caching.
3. Try MTP acceleration.
4. Try speculative decoding with a pruned draft model.
5. Try 1M context support.

## Acknowledgement

This project builds on DeepSeek-V4-Flash model assets and the surrounding open-source PyTorch, CUDA, safetensors, and transformers ecosystem. The service/runtime organization and performance scripts in this repository are engineering work for deploying and benchmarking the standalone inference path.
