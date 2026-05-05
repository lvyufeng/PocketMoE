# DeepSeek-V4-Flash on 2080ti

[English](README.md) | [中文](README_CN.md)

## Description

I have a small 4-GPU RTX 2080 Ti server built around 2023, with a total hardware cost below CNY 20,000. I previously tried deployment approaches such as ktransformers and llama.cpp for large-model inference, but low-bit quantized versions often reduce model quality or runtime behavior. At the same time, Turing GPUs are no longer supported by many mainstream acceleration libraries such as FlashAttention and FlashInfer, nor by newer compiler stacks such as TileLang. Newer DeepSeek models also rely on data formats such as FP4 and FP8 that are only natively supported by newer GPUs, making it almost impossible to run DeepSeek directly on RTX 2080 Ti.

This project starts from that constraint. It tries to make old RTX 2080 Ti cards run DeepSeek-V4 by combining heterogeneous CPU+GPU execution with custom DeepSeek-V4 components, including FP4/FP8 unpacking, sparse attention, indexer kernels, and other runtime paths.

## Motivation

DeepSeek-V4 is one of the strongest Chinese-developed models currently available. DeepSeek-V4-Flash keeps strong model quality while using fewer total parameters: 284B total parameters with only 13B active parameters. An RTX 2080 Ti still has meaningful raw compute: roughly 13.4 TFLOPS FP32 and 107.6 TFLOPS FP16 Tensor Core throughput per card, or about 53.8 TFLOPS FP32 and 430 TFLOPS FP16 Tensor Core throughput across 4 cards. Therefore, trying to run DeepSeek-V4-Flash on this hardware is still a realistic engineering target.

For DeepSeek-style MoE models, most weights are routed expert weights. If the memory-capacity problem can be handled, inference deployment becomes possible. This project uses GPU+CPU heterogeneous execution to support DeepSeek-V4-Flash inference, while also optimizing prefill and decode performance.

## Hardware and limitations

The measurements in this repository were taken on:

- GPU: 4 x NVIDIA GeForce RTX 2080 Ti, 22 GiB each, Turing architecture.
- CPU: dual-socket Intel Xeon E5-2696 v4 @ 2.20 GHz.
- CPU topology: 88 logical CPUs, 2 sockets, 22 cores/socket, 2 threads/core.
- System memory: 1 TiB.
- Runtime mode: `torchrun --nproc-per-node 4`, one rank per GPU.

Instruction set and datatype limitations:

- CPU supports AVX2/FMA/F16C, but not AVX-512, AMX, or BF16/FP16 matrix instructions.
- RTX 2080 Ti supports CUDA Tensor Cores for FP16, but does not have native BF16, FP8, or FP4 tensor cores.
- DeepSeek-V4 uses FP4/FP8-style formats and sparse/indexed attention paths that are designed for newer GPUs, so this project implements custom unpacking and kernels instead of relying on modern library support.

Memory and interconnect constraints:

- Total GPU memory is about 88 GiB across 4 cards, but each rank only has about 22 GiB locally.
- The machine has no NVLink between GPUs. `nvidia-smi topo -m` reports PHB links within the same CPU socket pair and SYS links across sockets.
- Each GPU is connected as PCIe Gen3 x16 at maximum link width. The theoretical one-direction PCIe payload bandwidth is about 15.75 GB/s per GPU.
- GPU-GPU communication therefore goes through PCIe/host bridges, and cross-socket GPU-GPU traffic also crosses the CPU interconnect.
- GPU-CPU expert staging is limited by PCIe bandwidth and NUMA placement, which is why active-expert H2D staging and CPU/GPU overlap are central to the runtime.

## Current performance

Best currently validated path:

- Routed MoE expert weights live on CPU.
- Decode uses active routed-expert H2D staging and GPU MoE compute.
- Shared experts use INT8 GPU kernels.
- Logical PD scheduler is enabled.
- OpenAI service uses the same default optimized environment as the best scheduler script.

Representative benchmark results on this machine:

| Case | Prefill tokens | Decode tokens | Prefill | Decode TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| short_short | 1 | 7 | ~2.05s in latest smoke | 2.59 tok/s | `PD_CASE=short_short` smoke |
| short_short | 1 | 7 | ~4.04s | 2.67 tok/s | prior all-case run with active GPU MoE |
| short_long | 1 | 9 | ~4.67s | 2.71 tok/s | prior all-case run with active GPU MoE |
| long_short | ~2k | 7 | ~14.00s | 2.73 tok/s | prior all-case run with active GPU MoE |
| long_long | ~2k | 63 | ~12.75s | 2.64 tok/s | prior all-case run with active GPU MoE |

The short prefill number is noisy on this machine, so compare decode TPS and long-prompt cases when evaluating optimization changes.

## Quickstart

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
PYTHONPATH=$PWD python -m src.cli.convert_checkpoint \
  --hf-ckpt-path /path/to/original/DeepSeek-V4-Flash \
  --save-path checkpoints/DeepSeek-V4-Flash-w8a8
CKPT_PATH=$PWD/checkpoints/DeepSeek-V4-Flash-w8a8 bash scripts/run_openai_server.sh
```

The server exposes an OpenAI-compatible subset:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `stream=false` non-streaming responses
- `stream=true` token-level SSE responses
- `stream_options.include_usage=true` final usage/timing chunk

Useful runtime environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CKPT_PATH` | `checkpoints/DeepSeek-V4-Flash-w8a8` | Converted W8A8 runtime checkpoint. |
| `TORCHRUN` | `torchrun` | Torch distributed launcher. |
| `PYTHON` | `python` | Python executable used by utility scripts. |
| `HOST` | `0.0.0.0` | Server bind host. |
| `PORT` | `8000` | HTTP server port. |
| `MASTER_PORT` | script-specific | Torch distributed rendezvous port. |
| `NPROC_PER_NODE` | `4` | Number of local ranks / GPUs. |
| `DSV4_TMP_DIR` | `.tmp` | Temporary benchmark and generated prompt directory. |

## How to run

### 1. Install Python dependencies

Use Python 3.11 with PyTorch/CUDA available. The validated environment uses Python 3.11:

```bash
python -m pip install -r requirements.txt
```

### 2. Build native extensions

```bash
python setup.py build_ext
```

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

## Known limitations and troubleshooting

Known limitations:

- The OpenAI API is intentionally minimal and currently targets single-request serving. Multi-request concurrency is still on the to-do list.
- Fine-grained streaming tool-call deltas are not implemented; tool calls are emitted after final parsing.
- The runtime is specialized for the converted `DeepSeek-V4-Flash-w8a8` checkpoint format.
- Performance numbers are hardware- and NUMA-sensitive; short prefill timings are especially noisy.

Troubleshooting:

- `checkpoint not found`: set `CKPT_PATH` or convert/place the checkpoint under `checkpoints/DeepSeek-V4-Flash-w8a8`.
- `ModuleNotFoundError: src`: run commands from the repository root, or set `PYTHONPATH=$PWD`.
- Native extension unavailable: run `python setup.py build_ext` and check that `.so` files exist under `build/extensions/`.
- `torchrun` not found: set `TORCHRUN=/path/to/torchrun`.
- Port conflicts: change `PORT` for HTTP and `MASTER_PORT` for torch distributed.
- Service hangs after client disconnect: the server may finish draining the in-flight generation to keep all ranks synchronized.

## To do list

- [ ] Support multi-request concurrency.
- [ ] Add hot/cold expert statistics and caching.
- [ ] Try MTP acceleration.
- [ ] Try speculative decoding with a pruned draft model.
- [ ] Try 1M context support.

## Acknowledgement

This project builds on DeepSeek-V4-Flash model assets and the surrounding open-source PyTorch, CUDA, safetensors, and transformers ecosystem. The service/runtime organization and performance scripts in this repository are engineering work for deploying and benchmarking the standalone inference path.
