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

Best currently validated FP4 resident path:

- Routed MoE expert weights live on CPU.
- Decode uses active routed-expert H2D staging and GPU MoE compute.
- Decode keeps cross-layer MoE prefetch disabled by default because the latest long-prompt A/B is faster without it.
- Shared experts use paired INT8 GPU kernels and the PD shared-expert FP16 path.
- Sparse attention, C4 indexer, custom MoE finalize, async all-reduce, and HC pre/post CUDA kernels are enabled in the optimized FP4 resident path.
- Logical PD scheduler is enabled.
- OpenAI service uses the same default optimized environment as the best scheduler script.

Representative FP4 benchmark results on this machine:

| Scenario | Prompt / prefill tokens | Decode tokens | Prefill | Decode TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Maximum validated context | 65,536 | 2 | 257.45s (~255 tok/s) | n/a | OpenAI path, content check returned `OK`. |
| Long prompt decode | 2,148 | 63 | 6.69s (~321 tok/s after warmup) | 3.49 tok/s mean | FP4 resident OpenAI path, `scripts/run_fp4_resident_best.sh`, 3 fresh runs: 3.464/3.500/3.507 tok/s. |
| Short prompt decode | 29 | 127 | ~1.7-3.2s observed | 3.16 tok/s mean | Earlier OpenAI short-prompt reference; short prefill timing is noisier than the long-prompt case. |

The runtime has validated 65,536-token prompts on this 4 x RTX 2080 Ti machine. The server script keeps `MAX_MODEL_LEN=4096` by default for normal serving; set `MAX_MODEL_LEN=65536` explicitly when testing the maximum context path. Short prefill numbers are noisy on this machine, so compare decode TPS and long-prompt cases when evaluating optimization changes.

### GGUF Q2 TP resident path

The GGUF IQ2_XXS/Q2_K path is also supported through `scripts/run_gguf_q2_tp_resident.sh`. On 4 GPUs the script defaults to `PARTITION_POLICY=baseline_4gpu`, grouped GPU prefill, IQ2_XXS W1/W3 DP4A expert-tile kernels, Q2_K W2 DP4A expert-tile kernels, active-expert decode, the decode slot cache, async all-reduce/fused finalize, and 4096-token prefill chunks. Routed GGUF expert weights remain CPU-resident and are staged to GPU as quantized blocks rather than expanded fp32 weights.

Resident OpenAI-compatible benchmark, no explicit warmup, `CASE=all REPEAT=2`, 4 x RTX 2080 Ti:

| Case | Request | Prompt tokens | Service decode tokens | Prefill | Decode TPS | Wall time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_short` | 1 | 5 | 7 | 3.17s (1.58 tok/s) | 2.94 | 5.64s | Cold decode cache. |
| `short_short` | 2 | 5 | 7 | 2.46s (2.03 tok/s) | 4.83 | 3.98s | Warm slot/cache path; meets the 4.5 tok/s short-decode target. |
| `short_long` | 1 | 5 | 9 | 2.46s (2.03 tok/s) | 4.49 | 4.53s | Near the decode target even cold. |
| `short_long` | 2 | 5 | 9 | 2.46s (2.03 tok/s) | 4.89 | 4.37s | Warm slot/cache path. |
| `long_short` | 1 | 2,148 | 7 | 14.17s (151.58 tok/s) | 3.26 | 16.47s | Cold long prefill / decode-cache path. |
| `long_short` | 2 | 2,148 | 7 | 9.94s (216.17 tok/s) | 4.44 | 11.66s | Warm prefill staging; just below the 4.5 tok/s decode target. |
| `long_long` | 1 | 2,148 | 63 | 10.01s (214.52 tok/s) | 3.71 | 27.29s | Longer decode remains below the short-decode target. |
| `long_long` | 2 | 2,148 | 63 | 10.08s (213.05 tok/s) | 3.75 | 27.18s | Warm prefill does not fully fix long decode. |

Single-GPU 2080 Ti + host-memory mode is also available by setting `NPROC_PER_NODE=1`; it is intended for smoke/demo use with very short prompts, not practical long-prompt serving. See [GGUF_Q2_SINGLE_GPU.md](GGUF_Q2_SINGLE_GPU.md) for run commands, memory usage, context limits, and benchmark results.

Real HTTP `stream=true` SSE test with different realistic prompts and no warmup:

| Order | Prompt type | Prompt tokens | Service decode tokens | Client TTFC | Prefill | Decode TPS | Client wall time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Short Chinese chat | 33 | 95 | 7.52s | 7.45s (4.43 tok/s) | 3.93 | 31.94s |
| 2 | README summary | 918 | 159 | 7.40s | 7.32s (125.41 tok/s) | 3.82 | 49.44s |
| 3 | Long technical summary | 2,148 | 115 | 10.15s | 10.01s (214.64 tok/s) | 3.81 | 40.68s |

Current GGUF Q2 conclusion: the prefill path is close to the best result reached by the current architecture for this checkpoint and hardware. Repeated or later requests become much faster in prefill/TTFC because the staged GGUF expert path and OS/GPU caches are hot, but decode remains limited by active-expert cache misses/H2D copies plus TP all-reduce/finalize and long-context attention cost. Short warm-cache decode can reach about 4.8 tok/s, while realistic heterogeneous streaming and long-output decode are typically about 3.7-3.9 tok/s. Further large gains likely require decode-side work such as better active-expert cache scheduling or reducing communication/copy cost, not more prefill kernel tuning.

## Quickstart

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
PYTHONPATH=$PWD python -m src.cli.convert_checkpoint \
  --hf-ckpt-path /path/to/original/DeepSeek-V4-Flash \
  --save-path checkpoints/DeepSeek-V4-Flash-w8a8
CKPT_PATH=$PWD/checkpoints/DeepSeek-V4-Flash-w8a8 bash scripts/run_openai_server.sh
```

The server exposes an OpenAI-compatible chat-completions subset:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `stream=false` non-streaming responses
- `stream=true` token-level SSE responses
- `stream_options.include_usage=true` final usage/timing chunk
- common OpenAI request parameters: `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, `stop`, `n`, `logprobs`, `top_logprobs`, `user`, and `parallel_tool_calls`
- standard OpenAI `tools`, `tool_choice`, assistant `tool_calls`, and `role=tool` continuation messages
- `response_format` and reasoning effort prompt hints

Notes: streaming currently supports `n=1`; streaming `logprobs` are rejected; tool-call streaming emits the parsed `delta.tool_calls` once near the end instead of character-by-character argument deltas.

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
| `CKPT_FORMAT` | `auto` | Checkpoint format passed to the server: `auto`, `safetensors`, or `gguf`. |
| `TOKENIZER_PATH` | unset | Required for GGUF checkpoints; optional tokenizer override for other formats. |
| `PARTITION_POLICY` | `legacy` | Runtime placement policy: `legacy`, `baseline_4gpu`, or `layer_pp_4gpu`. |
| `SERVING_PROFILE` | `safe1` | Serving admission profile: `safe1`, `latency2`, or `throughput4`. |
| `DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH` | `0` | Cross-layer decode MoE prefetch; default is off for the current long-prompt best path. |

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

### 6. Optional GGUF Q2 layer-pipeline run

The runtime can also load the IQ2/Q2 GGUF checkpoint path used during development. Keep the GGUF file outside git, provide the tokenizer directory separately, and launch the layer-pipeline script:

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash-tokenizer \
bash scripts/run_gguf_q2_layer_pp.sh
```

The OpenAI server path also accepts `--ckpt-format gguf --tokenizer-path /path/to/tokenizer` when launched manually.

### 7. Call the API

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

- The OpenAI API targets chat-completions compatibility for common clients, but it is not a complete OpenAI API implementation.
- Fine-grained streaming tool-call deltas are not implemented; parsed tool calls are emitted once after final parsing.
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

## Friendly links

- [linux.do](https://linux.do/)

## Acknowledgement

This project builds on DeepSeek-V4-Flash model assets and the surrounding open-source PyTorch, CUDA, safetensors, and transformers ecosystem. The service/runtime organization and performance scripts in this repository are engineering work for deploying and benchmarking the standalone inference path.
