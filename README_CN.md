# DeepSeek-V4-Flash on 2080ti

[English](README.md) | [中文](README_CN.md)

## 项目描述

手里有一台 2023 年组的 4 卡 RTX 2080 Ti 小服务器，整机成本小于 2 万元。之前尝试过用 ktransformers、llama.cpp 等方案做大模型推理部署，但低比特量化版本会带来模型效果或运行表现下降。同时，由于 Turing 架构 GPU 早已不被业界主流加速库支持，例如 FlashAttention、FlashInfer，或者最新 compiler 如 TileLang。此外，最新的 DeepSeek 模型使用了 FP4、FP8 等只有较新 GPU 才原生支持的数据格式，因此在 RTX 2080 Ti 上直接执行 DeepSeek 几乎不可行。

本项目就是在这个背景下，尝试通过 GPU+CPU 异构执行、自定义 FP4/FP8 unpack、自定义 sparse attention、自定义 indexer 等 DeepSeek-V4 组件，让 RTX 2080 Ti 这样的老卡也能运行 DeepSeek-V4。

## 背景与目标

DeepSeek-V4 是当前性能最好的国产模型之一，而 DeepSeek-V4-Flash 在保证较好性能的情况下，使用了较少的参数量：总参数量 284B，激活参数量只有 13B。RTX 2080 Ti 的理论算力仍然不低：单卡约 13.4 TFLOPS FP32、约 107.6 TFLOPS FP16 Tensor Core；4 卡合计约 53.8 TFLOPS FP32、约 430 TFLOPS FP16 Tensor Core。因此，其实完全可以尝试 DeepSeek-V4-Flash 的推理。

对于 DeepSeek 这类 MoE 模型而言，大规模权重主要用于 routed experts。因此，只要想办法解决显存容量问题，就可以尝试进行推理部署。本项目尝试用 GPU+CPU 异构的方式，在支持 DeepSeek-V4-Flash 推理的同时，继续提高 prefill 和 decode 性能。


## 硬件与限制

当前 README 中的性能数据来自以下机器：

- GPU：4 x NVIDIA GeForce RTX 2080 Ti，每张 22 GiB，Turing 架构。
- CPU：双路 Intel Xeon E5-2696 v4 @ 2.20 GHz。
- CPU 拓扑：88 个逻辑 CPU，2 socket，每 socket 22 core，每 core 2 thread。
- 系统内存：1 TiB。
- 运行方式：`torchrun --nproc-per-node 4`，每张 GPU 一个 rank。

指令集和数据类型限制：

- CPU 支持 AVX2/FMA/F16C，但不支持 AVX-512、AMX，也没有 BF16/FP16 矩阵指令。
- RTX 2080 Ti 支持 FP16 CUDA Tensor Core，但没有原生 BF16、FP8、FP4 Tensor Core。
- DeepSeek-V4 使用 FP4/FP8 风格格式，以及 sparse/indexed attention 等更偏向新 GPU 的路径，因此本项目需要自定义 unpack 和 kernel，而不能直接依赖现代加速库。

内存和互联限制：

- 4 卡总显存约 88 GiB，但每个 rank 本地只有约 22 GiB。
- 这台机器 GPU 之间没有 NVLink。`nvidia-smi topo -m` 显示同 socket 内 GPU 之间是 PHB，跨 socket GPU 之间是 SYS。
- 每张 GPU 最大链路为 PCIe Gen3 x16，理论单向 PCIe payload 带宽约 15.75 GB/s。
- GPU-GPU 通信需要经过 PCIe/host bridge，跨 socket 通信还要经过 CPU 互联。
- GPU-CPU expert staging 受 PCIe 带宽和 NUMA 位置限制，因此 active-expert H2D staging 以及 CPU/GPU overlap 是本项目 runtime 的核心。

## 当前性能

当前验证过的最佳路径：

- routed MoE expert 权重在 CPU。
- decode 使用 active routed-expert H2D staging + GPU MoE compute。
- cross-layer MoE prefetch 会预测下一层的本 rank local experts，并把它们的 H2D staging 与当前层计算重叠。
- shared experts 使用 INT8 GPU kernel。
- 启用 logical PD scheduler。
- OpenAI 服务默认使用与 best scheduler 脚本一致的优化环境变量。

代表性 benchmark 结果：

| 场景 | Prompt / prefill tokens | Decode tokens | Prefill | Decode TPS | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 已验证最大上下文 | 65,536 | 2 | 257.45s（约 255 tok/s） | n/a | OpenAI 路径，内容校验返回 `OK`。 |
| 长 prompt decode | 2,148 | 63 | 7.85s（约 274 tok/s） | 2.66 tok/s | 2K prompt，cross-layer prefetch 前的参考值。 |
| 当前 decode 最优 | 29 | 127 | 实测约 1.7-3.2s | 3.16 tok/s mean | OpenAI 路径，K10/local-limit2 cross-layer prefetch，3 次 fresh run：3.135/3.168/3.170 tok/s。 |

当前 runtime 已在这台 4 x RTX 2080 Ti 机器上验证 65,536-token prompt。服务脚本为了日常 serving 默认仍使用 `MAX_MODEL_LEN=4096`；测试最大上下文时需要显式设置 `MAX_MODEL_LEN=65536`。短 prompt 的 prefill 在这台机器上噪声较大，比较优化效果时建议优先看 decode TPS 和 long prompt case。

## 快速开始

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
PYTHONPATH=$PWD python -m src.cli.convert_checkpoint \
  --hf-ckpt-path /path/to/original/DeepSeek-V4-Flash \
  --save-path checkpoints/DeepSeek-V4-Flash-w8a8
CKPT_PATH=$PWD/checkpoints/DeepSeek-V4-Flash-w8a8 bash scripts/run_openai_server.sh
```

服务暴露 OpenAI 兼容子集：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `stream=false` 非流式响应
- `stream=true` token-level SSE 响应
- `stream_options.include_usage=true` 在最终 chunk 返回 usage/timing

常用运行环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CKPT_PATH` | `checkpoints/DeepSeek-V4-Flash-w8a8` | 转换后的 W8A8 runtime checkpoint。 |
| `TORCHRUN` | `torchrun` | Torch distributed launcher。 |
| `PYTHON` | `python` | 工具脚本使用的 Python。 |
| `HOST` | `0.0.0.0` | 服务监听地址。 |
| `PORT` | `8000` | HTTP 服务端口。 |
| `MASTER_PORT` | 脚本各自默认 | torch distributed rendezvous 端口。 |
| `NPROC_PER_NODE` | `4` | 本机 rank / GPU 数量。 |
| `DSV4_TMP_DIR` | `.tmp` | benchmark 临时文件和生成 prompt 目录。 |

## 如何运行

### 1. 安装 Python 依赖

使用 Python 3.11，并确保 PyTorch/CUDA 可用。当前验证环境使用 Python 3.11：

```bash
python -m pip install -r requirements.txt
```

### 2. 编译 native extensions

```bash
python setup.py build_ext
```

### 3. 转换 checkpoint

运行时使用的 `DeepSeek-V4-Flash-w8a8` 不是原始 upstream checkpoint，而是由原始 Hugging Face/model checkpoint 转换出来的 W8A8 runtime 格式。

```bash
PYTHONPATH=$PWD python -m src.cli.convert_checkpoint \
  --hf-ckpt-path /path/to/original/DeepSeek-V4-Flash \
  --save-path checkpoints/DeepSeek-V4-Flash-w8a8
```

然后把转换后的目录作为 `CKPT_PATH`：

```bash
export CKPT_PATH=$PWD/checkpoints/DeepSeek-V4-Flash-w8a8
```

### 4. 提供 checkpoint 路径

可以把 checkpoint 放在默认的仓库相对路径：

```text
checkpoints/DeepSeek-V4-Flash-w8a8
```

也可以显式传入：

```bash
export CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8
```

如果 `torchrun` 或 `python` 不在 `PATH` 上，也可以显式传入：

```bash
export TORCHRUN=/path/to/torchrun
export PYTHON=/path/to/python
```

### 5. 启动 OpenAI 兼容服务

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
bash scripts/run_openai_server.sh
```

常用覆盖项：

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
HOST=127.0.0.1 \
PORT=8000 \
MASTER_PORT=29920 \
NPROC_PER_NODE=4 \
bash scripts/run_openai_server.sh
```

### 6. 调用 API

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

非流式 chat completion：

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

Token-level 流式输出：

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

### 7. 跑 benchmark smoke

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
DEEPSEEK_GPU_MOE_DECODE_ACTIVE=1 \
PD_CASE=short_short \
bash scripts/run_best_scheduler.sh
```

更完整的测试可以设置 `PD_CASE=all`，或者指定 `short_short`、`short_long`、`long_short`、`long_long`。
临时日志默认写到 `.tmp/`，也可以通过 `DSV4_TMP_DIR=/path/to/tmp` 覆盖。

### 8. 跑测试

```bash
PYTHONPATH=$PWD python tests/test_moe_single_token.py
PYTHONPATH=$PWD python tests/test_fused_attn_prefuse.py
PYTHONPATH=$PWD python tests/test_int8_gemm_imma.py
PYTHONPATH=$PWD python tests/test_encoding_dsv4.py
```

## 已知限制与排障

已知限制：

- OpenAI API 当前是最小可用子集，主要面向单请求服务；多请求并发仍在 to-do list 中。
- 还没有实现细粒度 tool-call delta streaming；tool calls 会在最终解析后一次性返回。
- runtime 专门面向转换后的 `DeepSeek-V4-Flash-w8a8` checkpoint 格式。
- 性能对硬件和 NUMA 很敏感；短 prompt 的 prefill timing 尤其容易波动。

常见排障：

- `checkpoint not found`：设置 `CKPT_PATH`，或者把转换后的 checkpoint 放到 `checkpoints/DeepSeek-V4-Flash-w8a8`。
- `ModuleNotFoundError: src`：从仓库根目录运行命令，或者设置 `PYTHONPATH=$PWD`。
- native extension 不可用：运行 `python setup.py build_ext`，并检查 `build/extensions/` 下是否有 `.so`。
- `torchrun` 找不到：设置 `TORCHRUN=/path/to/torchrun`。
- 端口冲突：HTTP 使用 `PORT`，torch distributed 使用 `MASTER_PORT`，分别修改。
- 客户端断开后服务仍在计算：服务可能会继续 drain 当前 generation，以保持所有 rank 同步。

## To do list

- [ ] 支持多请求并发。
- [ ] 支持冷热专家统计和缓存。
- [ ] 尝试支持 MTP 加速。
- [ ] 尝试使用裁剪模型做投机推理。
- [ ] 尝试 1M 上下文支持。

## Acknowledgement

本项目基于 DeepSeek-V4-Flash 模型资产，以及 PyTorch、CUDA、safetensors、transformers 等开源生态。仓库中的服务化组织、运行时整理和性能脚本主要用于部署、验证和 benchmark standalone inference 路径。
