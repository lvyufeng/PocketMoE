# DSV4 Inference

[English](README.md) | [中文](README_CN.md)

DeepSeek-V4-Flash W8A8 独立推理运行时，包含 4-rank OpenAI 兼容服务。

## 背景与目标

本项目把 DeepSeek-V4-Flash W8A8 的 standalone inference 路径整理成可复用、可部署、可 benchmark 的推理服务。当前重点是在 4 卡 tensor parallel 机器上保留已有 decode 优化，并提供清晰的 OpenAI 兼容接口：

- OpenAI 兼容 `/v1/chat/completions` API。
- 真正 token-level SSE 流式输出，而不是生成完以后再切字符串。
- routed MoE expert 权重保留在 CPU，decode 阶段按 active expert 做 H2D staging 并在 GPU 上计算。
- 默认脚本包含当前 best-known PD scheduler 配置。
- 项目内不依赖本机绝对路径，checkpoint、python、torchrun 路径都通过环境变量传入。

## 硬件与当前性能

当前 README 中的性能数据来自以下机器：

- GPU：4 x NVIDIA GeForce RTX 2080 Ti，每张 22 GiB。
- CPU：双路 Intel Xeon E5-2696 v4 @ 2.20 GHz。
- CPU 拓扑：88 个逻辑 CPU，2 socket，每 socket 22 core，每 core 2 thread。
- 运行方式：`torchrun --nproc-per-node 4`，每张 GPU 一个 rank。

当前验证过的最佳路径：

- routed MoE expert 权重在 CPU。
- decode 使用 active routed-expert H2D staging + GPU MoE compute。
- shared experts 使用 INT8 GPU kernel。
- 启用 logical PD scheduler。
- OpenAI 服务默认使用与 best scheduler 脚本一致的优化环境变量。

代表性 benchmark 结果：

| Case | Prefill | Decode TPS | 说明 |
| --- | ---: | ---: | --- |
| short_short | 最新 smoke 约 2.05s | 2.59 tok/s | `PD_CASE=short_short`，7 个 decode token |
| short_short | 约 4.04s | 2.67 tok/s | active GPU MoE 的 prior all-case run |
| short_long | 约 4.67s | 2.71 tok/s | active GPU MoE 的 prior all-case run |
| long_short | 约 14.00s | 2.73 tok/s | active GPU MoE 的 prior all-case run |
| long_long | 约 12.75s | 2.64 tok/s | active GPU MoE 的 prior all-case run |

短 prompt 的 prefill 在这台机器上噪声较大，比较优化效果时建议优先看 decode TPS 和 long prompt case。

## 如何运行

### 1. 安装 Python 依赖

使用你自己的 Python 环境，确保 PyTorch/CUDA 可用：

```bash
python -m pip install -r requirements.txt
```

### 2. 编译 native extensions

所有 C++/CUDA 扩展都通过根目录的单一 `setup.py` 编译：

```bash
python setup.py build_ext
```

编译出的 `.so` 会复制到 `build/extensions/`，这是本地构建产物，不应该提交到 Git。

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

## To do list

1. 支持多请求并发。
2. 支持冷热专家统计和缓存。
3. 尝试支持 MTP 加速。
4. 尝试使用裁剪模型做投机推理。
5. 尝试 1M 上下文支持。

## Acknowledgement

本项目基于 DeepSeek-V4-Flash 模型资产，以及 PyTorch、CUDA、safetensors、transformers 等开源生态。仓库中的服务化组织、运行时整理和性能脚本主要用于部署、验证和 benchmark standalone inference 路径。
