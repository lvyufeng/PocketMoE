#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEEPSEEK_GPU_MOE_DECODE_ACTIVE="${DEEPSEEK_GPU_MOE_DECODE_ACTIVE:-1}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH:-1}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K:-10}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT:-2}"

TORCHRUN="${TORCHRUN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29920}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEFAULT_CKPT_PATH="$ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
CONFIG="${CONFIG:-$ROOT/configs/config_w8a8.json}"

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  echo "Set CKPT_PATH or place the checkpoint under $DEFAULT_CKPT_PATH" >&2
  exit 1
fi
MODEL_ID="${MODEL_ID:-deepseek-v4-flash-w8a8}"
ROUTED_EXPERTS_DEVICE="${ROUTED_EXPERTS_DEVICE:-cpu}"
PD_MODE="${PD_MODE:-scheduler}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

PYTHONPATH="$ROOT" exec "$TORCHRUN" \
  --master-port "$MASTER_PORT" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --module src.server.openai \
  --host "$HOST" \
  --port "$PORT" \
  --ckpt-path "$CKPT_PATH" \
  --config "$CONFIG" \
  --model "$MODEL_ID" \
  --routed-experts-device "$ROUTED_EXPERTS_DEVICE" \
  --pd-mode "$PD_MODE" \
  --max-model-len "$MAX_MODEL_LEN"
