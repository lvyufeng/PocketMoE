#!/usr/bin/env bash
# Best-known logical PD scheduler config for DeepSeek-V4-Flash-w8a8.
#
# Architecture invariants:
# - Single model, single request goes through prefill -> decode naturally.
# - Phase-aware attention INT8 (prefill + decode each enable WQ_A/WQ_B/WKV/WO_A/WO_B/INDEXER_WQ_B).
# - GPU prefill MoE staging with grouped GEMM and a 3-layer LRU cache.
# - Routed CPU MoE stays in-process; no external shared-memory server, no shared-weight files.
#
# Frozen optimizations (each entry: what changed, expected effect, escape hatch).
# 1. GPU prefill MoE arena: single-copy pinned int8 expert weights replace the
#    duplicate per-expert tensors. nn.Parameter storages are released after copy
#    so the arena is the unique physical store. Saves ~10.5 GB host RSS, kills
#    ~29 s/long-prefill of cudaHostRegister, and feeds GPU prefill via one
#    non-blocking H2D copy per layer.
#    Code: cpu_routed_backend.py uses _env_enabled_default_on() so it ships ON.
#    Disable: DEEPSEEK_GPU_PREFILL_MOE_ARENA=0
# 2. CPU MoE INT8 thread_local scratch buffers: replaces 4 std::vector mallocs
#    per call inside int8_single_token_topk_persistent_impl with persistent
#    thread_local vectors. Decode TPS 1.78 -> 1.89 (+6.2%) on long_long under
#    schedutil. Effect partly diluted under performance governor (see #4).
#    Code: deepseek_cpu_moe_ext.cpp; rebuilt in-place. No env knob.
# 3. OMP_THREADS=12 for CPU MoE decode: sweep 8/12/16/22 picked 12 as the
#    sweet spot. Decode TPS 1.52 -> 1.82 (+19.9%).
#    Disable: DEEPSEEK_PD_DECODE_OMP_THREADS=<n>
# 4. Pin CPU governor to "performance" while the benchmark runs (opt-in).
#    schedutil keeps cores at 1.2-2.1 GHz (max 3.7) during decode because the
#    OMP workers idle most of each step; switching to performance unblocks
#    turbo and lifts long_long decode TPS from ~1.63 to ~1.89 (+15%). The
#    governor is restored on exit so this is not a permanent change to the
#    machine.
#    Enable: DEEPSEEK_BENCH_GOVERNOR_PIN=1 (requires passwordless sudo for
#    `tee /sys/.../scaling_governor` and the intel_pstate knobs).
#
# Tried but NOT adopted (rationale documented so we do not retry blindly):
# - pmaddubsw sign-trick rewrite of dot_i8_avx2 / dot_i8_avx2_pair.
#   dot_microbench.cpp shows ~2.0x throughput on the inner kernel
#   (30 -> 60 GiB/s @ dim=2048/4096) and is exact-match in [-127,127] domain.
#   End-to-end on long_long, however, the gain disappears even under the
#   performance governor: A/B over 7 runs gave baseline {1.846, 1.914, 1.536,
#   1.926} vs pmaddubsw {1.930, 1.865, 1.925} TPS -- mean delta +0.6%, well
#   inside the per-run +/-10% jitter. The kernel is ~7% of decode wallclock,
#   and OMP barrier + GPU phase switch + allreduce dominate. Re-evaluate only
#   if the OMP synchronization path becomes much shorter or VNNI hardware
#   replaces this AVX2-only Broadwell node.
#   Source preserved as research artefact: dot_microbench.cpp.
#
#
# Reference numbers (4x RTX 2080 Ti, /tmp/dsv4_long_input_single.txt ~2k tokens,
# arena ON + thread_local kernel + OMP=12, generate.py max_new=64 unless noted):
#   schedutil governor (default):
#     long/long  decode 1.63-1.85 TPS (high run-to-run jitter)
#   performance governor (DEEPSEEK_BENCH_GOVERNOR_PIN=1):
#     long/long  decode 1.85-1.93 TPS  (steady; outlier 1.54 if machine load spikes)
#   short/short  prefill 4.81s   decode 1.81 TPS  (7 decoded tokens)
#   short/long   prefill 3.63s   decode 1.80 TPS  (9 decoded tokens)
#   long/short   prefill 13.02s  decode 1.72 TPS  (7 decoded tokens)
#   long/long    prefill 13.6s   decode 1.87 TPS  (63 decoded tokens, sustained)
set -eo pipefail

ROOT="/mnt/data1/dsv4_inference/inference"
LOCK_FILE="${LOCK_FILE:-/tmp/dsv4_best_scheduler.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another best-scheduler benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Optional CPU governor pinning (frozen optimization #4).
#
# When DEEPSEEK_BENCH_GOVERNOR_PIN=1 we pin every CPU's governor to
# "performance" plus intel_pstate min_perf_pct=100 / no_turbo=0 for the
# duration of the script, then restore the previous values on exit. This
# requires passwordless sudo for `tee` against the relevant sysfs files.
#
# We probe `sudo -n true` first; if that fails (no passwordless sudo) we log
# a warning and continue without pinning rather than blocking interactively.
# Restoration runs from a `trap EXIT` so a Ctrl-C still hands the machine
# back in its original state.
# ---------------------------------------------------------------------------
GOV_DIR="/sys/devices/system/cpu"
GOV_BACKUP_FILE=""
GOV_MIN_PERF_BACKUP=""
GOV_NO_TURBO_BACKUP=""
GOV_PINNED=0

restore_governor() {
  if [[ "$GOV_PINNED" != "1" ]]; then
    return
  fi
  echo "restoring CPU governor and intel_pstate knobs..." >&2
  if [[ -s "$GOV_BACKUP_FILE" ]]; then
    while IFS=$'\t' read -r cpu_path prev_gov; do
      [[ -z "$cpu_path" ]] && continue
      echo "$prev_gov" | sudo -n tee "$cpu_path" >/dev/null 2>&1 || true
    done < "$GOV_BACKUP_FILE"
    rm -f "$GOV_BACKUP_FILE"
  fi
  if [[ -n "$GOV_MIN_PERF_BACKUP" && -e /sys/devices/system/cpu/intel_pstate/min_perf_pct ]]; then
    echo "$GOV_MIN_PERF_BACKUP" | sudo -n tee /sys/devices/system/cpu/intel_pstate/min_perf_pct >/dev/null 2>&1 || true
  fi
  if [[ -n "$GOV_NO_TURBO_BACKUP" && -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo "$GOV_NO_TURBO_BACKUP" | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null 2>&1 || true
  fi
  GOV_PINNED=0
}
trap restore_governor EXIT INT TERM

if [[ "${DEEPSEEK_BENCH_GOVERNOR_PIN:-0}" == "1" ]]; then
  if ! sudo -n true 2>/dev/null; then
    echo "DEEPSEEK_BENCH_GOVERNOR_PIN=1 set but passwordless sudo unavailable; skipping governor pinning" >&2
  else
    GOV_BACKUP_FILE="$(mktemp /tmp/dsv4_gov_backup.XXXXXX)"
    for f in "$GOV_DIR"/cpu*/cpufreq/scaling_governor; do
      [[ -e "$f" ]] || continue
      printf '%s\t%s\n' "$f" "$(cat "$f")" >> "$GOV_BACKUP_FILE"
      echo performance | sudo -n tee "$f" >/dev/null
    done
    if [[ -e /sys/devices/system/cpu/intel_pstate/min_perf_pct ]]; then
      GOV_MIN_PERF_BACKUP="$(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct)"
      echo 100 | sudo -n tee /sys/devices/system/cpu/intel_pstate/min_perf_pct >/dev/null
    fi
    if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
      GOV_NO_TURBO_BACKUP="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
      echo 0 | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
    fi
    GOV_PINNED=1
    sample_gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    echo "CPU governor pinned to performance (will restore on exit). cpu0=$sample_gov" >&2
  fi
fi

PYTHON="/home/lvyufeng/miniconda3/envs/deepseek/bin/python"
TORCHRUN="/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun"
CKPT_PATH="${CKPT_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash-w8a8}"
CONFIG="${CONFIG:-$ROOT/config_w8a8.json}"

SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$ROOT/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-/tmp/dsv4_long_input_single.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-64}"

PD_CASE="${PD_CASE:-all}"   # all | short_short | short_long | long_short | long_long
LOG_DIR="${LOG_DIR:-/tmp/dsv4_best_scheduler}"
mkdir -p "$LOG_DIR"

# Generate the long single-prompt input file if it does not exist yet.
if [[ ! -s "$LONG_INPUT_FILE" ]]; then
  "$PYTHON" - <<PY
from transformers import AutoTokenizer
ckpt = "$CKPT_PATH"
tok = AutoTokenizer.from_pretrained(ckpt)
base = (
    "请阅读下面这段重复的技术背景，并在最后用中文总结它的核心观点。\n"
    "DeepSeek V4 Flash standalone inference is being optimized on four RTX 2080Ti GPUs. "
    "Routed MoE experts remain on CPU, exact top-6 routing must be preserved, and large native operators are preferred over Python hot paths. "
    "The validation must include both short and long sequence correctness and performance, with full-network benchmarks run serially.\n"
)
parts = []
token_count = 0
while token_count < 2048:
    parts.append(base.rstrip("\n"))
    token_count = len(tok.encode("\n".join(parts)))
text = "\n".join(parts) + "\n请用三句话总结以上内容。"
with open("$LONG_INPUT_FILE", "w") as f:
    f.write(text)
print(f"wrote $LONG_INPUT_FILE tokens={len(tok.encode(text))}", flush=True)
PY
fi

# Best-known environment for logical PD scheduler.
# Note: keep DEEPSEEK_CPU_MOE_*EXTERNAL*=0 — routed CPU MoE stays in-process
# so we only load expert weights once and skip shared-memory plumbing.
export_best_env() {
  export DEEPSEEK_PD_PHASE_AUTO_SELECT=1
  export DEEPSEEK_GPU_PREFILL_MOE=1
  export DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1
  export DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1
  export DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS="${DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS:-3}"
  export DEEPSEEK_GPU_PREFILL_MOE_ARENA="${DEEPSEEK_GPU_PREFILL_MOE_ARENA:-1}"  # default-on; explicit for clarity
  export DEEPSEEK_INT8_IMPL=cuda_ext
  export DEEPSEEK_MOE_ASYNC_ALLREDUCE=1
  export DEEPSEEK_SHARED_EXPERT_INT8=1
  export DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1
  export DEEPSEEK_FUSED_C4_INDEXER_CUDA=1
  export DEEPSEEK_HC_PRE_CUDA=1
  export DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD="${DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD:-1}"
  export DEEPSEEK_CPU_TOPK_PERSISTENT="${DEEPSEEK_CPU_TOPK_PERSISTENT:-1}"
  export DEEPSEEK_PD_DECODE_OMP_THREADS="${DEEPSEEK_PD_DECODE_OMP_THREADS:-12}"

  # Phase-aware attention INT8 (matches the verified best path).
  for module in WQ_A WQ_B WKV WO_A WO_B INDEXER_WQ_B; do
    export "DEEPSEEK_PD_PREFILL_${module}_INT8=1"
    export "DEEPSEEK_PD_DECODE_${module}_INT8=1"
  done

  # Make absolutely sure no stray external CPU MoE plumbing is on.
  export DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=0
  export DEEPSEEK_CPU_MOE_INPROC_SERVER=0
  export DEEPSEEK_CPU_MOE_SHARED_WEIGHTS=0
}

run_case() {
  local name="$1"
  local input="$2"
  local max_new="$3"
  local port="$4"
  local log="$LOG_DIR/${name}.log"

  echo "=== $name (input=$input, max_new=$max_new) ==="
  rm -f "$log"
  (
    export_best_env
    PYTHONPATH="$ROOT" "$TORCHRUN" \
      --master-port "$port" \
      --nproc-per-node 4 \
      "$ROOT/generate.py" \
      --ckpt-path "$CKPT_PATH" \
      --config "$CONFIG" \
      --input-file "$input" \
      --max-new-tokens "$max_new" \
      --temperature 0 \
      --routed-experts-device cpu \
      --pd-mode scheduler \
      > "$log" 2>&1
  )
  grep -E 'generate time:|prefill time:|^Completion:' "$log" || true
}

case "$PD_CASE" in
  all)
    run_case short_short "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" 29881
    run_case short_long  "$SHORT_INPUT_FILE" "$LONG_MAX_NEW_TOKENS"  29882
    run_case long_short  "$LONG_INPUT_FILE"  "$SHORT_MAX_NEW_TOKENS" 29883
    run_case long_long   "$LONG_INPUT_FILE"  "$LONG_MAX_NEW_TOKENS"  29884
    ;;
  short_short) run_case short_short "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" 29881 ;;
  short_long)  run_case short_long  "$SHORT_INPUT_FILE" "$LONG_MAX_NEW_TOKENS"  29882 ;;
  long_short)  run_case long_short  "$LONG_INPUT_FILE"  "$SHORT_MAX_NEW_TOKENS" 29883 ;;
  long_long)   run_case long_long   "$LONG_INPUT_FILE"  "$LONG_MAX_NEW_TOKENS"  29884 ;;
  *)
    echo "unknown PD_CASE=$PD_CASE (expected: all|short_short|short_long|long_short|long_long)" >&2
    exit 2
    ;;
esac
