from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence
import os
import threading

import torch
from torch.autograd.profiler import record_function

from kernel import _dequant_fp4_weight_torch, _quantize_int8_weight_torch


_EXT_DIR = Path(__file__).resolve().parent
_EXT_PATH = _EXT_DIR / "deepseek_cpu_moe_ext.so"
_NATIVE_MOD = None
_RUNTIME_OMP_THREADS: int | None = None
_WORKER_EXECUTOR: ThreadPoolExecutor | None = None
_WORKER_LOCK = threading.Lock()
_FP4_BLOCK_SIZE = 32


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _apply_native_runtime_config(module) -> None:
    if module is None or _RUNTIME_OMP_THREADS is None:
        return
    setter = getattr(module, "set_omp_num_threads", None)
    if setter is not None:
        setter(int(_RUNTIME_OMP_THREADS))


def configure_cpu_routed_runtime(omp_threads: int | None = None) -> None:
    global _RUNTIME_OMP_THREADS
    if omp_threads is not None:
        _RUNTIME_OMP_THREADS = int(omp_threads) if omp_threads > 0 else None
        if _RUNTIME_OMP_THREADS is not None:
            os.environ["OMP_NUM_THREADS"] = str(_RUNTIME_OMP_THREADS)
    _apply_native_runtime_config(_NATIVE_MOD)


def _get_worker_executor() -> ThreadPoolExecutor:
    global _WORKER_EXECUTOR
    if _WORKER_EXECUTOR is None:
        with _WORKER_LOCK:
            if _WORKER_EXECUTOR is None:
                _WORKER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-cpu-moe")
    return _WORKER_EXECUTOR


def _find_extension_path() -> Path | None:
    import sysconfig

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix:
        abi_path = _EXT_DIR / f"deepseek_cpu_moe_ext{ext_suffix}"
        if abi_path.exists():
            return abi_path
    if _EXT_PATH.exists():
        return _EXT_PATH
    matches = sorted(_EXT_DIR.glob("deepseek_cpu_moe_ext*.so"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def _load_native_mod():
    global _NATIVE_MOD
    if _NATIVE_MOD is not None:
        return _NATIVE_MOD
    ext_path = _find_extension_path()
    if ext_path is None:
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("deepseek_cpu_moe_ext", ext_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NATIVE_MOD = module
        _apply_native_runtime_config(module)
        return module
    except Exception:
        return None


@dataclass
class _PendingTask:
    slot: int
    batch_size: int
    device: torch.device
    future: Future | None


def _run_cpu_task(
    ready_event: torch.cuda.Event | None,
    runner,
    input_cpu: torch.Tensor,
    expert_ids_cpu: torch.Tensor,
    weights_cpu: torch.Tensor,
    output_cpu: torch.Tensor,
) -> None:
    with torch.inference_mode():
        if ready_event is not None:
            ready_event.synchronize()
        runner(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)


class CPURoutedExpertsBuffer:
    capture_bs = []
    capture_buffers: Dict[int, tuple] = {}
    temp_bs = 0
    temp_buffer: tuple = tuple()
    buffer_depth = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok: int, output_dim: int):
        batch_size, hidden_size = hidden_states.shape
        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            return cls.temp_buffer

        pin_memory = True
        input_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        expert_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", pin_memory=pin_memory, dtype=torch.long)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, output_dim), device="cpu", pin_memory=pin_memory, dtype=torch.float32).clone()
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((batch_size, output_dim), device=hidden_states.device, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, output_gpu)
        if batch_size in cls.capture_bs:
            cls.capture_buffers[batch_size] = cur_buffer
        cls.temp_bs = batch_size
        cls.temp_buffer = cur_buffer
        return cur_buffer


class CPURoutedItemsBuffer:
    capture_keys = []
    capture_buffers: Dict[tuple[int, int, int], tuple] = {}
    temp_key: tuple[int, int, int] | None = None
    temp_buffer: tuple = tuple()
    buffer_depth = 2

    @classmethod
    def get_buffer(cls, item_count: int, hidden_size: int, output_dim: int, device: torch.device):
        key = (item_count, hidden_size, output_dim)
        if key in cls.capture_buffers:
            return cls.capture_buffers[key]
        if key == cls.temp_key:
            return cls.temp_buffer

        pin_memory = True
        input_cpu = [
            torch.zeros((item_count, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        expert_ids_cpu = [
            torch.zeros((item_count, 1), device="cpu", pin_memory=pin_memory, dtype=torch.long)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((item_count, 1), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((item_count, output_dim), device="cpu", pin_memory=pin_memory, dtype=torch.float32).clone()
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((item_count, output_dim), device=device, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, output_gpu)
        if key in cls.capture_keys:
            cls.capture_buffers[key] = cur_buffer
        cls.temp_key = key
        cls.temp_buffer = cur_buffer
        return cur_buffer


class CPURoutedExpertsBackend:
    def __init__(
        self,
        layer_idx: int,
        experts: Sequence,
        experts_start_idx: int,
        experts_end_idx: int,
        num_experts_per_tok: int,
        output_dim: int,
    ):
        self.layer_idx = layer_idx
        self.experts = experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_end_idx
        self.num_experts_per_tok = num_experts_per_tok
        self.output_dim = output_dim
        self._pending: _PendingTask | None = None
        self._native_mod = _load_native_mod()
        self._native_ready = False
        self._native_enabled = False
        self._inter_dim = 0
        self._swiglu_limit = 0.0
        self._cpu_expert_int8_enabled = _env_enabled("DEEPSEEK_CPU_EXPERT_INT8")
        self._native_int8_enabled = False
        self._native_w1_ptrs = None
        self._native_w2_ptrs = None
        self._native_w3_ptrs = None
        self._native_s1_ptrs = None
        self._native_s2_ptrs = None
        self._native_s3_ptrs = None
        self._native_int8_w1_ptrs = None
        self._native_int8_w2_ptrs = None
        self._native_int8_w3_ptrs = None
        self._native_int8_s1_ptrs = None
        self._native_int8_s2_ptrs = None
        self._native_int8_s3_ptrs = None
        self._expert_int8_cache = {}
        self._int8_weights_prepared = False
        self._executor = _get_worker_executor()
        self._copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._decode_inline_threshold = 4

    def _copy_inputs_to_cpu(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        (
            input_cpu,
            expert_ids_cpu,
            weights_cpu,
            output_cpu,
            output_gpu,
        ) = CPURoutedExpertsBuffer.get_buffer(hidden_states, self.num_experts_per_tok, self.output_dim)
        current_slot = self.layer_idx % CPURoutedExpertsBuffer.buffer_depth
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat_hidden_states.shape[0]
        if hidden_states.device.type == "cuda" and self._copy_stream is not None:
            current_stream = torch.cuda.current_stream(hidden_states.device)
            self._copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._copy_stream):
                input_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
                expert_ids_cpu[current_slot].copy_(topk_ids.to(torch.long), non_blocking=True)
                weights_cpu[current_slot].copy_(topk_weights.to(torch.float32), non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(self._copy_stream)
        else:
            input_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
            expert_ids_cpu[current_slot].copy_(topk_ids.to(torch.long), non_blocking=True)
            weights_cpu[current_slot].copy_(topk_weights.to(torch.float32), non_blocking=True)
            ready_event = None
            if hidden_states.device.type == "cuda":
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(hidden_states.device))
        return current_slot, batch_size, hidden_states.device, ready_event, input_cpu, expert_ids_cpu, weights_cpu, output_cpu, output_gpu

    def _copy_output_to_device(
        self,
        current_slot: int,
        batch_size: int,
        device: torch.device,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        (
            _input_cpu,
            _expert_ids_cpu,
            _weights_cpu,
            output_cpu,
            output_gpu,
        ) = CPURoutedExpertsBuffer.get_buffer(hidden_states.view(-1, hidden_states.shape[-1]), self.num_experts_per_tok, self.output_dim)
        if device.type == "cuda":
            output_gpu[current_slot][:batch_size].copy_(output_cpu[current_slot][:batch_size], non_blocking=True)
            return output_gpu[current_slot][:batch_size]
        return output_cpu[current_slot][:batch_size]

    def _should_run_decode_inline(self, batch_size: int) -> bool:
        return batch_size <= self._decode_inline_threshold

    def _should_run_int8_native_decode(self, batch_size: int) -> bool:
        return self._int8_weights_prepared and self._native_int8_enabled and batch_size <= 16

    def _materialize_int8_expert(self, expert_id: int):
        cached = self._expert_int8_cache.get(expert_id)
        if cached is not None:
            return cached
        expert = self.experts[expert_id]
        if expert is None:
            return None
        expert._materialize_cpu_weights()
        if not hasattr(expert._cpu_w1, "layout_tensor"):
            return None
        w1 = _dequant_fp4_weight_torch(expert._cpu_w1, expert._cpu_w1_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
        w2 = _dequant_fp4_weight_torch(expert._cpu_w2, expert._cpu_w2_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
        w3 = _dequant_fp4_weight_torch(expert._cpu_w3, expert._cpu_w3_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
        w1_q, w1_s = _quantize_int8_weight_torch(w1)
        w2_q, w2_s = _quantize_int8_weight_torch(w2)
        w3_q, w3_s = _quantize_int8_weight_torch(w3)
        cached = (w1_q, w1_s, w2_q, w2_s, w3_q, w3_s)
        self._expert_int8_cache[expert_id] = cached
        return cached

    def prepare_int8_weights(self, expert_ids: Sequence[int] | None = None) -> bool:
        if not self._cpu_expert_int8_enabled:
            return False
        if not self._ensure_native_ready() or not self._native_int8_enabled:
            return False
        if expert_ids is None:
            expert_ids = range(self.experts_start_idx, self.experts_end_idx)
        ok = True
        prepared_any = False
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                continue
            expert = self.experts[expert_id]
            if expert is None:
                continue
            ok = self._materialize_int8_expert(expert_id) is not None and ok
            prepared_any = True
        if prepared_any:
            self._refresh_int8_ptrs()
        self._int8_weights_prepared = bool(self._expert_int8_cache) and ok
        return self._int8_weights_prepared

    def _refresh_int8_ptrs(self) -> None:
        if self._native_int8_w1_ptrs is None:
            num_experts = len(self.experts)
            self._native_int8_w1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_w2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_w3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_s1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_s2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_s3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        for expert_id, packed in self._expert_int8_cache.items():
            w1_q, w1_s, w2_q, w2_s, w3_q, w3_s = packed
            self._native_int8_w1_ptrs[expert_id] = w1_q.data_ptr()
            self._native_int8_w2_ptrs[expert_id] = w2_q.data_ptr()
            self._native_int8_w3_ptrs[expert_id] = w3_q.data_ptr()
            self._native_int8_s1_ptrs[expert_id] = w1_s.data_ptr()
            self._native_int8_s2_ptrs[expert_id] = w2_s.data_ptr()
            self._native_int8_s3_ptrs[expert_id] = w3_s.data_ptr()

    def _run_reference_items_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        output_cpu.zero_()
        if input_cpu.numel() == 0:
            return
        expert_ids_flat = expert_ids_cpu.reshape(-1)
        sort_order = torch.argsort(expert_ids_flat)
        rows = torch.arange(expert_ids_flat.numel(), device=expert_ids_flat.device)[sort_order]
        expert_ids = expert_ids_flat[sort_order]
        unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        offset = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                offset += count
                continue
            group_rows = rows[offset:offset + count]
            expert = self.experts[expert_id]
            output_cpu[group_rows] = expert.forward_cpu(
                input_cpu[group_rows],
                weights_cpu[group_rows],
                x_cpu=input_cpu[group_rows],
            )
            offset += count

    def dispatch_forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        item_count = hidden_states.shape[0]
        if item_count == 0:
            return torch.zeros((0, self.output_dim), device=hidden_states.device, dtype=torch.float32)
        (
            input_cpu,
            expert_ids_cpu,
            weights_cpu,
            output_cpu,
            output_gpu,
        ) = CPURoutedItemsBuffer.get_buffer(item_count, hidden_states.shape[-1], self.output_dim, hidden_states.device)
        current_slot = self.layer_idx % CPURoutedItemsBuffer.buffer_depth
        input_cpu[current_slot].copy_(hidden_states.reshape(item_count, -1), non_blocking=False)
        expert_ids_cpu[current_slot].copy_(expert_ids.to(torch.long).reshape(item_count, 1), non_blocking=False)
        weights_cpu[current_slot].copy_(route_weights.to(torch.float32).reshape(item_count, 1), non_blocking=False)
        native_ready = self._ensure_native_ready()
        if native_ready and self._should_run_int8_native_decode(item_count):
            runner = self._run_native_int8_decode_forward
        else:
            runner = self._run_native_forward if native_ready else self._run_reference_items_forward
        runner(
            input_cpu[current_slot],
            expert_ids_cpu[current_slot],
            weights_cpu[current_slot],
            output_cpu[current_slot],
        )
        if hidden_states.device.type == "cuda":
            output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=False)
            return output_gpu[current_slot]
        return output_cpu[current_slot]

    def dispatch_forward_reference(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.dispatch_forward(hidden_states, expert_ids, route_weights)

    def _ensure_native_ready(self) -> bool:
        if self._native_ready:
            return self._native_enabled
        self._native_ready = True
        if self._native_mod is None:
            return False

        _apply_native_runtime_config(self._native_mod)
        template_expert = None
        num_experts = len(self.experts)
        w1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        w2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        w3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        s1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        s2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        s3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        native_int8_enabled = self._cpu_expert_int8_enabled and hasattr(self._native_mod, "routed_int8_moe_forward")
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[expert_id]
            if expert is None:
                continue
            expert._materialize_cpu_weights()
            if expert.w1.weight.dtype != torch.float4_e2m1fn_x2:
                self._native_enabled = False
                self._native_int8_enabled = False
                return False
            if template_expert is None:
                template_expert = expert
            w1_ptrs[expert_id] = expert._cpu_w1.layout_tensor.data_ptr()
            w2_ptrs[expert_id] = expert._cpu_w2_tiled.data_ptr() if expert._cpu_w2_tiled is not None else expert._cpu_w2.layout_tensor.data_ptr()
            w3_ptrs[expert_id] = expert._cpu_w3.layout_tensor.data_ptr()
            s1_ptrs[expert_id] = expert._cpu_w1_scale.data_ptr()
            s2_ptrs[expert_id] = expert._cpu_w2_scale_tiled.data_ptr() if expert._cpu_w2_scale_tiled is not None else expert._cpu_w2_scale.data_ptr()
            s3_ptrs[expert_id] = expert._cpu_w3_scale.data_ptr()

        if template_expert is None:
            self._native_enabled = False
            self._native_int8_enabled = False
            return False

        self._inter_dim = template_expert._cpu_w1.plain_shape[0]
        self._swiglu_limit = float(template_expert.swiglu_limit)
        self._native_w1_ptrs = w1_ptrs
        self._native_w2_ptrs = w2_ptrs
        self._native_w3_ptrs = w3_ptrs
        self._native_s1_ptrs = s1_ptrs
        self._native_s2_ptrs = s2_ptrs
        self._native_s3_ptrs = s3_ptrs
        self._native_enabled = True
        self._native_int8_enabled = native_int8_enabled
        return True

    def _run_native_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        self._native_mod.routed_fp4_moe_forward(
            input_cpu.data_ptr(),
            expert_ids_cpu.data_ptr(),
            weights_cpu.data_ptr(),
            output_cpu.data_ptr(),
            input_cpu.shape[0],
            input_cpu.shape[1],
            expert_ids_cpu.shape[1],
            self._inter_dim,
            len(self.experts),
            self.experts_start_idx,
            self.experts_end_idx,
            self._native_w1_ptrs.data_ptr(),
            self._native_w2_ptrs.data_ptr(),
            self._native_w3_ptrs.data_ptr(),
            self._native_s1_ptrs.data_ptr(),
            self._native_s2_ptrs.data_ptr(),
            self._native_s3_ptrs.data_ptr(),
            self._swiglu_limit,
        )

    def _run_native_int8_decode_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        if not self._native_int8_enabled:
            self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
            return
        self._refresh_int8_ptrs()
        self._native_mod.routed_int8_moe_forward(
            input_cpu.data_ptr(),
            expert_ids_cpu.data_ptr(),
            weights_cpu.data_ptr(),
            output_cpu.data_ptr(),
            input_cpu.shape[0],
            input_cpu.shape[1],
            expert_ids_cpu.shape[1],
            self._inter_dim,
            len(self.experts),
            self.experts_start_idx,
            self.experts_end_idx,
            self._native_int8_w1_ptrs.data_ptr(),
            self._native_int8_w2_ptrs.data_ptr(),
            self._native_int8_w3_ptrs.data_ptr(),
            self._native_int8_s1_ptrs.data_ptr(),
            self._native_int8_s2_ptrs.data_ptr(),
            self._native_int8_s3_ptrs.data_ptr(),
            self._swiglu_limit,
        )

    def _run_reference_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        output_cpu.zero_()
        for top_idx in range(expert_ids_cpu.size(1)):
            ids_col = expert_ids_cpu[:, top_idx]
            local_mask = (ids_col >= self.experts_start_idx) & (ids_col < self.experts_end_idx)
            if not local_mask.any():
                continue
            rows = torch.nonzero(local_mask, as_tuple=False).flatten()
            expert_ids = ids_col[rows]
            sort_order = torch.argsort(expert_ids)
            rows = rows[sort_order]
            expert_ids = expert_ids[sort_order]
            unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
            offset = 0
            for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
                group_rows = rows[offset:offset + count]
                expert = self.experts[expert_id]
                output_cpu[group_rows] += expert.forward_cpu(
                    input_cpu[group_rows],
                    weights_cpu[group_rows, top_idx:top_idx + 1],
                    x_cpu=input_cpu[group_rows],
                )
                offset += count

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("CPU routed expert task already pending")
        current_slot, batch_size, device, ready_event, input_cpu, expert_ids_cpu, weights_cpu, output_cpu, _output_gpu = self._copy_inputs_to_cpu(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        native_ready = self._ensure_native_ready()
        if native_ready and self._should_run_int8_native_decode(batch_size):
            runner = self._run_native_int8_decode_forward
        else:
            runner = self._run_native_forward if native_ready else self._run_reference_forward
        if self._should_run_decode_inline(batch_size):
            if ready_event is not None:
                ready_event.synchronize()
            runner(
                input_cpu[current_slot][:batch_size],
                expert_ids_cpu[current_slot][:batch_size],
                weights_cpu[current_slot][:batch_size],
                output_cpu[current_slot][:batch_size],
            )
            self._pending = _PendingTask(current_slot, batch_size, device, future=None)
            return
        future = self._executor.submit(
            _run_cpu_task,
            ready_event,
            runner,
            input_cpu[current_slot][:batch_size],
            expert_ids_cpu[current_slot][:batch_size],
            weights_cpu[current_slot][:batch_size],
            output_cpu[current_slot][:batch_size],
        )
        self._pending = _PendingTask(current_slot, batch_size, device, future)

    def sync_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._pending is None:
            raise RuntimeError("No pending CPU routed expert task")
        current_slot = self._pending.slot
        batch_size = self._pending.batch_size
        device = self._pending.device
        future = self._pending.future
        self._pending = None
        if future is not None:
            future.result()
        return self._copy_output_to_device(current_slot, batch_size, device, hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.submit_forward(hidden_states, topk_ids, topk_weights)
        return self.sync_forward(hidden_states)
