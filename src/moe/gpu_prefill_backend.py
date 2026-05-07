import os
import time
from collections import OrderedDict
from typing import Optional

import torch

from src.kernels.cuda_loader import load_cuda_kernel


class _ExpertSlabBudget:
    """Process-wide byte budget for per-layer compact expert slabs.
    Each backend can request to reserve a slab of `nbytes` bytes; the budget
    refuses requests once the running total would exceed the cap, in which
    case the backend falls back to its layer-LRU path."""

    def __init__(self):
        self._cap = int(os.getenv("DEEPSEEK_GPU_MOE_EXPERT_SLAB_BUDGET_BYTES", str(int(3.0 * 1024 ** 3))))
        self._used = 0
        self._announced = False
        self._profile = os.getenv("DEEPSEEK_GPU_MOE_EXPERT_CACHE_PROFILE", "0").lower() in {"1", "true", "yes"}

    def try_reserve(self, nbytes: int) -> bool:
        if nbytes <= 0:
            return False
        if self._used + nbytes > self._cap:
            return False
        self._used += nbytes
        if self._profile and not self._announced:
            print(
                f"gpu_moe_slab_budget cap={self._cap / 1024 ** 3:.2f}GB "
                f"first_reserve={nbytes / 1024 ** 2:.2f}MB",
                flush=True,
            )
            self._announced = True
        return True


_SLAB_BUDGET = _ExpertSlabBudget()


class GPUPrefillMoEBackend:
    _cache_lru = OrderedDict()
    _max_cached_layers = int(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", "3"))
    _pinned_workspaces = {}
    _pinned_workspace_events = {}
    _pinned_workspace_next = {}
    _copy_streams = {}
    _pinned_stage_enabled = os.getenv("DEEPSEEK_GPU_PREFILL_MOE_PINNED_STAGE", "1").lower() in {"1", "true", "yes"}
    _pinned_buffer_count = max(1, int(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_PINNED_BUFFERS", "2")))
    _dense_stage_threshold = float(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_DENSE_STAGE_THRESHOLD", "0.75"))

    def __init__(self, cpu_backend, dim: int, num_experts: int, experts_start_idx: int, experts_end_idx: int):
        self.cpu_backend = cpu_backend
        self.dim = dim
        self.num_experts = num_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_end_idx
        self.n_local_experts = experts_end_idx - experts_start_idx
        self._cuda_ext = None
        self._prepared_device: Optional[torch.device] = None
        self._w1q = self._w1s = None
        self._w2q = self._w2s = None
        self._w3q = self._w3s = None
        self._staged_local_experts = set()
        self._stage_event: Optional[torch.cuda.Event] = None
        self._stage_pending = False
        self.profile_enabled = os.getenv("DEEPSEEK_GPU_PREFILL_MOE_PROFILE", "0").lower() in {"1", "true", "yes"}
        self.grouped_gemm_enabled = os.getenv("DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM", "0").lower() in {"1", "true", "yes"}
        self.bucketed_gemm_enabled = os.getenv("DEEPSEEK_GPU_PREFILL_MOE_BUCKETED_GEMM", "0").lower() in {"1", "true", "yes"}
        self.single_token_group_enabled = os.getenv("DEEPSEEK_GPU_MOE_SINGLE_TOKEN_GROUP", "1").lower() in {"1", "true", "yes"}

    @staticmethod
    def _device_index(device: torch.device) -> int:
        return int(device.index if device.index is not None else torch.cuda.current_device())

    def _cache_key(self, device: torch.device):
        return (int(self.cpu_backend.layer_idx), self._device_index(device), self.experts_start_idx, self.experts_end_idx)

    def _synchronize_pending_stage(self) -> None:
        if self._stage_pending and self._stage_event is not None:
            self._stage_event.synchronize()
        self._stage_pending = False

    def release_cache(self) -> None:
        self._synchronize_pending_stage()
        key = self._cache_key(self._prepared_device) if self._prepared_device is not None else None
        if key is not None:
            self._cache_lru.pop(key, None)
        self._w1q = self._w1s = None
        self._w2q = self._w2s = None
        self._w3q = self._w3s = None
        self._staged_local_experts.clear()
        self._stage_event = None
        self._prepared_device = None

    def _touch_cache(self, device: torch.device) -> None:
        key = self._cache_key(device)
        self._cache_lru.pop(key, None)
        self._cache_lru[key] = self
        while self._max_cached_layers > 0 and len(self._cache_lru) > self._max_cached_layers:
            _old_key, old_backend = self._cache_lru.popitem(last=False)
            if old_backend is not self:
                old_backend.release_cache()

    @classmethod
    def _get_copy_stream(cls, device: torch.device) -> torch.cuda.Stream:
        dev_index = cls._device_index(device)
        stream = cls._copy_streams.get(dev_index)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            cls._copy_streams[dev_index] = stream
        return stream

    @classmethod
    def _workspace_key(cls, device: torch.device, n_local_experts: int, packed) -> tuple:
        e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
        dev_index = cls._device_index(device)
        return (
            dev_index,
            n_local_experts,
            tuple(e_w1q.shape), tuple(e_w1s.shape),
            tuple(e_w2q.shape), tuple(e_w2s.shape),
            tuple(e_w3q.shape), tuple(e_w3s.shape),
        )

    @classmethod
    def _new_pinned_workspace(cls, n_local_experts: int, packed):
        e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
        return (
            torch.empty((n_local_experts, *e_w1q.shape), device="cpu", dtype=e_w1q.dtype, pin_memory=True),
            torch.empty((n_local_experts, *e_w1s.shape), device="cpu", dtype=e_w1s.dtype, pin_memory=True),
            torch.empty((n_local_experts, *e_w2q.shape), device="cpu", dtype=e_w2q.dtype, pin_memory=True),
            torch.empty((n_local_experts, *e_w2s.shape), device="cpu", dtype=e_w2s.dtype, pin_memory=True),
            torch.empty((n_local_experts, *e_w3q.shape), device="cpu", dtype=e_w3q.dtype, pin_memory=True),
            torch.empty((n_local_experts, *e_w3s.shape), device="cpu", dtype=e_w3s.dtype, pin_memory=True),
        )

    @classmethod
    def _get_pinned_workspace(cls, device: torch.device, n_local_experts: int, packed):
        if not cls._pinned_stage_enabled:
            return None, None, None
        key = cls._workspace_key(device, n_local_experts, packed)
        pool = cls._pinned_workspaces.get(key)
        if pool is None:
            pool = [cls._new_pinned_workspace(n_local_experts, packed) for _ in range(cls._pinned_buffer_count)]
            cls._pinned_workspaces[key] = pool
            cls._pinned_workspace_events[key] = [None for _ in range(cls._pinned_buffer_count)]
            cls._pinned_workspace_next[key] = 0
        events = cls._pinned_workspace_events[key]
        start = cls._pinned_workspace_next[key]
        idx = start
        for offset in range(len(pool)):
            candidate = (start + offset) % len(pool)
            event = events[candidate]
            if event is None or event.query():
                idx = candidate
                break
        else:
            idx = start
            events[idx].synchronize()
        cls._pinned_workspace_next[key] = (idx + 1) % len(pool)
        return pool[idx], key, idx

    @classmethod
    def _record_pinned_workspace_event(cls, key, idx, event) -> None:
        if key is not None and idx is not None:
            cls._pinned_workspace_events[key][idx] = event

    def _ensure_storage(self, device: torch.device, packed) -> None:
        if self._w1q is not None and self._prepared_device == device:
            return
        self._synchronize_pending_stage()
        e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
        self._w1q = torch.empty((self.n_local_experts, *e_w1q.shape), device=device, dtype=e_w1q.dtype)
        self._w1s = torch.empty((self.n_local_experts, *e_w1s.shape), device=device, dtype=e_w1s.dtype)
        self._w2q = torch.empty((self.n_local_experts, *e_w2q.shape), device=device, dtype=e_w2q.dtype)
        self._w2s = torch.empty((self.n_local_experts, *e_w2s.shape), device=device, dtype=e_w2s.dtype)
        self._w3q = torch.empty((self.n_local_experts, *e_w3q.shape), device=device, dtype=e_w3q.dtype)
        self._w3s = torch.empty((self.n_local_experts, *e_w3s.shape), device=device, dtype=e_w3s.dtype)
        self._staged_local_experts.clear()
        self._prepared_device = device

    def _expert_ids_for_stage(self, local_experts: torch.Tensor | None) -> list[int]:
        if local_experts is None:
            expert_ids = list(range(self.experts_start_idx, self.experts_end_idx))
        elif local_experts.numel() == 0:
            return []
        else:
            expert_ids = [int(e) + self.experts_start_idx for e in local_experts.detach().to(device="cpu", dtype=torch.long).unique().tolist()]
        if self._dense_stage_threshold > 0.0 and len(expert_ids) >= int(self.n_local_experts * self._dense_stage_threshold):
            expert_ids = list(range(self.experts_start_idx, self.experts_end_idx))
        return expert_ids

    def _stage_local_experts(self, device: torch.device, local_experts: torch.Tensor | None = None, async_copy: bool = False) -> None:
        if self._cuda_ext is None:
            self._cuda_ext = load_cuda_kernel()
        if self._cuda_ext is None or not hasattr(self._cuda_ext, "int8_gemm_pair_forward"):
            raise RuntimeError("cuda_kernel int8 GEMM extension is unavailable")
        expert_ids = self._expert_ids_for_stage(local_experts)
        if not expert_ids:
            return
        missing = [expert_id for expert_id in expert_ids if expert_id - self.experts_start_idx not in self._staged_local_experts]
        if not missing:
            if self._prepared_device == device:
                self._touch_cache(device)
            return
        if self._stage_pending:
            self._synchronize_pending_stage()
        t0 = time.perf_counter() if self.profile_enabled else 0.0
        local_ids = sorted(expert_id - self.experts_start_idx for expert_id in missing)
        arena = self.cpu_backend.get_int8_arena() if hasattr(self.cpu_backend, "get_int8_arena") else None
        if arena is not None:
            first_packed = tuple(buf[0] for buf in arena)
            self._ensure_storage(device, first_packed)
            t_prepare = time.perf_counter() if self.profile_enabled else 0.0
            t_pin = t_prepare
            current_stream = torch.cuda.current_stream(device)
            copy_stream = self._get_copy_stream(device) if async_copy else current_stream
            if async_copy:
                copy_stream.wait_stream(current_stream)
            event = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                if len(local_ids) == self.n_local_experts and local_ids[0] == 0 and local_ids[-1] == self.n_local_experts - 1:
                    self._w1q.copy_(arena[0], non_blocking=True)
                    self._w1s.copy_(arena[1], non_blocking=True)
                    self._w2q.copy_(arena[2], non_blocking=True)
                    self._w2s.copy_(arena[3], non_blocking=True)
                    self._w3q.copy_(arena[4], non_blocking=True)
                    self._w3s.copy_(arena[5], non_blocking=True)
                else:
                    for local_id in local_ids:
                        self._w1q[local_id].copy_(arena[0][local_id], non_blocking=True)
                        self._w1s[local_id].copy_(arena[1][local_id], non_blocking=True)
                        self._w2q[local_id].copy_(arena[2][local_id], non_blocking=True)
                        self._w2s[local_id].copy_(arena[3][local_id], non_blocking=True)
                        self._w3q[local_id].copy_(arena[4][local_id], non_blocking=True)
                        self._w3s[local_id].copy_(arena[5][local_id], non_blocking=True)
                event.record(copy_stream)
            if async_copy:
                for tensor in (self._w1q, self._w1s, self._w2q, self._w2s, self._w3q, self._w3s):
                    tensor.record_stream(copy_stream)
            for expert_id in missing:
                self._staged_local_experts.add(expert_id - self.experts_start_idx)
            self._stage_event = event
            self._stage_pending = True
            self._prepared_device = device
            self._touch_cache(device)
            if self.profile_enabled:
                t_done = time.perf_counter()
                print(
                    f"gpu_prefill_moe_stage layer={self.cpu_backend.layer_idx} experts={len(missing)} async={int(async_copy)} arena=1 "
                    f"prepare={t_prepare - t0:.4f}s pin=0.0000s copy={t_done - t_pin:.4f}s total={t_done - t0:.4f}s",
                    flush=True,
                )
            return
        if not self.cpu_backend.prepare_int8_weights(missing):
            raise RuntimeError("CPU expert int8 weights are not prepared")
        t_prepare = time.perf_counter() if self.profile_enabled else 0.0
        first_packed = None
        for expert_id in missing:
            packed = self.cpu_backend._expert_int8_cache.get(expert_id)
            if packed is not None:
                first_packed = packed
                break
        if first_packed is None:
            raise RuntimeError("missing int8 cache for active experts")
        self._ensure_storage(device, first_packed)
        # Fast path: if the CPU backend already keeps every local expert in a per-layer pinned
        # arena, do a single non-blocking H2D copy per buffer and skip the workspace memcpy.
        arena = None
        if hasattr(self.cpu_backend, "get_int8_arena"):
            arena = self.cpu_backend.get_int8_arena()
        if arena is not None and len(local_ids) == self.n_local_experts and local_ids[0] == 0 and local_ids[-1] == self.n_local_experts - 1:
            t_alloc = time.perf_counter() if self.profile_enabled else 0.0
            t_pin = t_alloc  # no per-expert memcpy in this path
            current_stream = torch.cuda.current_stream(device)
            copy_stream = self._get_copy_stream(device) if async_copy else current_stream
            if async_copy:
                copy_stream.wait_stream(current_stream)
            event = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                self._w1q.copy_(arena[0], non_blocking=True)
                self._w1s.copy_(arena[1], non_blocking=True)
                self._w2q.copy_(arena[2], non_blocking=True)
                self._w2s.copy_(arena[3], non_blocking=True)
                self._w3q.copy_(arena[4], non_blocking=True)
                self._w3s.copy_(arena[5], non_blocking=True)
                event.record(copy_stream)
            if async_copy:
                for tensor in (self._w1q, self._w1s, self._w2q, self._w2s, self._w3q, self._w3s):
                    tensor.record_stream(copy_stream)
            for expert_id in missing:
                self._staged_local_experts.add(expert_id - self.experts_start_idx)
            self._stage_event = event
            self._stage_pending = True
            self._prepared_device = device
            self._touch_cache(device)
            if self.profile_enabled:
                t_done = time.perf_counter()
                print(
                    f"gpu_prefill_moe_stage layer={self.cpu_backend.layer_idx} experts={len(missing)} async={int(async_copy)} arena=1 "
                    f"prepare={t_prepare - t0:.4f}s pin=0.0000s copy={t_done - t_pin:.4f}s total={t_done - t0:.4f}s",
                    flush=True,
                )
            return
        pinned_ws, pinned_key, pinned_idx = self._get_pinned_workspace(device, self.n_local_experts, first_packed)
        t_alloc = time.perf_counter() if self.profile_enabled else 0.0
        if pinned_ws is not None:
            for expert_id in missing:
                packed = self.cpu_backend._expert_int8_cache.get(expert_id)
                if packed is None:
                    raise RuntimeError(f"missing int8 cache for expert {expert_id}")
                local_id = expert_id - self.experts_start_idx
                e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
                pinned_ws[0][local_id].copy_(e_w1q)
                pinned_ws[1][local_id].copy_(e_w1s)
                pinned_ws[2][local_id].copy_(e_w2q)
                pinned_ws[3][local_id].copy_(e_w2s)
                pinned_ws[4][local_id].copy_(e_w3q)
                pinned_ws[5][local_id].copy_(e_w3s)
        t_pin = time.perf_counter() if self.profile_enabled else 0.0
        current_stream = torch.cuda.current_stream(device)
        copy_stream = self._get_copy_stream(device) if async_copy else current_stream
        if async_copy:
            copy_stream.wait_stream(current_stream)
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            if pinned_ws is not None:
                if len(local_ids) == self.n_local_experts and local_ids[0] == 0 and local_ids[-1] == self.n_local_experts - 1:
                    self._w1q.copy_(pinned_ws[0], non_blocking=True)
                    self._w1s.copy_(pinned_ws[1], non_blocking=True)
                    self._w2q.copy_(pinned_ws[2], non_blocking=True)
                    self._w2s.copy_(pinned_ws[3], non_blocking=True)
                    self._w3q.copy_(pinned_ws[4], non_blocking=True)
                    self._w3s.copy_(pinned_ws[5], non_blocking=True)
                else:
                    for local_id in local_ids:
                        self._w1q[local_id].copy_(pinned_ws[0][local_id], non_blocking=True)
                        self._w1s[local_id].copy_(pinned_ws[1][local_id], non_blocking=True)
                        self._w2q[local_id].copy_(pinned_ws[2][local_id], non_blocking=True)
                        self._w2s[local_id].copy_(pinned_ws[3][local_id], non_blocking=True)
                        self._w3q[local_id].copy_(pinned_ws[4][local_id], non_blocking=True)
                        self._w3s[local_id].copy_(pinned_ws[5][local_id], non_blocking=True)
            else:
                for expert_id in missing:
                    packed = self.cpu_backend._expert_int8_cache.get(expert_id)
                    if packed is None:
                        raise RuntimeError(f"missing int8 cache for expert {expert_id}")
                    local_id = expert_id - self.experts_start_idx
                    e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
                    self._w1q[local_id].copy_(e_w1q, non_blocking=True)
                    self._w1s[local_id].copy_(e_w1s, non_blocking=True)
                    self._w2q[local_id].copy_(e_w2q, non_blocking=True)
                    self._w2s[local_id].copy_(e_w2s, non_blocking=True)
                    self._w3q[local_id].copy_(e_w3q, non_blocking=True)
                    self._w3s[local_id].copy_(e_w3s, non_blocking=True)
            event.record(copy_stream)
        if async_copy:
            for tensor in (self._w1q, self._w1s, self._w2q, self._w2s, self._w3q, self._w3s):
                tensor.record_stream(copy_stream)
        self._record_pinned_workspace_event(pinned_key, pinned_idx, event)
        for expert_id in missing:
            self._staged_local_experts.add(expert_id - self.experts_start_idx)
        self._stage_event = event
        self._stage_pending = True
        self._prepared_device = device
        self._touch_cache(device)
        if self.profile_enabled:
            t_done = time.perf_counter()
            print(
                f"gpu_prefill_moe_stage layer={self.cpu_backend.layer_idx} experts={len(missing)} async={int(async_copy)} "
                f"prepare={t_prepare - t0:.4f}s pin={t_pin - t_alloc:.4f}s copy={t_done - t_pin:.4f}s total={t_done - t0:.4f}s",
                flush=True,
            )

    def prefetch(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        self._stage_local_experts(device, None, async_copy=True)

    def prefetch_active_experts(self, device: torch.device, indices: torch.Tensor) -> None:
        if device.type != "cuda":
            return
        flat_ids = indices.reshape(-1)
        local_ids = flat_ids - self.experts_start_idx
        mask = (local_ids >= 0) & (local_ids < self.n_local_experts)
        if not bool(mask.any().item()):
            return
        self._stage_local_experts(device, local_ids[mask].to(torch.long), async_copy=True)

    def wait_for_stage(self, device: torch.device) -> None:
        if self._stage_pending and self._stage_event is not None:
            torch.cuda.current_stream(device).wait_event(self._stage_event)
            self._stage_pending = False

    def _record_staged_weights_on_current_stream(self, device: torch.device) -> None:
        stream = torch.cuda.current_stream(device)
        for tensor in (self._w1q, self._w1s, self._w2q, self._w2s, self._w3q, self._w3s):
            tensor.record_stream(stream)

    def _group_routes(self, indices: torch.Tensor, weights: torch.Tensor):
        tokens, topk = indices.shape
        flat_ids = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_tokens = torch.arange(tokens, device=indices.device, dtype=torch.long).repeat_interleave(topk)
        local_ids = flat_ids - self.experts_start_idx
        mask = (local_ids >= 0) & (local_ids < self.n_local_experts)
        if not bool(mask.any().item()):
            return None
        local_ids = flat_ids[mask] - self.experts_start_idx
        route_weights = flat_weights[mask].to(torch.float32)
        route_tokens = flat_tokens[mask]
        order = torch.argsort(local_ids)
        local_ids = local_ids[order].to(torch.long)
        route_weights = route_weights[order].contiguous()
        route_tokens = route_tokens[order].contiguous()
        boundaries = torch.arange(self.n_local_experts + 1, device=indices.device, dtype=torch.long)
        seg_starts = torch.searchsorted(local_ids, boundaries).to(torch.int32).contiguous()
        return local_ids.unique(), route_tokens, route_weights, seg_starts

    def _group_routes_single_token(self, indices: torch.Tensor, weights: torch.Tensor):
        flat_ids = indices[0]
        local_ids = flat_ids - self.experts_start_idx
        mask = (local_ids >= 0) & (local_ids < self.n_local_experts)
        if not bool(mask.any().item()):
            return None
        local_ids = local_ids[mask].to(torch.long)
        route_weights = weights[0][mask].to(torch.float32)
        order = torch.argsort(local_ids)
        local_ids = local_ids[order].contiguous()
        route_weights = route_weights[order].contiguous()
        route_tokens = torch.zeros(local_ids.numel(), device=indices.device, dtype=torch.long)
        counts = torch.bincount(local_ids, minlength=self.n_local_experts).to(torch.int32)
        seg_starts = torch.empty(self.n_local_experts + 1, device=indices.device, dtype=torch.int32)
        seg_starts[0] = 0
        torch.cumsum(counts, dim=0, out=seg_starts[1:])
        return local_ids.unique(), route_tokens, route_weights, seg_starts

    def prefetch_active_experts_to_cache(self, device: torch.device, indices: torch.Tensor) -> int:
        """No-op shim. The earlier per-expert global GPU cache was reverted
        because Python-side compact assembly (torch.stack + scattered allocs)
        cost more than it saved. The active GPU MoE decode path now stays on
        the layer-LRU staging in _stage_local_experts."""
        return 0

    def forward(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("GPU prefill MoE requires CUDA input")
        if self._cuda_ext is None:
            self._cuda_ext = load_cuda_kernel()
            if self._cuda_ext is None or not hasattr(self._cuda_ext, "int8_gemm_pair_forward"):
                raise RuntimeError("cuda_kernel int8 GEMM extension is unavailable")
        t0 = time.perf_counter() if self.profile_enabled else 0.0
        indices, weights = self.cpu_backend._apply_topk_limit(indices, weights)
        if (
            self.single_token_group_enabled
            and x.shape[0] == 1
            and indices.shape[0] == 1
            and hasattr(self._cuda_ext, "moe_single_token_int8_forward")
        ):
            indices_row = indices[0].contiguous().to(torch.int64)
            weights_row = weights[0].contiguous().to(torch.float32)
            local_ids = indices_row - self.experts_start_idx
            local_mask = (local_ids >= 0) & (local_ids < self.n_local_experts)
            if not bool(local_mask.any().item()):
                return torch.zeros((1, self.dim), device=x.device, dtype=torch.float32)
            local_ids = local_ids[local_mask].to(torch.long).unique()
            route_count = int(local_ids.numel())
            if self._w1q is None or self._prepared_device != x.device:
                self._stage_local_experts(x.device, local_ids)
            else:
                unstaged_local_ids = [local_id for local_id in local_ids.tolist() if int(local_id) not in self._staged_local_experts]
                if unstaged_local_ids:
                    self._stage_local_experts(
                        x.device,
                        torch.tensor(unstaged_local_ids, device=indices.device, dtype=torch.long),
                    )
            if self.profile_enabled:
                torch.cuda.synchronize(x.device)
                t_compute = time.perf_counter()
            self.wait_for_stage(x.device)
            self._record_staged_weights_on_current_stream(x.device)
            self._touch_cache(x.device)
            y = self._cuda_ext.moe_single_token_int8_forward(
                x.contiguous(),
                indices_row,
                weights_row,
                self._w1q,
                self._w1s,
                self._w2q,
                self._w2s,
                self._w3q,
                self._w3s,
                int(self.experts_start_idx),
                float(swiglu_limit),
            )
            if self.profile_enabled:
                torch.cuda.synchronize(x.device)
                t_done = time.perf_counter()
                print(
                    f"gpu_prefill_moe_profile layer={self.cpu_backend.layer_idx} tokens={x.shape[0]} "
                    f"routes={route_count} group={t_compute - t0:.4f}s compute_scatter={t_done - t_compute:.4f}s total={t_done - t0:.4f}s",
                    flush=True,
                )
            return y

        if hasattr(self._cuda_ext, "moe_group_routes"):
            grouped = self._cuda_ext.moe_group_routes(
                indices.contiguous(),
                weights.contiguous().to(torch.float32),
                int(self.experts_start_idx),
                int(self.n_local_experts),
            )
        else:
            grouped = self._group_routes(indices, weights)
        if grouped is None:
            return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
        _local_ids, route_tokens, route_weights, seg_starts = grouped
        if _local_ids.numel() == 0:
            return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
        if self._w1q is None or self._prepared_device != x.device:
            self._stage_local_experts(x.device, _local_ids)
        else:
            unstaged_local_ids = [local_id for local_id in _local_ids.tolist() if int(local_id) not in self._staged_local_experts]
            if unstaged_local_ids:
                self._stage_local_experts(
                    x.device,
                    torch.tensor(unstaged_local_ids, device=indices.device, dtype=torch.long),
                )
        if self.profile_enabled:
            torch.cuda.synchronize(x.device)
            t_compute = time.perf_counter()
        self.wait_for_stage(x.device)
        self._record_staged_weights_on_current_stream(x.device)
        self._touch_cache(x.device)
        if self.grouped_gemm_enabled and self.bucketed_gemm_enabled and hasattr(self._cuda_ext, "moe_prefill_int8_grouped_gemm_bucketed_forward"):
            y = self._cuda_ext.moe_prefill_int8_grouped_gemm_bucketed_forward(
                x.contiguous(),
                route_tokens,
                route_weights.contiguous(),
                seg_starts,
                self._w1q,
                self._w1s,
                self._w2q,
                self._w2s,
                self._w3q,
                self._w3s,
                float(swiglu_limit),
            )
        elif self.grouped_gemm_enabled and hasattr(self._cuda_ext, "moe_prefill_int8_grouped_gemm_forward"):
            y = self._cuda_ext.moe_prefill_int8_grouped_gemm_forward(
                x.contiguous(),
                route_tokens,
                route_weights.contiguous(),
                seg_starts,
                self._w1q,
                self._w1s,
                self._w2q,
                self._w2s,
                self._w3q,
                self._w3s,
                float(swiglu_limit),
            )
        elif hasattr(self._cuda_ext, "moe_prefill_int8_fused_forward"):
            y = self._cuda_ext.moe_prefill_int8_fused_forward(
                x.contiguous(),
                route_tokens,
                route_weights.contiguous(),
                seg_starts,
                self._w1q,
                self._w1s,
                self._w2q,
                self._w2s,
                self._w3q,
                self._w3s,
                float(swiglu_limit),
            )
        else:
            x_sorted = x.index_select(0, route_tokens).contiguous()
            y_sorted = self._cuda_ext.moe_prefill_int8_grouped_forward(
                x_sorted,
                route_weights.contiguous(),
                seg_starts,
                self._w1q,
                self._w1s,
                self._w2q,
                self._w2s,
                self._w3q,
                self._w3s,
                float(swiglu_limit),
            )
            y = torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
            y.index_add_(0, route_tokens, y_sorted)
        if self.profile_enabled:
            torch.cuda.synchronize(x.device)
            t_done = time.perf_counter()
            print(
                f"gpu_prefill_moe_profile layer={self.cpu_backend.layer_idx} tokens={x.shape[0]} "
                f"routes={route_tokens.numel()} group={t_compute - t0:.4f}s compute_scatter={t_done - t_compute:.4f}s total={t_done - t0:.4f}s",
                flush=True,
            )
        return y
