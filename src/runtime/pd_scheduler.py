"""Logical PD-separation scheduler for the standalone TP=4 inference runtime."""

from __future__ import annotations

import collections
import os
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterator, List, Optional, Tuple


_ATTN_INT8_SUFFIXES = (
    "WQ_A_INT8",
    "WQ_B_INT8",
    "WKV_INT8",
    "WO_A_INT8",
    "WO_B_INT8",
    "INDEXER_WQ_B_INT8",
)

_VALID_PHASES = ("prefill", "decode")

@dataclass
class Request:
    request_id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    phase: str = "prefill"
    kv_owned: bool = False
    next_prefill_offset: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class PDExecutionConfig:
    scheduler: Optional["PDScheduler"] = None
    prefill_chunk_tokens: int = 0
    decode_first: bool = True

    @property
    def phase_callback(self) -> Optional[Callable[[str], None]]:
        if self.scheduler is None or not self.scheduler.has_phase_overrides():
            return None
        return self.scheduler.apply_phase_resources

    @classmethod
    def from_env(cls, scheduler: Optional["PDScheduler"] = None) -> "PDExecutionConfig":
        prefill_chunk_tokens = int(os.getenv("DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS", "0") or "0")
        decode_first = os.getenv("DEEPSEEK_SERVING_DECODE_FIRST", "1").lower() not in {"0", "false", "no"}
        return cls(scheduler=scheduler, prefill_chunk_tokens=prefill_chunk_tokens, decode_first=decode_first)


class PDExecutionFacade:
    def __init__(self, generate_fn: Callable, generate_stream_fn: Callable, config: PDExecutionConfig) -> None:
        self._generate_fn = generate_fn
        self._generate_stream_fn = generate_stream_fn
        self.config = config

    @classmethod
    def from_env(
        cls,
        generate_fn: Callable,
        generate_stream_fn: Callable,
        scheduler: Optional["PDScheduler"] = None,
    ) -> "PDExecutionFacade":
        return cls(generate_fn, generate_stream_fn, PDExecutionConfig.from_env(scheduler))

    def run(
        self,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
        **generate_kwargs,
    ):
        return self._call_generate_with_kwargs(
            self._generate_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )

    def stream(
        self,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
        **generate_kwargs,
    ) -> Iterator[dict[str, Any]]:
        yield from self._call_generate_with_kwargs(
            self._generate_stream_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )

    def _call_generate(
        self,
        generate_fn: Callable,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
    ):
        return self._call_generate_with_kwargs(
            generate_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            {},
        )

    def _call_generate_with_kwargs(
        self,
        generate_fn: Callable,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
        extra_kwargs: dict[str, Any],
    ):
        kwargs: dict[str, Any] = dict(extra_kwargs)
        kwargs["prefill_chunk_tokens"] = self.config.prefill_chunk_tokens
        phase_callback = self.config.phase_callback
        if phase_callback is not None:
            kwargs["phase_callback"] = phase_callback
        return generate_fn(model, prompt_tokens, max_new_tokens, eos_id, temperature, **kwargs)


class PDScheduler:
    def __init__(self) -> None:
        self.decode_queue: Deque[Request] = collections.deque()
        self.prefill_queue: Deque[Request] = collections.deque()
        self._next_request_id: int = 0
        self._current_phase: Optional[str] = None
        self._current_runtime_threads: Optional[int] = None
        self._pause_server_during_prefill = os.getenv("DEEPSEEK_PD_PAUSE_SERVER_DURING_PREFILL", "0").lower() in {"1", "true", "yes"}
        rank = int(os.getenv("RANK", "0"))
        server_pid_env = os.getenv("DEEPSEEK_PD_SERVER_PID") if rank == 0 else None
        try:
            self._server_pid: Optional[int] = int(server_pid_env) if server_pid_env else None
        except ValueError:
            self._server_pid = None
        self._server_paused: Optional[bool] = None

    def submit(self, prompt_tokens: List[int], max_new_tokens: int) -> Request:
        request = Request(
            request_id=self._next_request_id,
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            phase="prefill",
        )
        self._next_request_id += 1
        self.prefill_queue.append(request)
        return request

    def has_work(self) -> bool:
        return bool(self.decode_queue) or bool(self.prefill_queue)

    def next_step(self) -> Optional[Tuple[Request, str]]:
        if self.decode_queue:
            return self.decode_queue[0], "decode"
        if self.prefill_queue:
            return self.prefill_queue[0], "prefill"
        return None

    def mark_prefill_done(self, request: Request) -> None:
        try:
            self.prefill_queue.remove(request)
        except ValueError:
            pass
        request.phase = "decode"
        request.kv_owned = True
        self.decode_queue.append(request)

    def mark_request_done(self, request: Request) -> None:
        try:
            self.decode_queue.remove(request)
        except ValueError:
            pass
        try:
            self.prefill_queue.remove(request)
        except ValueError:
            pass
        request.kv_owned = False

    def has_phase_overrides(self) -> bool:
        return (
            any(os.getenv(f"DEEPSEEK_PD_{phase.upper()}_{suffix}") for phase in _VALID_PHASES for suffix in ("CPUS", "OMP_THREADS", "NUMA_NODE"))
            or os.getenv("DEEPSEEK_PD_PAUSE_SERVER_DURING_PREFILL", "0").lower() in {"1", "true", "yes"}
            or os.getenv("DEEPSEEK_PD_PHASE_AUTO_SELECT", "0").lower() in {"1", "true", "yes"}
        )

    def apply_phase_resources(self, phase: str) -> None:
        if phase not in _VALID_PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        os.environ["DEEPSEEK_PD_ACTIVE_PHASE"] = phase
        if self._current_phase == phase:
            return
        self._current_phase = phase
        self._apply_phase_attention_env(phase)
        self._set_server_paused(phase == "prefill")

        cpus_env = os.getenv(f"DEEPSEEK_PD_{phase.upper()}_CPUS")
        if cpus_env and hasattr(os, "sched_setaffinity"):
            cpus = _parse_cpu_list(cpus_env)
            if cpus:
                try:
                    os.sched_setaffinity(0, set(cpus))
                except OSError:
                    pass

        omp_env = os.getenv(f"DEEPSEEK_PD_{phase.upper()}_OMP_THREADS")
        if omp_env:
            try:
                self._current_runtime_threads = max(1, int(omp_env))
                os.environ["OMP_NUM_THREADS"] = str(self._current_runtime_threads)
                try:
                    import src.runtime.moe.cpu_backend as cpu_routed_backend
                    cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=self._current_runtime_threads)
                except Exception:
                    pass
            except ValueError:
                pass

        numa_env = os.getenv(f"DEEPSEEK_PD_{phase.upper()}_NUMA_NODE")
        if numa_env:
            _try_apply_numa_node(numa_env)

    def _apply_phase_attention_env(self, phase: str) -> None:
        for suffix in _ATTN_INT8_SUFFIXES:
            phase_key = f"DEEPSEEK_PD_{phase.upper()}_{suffix}"
            if phase_key in os.environ:
                os.environ[f"DEEPSEEK_ACTIVE_{suffix}"] = os.environ[phase_key]
            else:
                os.environ.pop(f"DEEPSEEK_ACTIVE_{suffix}", None)

    def _set_server_paused(self, paused: bool) -> None:
        if not self._pause_server_during_prefill or self._server_pid is None:
            return
        if self._server_paused is paused:
            return
        try:
            os.kill(self._server_pid, signal.SIGSTOP if paused else signal.SIGCONT)
            self._server_paused = paused
        except ProcessLookupError:
            self._server_pid = None
        except OSError:
            pass


def _parse_cpu_list(spec: str) -> List[int]:
    cpus: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                cpus.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return sorted(set(cpus))


def _try_apply_numa_node(spec: str) -> None:
    try:
        nodes = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        return
    if not nodes:
        return
    try:
        import ctypes
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError:
        return
    try:
        libnuma.numa_run_on_node.restype = ctypes.c_int
        libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
        libnuma.numa_run_on_node(nodes[0])
    except Exception:
        return


def run_single_request(
    generate_fn: Callable,
    model,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
    scheduler: Optional[PDScheduler] = None,
    **generate_kwargs,
):
    prefill_chunk_tokens = int(generate_kwargs.pop("prefill_chunk_tokens", 0) or 0)
    facade = PDExecutionFacade(
        generate_fn,
        lambda *args, **kwargs: iter(()),
        PDExecutionConfig(scheduler=scheduler or PDScheduler(), prefill_chunk_tokens=prefill_chunk_tokens),
    )
    if generate_kwargs:
        return facade._call_generate_with_kwargs(
            generate_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )
    return facade.run(model, prompt_tokens, max_new_tokens, eos_id, temperature)


def run_single_stream_request(
    generate_stream_fn: Callable,
    model,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
    scheduler: Optional[PDScheduler] = None,
    **generate_kwargs,
):
    prefill_chunk_tokens = int(generate_kwargs.pop("prefill_chunk_tokens", 0) or 0)
    facade = PDExecutionFacade(
        lambda *args, **kwargs: None,
        generate_stream_fn,
        PDExecutionConfig(scheduler=scheduler or PDScheduler(), prefill_chunk_tokens=prefill_chunk_tokens),
    )
    if generate_kwargs:
        yield from facade._call_generate_with_kwargs(
            generate_stream_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )
        return
    yield from facade.stream(model, prompt_tokens, max_new_tokens, eos_id, temperature)
