from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator


_DONE = object()


class _RequestResult:
    def __init__(self, stream: bool) -> None:
        self.stream = stream
        self._done = threading.Event()
        self._events: queue.Queue[Any] = queue.Queue(maxsize=int(os.getenv("DEEPSEEK_SERVING_EVENT_QUEUE", "1024")))
        self._result: Any = None
        self._exception: BaseException | None = None

    def set_result(self, result: Any) -> None:
        self._result = result
        self._done.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._events.put(exc)
        self._done.set()

    def put_event(self, event: dict[str, Any]) -> None:
        self._events.put(event)

    def close_events(self) -> None:
        self._events.put(_DONE)
        self._done.set()

    def result(self) -> Any:
        self._done.wait()
        if self._exception is not None:
            raise self._exception
        return self._result

    def events(self) -> Iterator[dict[str, Any]]:
        while True:
            item = self._events.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


@dataclass
class _RequestMetadata:
    prompt_tokens: int
    max_tokens: int
    stream: bool
    profile: str
    decode_first: bool


@dataclass
class _QueuedRequest:
    payload: dict[str, Any]
    stream: bool
    result: _RequestResult
    metadata: _RequestMetadata


class DeepSeekServingEngine:
    def __init__(
        self,
        runtime: dict[str, Any],
        broadcast_payload: Callable[[dict[str, Any], dict[str, Any]], None],
        run_payload: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        run_payload_stream: Callable[[dict[str, Any], dict[str, Any]], Iterator[dict[str, Any]]],
    ) -> None:
        self._runtime = runtime
        self._broadcast_payload = broadcast_payload
        self._run_payload = run_payload
        self._run_payload_stream = run_payload_stream
        self._max_queue = int(os.getenv("DEEPSEEK_SERVING_MAX_QUEUE", "32"))
        self._max_running_requests = max(1, int(os.getenv("DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS", "1")))
        self._batch_wait_ms = max(0.0, float(os.getenv("DEEPSEEK_SERVING_BATCH_WAIT_MS", "2")))
        self._max_total_tokens = int(os.getenv("DEEPSEEK_SERVING_MAX_TOTAL_TOKENS", "0"))
        self._serving_profile = os.getenv("SERVING_PROFILE", "custom")
        self._decode_first = os.getenv("DEEPSEEK_SERVING_DECODE_FIRST", "1").lower() not in {"0", "false", "no"}
        self._queue: queue.Queue[_QueuedRequest | None] = queue.Queue(maxsize=self._max_queue)
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._run, name="ds4-serving-engine", daemon=True)
        self._worker.start()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._enqueue(payload, stream=False)
        return request.result.result()

    def submit_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        request = self._enqueue(payload, stream=True)
        return request.result.events()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=5.0)

    def _enqueue(self, payload: dict[str, Any], stream: bool) -> _QueuedRequest:
        if self._closed.is_set():
            raise RuntimeError("serving engine is closed")
        self._admit(payload)
        result = _RequestResult(stream=stream)
        request = _QueuedRequest(payload=payload, stream=stream, result=result, metadata=self._metadata(payload, stream))
        try:
            self._queue.put_nowait(request)
        except queue.Full as exc:
            raise RuntimeError("serving request queue is full") from exc
        return request

    def _metadata(self, payload: dict[str, Any], stream: bool) -> _RequestMetadata:
        prompt_ids = payload.get("_prompt_ids") or []
        return _RequestMetadata(
            prompt_tokens=len(prompt_ids),
            max_tokens=int(payload.get("max_tokens") or 0),
            stream=stream,
            profile=self._serving_profile,
            decode_first=self._decode_first,
        )

    def _admit(self, payload: dict[str, Any]) -> None:
        prompt_ids = payload.get("_prompt_ids") or []
        max_tokens = int(payload.get("max_tokens") or 0)
        total_tokens = len(prompt_ids) + max_tokens
        if self._max_total_tokens > 0 and total_tokens > self._max_total_tokens:
            raise RuntimeError(
                f"request token budget {total_tokens} exceeds DEEPSEEK_SERVING_MAX_TOTAL_TOKENS={self._max_total_tokens}"
            )
        if self._max_running_requests > 1:
            model = self._runtime.get("model")
            max_batch = int(getattr(model, "max_batch_size", 1)) if model is not None else 1
            if self._max_running_requests > max_batch:
                raise RuntimeError(
                    f"DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS={self._max_running_requests} exceeds model max_batch_size={max_batch}"
                )

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return
            if request.stream:
                try:
                    self._broadcast_payload(request.payload, self._runtime)
                    for event in self._run_payload_stream(self._runtime, request.payload):
                        request.result.put_event(event)
                    request.result.close_events()
                except BaseException as exc:
                    request.result.set_exception(exc)
                continue

            batch = [request]
            if self._max_running_requests > 1:
                batch = self._collect_batch(request)
            try:
                if len(batch) == 1:
                    self._broadcast_payload(batch[0].payload, self._runtime)
                    batch[0].result.set_result(self._run_payload(self._runtime, batch[0].payload))
                else:
                    payload = self._batch_payload(batch)
                    self._broadcast_payload(payload, self._runtime)
                    results = self._run_payload(self._runtime, payload)
                    if not isinstance(results, list) or len(results) != len(batch):
                        raise RuntimeError("batched serving result shape mismatch")
                    for queued, result in zip(batch, results):
                        queued.result.set_result(result)
            except BaseException as exc:
                for queued in batch:
                    queued.result.set_exception(exc)

    def _collect_batch(self, first: _QueuedRequest) -> list[_QueuedRequest]:
        batch = [first]
        deadline = time.monotonic() + self._batch_wait_ms / 1000.0
        while len(batch) < self._max_running_requests:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                candidate = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if candidate is None:
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    self._queue.put(None)
                break
            if candidate.stream or not self._is_batch_compatible(first, candidate):
                try:
                    self._queue.put_nowait(candidate)
                except queue.Full:
                    self._queue.put(candidate)
                break
            batch.append(candidate)
        return batch

    def _is_batch_compatible(self, first: _QueuedRequest, candidate: _QueuedRequest) -> bool:
        if candidate.stream:
            return False
        first_max_tokens = int(first.payload.get("max_tokens") or 0)
        cand_max_tokens = int(candidate.payload.get("max_tokens") or 0)
        if first_max_tokens != cand_max_tokens:
            return False
        compare_keys = (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "stop",
            "logprobs",
            "top_logprobs",
            "n",
        )
        for key in compare_keys:
            if first.payload.get(key) != candidate.payload.get(key):
                return False
        return True

    def _batch_payload(self, batch: list[_QueuedRequest]) -> dict[str, Any]:
        payload0 = dict(batch[0].payload)
        payload0["op"] = "chat_completion_batch"
        payload0["_batch_size"] = len(batch)
        payload0["_prompt_ids_batch"] = [req.payload.get("_prompt_ids") or [] for req in batch]
        payload0["_max_tokens_batch"] = [int(req.payload.get("max_tokens") or 0) for req in batch]
        payload0["_thinking_mode_batch"] = [req.payload.get("thinking_mode") for req in batch]
        payload0["_reasoning_effort_batch"] = [req.payload.get("reasoning_effort") for req in batch]
        payload0["_request_ids_batch"] = [req.payload.get("request_id") for req in batch]
        payload0["_serving_profile"] = self._serving_profile
        payload0["_decode_first"] = self._decode_first
        payload0["_prompt_token_lengths_batch"] = [req.metadata.prompt_tokens for req in batch]
        payload0["_stream_batch"] = [req.metadata.stream for req in batch]
        return payload0
