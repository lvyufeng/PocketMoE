import json
import os
import sys
import threading
import time
import uuid
from argparse import ArgumentParser
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.moe.shared_weights import SharedCPUMoEWeightArena
from src.runtime.generation import (
    _bind_shared_cpu_moe_weights,
    _cpu_affinity_for_rank,
    _enable_numa_interleave,
    generate,
    generate_stream,
    load_original_hf_model,
)
from src.runtime.transformer import ModelArgs, Transformer
from src.runtime.pd_scheduler import PDScheduler, run_single_request, run_single_stream_request

current_dir = os.path.dirname(os.path.abspath(__file__))
from src.encoding.dsv4 import encode_messages, eos_token, parse_message_from_completion_text  # noqa: E402


DEFAULT_MODEL_ID = "deepseek-v4-flash-w8a8"


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _set_best_env_defaults() -> None:
    os.environ.setdefault("DEEPSEEK_PD_PHASE_AUTO_SELECT", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", "3")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_ARENA", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_BUCKETED_GEMM", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_BUCKET_EXPERTS", "16")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS", "512")
    os.environ.setdefault("DEEPSEEK_INT8_IMPL", "cuda_ext")
    os.environ.setdefault("DEEPSEEK_MOE_ASYNC_ALLREDUCE", "1")
    os.environ.setdefault("DEEPSEEK_SHARED_EXPERT_INT8", "1")
    os.environ.setdefault("DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_FUSED_C4_INDEXER_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_HC_PRE_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_HC_POST_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD", "0")
    os.environ.setdefault("DEEPSEEK_CPU_TOPK_PERSISTENT", "1")
    os.environ.setdefault("DEEPSEEK_PD_DECODE_OMP_THREADS", "12")
    os.environ.setdefault("DEEPSEEK_FUSED_ATTN_PREFUSE", "1")
    for module in ("WQ_A", "WQ_B", "WKV", "WO_B", "INDEXER_WQ_B"):
        os.environ.setdefault(f"DEEPSEEK_PD_PREFILL_{module}_INT8", "1")
        os.environ.setdefault(f"DEEPSEEK_PD_DECODE_{module}_INT8", "1")
    os.environ.setdefault("DEEPSEEK_PD_PREFILL_WO_A_INT8", "0")
    os.environ.setdefault("DEEPSEEK_PD_DECODE_WO_A_INT8", "1")
    os.environ.setdefault("DEEPSEEK_WO_A_FP16", "1")
    os.environ.setdefault("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER", "0")
    os.environ.setdefault("DEEPSEEK_CPU_MOE_INPROC_SERVER", "0")
    os.environ.setdefault("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS", "0")


def _setup_cpu_runtime(routed_experts_device: str, local_rank: int, world_size: int) -> None:
    if routed_experts_device != "cpu":
        torch.set_num_threads(8)
        return
    omp_threads_env = os.getenv("DEEPSEEK_CPU_OMP_THREADS")
    omp_threads = int(omp_threads_env) if omp_threads_env else None
    use_affinity = os.getenv("DEEPSEEK_CPU_AFFINITY", "1").lower() not in {"0", "false", "no"}
    rank0_server = _env_enabled("DEEPSEEK_CPU_MOE_RANK0_SERVER")
    inproc_server = _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER")
    centralized_cpu_server = rank0_server or inproc_server
    server_omp_threads_env = os.getenv("DEEPSEEK_CPU_MOE_SERVER_OMP_THREADS")
    nonserver_omp_threads = int(os.getenv("DEEPSEEK_CPU_MOE_NONSERVER_OMP_THREADS", "1"))
    affinity_cpus = None if (centralized_cpu_server and local_rank == 0) else (_cpu_affinity_for_rank(local_rank, world_size) if use_affinity else None)
    if affinity_cpus is not None:
        os.sched_setaffinity(0, affinity_cpus)
        cpu_threads = omp_threads or max(len(affinity_cpus), 1)
    else:
        if centralized_cpu_server and local_rank == 0:
            cpu_threads = omp_threads or int(server_omp_threads_env or "22")
        elif inproc_server:
            cpu_threads = omp_threads or max(nonserver_omp_threads, 1)
        else:
            cpu_threads = omp_threads or max((os.cpu_count() or 1) // world_size, 1)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    if centralized_cpu_server and local_rank == 0:
        os.environ.setdefault("OMP_PROC_BIND", "spread")
    else:
        os.environ.setdefault("OMP_PROC_BIND", "close")
    import src.moe.cpu_backend as cpu_routed_backend

    cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=cpu_threads)
    torch.set_num_threads(1)


def _init_runtime(args):
    _set_best_env_defaults()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(days=7))
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    _setup_cpu_runtime(args.routed_experts_device, local_rank, world_size)
    torch.manual_seed(33377335)

    with open(args.config) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = args.routed_experts_device
    model_args = ModelArgs(**config_data)
    model_args.max_batch_size = 1
    if args.max_model_len:
        model_args.max_seq_len = int(args.max_model_len)

    shared_cpu_moe_arena = None
    init_start = time.perf_counter()
    with torch.device("cuda"):
        model = Transformer(model_args)
    if args.routed_experts_device == "cpu" and SharedCPUMoEWeightArena.enabled():
        if _env_enabled("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_NUMA_INTERLEAVE"):
            _enable_numa_interleave()
        shared_root_dir = SharedCPUMoEWeightArena.root_dir_from_env()
        if not shared_root_dir:
            raise RuntimeError("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS=1 requires DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR")
        shared_cpu_moe_arena = _bind_shared_cpu_moe_weights(model, shared_root_dir, model_args, world_size, rank)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    print(f"init time: {time.perf_counter() - init_start:.3f}s", flush=True)

    print("load model", flush=True)
    load_start = time.perf_counter()
    load_original_hf_model(model, args.ckpt_path, world_size, rank)
    if args.routed_experts_device == "cpu":
        model.prepare_cpu_expert_int8()
        if shared_cpu_moe_arena is not None:
            shared_cpu_moe_arena.mark_ready()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"load time: {time.perf_counter() - load_start:.3f}s", flush=True)
    torch.set_default_device("cuda")

    control_group = dist.new_group(backend="gloo", timeout=timedelta(days=7)) if world_size > 1 else None
    scheduler = PDScheduler() if args.pd_mode == "scheduler" else None
    return {
        "model": model,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "model_id": args.model,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "control_group": control_group,
        "shared_cpu_moe_arena": shared_cpu_moe_arena,
    }


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(f"[Unsupported {block.get('type', 'content')}]" )
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _prepare_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for msg in body.get("messages", []):
        copied = dict(msg)
        if "content" in copied:
            copied["content"] = _normalize_content(copied["content"])
        messages.append(copied)
    if not messages:
        raise ValueError("messages must be a non-empty list")
    attach_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in {"user", "developer", "system"}:
            attach_idx = idx
            break
    if attach_idx is not None:
        if body.get("tools") is not None:
            messages[attach_idx]["tools"] = body["tools"]
        if body.get("response_format") is not None:
            messages[attach_idx]["response_format"] = body["response_format"]
    return messages


def _thinking_config(body: dict[str, Any]) -> tuple[str, str | None]:
    reasoning = body.get("reasoning")
    if reasoning is None:
        return "chat", None
    effort = None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort not in {None, "high", "max"}:
        effort = None
    return "thinking", effort


def _strip_special_eos(text: str) -> str:
    if text.endswith(eos_token):
        return text[: -len(eos_token)]
    return text


def _run_payload(runtime: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    torch.cuda.set_device(runtime["local_rank"])
    torch.set_default_device("cuda")
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    scheduler = runtime["scheduler"]
    messages = payload["messages"]
    thinking_mode = payload["thinking_mode"]
    reasoning_effort = payload.get("reasoning_effort")
    prompt_text = encode_messages(messages, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)
    prompt_ids = tokenizer.encode(prompt_text)
    max_tokens = int(payload.get("max_tokens") or 512)
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    runner = generate
    if scheduler is None:
        completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens = runner(
            model,
            [prompt_ids],
            max_tokens,
            tokenizer.eos_token_id,
            temperature,
        )
    else:
        completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens = run_single_request(
            runner,
            model,
            [prompt_ids],
            max_tokens,
            tokenizer.eos_token_id,
            temperature,
            scheduler=scheduler,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    completion_ids = completion_tokens[0]
    completion_text = tokenizer.decode(completion_ids)
    try:
        assistant_msg = parse_message_from_completion_text(completion_text, thinking_mode)
    except Exception:
        assistant_msg = {
            "role": "assistant",
            "content": _strip_special_eos(completion_text),
            "reasoning_content": "",
            "tool_calls": [],
        }
    content = assistant_msg.get("content") or ""
    reasoning = assistant_msg.get("reasoning_content") or ""
    tool_calls = assistant_msg.get("tool_calls") or []
    return {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "timings": {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / max(decode_time, 1e-9),
        },
    }


def _make_payload(body: dict[str, Any]) -> dict[str, Any]:
    thinking_mode, reasoning_effort = _thinking_config(body)
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 512))
    return {
        "op": "chat_completion",
        "request_id": f"chatcmpl-{uuid.uuid4().hex}",
        "messages": _prepare_messages(body),
        "max_tokens": int(max_tokens or 512),
        "temperature": float(body.get("temperature", 0.0) or 0.0),
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort,
        "stream": bool(body.get("stream", False)),
        "stream_options": body.get("stream_options") or {},
    }


def _run_payload_stream(runtime: dict[str, Any], payload: dict[str, Any]):
    torch.cuda.set_device(runtime["local_rank"])
    torch.set_default_device("cuda")
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    scheduler = runtime["scheduler"]
    prompt_text = encode_messages(
        payload["messages"],
        thinking_mode=payload["thinking_mode"],
        reasoning_effort=payload.get("reasoning_effort"),
    )
    prompt_ids = tokenizer.encode(prompt_text)
    max_tokens = int(payload.get("max_tokens") or 512)
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    if scheduler is None:
        events = generate_stream(
            model,
            [prompt_ids],
            max_tokens,
            tokenizer.eos_token_id,
            temperature,
        )
    else:
        events = run_single_stream_request(
            generate_stream,
            model,
            [prompt_ids],
            max_tokens,
            tokenizer.eos_token_id,
            temperature,
            scheduler=scheduler,
        )
    for event in events:
        if event.get("type") == "done":
            event = dict(event)
            event["prompt_tokens"] = len(prompt_ids)
        yield event


class _StreamingDecoder:
    def __init__(self, tokenizer, thinking_mode: str):
        self.tokenizer = tokenizer
        self.thinking_mode = thinking_mode
        self.token_ids: list[int] = []
        self.emitted_text = ""
        self.content_emitted = ""
        self.reasoning_emitted = ""

    def append(self, token_ids: list[int]) -> list[dict[str, str]]:
        self.token_ids.extend(int(t) for t in token_ids)
        decoded = _strip_special_eos(self.tokenizer.decode(self.token_ids))
        if len(decoded) < len(self.emitted_text):
            return []
        delta = decoded[len(self.emitted_text):]
        self.emitted_text = decoded
        if not delta:
            return []
        if self.thinking_mode != "thinking":
            self.content_emitted += delta
            return [{"content": delta}]
        outputs: list[dict[str, str]] = []
        marker = "</think>"
        marker_idx = decoded.find(marker)
        if marker_idx < 0:
            new_reasoning = decoded[len(self.reasoning_emitted):]
            if new_reasoning:
                self.reasoning_emitted = decoded
                outputs.append({"reasoning_content": new_reasoning})
            return outputs
        reasoning_text = decoded[:marker_idx]
        content_text = decoded[marker_idx + len(marker):]
        if len(reasoning_text) > len(self.reasoning_emitted):
            outputs.append({"reasoning_content": reasoning_text[len(self.reasoning_emitted):]})
            self.reasoning_emitted = reasoning_text
        if len(content_text) > len(self.content_emitted):
            outputs.append({"content": content_text[len(self.content_emitted):]})
            self.content_emitted = content_text
        return outputs

    def final_message(self, thinking_mode: str) -> dict[str, Any]:
        text = self.tokenizer.decode(self.token_ids)
        try:
            return parse_message_from_completion_text(text, thinking_mode)
        except Exception:
            return {
                "role": "assistant",
                "content": _strip_special_eos(text),
                "reasoning_content": "",
                "tool_calls": [],
            }


def _broadcast_payload(payload: dict[str, Any], runtime: dict[str, Any]) -> None:
    if runtime["world_size"] > 1:
        dist.broadcast_object_list([payload], src=0, group=runtime["control_group"])


def _completion_response(runtime: dict[str, Any], payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    created = int(time.time())
    message = {
        "role": "assistant",
        "content": result["content"],
    }
    if result["reasoning_content"]:
        message["reasoning_content"] = result["reasoning_content"]
    if result["tool_calls"]:
        message["tool_calls"] = result["tool_calls"]
    return {
        "id": payload["request_id"],
        "object": "chat.completion",
        "created": created,
        "model": runtime["model_id"],
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if result["tool_calls"] else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "deepseek_timings": result["timings"],
    }


def _chunk_payload(runtime: dict[str, Any], payload: dict[str, Any], delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    choice = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {
        "id": payload["request_id"],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": runtime["model_id"],
        "choices": [choice],
    }


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sse_line(obj: Any) -> bytes:
    if obj == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + _json_bytes(obj) + b"\n\n"


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekOpenAIServer/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, status: int, obj: Any) -> None:
        data = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        runtime = self.server.runtime
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "rank": runtime["rank"], "world_size": runtime["world_size"], "model": runtime["model_id"]})
            return
        if self.path == "/v1/models":
            now = int(time.time())
            self._send_json(200, {"object": "list", "data": [{"id": runtime["model_id"], "object": "model", "created": now, "owned_by": "local"}]})
            return
        self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        runtime = self.server.runtime
        try:
            body = self._read_json_body()
            stream = bool(body.get("stream", False))
            payload = _make_payload(body)
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        if not stream:
            with self.server.request_lock:
                try:
                    _broadcast_payload(payload, runtime)
                    result = _run_payload(runtime, payload)
                except Exception as exc:
                    self._send_json(500, {"error": {"message": str(exc), "type": "server_error"}})
                    return
            self._send_json(200, _completion_response(runtime, payload, result))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        disconnected = False
        decoder = _StreamingDecoder(runtime["tokenizer"], payload["thinking_mode"])
        self.wfile.write(_sse_line(_chunk_payload(runtime, payload, {"role": "assistant"})))
        self.wfile.flush()
        done_event = None
        with self.server.request_lock:
            try:
                _broadcast_payload(payload, runtime)
                for event in _run_payload_stream(runtime, payload):
                    if event.get("type") == "token":
                        if disconnected:
                            continue
                        for delta in decoder.append(event.get("token_ids", [])):
                            try:
                                self.wfile.write(_sse_line(_chunk_payload(runtime, payload, delta)))
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                disconnected = True
                    elif event.get("type") == "done":
                        done_event = event
            except Exception as exc:
                if not disconnected:
                    self._send_json(500, {"error": {"message": str(exc), "type": "server_error"}})
                return
        if done_event is None:
            return
        final_msg = decoder.final_message(payload["thinking_mode"])
        final_tool_calls = final_msg.get("tool_calls") or []
        if not disconnected and final_tool_calls:
            self.wfile.write(_sse_line(_chunk_payload(runtime, payload, {"tool_calls": final_tool_calls})))
        finish = "tool_calls" if final_tool_calls else done_event.get("finish_reason", "stop")
        if not disconnected:
            include_usage = bool(payload.get("stream_options", {}).get("include_usage", False))
            final_chunk = _chunk_payload(runtime, payload, {}, finish_reason=finish)
            if include_usage:
                final_chunk["usage"] = {
                    "prompt_tokens": done_event["prompt_tokens"],
                    "completion_tokens": len(done_event["completion_tokens"][0]),
                    "total_tokens": done_event["prompt_tokens"] + len(done_event["completion_tokens"][0]),
                }
                final_chunk["deepseek_timings"] = {
                    "prefill_time": done_event["prefill_time"],
                    "decode_time": done_event["decode_time"],
                    "prefill_tokens": done_event["prefill_tokens"],
                    "decode_tokens": done_event["decode_tokens"],
                    "decode_tokens_per_second": done_event["decode_tokens"] / max(done_event["decode_time"], 1e-9),
                }
            self.wfile.write(_sse_line(final_chunk))
            self.wfile.write(_sse_line("[DONE]"))
            self.wfile.flush()
        self.close_connection = True


class DeepSeekHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, runtime):
        super().__init__(server_address, handler_class)
        self.runtime = runtime
        self.request_lock = threading.Lock()


def _worker_loop(runtime: dict[str, Any]) -> None:
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0, group=runtime["control_group"])
        payload = box[0]
        if not isinstance(payload, dict):
            continue
        if payload.get("op") == "shutdown":
            break
        if payload.get("op") == "chat_completion":
            if payload.get("stream"):
                for _event in _run_payload_stream(runtime, payload):
                    pass
            else:
                _run_payload(runtime, payload)


def _serve(runtime: dict[str, Any], host: str, port: int) -> None:
    server = DeepSeekHTTPServer((host, port), OpenAIHandler, runtime)
    print(f"OpenAI-compatible server listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        if runtime["world_size"] > 1:
            _broadcast_payload({"op": "shutdown"}, runtime)
        server.server_close()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ckpt-path", type=str, default=os.environ.get("CKPT_PATH", os.path.abspath(os.path.join(current_dir, "../../checkpoints/DeepSeek-V4-Flash-w8a8"))))
    parser.add_argument("--config", type=str, default=os.path.abspath(os.path.join(current_dir, "../../configs/config_w8a8.json")))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--routed-experts-device", type=str, choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--pd-mode", type=str, choices=["off", "scheduler"], default="scheduler")
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    runtime = _init_runtime(args)
    if runtime["rank"] == 0:
        _serve(runtime, args.host, args.port)
    else:
        _worker_loop(runtime)
    if dist.is_initialized():
        dist.destroy_process_group()
    if runtime.get("shared_cpu_moe_arena") is not None:
        runtime["shared_cpu_moe_arena"].close(unlink=True)


if __name__ == "__main__":
    main()
