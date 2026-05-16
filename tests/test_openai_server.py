import json
import threading
from urllib import request

from src.server.openai import DeepSeekHTTPServer, OpenAIHandler


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return list(range(max(1, len(text.split()))))

    def decode(self, token_ids):
        if token_ids == [101]:
            return "hello<｜end▁of▁sentence｜>"
        if token_ids == [201]:
            return "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">sh</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        return "".join(chr(i) for i in token_ids)


class FakeEngine:
    def submit(self, payload):
        return {
            "prompt_tokens": len(payload.get("_prompt_ids") or []),
            "completion_tokens": 1,
            "content": "hello",
            "reasoning_content": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "token_texts": ["hello"] if payload.get("logprobs") else [],
            "token_logprobs": [-0.1] if payload.get("logprobs") else [],
            "top_logprobs": [[{"token": "hello", "logprob": -0.1}]] if payload.get("logprobs") else [],
            "timings": {"prefill_time": 0.1, "decode_time": 0.2, "prefill_tokens": 1, "decode_tokens": 1},
        }

    def submit_stream(self, payload):
        if payload.get("stop"):
            yield {"type": "token", "token_ids": [97]}
            yield {"type": "token", "token_ids": [98]}
            yield {"type": "token", "token_ids": [99]}
            yield {"type": "done", "prompt_tokens": len(payload.get("_prompt_ids") or []), "completion_tokens": [[97, 98, 99]], "prefill_time": 0.1, "decode_time": 0.2, "prefill_tokens": 1, "decode_tokens": 3, "finish_reason": "length"}
            return
        yield {"type": "token", "token_ids": [201]}
        yield {"type": "done", "prompt_tokens": len(payload.get("_prompt_ids") or []), "completion_tokens": [[201]], "prefill_time": 0.1, "decode_time": 0.2, "prefill_tokens": 1, "decode_tokens": 1, "finish_reason": "stop"}

    def close(self):
        pass


def _server():
    runtime = {"rank": 0, "world_size": 1, "model_id": "fake-model", "tokenizer": FakeTokenizer()}
    server = DeepSeekHTTPServer(("127.0.0.1", 0), OpenAIHandler, runtime, FakeEngine())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post_json(url, body):
    req = request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def test_health_models_and_non_stream_chat():
    server, base = _server()
    try:
        with request.urlopen(base + "/health", timeout=10) as resp:
            health = json.loads(resp.read().decode())
        assert health["status"] == "ok"
        with request.urlopen(base + "/v1/models", timeout=10) as resp:
            models = json.loads(resp.read().decode())
        assert models["data"][0]["id"] == "fake-model"
        result = _post_json(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["usage"]["completion_tokens"] == 1
        with_logprobs = _post_json(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "logprobs": True, "top_logprobs": 1})
        assert with_logprobs["choices"][0]["logprobs"]["content"][0]["token"] == "hello"
    finally:
        server.shutdown()
        server.server_close()


def test_streaming_stop_sequence_truncates_output():
    server, base = _server()
    try:
        body = {"messages": [{"role": "user", "content": "letters"}], "stream": True, "stop": "bc"}
        req = request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
        lines = [line[len("data: "):] for line in raw.splitlines() if line.startswith("data: ")]
        chunks = [json.loads(line) for line in lines if line != "[DONE]"]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert content == "a"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        server.shutdown()
        server.server_close()



def test_streaming_tool_call_final_delta_and_usage():
    server, base = _server()
    try:
        body = {
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            "tool_choice": "required",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
        lines = [line[len("data: "):] for line in raw.splitlines() if line.startswith("data: ")]
        chunks = [json.loads(line) for line in lines if line != "[DONE]"]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert tool_chunks
        tool_call = tool_chunks[-1]["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["id"].startswith("call_")
        assert tool_call["function"]["name"] == "weather"
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert "usage" in chunks[-1]
        assert lines[-1] == "[DONE]"
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_health_models_and_non_stream_chat()
    test_streaming_tool_call_final_delta_and_usage()
