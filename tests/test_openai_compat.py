"""End-to-end check of the free-LLM path against a local fake OpenAI-compatible server (no network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from bacm.config import DynamicsConfig, RunConfig, TaskConfig, get_arch
from bacm.llm.openai_compat import OpenAICompatClient
from bacm.sim.orchestrator import Orchestrator


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        if "tools" not in body:   # manager worker prompt
            msg = {"role": "assistant", "content": "redis"}
        else:
            first = next(m["content"] for m in msgs if m["role"] == "user")
            keys = [l.split("]")[0].split("[")[1] for l in first.splitlines() if l.startswith("- [")]
            calls = [c for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]
            answered = [c for c in calls if c["function"]["name"] == "answer"]
            if not calls:
                tc = {"name": "read_file", "arguments": json.dumps({"path": "api/routes_0.py"})}
            elif len(answered) < len(keys):
                tc = {"name": "answer", "arguments": json.dumps({"key": keys[len(answered)], "value": "redis"})}
            else:
                tc = None
            msg = {"role": "assistant", "content": None if tc else "done"}
            if tc:
                msg["tool_calls"] = [{"id": f"c{len(calls)}", "type": "function", "function": tc}]
        out = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_openai_compat_backend_end_to_end(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        for arch in ("independent", "bacm"):
            client = OpenAICompatClient("custom", "fake-model", url, cache_dir=str(tmp_path))
            cfg = RunConfig(arch=get_arch(arch), seed=1, backend="openai_compat", model="fake-model",
                            task=TaskConfig(n_agents=2, questions_per_agent=2),
                            dynamics=DynamicsConfig(n_changes=1))
            o = Orchestrator(cfg, client)
            m = o.run()
            assert m["llm_calls"] >= 6 and m["tokens_in"] > 0
            assert len({(a, k) for a, k, *_ in o.tracker.answers}) == 4
        # second identical run is served from the on-disk cache
        client2 = OpenAICompatClient("custom", "fake-model", url, cache_dir=str(tmp_path))
        Orchestrator(cfg, client2).run()
        assert client2.cache_hits == client2.calls > 0
    finally:
        srv.shutdown()


class KeyAware(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        key = self.headers.get("Authorization", "").removeprefix("Bearer ")
        if key == "bad":
            code, out = 401, b'{"error": {"message": "invalid credentials"}}'
        elif key == "quota":
            code, out = 429, (b'{"error": {"message": "Quota exceeded for metric: generate_content_free_tier_requests,'
                              b' limit: 500. Please retry in 10h46m10s."}}')
        else:
            code, out = 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_key_failover_and_quota_errors():
    import pytest

    from bacm.llm.openai_compat import AuthError, QuotaExhausted, safe_name
    srv = HTTPServer(("127.0.0.1", 0), KeyAware)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        c = OpenAICompatClient("custom", "m", url, api_key="bad,quota,good", cache_dir=None)
        for _ in range(3):
            blocks, *_ = c.complete("s", [{"role": "user", "content": "hi"}])
            assert blocks[0].text == "ok"
        assert set(c.dead_keys) == {"bad", "quota"}
        with pytest.raises(AuthError):
            OpenAICompatClient("custom", "m", url, api_key="bad", cache_dir=None).complete("s", [{"role": "user", "content": "x"}])
        with pytest.raises(QuotaExhausted):
            OpenAICompatClient("custom", "m", url, api_key="quota,bad", cache_dir=None).complete("s", [{"role": "user", "content": "x"}])
    finally:
        srv.shutdown()
    assert safe_name("qwen2.5:3b") == "qwen2.5_3b" and safe_name("openai/gpt-oss-120b") == "openai_gpt-oss-120b"
