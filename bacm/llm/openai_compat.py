"""Client for any OpenAI-compatible chat-completions endpoint: local Ollama (free, unlimited)
and free-tier hosted APIs (Groq, Google Gemini, OpenRouter ':free' models, Cerebras, ...).

Agents keep a provider-neutral, Anthropic-style message history (text / tool_use /
tool_result blocks); this client translates it to the OpenAI wire format per call.

Built for free tiers: requests/minute throttling, retry with backoff on 429/5xx, and an
on-disk response cache so a crashed or rate-limited suite can be resumed without
spending quota twice (cached calls replay the originally recorded token usage).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PRESETS = {
    # name: (base_url, env var holding the key or None, default model)
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5-nano"),
}

# USD per million tokens (input, cached input, output) for the cost report and the spending guard.
# Source: OpenAI pricing page, Oct 2026. Unknown models cost 0 here, so the guard cannot protect them.
PRICES = {
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
}


@dataclass
class Block:
    type: str                      # "text" | "tool_use"
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = ""
    raw: dict | None = None    # provider fields to echo back verbatim (e.g. Gemini thought signatures)


def _to_openai(system: str, messages: list, tools: list | None):
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(getattr(b, "text", "") or "" for b in content if getattr(b, "type", "") == "text")
            calls = [getattr(b, "raw", None) or {"id": b.id, "type": "function",
                                                 "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                     for b in content if getattr(b, "type", "") == "tool_use"]
            msg = {"role": "assistant", "content": text or (None if calls else "")}   # null only with tool calls
            extra = next((b.raw for b in content if getattr(b, "type", "") == "text" and getattr(b, "raw", None)), None)
            if extra:
                msg["extra_content"] = extra
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        texts = []
        for b in content:   # user turn: tool results become role=tool messages
            if b.get("type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b["content"])})
            elif b.get("type") == "text":
                texts.append(b["text"])
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in (tools or [])]
    return out, oa_tools


class OpenAICompatClient:
    def __init__(self, provider: str = "ollama", model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, rpm: float | None = None, cache_dir: str | None = ".llm_cache",
                 max_tokens: int = 2048, timeout: float = 300, provider_name: str | None = None,
                 max_usd: float | None = None, reasoning_effort: str | None = None):
        self.provider = provider
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self.reasoning_effort = reasoning_effort
        url, env, default_model = PRESETS.get(provider, (None, None, None))
        self.base_url = (base_url or url or "").rstrip("/")
        if not self.base_url:
            raise ConfigError(f"unknown provider {provider!r}: pass --base-url")
        # several keys (comma-separated in <ENV>S or <ENV>) are used round-robin, e.g. GEMINI_API_KEYS=k1,k2
        keys = api_key or (os.environ.get(env + "S") or os.environ.get(env) if env else None)
        self.api_keys = [k.strip() for k in keys.split(",") if k.strip()] if keys else []
        self._key_i = 0
        self.dead_keys: dict[str, str] = {}   # key -> why it is no longer used
        self.api_key = self.api_keys[0] if self.api_keys else None
        if env and not self.api_key:
            raise ConfigError(f"set {env} (or {env}S=key1,key2 for several keys) for provider {provider!r}")
        self.model = model or default_model
        if not self.model:
            raise ConfigError(f"provider {provider!r} needs --model (e.g. a ':free' model id on OpenRouter)")
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.cache = Path(cache_dir) / safe_name(self.model) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._last = 0.0
        self.calls = self.cache_hits = 0
        self.last_was_cached = False
        self.last_latency = 0.0     # pure request time of the last call (excludes throttling waits)
        self.extra = {}             # extra request fields, e.g. {"reasoning_effort": "low"}

    def complete(self, system: str, messages: list, tools: list | None = None, effort: str | None = None):
        """Returns (blocks, input_tokens, output_tokens, stop_reason) like AnthropicClient. `effort` overrides the
        client's reasoning effort for this call (OpenAI GPT-5 family only)."""
        msgs, oa_tools = _to_openai(system, messages, tools)
        if self.provider == "openai":   # GPT-5 family: no temperature; reasoning tokens count toward the cap
            body = {"model": self.model, "messages": msgs, "max_completion_tokens": max(self.max_tokens, 8192),
                    "reasoning_effort": effort or self.reasoning_effort or "minimal", **self.extra}
        else:
            body = {"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": self.max_tokens,
                    **self.extra}
        if oa_tools:
            body["tools"] = oa_tools
            body["tool_choice"] = "auto"
        raw = json.dumps(body, sort_keys=True)
        path = self.cache / (hashlib.sha256(raw.encode()).hexdigest() + ".json") if self.cache else None
        if path and path.exists():
            self.cache_hits += 1
            self.last_was_cached = True
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.last_was_cached = False
            data = self._post(raw)
            data["_latency"] = self.last_latency
            if path:
                data = self._store_first(path, data)
        self.last_latency = data.get("_latency", 0.0)
        self.calls += 1
        self._charge(data.get("usage") or {}, replayed=path is not None and self.last_was_cached)
        choice = data["choices"][0]
        msg = choice["message"]
        blocks = []
        if msg.get("content") or msg.get("extra_content"):
            blocks.append(Block("text", text=msg.get("content") or "", raw=msg.get("extra_content")))
        for i, tc in enumerate(msg.get("tool_calls") or []):
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tc = dict(tc, id=tc.get("id") or f"call_{self.calls}_{i}")
            blocks.append(Block("tool_use", name=tc["function"]["name"], input=args if isinstance(args, dict) else {},
                                id=tc["id"], raw=tc))
        usage = data.get("usage") or {}
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else choice.get("finish_reason", "stop")
        tout = usage.get("completion_tokens", 0)
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        if usage.get("total_tokens", 0) > usage.get("prompt_tokens", 0) + tout:   # hidden thinking billed separately
            tout = usage["total_tokens"] - usage.get("prompt_tokens", 0)
        elif reasoning and reasoning > tout:
            tout += reasoning
        return blocks, usage.get("prompt_tokens", 0), tout, stop

    def _charge(self, usage: dict, replayed: bool) -> None:
        """Track spend of NEW requests only (replays from the cache are free) and stop at the budget."""
        if replayed:
            return
        pin, pcache, pout = PRICES.get(self.model, (0.0, 0.0, 0.0))
        tin = usage.get("prompt_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        tout = max(usage.get("completion_tokens", 0), usage.get("total_tokens", 0) - tin)
        cost = ((tin - cached) * pin + cached * pcache + tout * pout) / 1e6
        self.spent_usd += cost
        if self.cache is None:
            total = self.spent_usd
        else:   # one append-only ledger shared by every worker process, so the guard is global
            ledger = self.cache / "_spend.log"
            with open(ledger, "a", encoding="utf-8") as f:
                f.write(f"{cost:.8f}\n")
            total = sum(float(x) for x in ledger.read_text(encoding="utf-8").split())
        if self.max_usd is not None and total > self.max_usd:
            raise BudgetExceeded(f"spending guard: ${total:.2f} spent on {self.model} exceeds the --max-usd limit "
                                 f"of ${self.max_usd:.2f}. Finished calls are cached; raise the limit to continue.")

    @staticmethod
    def _store_first(path: Path, data: dict) -> dict:
        """Write-once cache entry. If a parallel run stored the same request first, use *its* response,
        so every run's transcript can be replayed exactly from the cache."""
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            os.link(tmp, path)            # atomic and fails if the entry already exists
        except FileExistsError:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError:                   # file systems without hard links
            if not path.exists():
                os.replace(tmp, path)
                return data
            data = json.loads(path.read_text(encoding="utf-8"))
        finally:
            tmp.unlink(missing_ok=True)
        return data

    def _next_key(self) -> str | None:
        alive = [k for k in self.api_keys if k not in self.dead_keys]
        if not self.api_keys:
            return None
        if not alive:
            reasons = "; ".join(f"key {k[:10]}...: {why}" for k, why in self.dead_keys.items())
            if any("quota" in why for why in self.dead_keys.values()):
                raise QuotaExhausted(f"all API keys are out of free quota for {self.model}. {reasons}")
            raise AuthError(f"no usable API key for {self.model}. {reasons}")
        key = alive[self._key_i % len(alive)]
        self._key_i += 1
        return key

    def _post(self, raw: str) -> dict:
        """POST with key round-robin, per-key failover (revoked key / daily quota), backoff on transient errors."""
        delay = 2.0
        per_minute_waits = 0
        for attempt in range(12 + 60):
            headers = {"Content-Type": "application/json"}
            key = self._next_key()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=raw.encode(), headers=headers)
            try:
                t0 = time.monotonic()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                self.last_latency = time.monotonic() - t0
                return out
            except urllib.error.HTTPError as e:
                body = e.read()[:2000].decode("utf-8", "replace")
                if e.code in (401, 403) and key:
                    self.dead_keys[key] = f"rejected ({e.code})"   # revoked/invalid key: stop using it
                    continue
                if e.code == 429 and key and _is_daily_quota(body):
                    self.dead_keys[key] = "daily free quota used up" + _retry_hint(body)
                    continue
                if e.code == 429 and per_minute_waits < 60:   # per-minute limit: wait as told, then retry
                    per_minute_waits += 1
                    retry_after = e.headers.get("Retry-After")
                    wait_s = _retry_seconds(body) or (float(retry_after) if retry_after and retry_after.isdigit()
                                                      else delay)
                    time.sleep(min(wait_s + 1.0, 120))
                    delay = min(delay * 2, 120)
                    continue
                if e.code in (500, 502, 503, 504) and attempt < 11:
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise LLMError(f"LLM API error {e.code}: {body[:400]}") from e
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                    http.client.HTTPException) as e:
                if attempt < 11:
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise LLMError(f"cannot reach {self.base_url}: {e}. Is the server running / network up?") from e
        raise LLMError("LLM API: too many retries")


class LLMError(RuntimeError):
    """A failed LLM request (after retries)."""


class ConfigError(LLMError):
    """Bad provider/model/key configuration."""


class BudgetExceeded(LLMError):
    """The --max-usd spending guard tripped. Finished calls are cached; raise the limit to continue."""


class AuthError(LLMError):
    """Every configured key was rejected."""


class QuotaExhausted(LLMError):
    """Every configured key is out of free-tier quota; resume later (responses so far are cached)."""


def _is_daily_quota(body: str) -> bool:
    """Daily caps (wait hours) vs per-minute limits (wait seconds). Gemini names the quota explicitly."""
    b = body.lower()
    if "perday" in b or "per day" in b or re.search(r"retry in \d+h", b):
        return True
    if "perminute" in b or "per minute" in b:
        return False
    secs = _retry_seconds(body)
    return secs is not None and secs > 600


def _retry_seconds(body: str) -> float | None:
    m = re.search(r'"retryDelay":\s*"([\d.]+)s"', body) or re.search(r"retry in ([\d.]+)s", body)
    return float(m.group(1)) if m else None


def _retry_hint(body: str) -> str:
    m = re.search(r"retry in ([0-9hms. ]+)", body)
    return f" (retry in {m.group(1).strip()})" if m else ""


def safe_name(name: str) -> str:
    """File-system safe on Windows/macOS/Linux (model ids contain ':' and '/')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)
