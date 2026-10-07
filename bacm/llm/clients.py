"""LLM client for the real-agent backend.

NOTE: calls made here are billed to an Anthropic API key / Console credits,
NOT to a Claude Pro/Max subscription. The default backend ("sim") never calls this.
"""
from __future__ import annotations

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicClient:
    def __init__(self, model: str | None = None, effort: str | None = None, max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install anthropic  (needed only for --backend anthropic)") from e
        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list, tools: list | None = None):
        """One agent step. Returns (response.content, input_tokens, output_tokens, stop_reason)."""
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                      betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if tools:
            kwargs["tools"] = tools
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.beta.messages.create(**kwargs)
        u = resp.usage
        tin = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        return resp.content, tin, u.output_tokens, resp.stop_reason
