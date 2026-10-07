"""The context manager's worker: a small, *fresh-context* LLM process.

Re-reading one file or re-applying one derivation rule needs only that file or
those parent facts in context (a few hundred to a few thousand tokens), whereas
waking an agent makes it redo the same work inside its whole accumulated context.
In the simulator the worker's extraction is subject to the same misread rate as agents.
"""
from __future__ import annotations

import random

from .knowledge import Source

SYSTEM_AGENT = "context-manager"
WORKER_PROMPT_TOKENS = 350
WORKER_OUT_TOKENS = 40


class Worker:
    def __init__(self, world, toolbox, tracker, cost, agent_cfg, seed: int, llm=None, indexed: bool = False):
        self.world = world
        self.toolbox = toolbox
        self.tracker = tracker
        self.cost = cost
        self.cfg = agent_cfg
        self.rng = random.Random(seed * 31 + 17)
        self.llm = llm   # real-LLM experiments: the worker is a real (small-context) model call too
        self.indexed = indexed   # deterministic symbol-index extraction/derivation: no LLM call, no tokens
        self.calls = 0

    def _ask(self, prompt: str, now: float, purpose: str) -> tuple[str, float]:
        import time
        t = time.perf_counter()
        eff = getattr(self, "effort", None)
        kw = {"effort": eff} if eff and hasattr(self.llm, "reasoning_effort") else {}
        blocks, tin, tout, _ = self.llm.complete(
            "You extract facts precisely. Reply with only the requested value, nothing else.",
            [{"role": "user", "content": prompt}], None, **kw)
        self.calls += 1
        self.tracker.on_llm(SYSTEM_AGENT, tin, tout, now, purpose)
        text = " ".join(getattr(b, "text", "") or "" for b in blocks).strip()
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        value = self.world.normalize(first.split("=")[-1] if "=" in first else first)   # keep case; sorted lists
        return value, getattr(self.llm, "last_latency", None) or time.perf_counter() - t

    def _llm(self, tin: int, now: float, purpose: str) -> float:
        self.calls += 1
        self.tracker.on_llm(SYSTEM_AGENT, tin, WORKER_OUT_TOKENS, now, purpose)
        c = self.cost
        return c.llm_base_latency + c.llm_latency_per_in_tok * tin + c.llm_latency_per_out_tok * WORKER_OUT_TOKENS

    def extract(self, key: str, now: float, hint: str | None = None):
        """Re-read a base fact from source. Provenance gives the file directly when known.
        Returns (value, Source, latency) or None."""
        paths = [hint] if hint else []
        paths += [p for p in self.world.locate(key) if p not in paths]
        lat = 0.0
        for path in paths:
            res = self.toolbox.call(SYSTEM_AGENT, "read_file", {"path": path}, now)
            lat += self.cost.tool_latency["read_file"]
            if self.llm is None and not self.indexed:
                lat += self._llm(WORKER_PROMPT_TOKENS + res.tokens, now, "worker:extract")
            parsed = self.world.parse(key, path, res.text) if res.data else None
            if parsed is None:
                continue
            value, line = parsed   # line = provenance span (located by the tool layer)
            if self.llm is not None:
                value, dt = self._ask(self.world.extract_prompt(key, path, res.text), now, "worker:extract")
                return value.rstrip(".;"), Source(path, res.data["version"], line), lat + dt
            if not self.indexed and self.rng.random() < self.cfg.p_misread:
                value = self.world.wrong_value(key, value, self.rng)
            return value, Source(path, res.data["version"], line), lat
        return None

    def derive(self, key: str, parent_values: dict[str, str], now: float):
        if self.indexed or (self.llm is not None and getattr(self.world, "bench", None) == "v2"):
            try:   # derived facts are typed rules over their parents: apply the rule, no model call
                return self.world.combine(key, parent_values), 0.001
            except (KeyError, ValueError, StopIteration):
                if self.indexed:
                    raise
        if self.llm is not None:
            value, dt = self._ask(self.world.derive_prompt(key, parent_values), now, "worker:derive")
            return value, dt
        tin = WORKER_PROMPT_TOKENS + 25 * len(parent_values)
        return self.world.combine(key, parent_values), self._llm(tin, now, "worker:derive")
