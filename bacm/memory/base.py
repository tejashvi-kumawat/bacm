"""Shared-memory interface + the three baseline architectures."""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

from ..config import ArchConfig
from ..core.knowledge import Claim, Level
from ..util import count_tokens


@dataclass
class Hit:
    key: str
    value: str
    evidence: str
    confidence: float
    text: str
    kid: str | None = None

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


class Memory:
    """Interface between agents and whatever is shared between branches."""

    supports_query = False   # can an agent look a fact up instead of re-deriving it?
    query_is_tool = True     # does a lookup cost a tool call (vs. already being in context)?

    def __init__(self, arch: ArchConfig, world, tracker, toolbox=None):
        self.arch = arch
        self.world = world
        self.tracker = tracker
        self.toolbox = toolbox
        # L3 events: (deliver_at, seq, agent, key, new_value or None for "invalidated, recompute")
        self._pending: list[tuple[float, int, str, str, str | None]] = []
        self._seq = 0
        self._args: dict[str, set[str]] = defaultdict(set)   # key argument -> keys (associative index)

    def configure(self, agent_cfg, seed: int, llm=None) -> None:
        """Receive run-level settings (used by the context manager's worker)."""

    # ------------------------------------------------------------- notifications
    def notify(self, agent_id: str, key: str, value: str | None, at: float) -> None:
        heapq.heappush(self._pending, (at, self._seq, agent_id, key, value))
        self._seq += 1

    def drain_notifications(self, now: float) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = defaultdict(dict)
        while self._pending and self._pending[0][0] <= now:
            _, _, agent, key, value = heapq.heappop(self._pending)
            out[agent][key] = value
        return out

    def next_notification_time(self) -> float | None:
        return self._pending[0][0] if self._pending else None

    # ----------------------------------------------------------------- lookups
    def query(self, agent_id: str, key: str, now: float) -> Hit | None:
        return None

    def _index(self, key: str) -> None:
        if "(" in key:
            self._args[key[key.index("(") + 1:-1]].add(key)

    def _lookup(self, agent_id: str, key: str, now: float) -> Hit | None:
        """A lookup used by prefetch (no demand accounting)."""
        return self.query(agent_id, key, now)

    SYSTEM = "context-manager"

    def known_value(self, key: str, now: float) -> str | None:
        hit = self._lookup(self.SYSTEM, key, now)
        return hit.value if hit else None

    def resolve_known(self, key: str, now: float, depth: int = 0) -> Hit | None:
        """Answer `key` from valid shared facts alone, with no LLM and no environment access: a stored
        fact, or a derived fact computed from its parents by the world's typed rules (and stored, with its
        dependencies, so that later changes reach it). None if some needed base fact is not known yet."""
        hit = self._lookup(self.SYSTEM, key, now)
        if hit is not None or depth > 6 or self.world.is_base_key(key) or not self.world.valid_key(key):
            return hit
        pkeys = self.world.parents(key, lambda k: self.known_value(k, now))
        if not pkeys or any("(None)" in p for p in pkeys):
            return None
        parents = []
        for p in pkeys:
            h = self.resolve_known(p, now, depth + 1)
            if h is None:
                return None
            parents.append(h)
        try:
            value = self.world.combine(key, {h.key: h.value for h in parents})
        except (KeyError, ValueError, StopIteration):
            return None
        claim = Claim(key, value, 0.95 * min(h.confidence for h in parents), "derived",
                      deps=[h.key for h in parents], dep_values={h.key: h.value for h in parents},
                      level=Level.L2)
        self.publish(self.SYSTEM, claim, now)
        return self._lookup(self.SYSTEM, key, now)

    def missing_facts(self, key: str, now: float, depth: int = 0) -> list[str]:
        """Base facts still needed to resolve `key` (a deterministic query plan)."""
        if self._lookup(self.SYSTEM, key, now) is not None:
            return []
        if self.world.is_base_key(key) or depth > 6:
            return [key]
        pkeys = self.world.parents(key, lambda k: self.known_value(k, now))
        out: list[str] = []
        for p in pkeys:
            if "(None)" in p:
                continue
            out += self.missing_facts(p, now, depth + 1)
        return out or [key]

    def subscribe(self, agent_id: str, key: str, now: float) -> None:
        """Register an agent's answer to `key` so changes to its supporting facts reach it."""

    def prefetch(self, agent_id: str, keys: list[str], now: float, max_items: int = 24) -> list[Hit]:
        """Associative retrieval for a question: the requested keys, facts about the same
        argument, and (multi-hop) facts about the values found, e.g.
        endpoint_datastore(/login) -> route(/login)=auth -> datastore(auth)."""
        out: dict[str, Hit] = {}
        seen: set[str] = set()
        frontier = list(keys)
        for _ in range(3):
            nxt: list[str] = []
            for key in frontier:
                if key in seen:
                    continue
                seen.add(key)
                hit = self._lookup(agent_id, key, now)
                if hit is not None:
                    out[key] = hit
                    nxt += sorted(self._args.get(hit.value, ()))
                if "(" in key:
                    nxt += sorted(self._args.get(key[key.index("(") + 1:-1], ()))
            frontier = nxt
        return list(out.values())[:max_items]

    def publish(self, agent_id: str, claim: Claim, now: float) -> str | None:
        return None

    def observe(self, agent_id: str, text: str, tokens: int, claim: Claim | None = None) -> None:
        """Called for every observation / thought an agent produces (used by fully shared context)."""

    def context_tokens(self, agent_id: str) -> int:
        """Extra tokens injected into this agent's prompt on every LLM call."""
        return 0

    def on_change(self, change, now: float) -> None:
        pass

    def reset(self) -> None:
        """Forget everything shared (used by the global-recomputation baseline)."""


class NullMemory(Memory):
    """Baseline 1: independent agents, A_i(C_0). Nothing is shared."""


class TranscriptMemory(Memory):
    """Baseline 2: fully shared context A_i(C). Every agent sees every other agent's
    observations, tool outputs and thoughts (including unverified hypotheses)."""

    supports_query = True
    query_is_tool = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.total_tokens = 0
        self.tokens_by_agent: dict[str, int] = defaultdict(int)
        self.log: list[tuple[str, str]] = []
        self.claims: dict[str, list[tuple[str, Claim]]] = defaultdict(list)

    def observe(self, agent_id, text, tokens, claim=None):
        self.total_tokens += tokens
        self.tokens_by_agent[agent_id] += tokens
        self.log.append((agent_id, text))
        if claim is not None:
            self.claims[claim.key].append((agent_id, claim))

    def publish(self, agent_id, claim, now):
        # claims are already in the transcript via observe(); nothing else to do
        return None

    def context_tokens(self, agent_id):
        return self.total_tokens - self.tokens_by_agent[agent_id]

    def shared_prompt(self, agent_id) -> str:
        """Text form of the shared transcript (used by real-LLM agents)."""
        lines = [f"[{a}] {t}" for a, t in self.log if a != agent_id]
        return "\n\n# Shared log of other agents\n" + "\n".join(lines) if lines else ""

    def query(self, agent_id, key, now):
        others = [(a, c) for a, c in self.claims.get(key, []) if a != agent_id]
        if not others:
            return None
        # agents tend to trust what was said last in the shared conversation
        a, c = others[-1]
        return Hit(key, c.value, c.evidence, c.confidence, f"{a} said: {c.text()}")


class SelectiveMemory(Memory):
    """Baseline 3: selective / gated shared memory bank (cf. Learning-to-Share, gated memory).
    Stores key -> value for claims above a confidence gate. No provenance, versions,
    dependencies or invalidation; last write wins."""

    supports_query = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bank: dict[str, tuple[str, Claim]] = {}

    def reset(self):
        self.bank.clear()

    def publish(self, agent_id, claim, now):
        if claim.level == Level.L0 or claim.confidence < self.arch.selective_min_confidence:
            self.tracker.on_admission(agent_id, claim, False)
            return None
        self.tracker.on_admission(agent_id, claim, True)
        self.bank[claim.key] = (agent_id, claim)
        self._index(claim.key)
        return claim.key

    def query(self, agent_id, key, now):
        if key not in self.bank:
            return None
        _, c = self.bank[key]
        return Hit(key, c.value, c.evidence, c.confidence, f"memory: {c.text()}", kid=key)
