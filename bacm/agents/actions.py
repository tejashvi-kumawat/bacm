"""Actions an agent yields to the orchestrator. The orchestrator executes them,
accounts for tokens/latency/cost, and sends the result back into the agent generator."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.knowledge import Claim


@dataclass
class LLMStep:
    """A simulated LLM call: input = the agent's current context, output = out_tokens."""
    purpose: str
    out_tokens: int


@dataclass
class LLMCall:
    """A real LLM call (LLM backends). Returns (text, usage)."""
    system: str
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    purpose: str = "step"


@dataclass
class Tool:
    op: str
    args: dict


@dataclass
class KBQuery:
    key: str
    in_context: bool = False   # True for fully shared context: lookup is free (already in prompt)


@dataclass
class Need:
    """Bookkeeping only (free): the agent needs a fact it does not hold locally."""
    key: str


@dataclass
class Publish:
    """Offer a claim to the shared layer (ignored by independent agents; free in transcript mode).
    piggyback=True: emitted in the same LLM turn as the next action (costs only its output tokens)."""
    claim: Claim
    piggyback: bool = False


@dataclass
class Note:
    """Real-LLM harvest: a small fresh-context call lists every fact in a file just read.
    Returns [(key, value, span)]. Shared memories cache notes per file@version."""
    path: str
    version: int
    text: str


@dataclass
class Prefetch:
    """Question start: the orchestrator injects relevant valid shared knowledge (no LLM turn)."""
    keys: list


@dataclass
class UseShared:
    """Bookkeeping (free): the agent used a prefetched shared fact for a need."""
    key: str
    hit: object


@dataclass
class Answer:
    key: str
    value: str
