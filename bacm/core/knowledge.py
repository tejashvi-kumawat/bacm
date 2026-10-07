"""Knowledge objects: K_i = (content, source, version, branch, dependencies, confidence, status)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    CANDIDATE = "CANDIDATE"   # L1: stored, not globally visible
    VERIFIED = "VERIFIED"     # L2: shared, queryable
    STALE = "STALE"           # invalidated by a source change (directly or via dependencies)
    CONFLICT = "CONFLICT"     # contradicts another claim, awaiting resolution
    REJECTED = "REJECTED"     # lost a conflict
    SUPERSEDED = "SUPERSEDED"  # overwritten (last-write-wins, used when conflict handling is off)


class Level(str, Enum):
    L0 = "L0-private"
    L1 = "L1-candidate"
    L2 = "L2-shared"
    L3 = "L3-critical"


# How strongly each kind of evidence supports a claim (V in the admission score).
EVIDENCE_STRENGTH = {"code": 1.0, "derived": 0.9, "memory": 0.8, "doc": 0.4, "hypothesis": 0.0}


@dataclass(frozen=True)
class Source:
    path: str
    version: int | None
    span: str | None = None   # the exact supporting line (span-level provenance)


@dataclass
class Claim:
    """What an agent offers to the shared layer."""

    key: str
    value: str
    confidence: float
    evidence: str                       # code | doc | derived | hypothesis | memory
    sources: list[Source] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)   # parent fact keys
    dep_values: dict[str, str] | None = None       # parent values the derivation actually used
    level: Level = Level.L1

    def text(self) -> str:
        return f"{self.key} = {self.value}"


@dataclass
class Knowledge:
    id: str
    key: str
    value: str
    evidence: str
    confidence: float
    produced_by: str | None          # agent id (None without provenance)
    branch: str | None
    sources: list[Source]
    dep_ids: list[str]
    created_at: float
    status: Status = Status.VERIFIED
    consumers: set[str] = field(default_factory=set)
    n_reuses: int = 0
    revalidations: int = 0
    corroborated: bool = False
    support: set[str] = field(default_factory=set)   # independent observers of this value

    @property
    def verification(self) -> float:
        return EVIDENCE_STRENGTH.get(self.evidence, 0.5)

    def render(self) -> str:
        """Compact text injected into an agent's context when the knowledge is reused."""
        s = f"[{self.id}] {self.key} = {self.value} (conf {self.confidence:.2f}, {self.status.value}"
        if self.sources:
            s += "; src " + ", ".join(f"{x.path}@v{x.version}" for x in self.sources)
        if self.dep_ids:
            s += "; deps " + ",".join(self.dep_ids)
        return s + ")"
