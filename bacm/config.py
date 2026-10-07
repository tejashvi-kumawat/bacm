"""Configuration objects for every experiment knob.

All configs are frozen dataclasses so a run is fully described (and reproducible)
by its RunConfig + seed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class ArchConfig:
    """Which context architecture the agents run under.

    memory:
      "none"       -> Baseline 1, independent agents (no sharing)
      "transcript" -> Baseline 2, fully shared context (every observation broadcast)
      "selective"  -> Baseline 3, selective/gated shared memory (key -> value, no provenance)
      "bacm"       -> Proposed branch-aware incremental context manager

    Knowledge-layer switches (only meaningful for "bacm"; the ablations of design-document section 19):
    S, P, V, D, I, C + span provenance + the manager's small-context worker.
    Interface / context-efficiency switches (usable by any memory that supports them).
    """

    name: str
    memory: str
    # ---- knowledge layer (bacm) ----
    admission: bool = True        # S: selective sharing / knowledge admission
    provenance: bool = True       # P: record source + producing branch
    versioning: bool = True       # V: record source versions, detect changes
    dependencies: bool = True     # D: dependency graph between knowledge objects
    invalidation: bool = True     # I: eager invalidation + L3 notifications
    conflict: bool = True         # C: conflict detection / resolution
    span_provenance: bool = True  # finer-grained provenance: revalidate if the supporting line is unchanged
    worker_recompute: bool = False  # W: manager re-derives the affected subgraph in a small fresh context
                                    #    and pushes corrected values (instead of waking agents to redo it)
    demand_driven: bool = False   # recompute / notify only facts some branch's questions depend on
                                  # (the *demanded* part of the affected subgraph); the rest stays STALE
    weak_verify: bool = False     # rejected weak-evidence claims are verified against code by the worker
    validate_answers: bool = False  # Validate->Share: a derived answer is re-derived from shared verified
                                    # premises in a small context; disagreement -> correction to the branch,
                                    # and the answer joins the dependency graph (so later changes reach it)
    cross_check: bool = False     # a base fact is independently re-read by the worker the first time
                                  # another branch reuses it (stops one misread spreading to every branch)
    # ---- interface / context efficiency ----
    prefetch: bool = False        # inject valid knowledge relevant to a question when the agent starts it
    parallel_lookup: bool = False  # kb_query issued in the same LLM turn as the search it would replace
    piggyback_publish: bool = False  # kb_publish emitted alongside the next action (no extra LLM turn)
    harvest: bool = False         # on reading a code file, extract every fact in it, not just the one needed
    evict: bool = False           # after extraction, drop raw file text from context, keep compact facts
    # ---- transparent memory (LLM backends): the tool layer, not the model, talks to the memory ----
    transparent: bool = False       # no kb_* tools: reads are harvested + published automatically
    serve_digest: bool = True       # a file already read by another branch is served as a compact digest
    auto_resolve: bool = True       # questions whose facts are known are answered by the manager (no LLM turn)
    status_hints: bool = True       # tool results list which facts a question still needs
    maintain_answers: bool = True   # answers are updated in place on source changes (no agent re-prompt)
    llm_extraction: bool = False    # facts from files via an LLM note call instead of the symbol index
    manager_effort: str | None = "low"   # reasoning effort of the manager's small-context calls (extraction,
                                         # re-derivation); agents keep the client's default (cheapest) effort
    reanswer_small: bool = True     # answers the manager cannot recompute are re-derived in a small fresh LLM
                                    # context from the current facts, instead of waking the agent
    # ---- baseline option ----
    global_recompute: bool = False  # after the team finishes, if anything changed, redo everything
    admission_threshold: float = 0.6
    selective_min_confidence: float = 0.6  # gate used by the "selective" baseline

    def __post_init__(self):
        if self.validate_answers and not (self.worker_recompute or self.weak_verify or self.cross_check):
            raise ValueError("validate_answers needs the manager's worker (enable worker_recompute)")
        if self.evict and not self.harvest:
            raise ValueError("evict requires harvest (facts must be kept when raw text is dropped)")


_FAST = dict(prefetch=True, parallel_lookup=True, piggyback_publish=True, harvest=True, evict=True)

ARCHITECTURES: dict[str, ArchConfig] = {
    "independent": ArchConfig("independent", "none"),
    # independent agents with the same local context hygiene (harvest + evict), no sharing
    "independent_evict": ArchConfig("independent_evict", "none", harvest=True, evict=True),
    "fully_shared": ArchConfig("fully_shared", "transcript"),
    "selective": ArchConfig("selective", "selective"),
    # selective memory given every interface/context optimisation the proposed system uses
    "selective_fast": ArchConfig("selective_fast", "selective", **_FAST),
    # strong dynamic baseline: knows *that* the repo changed, not *what* -> global recomputation
    "selective_global_recompute": ArchConfig("selective_global_recompute", "selective", global_recompute=True),
    # first version of the proposed system (knowledge layer only)
    "bacm_v1": ArchConfig("bacm_v1", "bacm"),
    # proposed system: knowledge layer + small-context worker + efficient interface
    "bacm": ArchConfig("bacm", "bacm", worker_recompute=True, weak_verify=True, cross_check=True,
                       demand_driven=True, **_FAST),
    # transparent-memory systems (LLM backends only)
    "selective_auto": ArchConfig("selective_auto", "selective", transparent=True, harvest=True, evict=True,
                                 maintain_answers=False),
    "bacm_auto": ArchConfig("bacm_auto", "bacm", transparent=True, harvest=True, evict=True,
                            worker_recompute=True, demand_driven=True),
    # proposed system + answer validation (aimed at real LLMs, whose own derivations can be wrong)
    "bacm_validate": ArchConfig("bacm_validate", "bacm", worker_recompute=True, weak_verify=True,
                                cross_check=True, validate_answers=True, demand_driven=True, **_FAST),
}

_B = ARCHITECTURES["bacm"]
_BA = ARCHITECTURES["bacm_auto"]


def _abl(suffix: str, **off) -> tuple[str, ArchConfig]:
    name = f"bacm-{suffix}"
    return name, replace(_B, name=name, **off)


# Full system minus one component at a time (design-document section 19).
ABLATIONS: dict[str, ArchConfig] = dict([
    ("bacm", _B),
    _abl("no_admission", admission=False),
    _abl("no_provenance", provenance=False),
    _abl("no_versioning", versioning=False),
    _abl("no_dependencies", dependencies=False),
    _abl("no_invalidation", invalidation=False),
    _abl("no_conflict", conflict=False),
    _abl("file_level", span_provenance=False),
    _abl("no_worker", worker_recompute=False),
    _abl("no_weak_verify", weak_verify=False),
    _abl("no_demand", demand_driven=False),
    _abl("no_cross_check", cross_check=False),
    _abl("no_prefetch", prefetch=False),
    _abl("no_parallel", parallel_lookup=False),
    _abl("no_piggyback", piggyback_publish=False),
    _abl("no_evict", evict=False),
    _abl("no_harvest", harvest=False, evict=False),
])
_SA = ARCHITECTURES["selective_auto"]
ABLATIONS["selective_auto-llm_extraction"] = replace(_SA, name="selective_auto-llm_extraction", llm_extraction=True)
# paper names: the realistic configuration (the model extracts facts; no symbol-index oracle) and its ablations
_BL = replace(_BA, name="bacm_llm", llm_extraction=True)
ABLATIONS["bacm_llm"] = _BL
ABLATIONS["selective_llm"] = replace(_SA, name="selective_llm", llm_extraction=True)
for _n, _off in {"no_digest": {"serve_digest": False}, "no_resolve": {"auto_resolve": False},
                 "no_hints": {"status_hints": False}, "no_maintain": {"maintain_answers": False},
                 "no_reanswer": {"reanswer_small": False}, "no_invalidation": {"invalidation": False},
                 "no_dependencies": {"dependencies": False}, "no_versioning": {"versioning": False},
                 "minimal_manager": {"manager_effort": None}}.items():
    ABLATIONS[f"bacm_llm-{_n}"] = replace(_BL, name=f"bacm_llm-{_n}", **_off)
ABLATIONS.update({
    n: replace(_BA, name=n, **off) for n, off in {
        "bacm_auto": {},
        "bacm_auto-no_digest": {"serve_digest": False},
        "bacm_auto-no_resolve": {"auto_resolve": False},
        "bacm_auto-no_hints": {"status_hints": False},
        "bacm_auto-no_maintain": {"maintain_answers": False},
        "bacm_auto-llm_extraction": {"llm_extraction": True},
        "bacm_auto-no_reanswer": {"reanswer_small": False},
        "bacm_auto-no_invalidation": {"invalidation": False},
        "bacm_auto-no_dependencies": {"dependencies": False},
        "bacm_auto-no_provenance": {"provenance": False},
        "bacm_auto-no_versioning": {"versioning": False},
    }.items()
})


def get_arch(name: str) -> ArchConfig:
    if name in ARCHITECTURES:
        return ARCHITECTURES[name]
    if name in ABLATIONS:
        return ABLATIONS[name]
    raise KeyError(f"unknown architecture {name!r}; choose from {sorted({**ARCHITECTURES, **ABLATIONS})}")


@dataclass(frozen=True)
class WorldConfig:
    """Synthetic repository that agents explore."""

    n_services: int = 8
    n_route_files: int = 3
    endpoints_per_file: int = 5
    n_features: int = 6
    endpoints_per_feature: int = 3
    filler_lines: tuple[int, int] = (60, 260)  # min/max filler lines per code file
    doc_error_rate: float = 0.35  # fraction of facts the (outdated) docs report wrongly


@dataclass(frozen=True)
class TaskConfig:
    n_agents: int = 4
    questions_per_agent: int = 5
    # 0 -> agents get (mostly) disjoint questions; 1 -> all agents draw from the same small pool.
    overlap: float = 0.5
    # mix of question types
    p_feature: float = 0.3
    p_endpoint_protocol: float = 0.3  # remainder -> endpoint_datastore


@dataclass(frozen=True)
class DynamicsConfig:
    """Source changes injected mid-run (design-document section 20)."""

    n_changes: int = 0          # fact-changing edits
    n_cosmetic: int = 0         # edits that touch a supporting file but not the fact
    trigger_progress: float = 0.5  # apply once this fraction of all questions has been answered


@dataclass(frozen=True)
class AgentConfig:
    """Behavioural parameters of the simulated agent policy."""

    p_doc_first: float = 0.35      # opens the (possibly outdated) docs before the code
    p_verify_doc: float = 0.5      # after reading docs, also reads the code to confirm
    p_misread: float = 0.03        # extracts a wrong value from code with high confidence
    p_hypothesis: float = 0.3      # emits a speculative guess before reading
    p_adopt_unverified: float = 0.7  # in fully shared context, adopts others' unverified claims
    p_hypothesis_correct: float = 0.35
    system_prompt_tokens: int = 700
    decide_out_tokens: int = 60
    extract_out_tokens: int = 120
    derive_out_tokens: int = 90
    answer_out_tokens: int = 80


@dataclass(frozen=True)
class CostModel:
    """Token prices (USD per million) and a latency model for simulated time."""

    price_in_per_mtok: float = 3.0
    price_out_per_mtok: float = 15.0
    llm_base_latency: float = 0.6
    llm_latency_per_in_tok: float = 0.00004
    llm_latency_per_out_tok: float = 0.015
    tool_latency: dict = field(default_factory=lambda: {
        "list_files": 0.02, "search": 0.15, "read_file": 0.05,
        "kb_query": 0.02, "kb_publish": 0.02, "answer": 0.0,
    })


@dataclass(frozen=True)
class RepoConfig:
    """A real open-source repository as the environment (instead of the synthetic one).
    rev_b: a later revision applied mid-run in dynamic experiments (None = static)."""

    name: str
    url: str
    rev_a: str
    rev_b: str | None = None
    cache_dir: str = ".repos"
    bench: str = "v1"      # "v1": hierarchy facts only; "v2": adds method-counting facts and questions


@dataclass(frozen=True)
class RunConfig:
    arch: ArchConfig
    seed: int = 0
    world: WorldConfig = WorldConfig()
    task: TaskConfig = TaskConfig()
    dynamics: DynamicsConfig = DynamicsConfig()
    agent: AgentConfig = AgentConfig()
    cost: CostModel = CostModel()
    repo: RepoConfig | None = None   # real-repository environment (LLM backends only)
    backend: str = "sim"         # "sim" | "anthropic" | "openai_compat"
    model: str | None = None     # model id for LLM backends
    max_llm_steps_per_agent: int = 60

    def with_(self, **kw) -> "RunConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)
