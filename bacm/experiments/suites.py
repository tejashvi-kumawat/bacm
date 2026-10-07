"""Experiment suites mirroring design-document sections 16-22.

Each suite yields (group_params, RunConfig) pairs. group_params are the
independent variables that identify a cell of the results table (seed excluded).
"""
from __future__ import annotations

from dataclasses import replace

from ..config import ABLATIONS, ARCHITECTURES, AgentConfig, DynamicsConfig, RunConfig, TaskConfig

BASELINES = ["independent", "fully_shared", "selective", "bacm"]          # design-document section 16
FAIR = ["independent_evict", "selective_fast"]   # baselines given the same context/interface optimisations
ALL_MAIN = [*BASELINES[:3], *FAIR, "selective_global_recompute", "bacm_v1", "bacm"]
DYNAMIC = DynamicsConfig(n_changes=3, n_cosmetic=2, trigger_progress=0.5)
CONFLICT_AGENT = AgentConfig(p_misread=0.15, p_doc_first=0.6, p_verify_doc=0.3)


def _cfg(arch: str, seed: int, **kw) -> RunConfig:
    a = ARCHITECTURES.get(arch) or ABLATIONS[arch]
    return RunConfig(arch=a, seed=seed, **kw)


def pilot(seeds):
    """Minimum viable experiment (section 29): baselines, static repo."""
    for s in seeds:
        for a in ALL_MAIN:
            yield {"arch": a}, _cfg(a, s)


def main(seeds):
    """Section 16: all baselines vs. proposed, static and dynamic environment."""
    for s in seeds:
        for env, dyn in (("static", DynamicsConfig()), ("dynamic", DYNAMIC)):
            for a in ALL_MAIN:
                yield {"arch": a, "env": env}, _cfg(a, s, dynamics=dyn)


def ablation(seeds):
    """Section 19: remove one component at a time (dynamic + conflict-prone setting)."""
    for s in seeds:
        for setting, kw in (("dynamic", {"dynamics": DYNAMIC}),
                            ("dynamic+conflict", {"dynamics": DYNAMIC, "agent": CONFLICT_AGENT})):
            for a in ["independent", "selective", *ABLATIONS]:
                yield {"arch": a, "setting": setting}, _cfg(a, s, **kw)


def dynamic(seeds):
    """Section 20: stale-knowledge experiments with increasing amounts of change."""
    for s in seeds:
        for n in (0, 1, 3, 5, 8):
            dyn = DynamicsConfig(n_changes=n, n_cosmetic=2 if n else 0)
            for a in [*ALL_MAIN, "bacm-no_worker", "bacm-no_dependencies", "bacm-no_invalidation",
                      "bacm-file_level"]:
                yield {"arch": a, "n_changes": n}, _cfg(a, s, dynamics=dyn)


def conflict(seeds):
    """Section 21: contradictory claims of different evidence strength."""
    for s in seeds:
        for p in (0.05, 0.15, 0.3):
            ag = replace(CONFLICT_AGENT, p_misread=p)
            for a in [*ALL_MAIN, "bacm-no_conflict", "bacm-no_admission", "bacm-no_cross_check",
                      "bacm-no_weak_verify"]:
                yield {"arch": a, "p_misread": p}, _cfg(a, s, agent=ag)


def scaling(seeds):
    """Section 22: N = 2, 4, 8, 16 agents."""
    for s in seeds:
        for n in (2, 4, 8, 16):
            for a in [*BASELINES, *FAIR]:
                yield {"arch": a, "n_agents": n}, _cfg(a, s, task=TaskConfig(n_agents=n))


def overlap(seeds):
    """How much do the branches' information needs overlap? (when is sharing worth it)"""
    for s in seeds:
        for ov in (0.0, 0.25, 0.5, 0.75, 1.0):
            for a in [*BASELINES, *FAIR]:
                yield {"arch": a, "overlap": ov}, _cfg(a, s, task=TaskConfig(overlap=ov))


def llm_pilot(seeds):
    """Small real-LLM validation (free local/hosted models): 3 agents x 3 questions, static + dynamic."""
    task = TaskConfig(n_agents=3, questions_per_agent=3)
    for s in seeds:
        for env, dyn in (("static", DynamicsConfig()), ("dynamic", DynamicsConfig(n_changes=2, n_cosmetic=1))):
            for a in ["independent", "independent_evict", "fully_shared", "selective", "selective_fast",
                      "bacm_v1", "bacm", "bacm_validate"]:
                yield {"arch": a, "env": env}, _cfg(a, s, dynamics=dyn, task=task)


REPO_ARCHS = ["independent", "independent_evict", "selective_fast", "bacm", "bacm_validate",
               "selective_auto", "bacm_auto", "selective_auto-llm_extraction", "bacm_auto-llm_extraction"]


def repobench(seeds, repos=None):
    """RepoQA-Dyn on real open-source repositories (LLM backends only). Each seed = a different set of
    questions and agent split; static and dynamic share the same questions per seed (paired).
    dynamic = the repository moves to a later real release mid-run (httpx/requests/flask change answers;
    click/rich change many files but no answers: tests that irrelevant changes cost little)."""
    from ..config import RepoConfig
    from ..env.realrepo import REPOS
    task = TaskConfig(n_agents=3, questions_per_agent=4)
    for s in seeds:
        for name, (url, a, b) in REPOS.items():
            if repos and name not in repos:
                continue
            repo = RepoConfig(name, url, a, b)
            for env, dyn in (("static", DynamicsConfig()), ("dynamic", DynamicsConfig(n_changes=1))):
                for arch in REPO_ARCHS:
                    yield {"arch": arch, "repo": name, "env": env}, _cfg(arch, s, task=task, dynamics=dyn, repo=repo)


REPO2_ARCHS = ["independent", "independent_evict", "selective_fast", "selective_llm", "bacm_llm", "bacm_auto",
               "bacm_llm-no_digest", "bacm_llm-no_resolve", "bacm_llm-no_hints", "bacm_llm-no_maintain",
               "bacm_llm-no_reanswer", "bacm_llm-no_invalidation", "bacm_llm-no_dependencies",
               "bacm_llm-minimal_manager"]


def repobench2(seeds, repos=None):
    """RepoQA-Dyn v2: five real repositories, five question kinds (hierarchy + method counting), 3 agents x 6
    questions, static and dynamic (a real release jump halfway). Facts are extracted by the model; the symbol-index
    variant (`bacm_auto`) is only an upper bound."""
    from ..config import RepoConfig
    from ..env.realrepo import REPOS
    task = TaskConfig(n_agents=3, questions_per_agent=6)
    for s in seeds:
        for name, (url, a, b) in REPOS.items():
            if repos and name not in repos:
                continue
            repo = RepoConfig(name, url, a, b, bench="v2")
            for env, dyn in (("static", DynamicsConfig()), ("dynamic", DynamicsConfig(n_changes=1))):
                for arch in REPO2_ARCHS:
                    yield {"arch": arch, "repo": name, "env": env}, _cfg(arch, s, task=task, dynamics=dyn, repo=repo)


SUITES = {f.__name__: f for f in (pilot, main, ablation, dynamic, conflict, scaling, overlap, llm_pilot, repobench,
                                  repobench2)}
LLM_SUITES = ("llm_pilot", "repobench", "repobench2")
SIM_SUITES = [n for n in SUITES if n not in LLM_SUITES]
