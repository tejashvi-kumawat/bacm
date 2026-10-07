"""Team tasks: each agent (branch) gets a subtask = a list of questions about the repo."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..config import DynamicsConfig, TaskConfig


@dataclass(frozen=True)
class Question:
    key: str          # canonical fact key, e.g. endpoint_datastore(/login)
    text: str         # natural-language form (used by LLM agents)


def question_text(key: str) -> str:
    arg = key[key.index("(") + 1:-1]
    if key.startswith("endpoint_datastore("):
        return f"Which datastore ultimately persists data for requests to the {arg} endpoint?"
    if key.startswith("endpoint_protocol("):
        return f"Which wire protocol does the service that owns the {arg} endpoint expose?"
    if key.startswith("feature_datastores("):
        return f"Which datastores (comma-separated, sorted) does the '{arg}' feature touch across its endpoints?"
    return f"What is {key}?"


def synthetic_question_types(world, cfg: TaskConfig) -> list[tuple[float, list[str]]]:
    return [
        (cfg.p_feature, [f"feature_datastores({f})" for f in world.features]),
        (cfg.p_endpoint_protocol, [f"endpoint_protocol({e})" for e in world.endpoints]),
        (1 - cfg.p_feature - cfg.p_endpoint_protocol, [f"endpoint_datastore({e})" for e in world.endpoints]),
    ]


def make_task(world, cfg: TaskConfig, seed: int) -> list[list[Question]]:
    """Each agent draws its questions from a shared pool; overlap controls the pool size."""
    r = random.Random(seed * 7919 + 13)
    types = [(w, list(c)) for w, c in world.question_types(cfg) if c and w > 0]   # keep world order (reproducibility)
    candidates = sorted({k for _, c in types for k in c})
    total = cfg.questions_per_agent * cfg.n_agents
    pool_size = max(cfg.questions_per_agent, math.ceil(total * (1.0 - cfg.overlap)))
    pool: list[str] = []
    attempts = 0
    while len(pool) < min(pool_size, len(candidates)) and attempts < 10_000:
        attempts += 1
        _, keys = r.choices(types, weights=[w for w, _ in types])[0]
        c = r.choice(keys)
        if c not in pool:
            pool.append(c)
    subtasks = []
    for _ in range(cfg.n_agents):
        keys = r.sample(pool, min(cfg.questions_per_agent, len(pool)))
        subtasks.append([Question(k, world.question_text(k)) for k in keys])
    return subtasks


def plan_changes(world, subtasks: list[list[Question]], dyn: DynamicsConfig, seed: int) -> list[tuple[str, str]]:
    """Choose which base facts (in the questions' dependency closure) will change mid-run (synthetic world).

    Returns a list of ("fact", key) / ("cosmetic", path) actions.
    """
    r = random.Random(seed * 104729 + 7)
    closure: set[str] = set()
    for qs in subtasks:
        for q in qs:
            closure |= world.dependency_closure(q.key)
    base = sorted(k for k in closure if world.is_base(k))
    # prefer facts many questions depend on: weight by how many questions include them
    weight = {k: sum(k in world.dependency_closure(q.key) for qs in subtasks for q in qs) for k in base}
    plan: list[tuple[str, str]] = []
    chosen: set[str] = set()
    for _ in range(min(dyn.n_changes, len(base))):
        ks = [k for k in base if k not in chosen]
        k = r.choices(ks, weights=[weight[x] for x in ks])[0]
        chosen.add(k)
        plan.append(("fact", k))
    paths = sorted({world.base_path[k] for k in base})
    for p in r.sample(paths, min(dyn.n_cosmetic, len(paths))):
        plan.append(("cosmetic", p))
    return plan
