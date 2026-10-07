import math

import pytest

from bacm.config import AgentConfig, DynamicsConfig, RunConfig, WorldConfig, get_arch
from bacm.core.knowledge import Claim, Level, Source, Status
from bacm.core.manager import ContextManager
from bacm.env.tools import ToolBox
from bacm.env.world import World, parse_fact
from bacm.metrics.tracker import Tracker
from bacm.sim.orchestrator import Orchestrator, run_once


def make_cm(arch="bacm_v1", seed=3, configure=False):
    w = World.generate(WorldConfig(), seed)
    tr = Tracker(w, RunConfig(arch=get_arch(arch)).cost)
    cm = ContextManager(get_arch(arch), w, tr, ToolBox(w, tr))
    if configure:
        cm.configure(AgentConfig(p_misread=0.0), seed)
    return w, tr, cm


def code_claim(w, key, value=None, conf=0.95):
    path = w.base_path[key]
    v, line = parse_fact(key, w.files[path].content)
    return Claim(key, value or v, conf, "code", [Source(path, w.version(path), line)], level=Level.L2)


def test_world_truth_and_parsing():
    w = World.generate(WorldConfig(), 0)
    for key, path in w.base_path.items():
        assert parse_fact(key, w.files[path].content)[0] == w.base[key]
    e = w.endpoints[0]
    assert w.truth(f"endpoint_datastore({e})") == w.base[f"datastore({w.base[f'route({e})']})"]


def test_mutation_bumps_version_and_changes_truth():
    w = World.generate(WorldConfig(), 0)
    key = next(k for k in w.base if k.startswith("datastore("))
    old, v0 = w.truth(key), w.version(w.base_path[key])
    ch = w.mutate_fact(key)
    assert w.truth(key) != old and w.version(w.base_path[key]) == v0 + 1 and ch.new_value == w.truth(key)


def test_reuse_and_admission_rejects_weak_claims():
    w, tr, cm = make_cm()
    key = next(k for k in w.base if k.startswith("datastore("))
    assert cm.query("A0", key, 0) is None
    assert cm.publish("A0", code_claim(w, key), 0) is not None
    hit = cm.query("A1", key, 1)
    assert hit and hit.value == w.truth(key)
    other = next(k for k in w.base if k.startswith("protocol("))
    assert cm.publish("A0", Claim(other, "rest", 0.3, "hypothesis", level=Level.L1), 2) is None
    assert cm.publish("A0", Claim(other, "rest", 0.3, "hypothesis", level=Level.L0), 2) is None


def _derived_setup(cm, w):
    e = w.endpoints[0]
    rk = f"route({e})"
    dk = f"datastore({w.base[rk]})"
    cm.publish("A0", code_claim(w, rk), 0)
    cm.publish("A0", code_claim(w, dk), 0)
    ek = f"endpoint_datastore({e})"
    cm.publish("A0", Claim(ek, w.truth(ek), 0.93, "derived", deps=[rk, dk], level=Level.L2), 1)
    cm.query("A2", ek, 2)   # A2 consumes only the derived fact
    return rk, dk, ek


def test_incremental_invalidation_propagates_through_dependencies():
    w, tr, cm = make_cm()
    rk, dk, ek = _derived_setup(cm, w)
    ch = w.mutate_fact(dk)
    cm.on_change(ch, 3)
    status = {k.key: k.status for k in cm.k.values()}
    assert status[dk] == Status.STALE and status[ek] == Status.STALE and status[rk] == Status.VERIFIED
    notes = cm.drain_notifications(1e9)
    assert ek in notes["A2"]          # consumer of the derived fact is told
    assert cm.query("A3", ek, 4) is None


def test_without_dependency_graph_derived_knowledge_stays_stale():
    w, tr, cm = make_cm("bacm-no_dependencies")
    rk, dk, ek = _derived_setup(cm, w)
    old = w.truth(ek)
    cm.on_change(w.mutate_fact(dk), 3)
    hit = cm.query("A3", ek, 4)
    assert hit is not None and hit.value == old != w.truth(ek)   # false reuse


def test_span_provenance_survives_cosmetic_edit_but_file_level_does_not():
    for arch, expect_valid in (("bacm_v1", True), ("bacm-file_level", False)):
        w, tr, cm = make_cm(arch)
        key = next(k for k in w.base if k.startswith("datastore("))
        cm.publish("A0", code_claim(w, key), 0)
        cm.on_change(w.cosmetic_edit(w.base_path[key]), 1)
        assert (cm.query("A1", key, 2) is not None) is expect_valid
        if not expect_valid:
            assert tr.invalidations and tr.invalidations[0]["unnecessary"]


def test_lazy_validation_without_eager_invalidation():
    w, tr, cm = make_cm("bacm-no_invalidation")
    key = next(k for k in w.base if k.startswith("datastore("))
    cm.publish("A0", code_claim(w, key), 0)
    cm.on_change(w.mutate_fact(key), 1)       # no eager action...
    assert not cm.drain_notifications(1e9)
    assert cm.query("A1", key, 2) is None      # ...but caught when reused


def test_outdated_claim_is_refused():
    w, tr, cm = make_cm()
    key = next(k for k in w.base if k.startswith("datastore("))
    claim = code_claim(w, key)                 # read before the change
    w.mutate_fact(key)
    assert cm.publish("A0", claim, 1) is None
    assert key in cm.drain_notifications(1e9)["A0"]


def test_conflict_resolved_by_verification():
    w, tr, cm = make_cm()
    key = next(k for k in w.base if k.startswith("datastore("))
    wrong = next(v for v in ("redis", "mysql") if v != w.truth(key))
    cm.publish("A0", code_claim(w, key, wrong, 0.9), 0)       # misread
    cm.query("A1", key, 1)
    cm.publish("A2", code_claim(w, key), 2)                    # correct read
    assert tr.conflicts and tr.conflicts[0]["correct"] and tr.conflicts[0]["method"] == "verification"
    assert cm.query("A3", key, 3).value == w.truth(key)
    assert key in cm.drain_notifications(1e9)["A1"]               # consumer of the wrong claim corrected


def test_conflict_off_is_last_write_wins():
    w, tr, cm = make_cm("bacm-no_conflict")
    key = next(k for k in w.base if k.startswith("datastore("))
    wrong = next(v for v in ("redis", "mysql") if v != w.truth(key))
    cm.publish("A0", code_claim(w, key), 0)
    cm.publish("A2", code_claim(w, key, wrong, 0.9), 2)
    assert cm.query("A3", key, 3).value == wrong


ALL = ["independent", "independent_evict", "fully_shared", "selective", "selective_fast",
       "selective_global_recompute", "bacm_v1", "bacm"]


@pytest.mark.parametrize("arch", ALL)
def test_end_to_end_runs_and_is_deterministic(arch):
    cfg = RunConfig(arch=get_arch(arch), seed=2, dynamics=DynamicsConfig(n_changes=2, n_cosmetic=1))
    m1, m2 = run_once(cfg), run_once(cfg)
    m1.pop("wall_clock_s"), m2.pop("wall_clock_s")
    assert m1 == m2 or all((a == b) or (isinstance(a, float) and math.isnan(a) and math.isnan(b))
                           for a, b in zip(m1.values(), m2.values()))
    assert 0 <= m1["question_accuracy"] <= 1


def test_sharing_reduces_reads_and_independent_never_reuses():
    ind = run_once(RunConfig(arch=get_arch("independent"), seed=5))
    bacm = run_once(RunConfig(arch=get_arch("bacm"), seed=5))
    assert ind["kb_hits"] == 0 and ind["reuse_rate"] == 0
    assert bacm["file_reads"] < ind["file_reads"] and bacm["tokens_total"] < ind["tokens_total"]


def test_bacm_repairs_stale_answers_better_than_selective():
    dyn = DynamicsConfig(n_changes=3, n_cosmetic=1)
    stale = {a: [] for a in ("selective", "bacm")}
    for seed in range(6):
        for a in stale:
            r = run_once(RunConfig(arch=get_arch(a), seed=seed, dynamics=dyn))["stale_answer_rate"]
            if not math.isnan(r):
                stale[a].append(r)
    assert sum(stale["bacm"]) / len(stale["bacm"]) < sum(stale["selective"]) / len(stale["selective"])


def test_worker_recomputes_affected_subgraph_and_pushes_values():
    w, tr, cm = make_cm("bacm", configure=True)
    rk, dk, ek = _derived_setup(cm, w)
    ch = w.mutate_fact(dk)
    cm.on_change(ch, 3)
    notes = cm.drain_notifications(1e9)
    assert notes["A2"][ek] == w.truth(ek)               # consumer receives the corrected derived value
    assert cm.query("A3", dk, 10).value == w.truth(dk)   # re-extracted base fact
    assert cm.query("A3", ek, 10).value == w.truth(ek)   # re-derived fact, reusable at once
    assert any(x[0] == "context-manager" and x[4].startswith("worker") for x in tr.llm)


def test_weak_claim_is_verified_instead_of_dropped():
    w, tr, cm = make_cm("bacm", configure=True)
    key = next(k for k in w.base if k.startswith("datastore("))
    wrong = next(v for v in ("redis", "mysql") if v != w.truth(key))
    assert cm.publish("A0", Claim(key, wrong, 0.55, "doc", level=Level.L2), 0) is None
    assert cm.drain_notifications(1e9)["A0"][key] == w.truth(key)
    assert cm.query("A1", key, 5).value == w.truth(key)


def test_prefetch_follows_values_multi_hop():
    w, tr, cm = make_cm()
    rk, dk, ek = _derived_setup(cm, w)
    ep = rk[len("route("):-1]
    other = "endpoint_protocol(" + ep + ")"
    keys = {h.key for h in cm.prefetch("A9", [other, rk], 5)}
    assert rk in keys and dk in keys and ek in keys     # route -> service -> datastore(service)


def test_efficient_bacm_saves_tokens_and_stays_accurate():
    dyn = DynamicsConfig(n_changes=3, n_cosmetic=2)
    tot = {a: [0.0, 0.0] for a in ("independent", "bacm")}
    for seed in range(5):
        for a in tot:
            m = run_once(RunConfig(arch=get_arch(a), seed=seed, dynamics=dyn))
            tot[a][0] += m["tokens_total"]
            tot[a][1] += m["question_accuracy"]
    assert tot["bacm"][0] < 0.8 * tot["independent"][0]
    assert tot["bacm"][1] > tot["independent"][1]


def test_conflict_vote_counts_independent_observers():
    w, tr, cm = make_cm("bacm", configure=True)
    key = next(k for k in w.base if k.startswith("datastore("))
    wrong = next(v for v in ("redis", "mysql") if v != w.truth(key))
    cm.publish("A0", code_claim(w, key), 0)
    cm.publish("A1", code_claim(w, key), 0)                    # corroborates -> 2 supporters
    cm.publish("A2", code_claim(w, key, wrong, 0.95), 1)       # lone misread
    assert tr.conflicts[-1]["correct"]
    assert cm.query("A3", key, 2).value == w.truth(key)


def test_answer_validation_corrects_a_wrong_derivation_and_tracks_it():
    w, tr, cm = make_cm("bacm_validate", configure=True)
    e = w.endpoints[0]
    rk = f"route({e})"
    dk = f"datastore({w.base[rk]})"
    cm.publish("A0", code_claim(w, rk), 0)
    cm.publish("A0", code_claim(w, dk), 0)
    ek = f"endpoint_datastore({e})"
    wrong = next(v for v in ("redis", "mysql") if v != w.truth(ek))
    cm.validate_answer("A1", ek, wrong, 1)                       # the agent mis-combined correct premises
    assert cm.drain_notifications(1e9)["A1"][ek] == w.truth(ek)
    cm.on_change(w.mutate_fact(dk), 2)                            # the answer is now in the graph
    assert cm.drain_notifications(1e9)["A1"][ek] == w.truth(ek)


def test_results_do_not_depend_on_hash_seed():
    """Exact reproducibility across processes (set iteration order changes with PYTHONHASHSEED)."""
    import os
    import subprocess
    import sys
    code = ("from bacm.config import RunConfig, get_arch; from bacm.experiments.suites import DYNAMIC, CONFLICT_AGENT;"
            "from bacm.sim.orchestrator import run_once;"
            "print([run_once(RunConfig(arch=get_arch(a), seed=s, dynamics=DYNAMIC, agent=CONFLICT_AGENT))['tokens_in']"
            " for s in range(3) for a in ('bacm', 'bacm_v1', 'fully_shared')])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout for h in ("1", "2", "3")}
    assert len(outs) == 1
