"""Event log for a single run + computation of every metric in design-document sections 17-22."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.knowledge import Level

SYSTEM_AGENT = "context-manager"


@dataclass
class Tracker:
    world: object
    cost: object
    llm: list = field(default_factory=list)          # (agent, in, out, t, purpose)
    tools: list = field(default_factory=list)        # dicts
    needs: list = field(default_factory=list)        # (agent, key, reusable, t)
    queries: list = field(default_factory=list)      # dicts
    answers: list = field(default_factory=list)      # (agent, key, value, t, correct_now)
    admissions: list = field(default_factory=list)
    invalidations: list = field(default_factory=list)
    revalidations: int = 0
    digest_reads: int = 0          # reads served from memory as a digest (transparent memory)
    auto_resolved: int = 0         # questions answered by the manager without an LLM turn
    answer_revisions: int = 0      # answers updated in place (source change / correction)
    answer_corrections: int = 0    # agent answers overridden by verified facts
    unverified_answers: int = 0    # agent answers the manager could not verify
    reanswers: int = 0             # answers re-derived by the manager in a small fresh context
    conflicts: list = field(default_factory=list)
    l3_events: list = field(default_factory=list)
    validations: list = field(default_factory=list)   # (agent, key, agreed, validated_value_correct)
    change_events: list = field(default_factory=list)  # (t, Change)
    knowledge_created: int = 0
    obtained: dict = field(default_factory=lambda: defaultdict(set))  # key -> agents that obtained it
    _seen_ops: dict = field(default_factory=dict)       # op signature -> first agent
    read_sets: dict = field(default_factory=lambda: defaultdict(set))
    _last_need: dict = field(default_factory=dict)      # (agent, key) -> was it reusable when needed

    def _truth(self, key):
        """Ground truth, or None for keys a real LLM invented (they can never be correct)."""
        try:
            return self.world.truth(key)
        except (KeyError, ValueError):
            return None

    # ------------------------------------------------------------- hooks
    def on_llm(self, agent, in_tok, out_tok, t, purpose="step"):
        self.llm.append((agent, in_tok, out_tok, t, purpose))

    def on_tool(self, agent, op, args, version, tokens, t):
        sig = (op, tuple(sorted((k, repr(v)) for k, v in args.items())), version)
        first = self._seen_ops.get(sig)
        redundant = first is not None
        self.tools.append({"agent": agent, "op": op, "args": dict(args), "version": version, "tokens": tokens,
                           "t": t, "redundant": redundant, "cross": redundant and first != agent})
        if first is None:
            self._seen_ops[sig] = agent
        if op == "read_file":
            self.read_sets[agent].add(args.get("path"))

    def on_need(self, agent, key, t):
        reusable = any(a != agent for a in self.obtained.get(key, ()))
        self.needs.append((agent, key, reusable, t))
        self._last_need[(agent, key)] = reusable

    def on_query(self, agent, key, hit, t):
        truth = self._truth(key)
        self.queries.append({"agent": agent, "key": key, "hit": hit is not None, "t": t,
                             "false": hit is not None and hit.value != truth,
                             "reusable": self._last_need.get((agent, key), False)})
        if hit is not None:
            self.obtained[key].add(agent)

    def on_obtained(self, agent, claim):
        if claim.level != Level.L0:
            self.obtained[claim.key].add(agent)

    def on_answer(self, agent, key, value, t):
        self.answers.append((agent, key, value, t, value == self._truth(key)))

    def on_admission(self, agent, claim, admitted, score=None):
        self.admissions.append((agent, claim.key, claim.evidence, admitted, score,
                                claim.value == self._truth(claim.key)))

    def on_knowledge_created(self, k):
        self.knowledge_created += 1

    def on_invalidation(self, k, reason, t):
        self.invalidations.append({"id": k.id, "key": k.key, "reason": reason, "t": t,
                                   "unnecessary": k.value == self._truth(k.key)})

    def on_revalidation(self, k):
        self.revalidations += 1

    def on_conflict(self, key, winner_value, method, t):
        self.conflicts.append({"key": key, "method": method, "t": t,
                               "correct": winner_value == self._truth(key)})

    def on_validation(self, agent, key, answered, validated):
        self.validations.append((agent, key, answered == validated, validated == self._truth(key)))

    def on_l3_event(self, change, n_stale, t):
        self.l3_events.append((t, change.path, n_stale))

    def on_change(self, change, t):
        self.change_events.append((t, change))

    # ----------------------------------------------------------- metrics
    def compute(self, subtasks, initial_truth: dict, makespan: float, wall: float, n_agents: int) -> dict:
        cm = self.cost
        tin = sum(x[1] for x in self.llm)
        tout = sum(x[2] for x in self.llm)
        agent_llm = [x for x in self.llm if x[0] != SYSTEM_AGENT]
        ctx = [x[1] for x in agent_llm] or [0]

        # final answer per (agent, key) = the last one submitted
        final: dict[tuple, tuple] = {}
        for a, k, v, t, _ in self.answers:
            final[(a, k)] = (v, t)
        truth_now = {q.key: self._truth(q.key) for qs in subtasks for q in qs}
        per_q = []
        agent_ok = []
        for i, qs in enumerate(subtasks):
            aid = f"A{i}"
            oks = []
            for q in qs:
                v = final.get((aid, q.key), (None, None))[0]
                oks.append(v == truth_now[q.key])
                per_q.append((aid, q.key, v, v == truth_now[q.key]))
            agent_ok.append(all(oks))
        n_q = len(per_q) or 1

        changed_keys = {k for k in truth_now if truth_now[k] != initial_truth[k]}
        affected = [p for p in per_q if p[1] in changed_keys]
        stale_answers = [p for p in affected if p[2] == initial_truth[p[1]]]
        t_change = self.change_events[0][0] if self.change_events else None
        ttc = None
        if affected and t_change is not None and all(p[3] for p in affected):
            ttc = max(max(0.0, final[(p[0], p[1])][1] - t_change) for p in affected)

        tools = [x for x in self.tools if x["agent"] != SYSTEM_AGENT]
        redundant = [x for x in tools if x["redundant"]]
        reads = [x for x in tools if x["op"] == "read_file"]
        red_reads = [x for x in reads if x["redundant"]]

        reusable_needs = [n for n in self.needs if n[2]]
        hits = [q for q in self.queries if q["hit"]]
        reused_on_opportunity = [q for q in hits if q["reusable"]]

        # correlated errors: pairs of agents answering the same question with the same wrong value
        by_key = defaultdict(list)
        for aid, key, v, ok in per_q:
            by_key[key].append((v, ok))
        pairs = corr = 0
        for vals in by_key.values():
            for (v1, ok1), (v2, ok2) in itertools.combinations(vals, 2):
                pairs += 1
                corr += (not ok1 and not ok2 and v1 == v2)
        sets = [self.read_sets.get(f"A{i}", set()) for i in range(n_agents)]
        jd = [1 - len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]

        cost = tin / 1e6 * cm.price_in_per_mtok + tout / 1e6 * cm.price_out_per_mtok
        return {
            "question_accuracy": sum(p[3] for p in per_q) / n_q,
            "agent_success_rate": sum(agent_ok) / max(1, len(agent_ok)),
            "team_success": float(all(agent_ok)),
            "tokens_in": tin,
            "tokens_out": tout,
            "tokens_total": tin + tout,
            "llm_calls": len(self.llm),
            "cost_usd": cost,
            "tool_calls": len(tools),
            "file_reads": len(reads),
            "redundant_ops": len(redundant),
            "redundant_cross_agent_ops": sum(x["cross"] for x in redundant),
            "redundancy_rate": len(redundant) / max(1, len(tools)),
            "redundant_file_reads": len(red_reads),
            "ctx_avg": sum(ctx) / len(ctx),
            "ctx_max": max(ctx),
            "latency_sim_s": makespan,
            "wall_clock_s": wall,
            "kb_queries": len(self.queries),
            "kb_hits": len(hits),
            "reuse_rate": len(reused_on_opportunity) / max(1, len(reusable_needs)),
            "reuse_opportunities": len(reusable_needs),
            "false_reuse_rate": sum(q["false"] for q in hits) / max(1, len(hits)),
            "knowledge_objects": self.knowledge_created,
            "admitted": sum(a[3] for a in self.admissions),
            "rejected": sum(not a[3] for a in self.admissions),
            "wrong_claims_admitted": sum(a[3] and not a[5] for a in self.admissions),
            "invalidations": len(self.invalidations),
            "unnecessary_invalidations": sum(x["unnecessary"] for x in self.invalidations),
            "revalidations": self.revalidations,
            "l3_events": len(self.l3_events),
            "digest_reads": self.digest_reads,
            "auto_resolved": self.auto_resolved,
            "answer_revisions": self.answer_revisions,
            "unverified_answers": self.unverified_answers,
            "reanswers": self.reanswers,
            "answer_validations": len(self.validations),
            "answer_corrections": sum(not v[2] for v in self.validations) + self.answer_corrections,
            "system_tokens": sum(x[1] + x[2] for x in self.llm if x[0] == SYSTEM_AGENT),
            "conflicts": len(self.conflicts),
            "conflict_accuracy": (sum(c["correct"] for c in self.conflicts) / len(self.conflicts))
            if self.conflicts else math.nan,
            "n_source_changes": sum(1 for _, c in self.change_events if c.key),
            "affected_answers": len(affected),
            "stale_answer_rate": len(stale_answers) / len(affected) if affected else math.nan,
            "affected_accuracy": sum(p[3] for p in affected) / len(affected) if affected else math.nan,
            "time_to_consistency_s": ttc if ttc is not None else (math.inf if affected else math.nan),
            "correlated_error_rate": corr / max(1, pairs),
            "read_set_diversity": sum(jd) / len(jd) if jd else math.nan,
        }
