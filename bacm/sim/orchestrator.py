"""Discrete-event orchestrator: C_0 -> branches B_1..B_n running in parallel.

Agents are generators that yield actions. Each agent has its own simulated clock;
the orchestrator always advances the agent that is earliest in time, so agents
genuinely race (Agent B only reuses what Agent A published *before* B asked).
Wall-clock time of real LLM calls is used as the step duration for LLM backends.
"""
from __future__ import annotations

import heapq
import random
import time

from ..agents.actions import Answer, KBQuery, LLMCall, LLMStep, Need, Note, Prefetch, Publish, Tool, UseShared
from ..config import RunConfig
from ..core.manager import ContextManager
from ..env.tasks import make_task
from ..env.tools import ToolBox
from ..env.world import World
from ..memory.base import NullMemory, SelectiveMemory, TranscriptMemory
from ..metrics.tracker import Tracker
from ..util import count_tokens

PIGGYBACK_OUT_TOKENS = 20

MEMORY_CLASSES = {"none": NullMemory, "transcript": TranscriptMemory,
                  "selective": SelectiveMemory, "bacm": ContextManager}
SHARE_MODE = {"none": "none", "transcript": "transcript", "selective": "tool", "bacm": "tool"}


def make_world(cfg: RunConfig):
    """The environment: the synthetic repository, or a real open-source repository (cfg.repo)."""
    if cfg.repo is not None:
        from ..env.realrepo import RealRepoWorld
        return RealRepoWorld.load(cfg.repo, cfg.seed)
    return World.generate(cfg.world, cfg.seed)


class Orchestrator:
    def __init__(self, cfg: RunConfig, llm_client=None):
        self.cfg = cfg
        self.world = make_world(cfg)
        self.subtasks = make_task(self.world, cfg.task, cfg.seed)
        self.change_plan = self.world.plan_changes(self.subtasks, cfg.dynamics, cfg.seed)
        self.initial_truth = {q.key: self.world.truth(q.key) for qs in self.subtasks for q in qs}
        self.tracker = Tracker(self.world, cfg.cost)
        self.toolbox = ToolBox(self.world, self.tracker)
        self.memory = MEMORY_CLASSES[cfg.arch.memory](cfg.arch, self.world, self.tracker, self.toolbox)
        self.memory.configure(cfg.agent, cfg.seed, llm_client if cfg.backend != "sim" else None)
        if hasattr(self.memory, "set_questions"):
            self.memory.set_questions({f"A{i}": [q.key for q in qs] for i, qs in enumerate(self.subtasks)})
        self.share_mode = SHARE_MODE[cfg.arch.memory]
        self.llm_client = llm_client
        self.c0 = self.world.initial_context()   # C_0: what every branch starts from
        self.agents = {}
        self.clock: dict[str, float] = {}
        self.llm_steps: dict[str, int] = {}
        self.answered: set[tuple[str, str]] = set()
        self.total_questions = sum(len(q) for q in self.subtasks)
        self.changes_applied = False
        self.mediator = None
        if cfg.arch.transparent:
            if cfg.backend == "sim":
                raise ValueError("transparent-memory architectures need an LLM backend")
            from .mediator import Mediator
            self.mediator = Mediator(self)
        self._build_agents()

    def _build_agents(self):
        route_files = sorted(p for p in self.world.files if p.startswith("api/routes_"))
        for i, qs in enumerate(self.subtasks):
            aid = f"A{i}"
            rng = random.Random(self.cfg.seed * 1_000_003 + i * 7_919)
            if self.cfg.backend == "sim":
                if self.cfg.repo is not None:
                    raise ValueError("real repositories need an LLM backend (--backend openai_compat/anthropic)")
                from ..agents.simulated import SimAgent
                agent = SimAgent(aid, qs, self.cfg.agent, rng, self.share_mode, self.c0, self.world.truth,
                                 self.world.services, self.world.features, route_files, self.cfg.arch)
            else:
                from ..agents.llm_agent import LLMAgent
                agent = LLMAgent(aid, qs, self.cfg, self.share_mode, self.c0, self.world)
            self.agents[aid] = agent
            self.clock[aid] = 0.0
            self.llm_steps[aid] = 0

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        t0 = time.perf_counter()
        heap: list[tuple[float, int, str]] = []
        seq = 0
        gens = {}
        pending = {}
        for aid, agent in self.agents.items():
            gens[aid] = agent.run()
            pending[aid] = None
            heapq.heappush(heap, (0.0, seq, aid))
            seq += 1
        idle: set[str] = set()
        self._seq = seq
        recomputed = False
        n_actions, aborted = 0, False
        max_actions = 2_000 * len(self.agents)   # livelock guard; normal runs use ~100 per agent
        while True:
            if not heap and self.memory.next_notification_time() is not None:
                # everyone is idle but L3 events are still in flight (e.g. the worker is recomputing)
                t_next = self.memory.next_notification_time()
                self._deliver(t_next, idle, gens, pending, heap)
                seq = self._seq
                continue
            if not heap:
                # Global-recomputation baseline: once everyone is done, if any source changed,
                # throw all shared + local state away and redo the whole task.
                if (self.cfg.arch.global_recompute and self.tracker.change_events and not recomputed):
                    recomputed = True
                    self.memory.reset()
                    t_restart = max(self.clock.values())
                    for aid, agent in self.agents.items():
                        agent.reset()
                        gens[aid], pending[aid] = agent.run(), None
                        heapq.heappush(heap, (t_restart, seq, aid))
                        seq += 1
                    idle.clear()
                    continue
                break
            now, _, aid = heapq.heappop(heap)
            n_actions += 1
            if n_actions > max_actions:
                aborted = True
                break
            try:
                action = gens[aid].send(pending[aid])
            except StopIteration:
                idle.add(aid)
                continue
            result, dt = self._execute(aid, action, now)
            pending[aid] = result
            self.clock[aid] = now + dt
            heapq.heappush(heap, (now + dt, seq, aid))
            seq += 1
            self._seq = seq
            self._deliver(now + dt, idle, gens, pending, heap)
            seq = self._seq
        makespan = max(self.clock.values()) if self.clock else 0.0
        metrics = self.tracker.compute(self.subtasks, self.initial_truth, makespan,
                                       time.perf_counter() - t0, len(self.agents))
        metrics["aborted"] = float(aborted)
        return metrics

    def _deliver(self, t, idle, gens, pending, heap):
        """Deliver L3 notifications due by time t; wake idle agents that must recompute."""
        for target, updates in self.memory.drain_notifications(t).items():
            if target not in self.agents:
                continue
            if self.mediator is not None:
                updates, wake = self.mediator.on_updates(target, updates, t)
                if not wake:
                    continue
            self.agents[target].inbox.update(updates)
            if target in idle:
                idle.discard(target)
                gens[target] = self.agents[target].run()
                pending[target] = None
                heapq.heappush(heap, (max(self.clock[target], t), self._seq, target))
                self._seq += 1

    # ------------------------------------------------------------ execute
    def _llm_latency(self, tin: int, tout: int) -> float:
        c = self.cfg.cost
        return c.llm_base_latency + c.llm_latency_per_in_tok * tin + c.llm_latency_per_out_tok * tout

    def _execute(self, aid: str, action, now: float):
        agent = self.agents[aid]
        lat = self.cfg.cost.tool_latency
        if isinstance(action, LLMStep):
            tin = agent.context_tokens() + self.memory.context_tokens(aid)
            self.tracker.on_llm(aid, tin, action.out_tokens, now, action.purpose)
            agent.add_context(action.out_tokens)
            self.memory.observe(aid, action.purpose, action.out_tokens)
            return None, self._llm_latency(tin, action.out_tokens)
        if isinstance(action, LLMCall):
            shared = self.memory.shared_prompt(aid) if hasattr(self.memory, "shared_prompt") else ""
            t = time.perf_counter()
            content, tin, tout, stop = self.llm_client.complete(action.system + shared, action.messages,
                                                                action.tools)
            dt = getattr(self.llm_client, "last_latency", None) or (time.perf_counter() - t)
            self.tracker.on_llm(aid, tin, tout, now, action.purpose)
            text = " ".join(getattr(b, "text", "") or f"[{getattr(b, 'name', '')} {getattr(b, 'input', '')}]"
                            for b in content)
            self.memory.observe(aid, text, tout)
            return (content, stop), dt
        if isinstance(action, Tool) and self.mediator is not None:
            res, dt = self.mediator.tool(aid, action, now)
            self.memory.observe(aid, f"{aid} {action.op}({action.args}):\n{res.text}", res.tokens)
            return res, dt
        if isinstance(action, Tool):
            res = self.toolbox.call(aid, action.op, action.args, now)
            agent.add_context(res.tokens)
            self.memory.observe(aid, f"{aid} {action.op}({action.args}):\n{res.text}", res.tokens)
            return res, lat.get(action.op, 0.05)
        if isinstance(action, Note):
            return self._note(aid, action, now)
        if isinstance(action, Prefetch):
            hits = self.memory.prefetch(aid, action.keys, now)
            return hits, lat["kb_query"]
        if isinstance(action, UseShared):
            self.tracker.on_query(aid, action.key, action.hit, now)
            return None, 0.0
        if isinstance(action, Need):
            self.tracker.on_need(aid, action.key, now)
            return None, 0.0
        if isinstance(action, KBQuery):
            hit = self.memory.query(aid, action.key, now)
            self.tracker.on_query(aid, action.key, hit, now)
            if action.in_context:
                return hit, 0.0
            agent.add_context(hit.tokens if hit else 8)
            return hit, lat["kb_query"]
        if isinstance(action, Publish):
            self.tracker.on_obtained(aid, action.claim)
            if self.share_mode == "none":
                return None, 0.0
            self.memory.observe(aid, action.claim.text(), count_tokens(action.claim.text()), action.claim)
            if action.piggyback:   # emitted inside another turn: pay only its output tokens
                self.tracker.on_llm(aid, 0, PIGGYBACK_OUT_TOKENS, now, "piggyback:kb_publish")
                agent.add_context(PIGGYBACK_OUT_TOKENS)
            kid = self.memory.publish(aid, action.claim, now)
            return kid, (lat["kb_publish"] if self.share_mode == "tool" else 0.0)
        if isinstance(action, Answer) and self.mediator is not None:
            return self.mediator.answer(aid, action.key, action.value, now), lat["answer"]
        if isinstance(action, Answer):
            self.tracker.on_answer(aid, action.key, action.value, now)
            self.answered.add((aid, action.key))
            if hasattr(self.memory, "validate_answer"):
                self.memory.validate_answer(aid, action.key, action.value, now)
            self._maybe_apply_changes(now)
            return None, lat["answer"]
        raise TypeError(f"unknown action {action!r}")

    def _note(self, aid: str, action, now: float):
        """Small fresh-context LLM call that notes every fact in a file (real-LLM harvest). With a shared
        memory the note for file@version is computed once and reused by every branch."""
        shared = self.share_mode != "none"
        cache = self.__dict__.setdefault("_notes", {})
        ck = (action.path, action.version) if shared else (aid, action.path, action.version)
        if ck in cache:
            return cache[ck], 0.0
        blocks, tin, tout, _ = self.llm_client.complete(
            "You extract facts from source files precisely.",
            [{"role": "user", "content": f"{self.world.note_prompt}\n\nFile {action.path}:\n{action.text}"}], None)
        self.tracker.on_llm(aid, tin, tout, now, "note")
        text = " ".join(getattr(b, "text", "") or "" for b in blocks)
        facts = self.world.note_facts(text, action.path, action.text.splitlines())
        cache[ck] = facts
        return facts, getattr(self.llm_client, "last_latency", 0.0)

    def _maybe_apply_changes(self, now: float) -> None:
        if self.changes_applied or not self.change_plan:
            return
        if len(self.answered) / self.total_questions < self.cfg.dynamics.trigger_progress:
            return
        self.changes_applied = True
        for item in self.change_plan:
            for ch in self.world.apply_change(item):
                self.tracker.on_change(ch, now)
                self.memory.on_change(ch, now)


def run_once(cfg: RunConfig, llm_client=None) -> dict:
    return Orchestrator(cfg, llm_client).run()
