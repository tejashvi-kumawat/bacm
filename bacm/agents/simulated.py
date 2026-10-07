"""A seeded, rule-based agent policy that mimics a ReAct-style LLM agent.

It genuinely searches, reads and parses repository files (so staleness and
redundancy emerge from the environment), while its *imperfections* - reading
outdated docs, misreading, speculating, trusting other agents - are controlled
by AgentConfig probabilities. Every decision is preceded by an LLMStep so token
usage grows with the agent's context exactly like a real tool-using agent.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import AgentConfig
from ..core.knowledge import Claim, Level, Source
from ..env.tasks import Question
from ..env.world import DATASTORES, DOC_FILE, PROTOCOLS, derive, harvest_facts, parse_doc, parse_fact
from ..util import count_tokens
from .actions import Answer, KBQuery, LLMStep, Need, Prefetch, Publish, Tool, UseShared

FACT_NOTE_TOKENS = 25        # a compact extracted fact kept in context instead of raw file text
HARVEST_OUT_PER_FACT = 15    # extra output tokens to note each additional fact seen in a file
NOTICE_TOKENS = 30           # an L3 notification entering the context
PIGGYBACK_OUT_TOKENS = 20


@dataclass
class LocalFact:
    value: str
    evidence: str
    confidence: float
    deps: list[str] = field(default_factory=list)
    paths: tuple[str, ...] = ()


class SimAgent:
    def __init__(self, aid: str, questions: list[Question], cfg: AgentConfig, rng: random.Random,
                 share_mode: str, initial_context: str, oracle, services: list[str],
                 feature_map: dict[str, list[str]], route_files: list[str], arch=None):
        """share_mode: none | transcript | tool.  oracle(key) is used ONLY to model the
        content of speculative hypotheses (whether a guess happens to be right)."""
        self.id = aid
        self.questions = questions
        self.cfg = cfg
        self.rng = rng
        self.share_mode = share_mode
        self.oracle = oracle
        self.services = services
        self.feature_map = feature_map
        self.route_files = route_files
        from ..config import ArchConfig
        self.arch = arch or ArchConfig("default", "none")
        self.tool_mode = share_mode == "tool"
        self.local: dict[str, LocalFact] = {}
        self.answers: dict[str, str] = {}
        self.inbox: dict[str, str | None] = {}     # key -> corrected value, or None = invalidated
        self.pref: dict[str, object] = {}          # prefetched shared knowledge not yet used
        self.seen_files: dict[str, object] = {}    # raw files still in this branch's context
        self.harvested: set[str] = set()           # files whose facts are all noted locally
        task_text = "\n".join(q.text for q in questions)
        self.ctx = self.ctx0 = cfg.system_prompt_tokens + count_tokens(initial_context) + count_tokens(task_text)
        self.done = False

    def reset(self) -> None:
        """Start the subtask again in a fresh context (global recomputation)."""
        self.local.clear()
        self.answers.clear()
        self.seen_files.clear()
        self.harvested.clear()
        self.pref.clear()
        self.inbox.clear()
        self.ctx = self.ctx0

    # -------------------------------------------------------------- context
    def context_tokens(self) -> int:
        return self.ctx

    def add_context(self, tokens: int) -> None:
        self.ctx += tokens

    # ----------------------------------------------------------------- main
    def run(self):
        while True:
            self._process_inbox()
            todo = [q for q in self.questions if q.key not in self.answers]
            if not todo:
                return
            q = todo[0]
            if self.tool_mode and self.arch.prefetch:
                hits = yield Prefetch(self._subgoals(q.key))
                for h in hits:
                    if h.key not in self.local and h.key not in self.pref:
                        self.pref[h.key] = h
                        self.add_context(h.tokens)
            value = yield from self.resolve(q.key)
            yield LLMStep("answer", self.cfg.answer_out_tokens)
            yield Answer(q.key, value)
            self.answers[q.key] = value

    def _subgoals(self, key: str) -> list[str]:
        arg = key[key.index("(") + 1:-1]
        if key.startswith("feature_datastores("):
            return [key] + [f"endpoint_datastore({e})" for e in self.feature_map[arg]]
        return [key, f"route({arg})"]

    def _process_inbox(self) -> None:
        """L3 notifications. A corrected value patches the fact in place; a bare invalidation drops it.
        Anything this branch derived from a changed fact is dropped, and affected answers are reopened."""
        if not self.inbox:
            return
        updates = dict(self.inbox)
        self.inbox.clear()
        self.add_context(NOTICE_TOKENS * len(updates))
        changed: set[str] = set()
        for k, v in updates.items():
            self.pref.pop(k, None)
            old = self.local.get(k)
            if old is not None and (v is None or old.value != v):
                for p in old.paths:   # raw text / notes from that file are outdated: re-read if needed
                    self.seen_files.pop(p, None)
                    self.harvested.discard(p)
            if v is None:
                if self.local.pop(k, None) is not None:
                    changed.add(k)
            elif old is None or old.value != v:
                self.local[k] = LocalFact(v, "memory", 0.95)
                changed.add(k)
        frontier = set(changed)
        dropped: set[str] = set()
        while frontier:
            k = frontier.pop()
            for d, f in list(self.local.items()):
                if k in f.deps and not (d in updates and updates[d] is not None):
                    del self.local[d]
                    dropped.add(d)
                    frontier.add(d)
        for key in list(self.answers):
            if key in changed | dropped and self.local.get(key) is None or (
                    key in self.local and self.local[key].value != self.answers[key]):
                del self.answers[key]

    # ------------------------------------------------------------ resolution
    def resolve(self, key: str):
        if key in self.local:
            return self.local[key].value
        yield Need(key)
        if key in self.pref:   # injected at question start: no lookup turn needed
            hit = self.pref.pop(key)
            yield UseShared(key, hit)
            self.local[key] = LocalFact(hit.value, "memory", hit.confidence)
            return hit.value
        is_base = key.startswith(("datastore(", "protocol(", "route("))
        if self.tool_mode and self.arch.parallel_lookup:
            if is_base:
                yield from self._maybe_hypothesize(key)
                yield LLMStep("decide:kb_query+search", self.cfg.decide_out_tokens)
                hit = yield KBQuery(key)
                results = yield Tool("search", {"query": self._query(key)})
                if hit is not None:
                    self.local[key] = LocalFact(hit.value, "memory", hit.confidence)
                    return hit.value
                return (yield from self._resolve_base(key, results))
            hit = yield KBQuery(key)   # issued in the same turn as the first sub-step
            if hit is not None:
                self.local[key] = LocalFact(hit.value, "memory", hit.confidence)
                return hit.value
            return (yield from self._resolve_derived(key))
        shared = yield from self._try_shared(key)
        if shared is not None:
            return shared
        if is_base:
            yield from self._maybe_hypothesize(key)
            return (yield from self._resolve_base(key))
        return (yield from self._resolve_derived(key))

    def _try_shared(self, key: str):
        if self.share_mode == "none":
            return None
        if self.share_mode == "tool":
            yield LLMStep("decide:kb_query", self.cfg.decide_out_tokens)
            hit = yield KBQuery(key)
        else:
            hit = yield KBQuery(key, in_context=True)
        if hit is None:
            return None
        if self.share_mode == "transcript" and hit.evidence in ("doc", "hypothesis"):
            if self.rng.random() >= self.cfg.p_adopt_unverified:
                return None
        self.local[key] = LocalFact(hit.value, "memory", hit.confidence)
        return hit.value

    def _share(self, *claims: Claim):
        if self.tool_mode and not self.arch.piggyback_publish:
            yield LLMStep("decide:kb_publish", self.cfg.decide_out_tokens)   # one turn publishes the batch
        for claim in claims:
            yield Publish(claim, piggyback=self.tool_mode and self.arch.piggyback_publish)

    def _resolve_derived(self, key: str):
        arg = key[key.index("(") + 1:-1]
        if key.startswith("feature_datastores("):
            parents = [f"endpoint_datastore({e})" for e in self._feature_endpoints(arg)]
            values = {}
            for p in parents:
                values[p] = yield from self.resolve(p)
        else:
            route_key = f"route({arg})"
            svc = yield from self.resolve(route_key)
            kind = "datastore" if key.startswith("endpoint_datastore(") else "protocol"
            parents = [route_key, f"{kind}({svc})"]
            values = {route_key: svc}
            values[parents[1]] = yield from self.resolve(parents[1])
        yield LLMStep("derive", self.cfg.derive_out_tokens)
        value = derive(key, values)
        conf = 0.98 * min(self.local[p].confidence for p in parents if p in self.local) if parents else 0.5
        evidence = "derived"
        if any(self.local.get(p) and self.local[p].evidence in ("doc", "hypothesis") for p in parents):
            evidence = "doc"   # a derivation is only as strong as its weakest premise
        self.local[key] = LocalFact(value, evidence, conf, parents)
        yield from self._share(Claim(key, value, conf, evidence, deps=parents, dep_values=values, level=Level.L2))
        return value

    def _feature_endpoints(self, feature: str) -> list[str]:
        return self.feature_map[feature]

    def _query(self, key: str) -> str:
        kind, arg = key[:key.index("(")], key[key.index("(") + 1:-1]
        return f"{arg} {kind.upper()}" if kind != "route" else f"{arg} ROUTES"

    def _maybe_hypothesize(self, key: str):
        if self.rng.random() < self.cfg.p_hypothesis:
            guess = self._guess(key)
            yield LLMStep("hypothesize", 40)
            yield Publish(Claim(key, guess, 0.3, "hypothesis", level=Level.L0))

    def _resolve_base(self, key: str, results=None):
        c = self.cfg
        kind = key[:key.index("(")]
        arg = key[key.index("(") + 1:-1]
        if results is None:
            yield LLMStep("decide:search", c.decide_out_tokens)
            results = yield Tool("search", {"query": self._query(key)})
        paths = results.data or []
        code_path = self._pick_code_path(kind, arg, paths)
        if code_path is None:
            yield LLMStep("decide:list_files", c.decide_out_tokens)
            listing = yield Tool("list_files", {})
            code_path = self._pick_code_path(kind, arg, listing.data, strict=False)

        value = evidence = None
        conf, sources = 0.0, []
        if kind != "route" and DOC_FILE in paths and self.rng.random() < c.p_doc_first:
            doc, fresh = yield from self._read(DOC_FILE)
            yield LLMStep("extract", c.extract_out_tokens)
            v = parse_doc(key, doc.text)
            if fresh and self.arch.evict:
                self.add_context(FACT_NOTE_TOKENS - doc.tokens)
            if v is not None:
                value, evidence, conf = v, "doc", 0.55
                sources = [Source(DOC_FILE, doc.data["version"], None)]
            if value is not None and self.rng.random() >= c.p_verify_doc:
                code_path = None  # satisfied with the docs
        if code_path is not None:
            for candidate in self._route_candidates(code_path, kind):
                if candidate in self.harvested:
                    continue   # all of its facts are already noted locally, and key is not among them
                res, fresh = yield from self._read(candidate)
                if self.arch.harvest and fresh and res.data:
                    found = yield from self._harvest(candidate, res)
                    if key in found:
                        return self.local[key].value
                    continue
                yield LLMStep("extract", c.extract_out_tokens)
                parsed = parse_fact(key, res.text)
                if parsed is None:
                    continue
                v, span = parsed
                conf = 0.95
                if self.rng.random() < c.p_misread:
                    v, conf = self._wrong(kind, v), 0.9
                value, evidence = v, "code"
                sources = [Source(candidate, res.data["version"], span)]
                break
        if value is None:   # nothing found: fall back to a guess
            value, evidence, conf = self._guess(key), "hypothesis", 0.3
        self.local[key] = LocalFact(value, evidence, conf, paths=tuple(s.path for s in sources))
        yield from self._share(Claim(key, value, conf, evidence, sources=sources, level=Level.L2))
        return value

    def _harvest(self, path: str, res):
        """Read once, note everything: every fact in the file becomes a local (and shared) fact."""
        facts = harvest_facts(path, res.text)
        yield LLMStep("extract", self.cfg.extract_out_tokens + HARVEST_OUT_PER_FACT * max(0, len(facts) - 1))
        claims, found = [], set()
        for key, v, span in facts:
            conf = 0.95
            if self.rng.random() < self.cfg.p_misread:
                v, conf = self._wrong(key[:key.index("(")], v), 0.9
            found.add(key)
            if key in self.local and self.local[key].evidence in ("code", "memory"):
                continue
            self.local[key] = LocalFact(v, "code", conf, paths=(path,))
            claims.append(Claim(key, v, conf, "code", sources=[Source(path, res.data["version"], span)],
                                level=Level.L2))
        self.harvested.add(path)
        if self.arch.evict:   # keep the compact notes, drop the raw text
            self.seen_files.pop(path, None)
            self.add_context(FACT_NOTE_TOKENS * len(facts) - res.tokens)
        if claims:
            yield from self._share(*claims)
        return found

    # --------------------------------------------------------------- helpers
    def _read(self, path: str):
        """Read a file, unless its raw content is still in this branch's context. Returns (result, fresh)."""
        if path in self.seen_files:
            return self.seen_files[path], False
        yield LLMStep("decide:read_file", self.cfg.decide_out_tokens)
        res = yield Tool("read_file", {"path": path})
        if not self.arch.evict:
            self.seen_files[path] = res
        return res, True

    def _pick_code_path(self, kind, arg, paths, strict=True):
        for p in paths:
            if kind == "route" and p.startswith("api/routes_"):
                return p
            if kind == "datastore" and p == f"services/{arg}/config.py":
                return p
            if kind == "protocol" and p == f"services/{arg}/server.py":
                return p
        if not strict and kind == "route":
            return next((p for p in paths if p.startswith("api/routes_")), None)
        return None

    def _route_candidates(self, first: str, kind: str) -> list[str]:
        if kind != "route":
            return [first]
        # the endpoint may live in any routing file; agents try them in order
        return [first] + [p for p in self.route_files if p != first]

    def _guess(self, key: str) -> str:
        truth = self.oracle(key)
        if self.rng.random() < self.cfg.p_hypothesis_correct:
            return truth
        return self._wrong(key[:key.index("(")], truth)

    def _wrong(self, kind: str, v: str) -> str:
        pool = {"datastore": DATASTORES, "protocol": PROTOCOLS}.get(kind, self.services)
        return self.rng.choice([x for x in pool if x != v])
