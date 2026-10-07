"""The proposed Branch-Aware Incremental Context Manager (design-document sections 5-12).

Responsibilities:
  * Query & reuse engine     - answer lookups (and question-level prefetches) from valid knowledge
  * Provenance tracking      - who/which branch/which source@version/which line produced a fact
  * Dependency graph G=(K,E) - which knowledge was derived from which
  * Incremental invalidation - on a source change, mark only the affected subgraph STALE
  * Incremental recompute    - a small fresh-context worker re-derives exactly that subgraph and
                               pushes the corrected values (L3 update) to the branches that used them
  * Selective sync (L0-L3)   - private thoughts stay local; only admitted claims are shared
  * Knowledge admission      - S(K) = w1 R + w2 C + w3 V + w4 D - w5 U  > tau; weak claims that are
                               refused are verified by the worker instead of silently dropped
  * Conflict resolution      - contradictory claims are kept with provenance, then resolved

Every component can be switched off via ArchConfig for ablations.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

from ..memory.base import Hit, Memory
from ..util import count_tokens
from .knowledge import EVIDENCE_STRENGTH, Claim, Knowledge, Level, Source, Status
from .worker import SYSTEM_AGENT, Worker

def _kid_order(kid: str) -> int:
    return int(kid[1:])


ADMISSION_WEIGHTS = {"R": 0.15, "C": 0.35, "V": 0.35, "D": 0.15, "U": 0.20}
CONFLICT_MARGIN = 0.15


class ContextManager(Memory):
    supports_query = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.k: dict[str, Knowledge] = {}
        self.by_key: dict[str, list[str]] = defaultdict(list)
        self.by_path: dict[str, set[str]] = defaultdict(set)
        self.dependents: dict[str, set[str]] = defaultdict(set)
        self.demand: Counter = Counter()
        self._producer: dict[str, str] = {}
        self._told: dict[str, set[str]] = {}
        self._n = 0
        self.worker: Worker | None = None
        self.busy_until = 0.0

    def configure(self, agent_cfg, seed, llm=None):
        if self.arch.worker_recompute or self.arch.weak_verify or self.arch.cross_check:
            indexed = self.arch.transparent and not self.arch.llm_extraction   # symbol-index maintenance
            self.worker = Worker(self.world, self.toolbox, self.tracker, self.tracker.cost, agent_cfg, seed,
                                 None if indexed else llm, indexed)
            self.worker.effort = self.arch.manager_effort if self.arch.transparent else None

    # ------------------------------------------------------------ helpers
    @property
    def can_detect_changes(self) -> bool:
        return self.arch.versioning and self.arch.provenance

    def _new_id(self) -> str:
        self._n += 1
        return f"K{self._n}"

    def _verified(self, key: str) -> list[Knowledge]:
        return [self.k[i] for i in reversed(self.by_key.get(key, [])) if self.k[i].status == Status.VERIFIED]

    def _current(self, key: str, now: float) -> Knowledge | None:
        return next((k for k in self._verified(key) if self._validate(k, now)), None)

    def _span_present(self, span: str | None, path: str) -> bool:
        """Is the supporting span still in the file? A multi-line span (e.g. a whole class body supporting a
        method count) must still appear verbatim."""
        if not span:
            return False
        lines = self.world.lines(path)
        if "\n" not in span:
            return span in lines
        text = "\n".join(lines)
        i = text.find(span)
        while i >= 0:   # the block must still be complete: the next non-blank line is not part of it
            rest = text[i + len(span):]
            nxt = next((ln for ln in rest.split("\n")[1:] if ln.strip()), None)
            if (rest == "" or rest.startswith("\n")) and (nxt is None or not nxt[0].isspace()):
                return True
            i = text.find(span, i + 1)
        return False

    def _span_still_present(self, src: Source) -> bool:
        return bool(self.arch.span_provenance and self._span_present(src.span, src.path))

    def _subscribers(self, k: Knowledge) -> list[str]:
        # sorted: iteration order of sets varies with PYTHONHASHSEED; results must be reproducible
        return sorted((k.consumers | {self._producer[k.id]}) - {SYSTEM_AGENT})

    # ---------------------------------------------------------- validation
    def _validate(self, k: Knowledge, now: float, _seen: set | None = None) -> bool:
        """Lazy validation at reuse time: are the sources (and parents) still current?"""
        if k.status != Status.VERIFIED:
            return False
        if not self.can_detect_changes:
            return True
        for i, src in enumerate(k.sources):
            if src.version is None:
                continue
            cur = self.world.version(src.path)
            if cur != src.version:
                if self._span_still_present(src):
                    k.sources[i] = Source(src.path, cur, src.span)
                    k.revalidations += 1
                    self.tracker.on_revalidation(k)
                else:
                    self._mark_stale([k], now, reason="lazy")
                    return False
        if self.arch.dependencies:
            _seen = _seen or set()
            for d in k.dep_ids:
                if d in _seen:
                    continue
                _seen.add(d)
                if not self._validate(self.k[d], now, _seen):
                    if k.status == Status.VERIFIED:
                        self._mark_stale([k], now, reason="lazy-dep")
                    return False
        return True

    def _mark_stale(self, roots: list[Knowledge], now: float, reason: str, notify: bool = False) -> list[Knowledge]:
        """Mark roots STALE and (with the dependency graph) everything downstream of them, BFS order."""
        stale: list[Knowledge] = []
        queue = deque(roots)
        seen = set()
        while queue:
            k = queue.popleft()
            if k.id in seen:
                continue
            seen.add(k.id)
            if k.status == Status.VERIFIED:
                k.status = Status.STALE
                stale.append(k)
                self.tracker.on_invalidation(k, reason, now)
            if self.arch.dependencies:
                queue.extend(self.k[d] for d in sorted(self.dependents.get(k.id, ()), key=_kid_order))
        if notify:
            for k in stale:
                for agent in self._subscribers(k):
                    self.notify(agent, k.key, None, now)
        return stale

    # --------------------------------------------------------------- query
    def _lookup(self, agent_id, key, now):
        k = self._current(key, now)
        if k is None:
            return None
        k.consumers.add(agent_id)
        k.n_reuses += 1
        if agent_id != self._producer[k.id]:
            k = self._cross_check(k, now) or k
        return Hit(key, k.value, k.evidence, k.confidence, k.render(), kid=k.id)

    def _cross_check(self, k: Knowledge, now: float) -> Knowledge | None:
        """First reuse by another branch: an independent re-read by the worker before the fact spreads.
        Agreement -> corroborated; disagreement -> conflict resolution (a third read decides)."""
        if not (self.arch.cross_check and self.worker and self.world.is_base(k.key)
                and self._producer[k.id] != SYSTEM_AGENT and not getattr(k, "corroborated", False)):
            return None
        hint = k.sources[0].path if k.sources else None
        res = self.worker.extract(k.key, now, hint)
        if res is None:
            return None
        value, src, lat = res
        if value == k.value:
            k.corroborated = True
            k.support.add(SYSTEM_AGENT + ":crosscheck")
            k.confidence = max(k.confidence, 0.99)
            return None
        other = self._store(SYSTEM_AGENT, k.key, value, "code", 0.95, [src], [], now)
        self._resolve_conflict([other, k], now + lat)
        win = self._current(k.key, now)
        if win is not None:
            win.corroborated = True
        return win

    def query(self, agent_id, key, now):
        self.demand[key] += 1
        return self._lookup(agent_id, key, now)

    # ----------------------------------------------------------- admission
    def admission_score(self, claim: Claim, n_missing_deps: int = 0) -> float:
        w = ADMISSION_WEIGHTS
        R = 1.0
        C = claim.confidence
        V = EVIDENCE_STRENGTH.get(claim.evidence, 0.5)
        if n_missing_deps:
            V *= 0.5   # derived from parents that were never admitted -> weak support
        D = min(1.0, self.demand[claim.key] / 2)
        U = 1.0 - C
        return w["R"] * R + w["C"] * C + w["V"] * V + w["D"] * D - w["U"] * U

    def _outdated(self, claim: Claim) -> bool:
        for s in claim.sources:
            if s.version is not None and s.version != self.world.version(s.path) and not (
                    self.arch.span_provenance and self._span_present(s.span, s.path)):
                return True
        return False

    def _tell(self, agent_id: str, key: str, now: float) -> None:
        """Point a branch at the current valid value of key (or just invalidate if none)."""
        cur = self._current(key, now)
        if cur is not None:
            cur.consumers.add(agent_id)
        self.notify(agent_id, key, cur.value if cur else None, now)

    # ------------------------------------------------------------- publish
    def publish(self, agent_id, claim, now):
        if claim.level == Level.L0 and self.arch.admission:
            return None   # private reasoning never leaves the branch
        if self.can_detect_changes and self._outdated(claim):
            # produced from a source version that has since changed: refuse, tell the branch the current value
            self.tracker.on_admission(agent_id, claim, False)
            self._tell(agent_id, claim.key, now)
            return None
        dep_ids, missing = [], 0
        if self.arch.dependencies:
            for dk in claim.deps:
                cur = self._current(dk, now)
                if cur is None:
                    missing += 1
                    stale_ids = {i for i in self.by_key.get(dk, ()) if self.k[i].status == Status.STALE}
                    if stale_ids - self._told.get(agent_id, set()):
                        # the branch derived from a premise known to be stale (tell it once per stale object)
                        self._told.setdefault(agent_id, set()).update(stale_ids)
                        self.notify(agent_id, dk, None, now)
                elif claim.dep_values and claim.dep_values.get(dk, cur.value) != cur.value:
                    missing += 1   # derived from a premise value the shared layer disagrees with
                    self._tell(agent_id, dk, now)
                else:
                    dep_ids.append(cur.id)
            if missing:
                # a derived claim is shared only if every premise is shared, valid and agrees:
                # otherwise it could never be invalidated when its premises change
                self.tracker.on_admission(agent_id, claim, False)
                return None
        if self.arch.admission:
            score = self.admission_score(claim, missing)
            admitted = score > self.arch.admission_threshold
            self.tracker.on_admission(agent_id, claim, admitted, score)
            if not admitted:
                self._verify_weak(agent_id, claim, now)
                return None
        else:
            self.tracker.on_admission(agent_id, claim, True)

        existing = [k for k in self._verified(claim.key) if self._validate(k, now)]
        for e in existing:
            if e.value == claim.value:   # corroboration: merge instead of duplicating
                if claim.evidence in ("code", "derived"):
                    e.support.add(agent_id)   # an independent observation of the same value
                e.confidence = max(e.confidence, claim.confidence)
                if EVIDENCE_STRENGTH.get(claim.evidence, 0) > e.verification:
                    e.evidence = claim.evidence
                e.consumers.add(agent_id)
                return e.id

        k = self._store(agent_id, claim.key, claim.value, claim.evidence, claim.confidence,
                        list(claim.sources), dep_ids, now)
        if existing:
            if self.arch.conflict:
                self._resolve_conflict([k] + existing, now)
            else:
                for e in existing:   # last write wins
                    e.status = Status.SUPERSEDED
        return k.id if k.status == Status.VERIFIED else None

    def _store(self, agent_id, key, value, evidence, confidence, sources, dep_ids, now) -> Knowledge:
        if not self.arch.provenance:
            sources = []
        if not self.arch.versioning:
            sources = [Source(s.path, None, None) for s in sources]
        k = Knowledge(
            id=self._new_id(), key=key, value=value, evidence=evidence, confidence=confidence,
            produced_by=agent_id if self.arch.provenance else None,
            branch=f"branch-{agent_id}" if self.arch.provenance else None,
            sources=sources, dep_ids=dep_ids, created_at=now,
        )
        self.k[k.id] = k
        k.support.add(agent_id)
        self._producer[k.id] = agent_id
        self.by_key[key].append(k.id)
        self._index(key)
        for s in sources:
            self.by_path[s.path].add(k.id)
        for d in dep_ids:
            self.dependents[d].add(k.id)
        self.tracker.on_knowledge_created(k)
        return k

    def _extract(self, key: str, now: float, hint: str | None = None):
        """Worker extraction for facts the manager itself will broadcast. With cross-checking,
        two independent reads must agree (a third breaks a tie): one misread must not be pushed to all."""
        first = self.worker.extract(key, now, hint)
        if first is None or not self.arch.cross_check:
            return first
        reads, lat = [first], first[2]
        while len(reads) < 3:
            values = [r[0] for r in reads]
            top = max(set(values), key=values.count)
            if values.count(top) >= 2:
                break
            r = self.worker.extract(key, now + lat, first[1].path)
            if r is None:
                break
            reads.append(r)
            lat += r[2]
        values = [r[0] for r in reads]
        top = max(set(values), key=values.count)
        best = next(r for r in reads if r[0] == top)
        return best[0], best[1], lat

    # ------------------------------------------------------ weak evidence
    def _verify_weak(self, agent_id: str, claim: Claim, now: float) -> None:
        """A refused doc/guess-based base claim is checked against code by the worker, and the
        branch is told the verified value - instead of the error silently staying in the branch."""
        if not (self.arch.weak_verify and self.worker and claim.level != Level.L0
                and claim.evidence in ("doc", "hypothesis") and self.world.is_base(claim.key)):
            return
        cur = self._current(claim.key, now)
        lat = 0.0
        if cur is None:
            res = self._extract(claim.key, now)
            if res is None:
                return
            value, src, lat = res
            cur = self._store(SYSTEM_AGENT, claim.key, value, "code", 0.95, [src], [], now)
        cur.consumers.add(agent_id)
        self.notify(agent_id, claim.key, cur.value, now + lat)

    # ------------------------------------------------------------ conflicts
    def _resolve_conflict(self, cands: list[Knowledge], now: float) -> None:
        for c in cands:
            c.status = Status.CONFLICT
        key = cands[0].key
        method = "evidence"

        def score(c: Knowledge) -> float:
            return 0.6 * c.verification + 0.4 * c.confidence

        ranked = sorted(cands, key=score, reverse=True)
        winner = None
        if score(ranked[0]) - score(ranked[1]) >= CONFLICT_MARGIN:
            winner = ranked[0]
        elif self.world.is_base(key) and self.arch.provenance and self.toolbox is not None:
            # equally credible claims -> trigger verification against the code source
            method = "verification"
            # provenance says where the claims came from; fall back to repository convention
            path = next((src.path for c in cands for src in c.sources if src.path != self.world.doc_file),
                        None) or (self.world.locate(key) or [None])[0]
            if self.worker is not None:
                winner = self._vote(cands, key, path, now)
                parsed = None
            else:
                res = self.toolbox.call(SYSTEM_AGENT, "read_file", {"path": path}, now)
                self.tracker.on_llm(SYSTEM_AGENT, res.tokens + 200, 40, now, purpose="verify")
                parsed = self.world.parse(key, path, res.text) if res.data else None
                version = res.data["version"] if res.data else None
            if parsed and winner is None:
                winner = next((c for c in cands if c.value == parsed[0]), None)
                if winner is None:   # nobody was right: record the verified value
                    winner = cands[0]
                    winner.value, winner.evidence, winner.confidence = parsed[0], "code", 0.99
                    winner.sources = [Source(path, version, parsed[1])]
        else:
            method = "recency"
            winner = max(cands, key=lambda c: c.created_at)
        for c in cands:
            c.status = Status.VERIFIED if c is winner else Status.REJECTED
        losers = [c for c in cands if c is not winner]
        # anyone who consumed a losing claim gets an L3 correction carrying the winning value
        for c in losers:
            for agent in self._subscribers(c):
                if winner is not None:
                    winner.consumers.add(agent)
                self.notify(agent, c.key, winner.value if winner else None, now)
            recompute = self.arch.worker_recompute and self.worker is not None
            down = self._mark_stale([self.k[d] for d in sorted(self.dependents.get(c.id, ()), key=_kid_order)],
                                    now, "conflict",
                                    notify=not recompute)
            if recompute and down:
                self._recompute(down, now)
        self.tracker.on_conflict(key, winner.value if winner else None, method, now)

    def _vote(self, cands: list[Knowledge], key: str, path: str, now: float) -> Knowledge:
        """Majority of independent observations. Each claim starts with one vote per independent
        branch that reported it; fallible verification reads add votes until one value leads by 2."""
        votes = {c.value: 0 for c in cands}
        for c in cands:
            votes[c.value] += len(c.support)
        by_value = {c.value: c for c in cands}
        t = now
        for _ in range(3):
            ranked = sorted(votes.values(), reverse=True) + [0]
            if ranked[0] - ranked[1] >= 2:
                break
            r = self.worker.extract(key, t, path)
            if r is None:
                break
            value, src, lat = r
            t += lat
            votes[value] = votes.get(value, 0) + 1
            if value not in by_value:   # nobody reported this value: record the read as a candidate
                nk = self._store(SYSTEM_AGENT, key, value, "code", 0.95, [src], [], t)
                nk.status = Status.CONFLICT
                cands.append(nk)
                by_value[value] = nk
        best = max(votes, key=lambda v: (votes[v], by_value[v].created_at))
        return by_value[best]

    # --------------------------------------------------------- source change
    def on_change(self, change, now):
        """Source change -> dependency lookup -> invalidate -> (recompute) -> notify (design-document s.12)."""
        if not (self.can_detect_changes and self.arch.invalidation):
            return   # without invalidation, lazy validation (if versioning) catches it at reuse time
        roots = []
        for kid in sorted(self.by_path.get(change.path, ()), key=_kid_order):
            k = self.k[kid]
            if k.status != Status.VERIFIED:
                continue
            for i, src in enumerate(k.sources):
                if src.path != change.path:
                    continue
                if self._span_still_present(src):
                    k.sources[i] = Source(src.path, change.new_version, src.span)
                    k.revalidations += 1
                    self.tracker.on_revalidation(k)
                else:
                    roots.append(k)
                    break
        recompute = self.arch.worker_recompute and self.worker is not None
        stale = self._mark_stale(roots, now, reason="eager", notify=not recompute)
        if recompute:
            self._recompute(stale, now)
        self.tracker.on_l3_event(change, len(stale), now)

    def subscribe(self, agent_id: str, key: str, now: float) -> None:
        k = self._current(key, now)
        if k is not None:
            k.consumers.add(agent_id)

    def set_questions(self, questions: dict[str, list[str]]) -> None:
        """Each branch's question keys (part of C_0), used to know which facts are in demand."""
        self.questions = questions

    def _known_value(self, key: str) -> str | None:
        ids = self.by_key.get(key)
        return self.k[ids[-1]].value if ids else None

    def _interest(self, agent: str) -> set[str]:
        """Facts this branch's questions depend on, following the latest known values."""
        keys: set[str] = set()
        frontier = list(getattr(self, "questions", {}).get(agent, []))
        while frontier and len(keys) < 500:
            k = frontier.pop()
            if k in keys:
                continue
            keys.add(k)
            frontier += [p for p in self.world.parents(k, self._known_value) if "(None)" not in p]
        return keys

    def _recompute(self, stale: list[Knowledge], now: float) -> None:
        """Incremental recompute of the affected subgraph, parents before children. With demand_driven,
        only facts that some branch's questions depend on are recomputed and announced."""
        t = max(now, self.busy_until)
        new_of: dict[str, Knowledge] = {}
        demand = self.arch.demand_driven and getattr(self, "questions", None)
        interest = {a: self._interest(a) for a in self.questions} if demand else {}

        def wanted(k: Knowledge) -> list[str]:
            subs = self._subscribers(k)
            return [a for a in subs if a not in interest or k.key in interest[a]] if demand else subs

        for old in stale:   # BFS order from the changed sources
            subs = wanted(old)
            if demand and not subs:
                continue   # nobody needs it: leave it STALE (recomputed on demand if someone asks)
            if self.world.is_base(old.key):
                hint = old.sources[0].path if old.sources else None
                res = self._extract(old.key, t, hint)
                if res is None:
                    continue
                value, src, lat = res
                t += lat
                nk = self._store(SYSTEM_AGENT, old.key, value, "code", 0.95, [src], [], t)
                nk.corroborated = self.arch.cross_check
            else:
                def get(key):   # a stale parent is rebuilt first, derived ones recursively (not only base facts)
                    nonlocal t
                    cur, t = self._ensure(key, t)
                    return cur
                def val(key):
                    cur = get(key)
                    return cur.value if cur else None
                try:
                    pkeys = self.world.parents(old.key, val)
                except KeyError:   # a key the model invented: nothing to rebuild
                    pkeys = [None]
                if not pkeys or None in pkeys:
                    continue
                parents = [None if "(None)" in p else get(p) for p in pkeys]
                if any(p is None for p in parents):
                    continue   # cannot rebuild yet: subscribers just get the invalidation
                value, lat = self.worker.derive(old.key, {p.key: p.value for p in parents}, t)
                t += lat
                nk = self._store(SYSTEM_AGENT, old.key, value, "derived", 0.95 * min(p.confidence for p in parents),
                                 [], [p.id for p in parents] if self.arch.dependencies else [], t)
            new_of[old.id] = nk
            for agent in subs:
                nk.consumers.add(agent)
                if nk.value != old.value:   # unchanged values need no message: the branch is still right
                    self.notify(agent, old.key, nk.value, t)
        for old in stale:
            if old.id not in new_of:
                for agent in wanted(old):
                    self.notify(agent, old.key, None, t)
        self.busy_until = t

    # ------------------------------------------------------- answer validation
    def _ensure(self, key: str, t: float, depth: int = 0):
        """Current valid knowledge for key, building it from shared premises if needed (worker, small
        context). Returns (Knowledge or None, time after the work)."""
        cur = self._current(key, t)
        if cur is not None or depth > 6:
            return cur, t
        if self.world.is_base(key):
            r = self._extract(key, t)
            if r is None:
                return None, t
            k = self._store(SYSTEM_AGENT, key, r[0], "code", 0.95, [r[1]], [], t + r[2])
            k.corroborated = self.arch.cross_check
            return k, t + r[2]
        memo: dict[str, Knowledge | None] = {}

        def get(pk):
            nonlocal t
            if pk not in memo:
                memo[pk], t = self._ensure(pk, t, depth + 1)
            return memo[pk]
        try:
            pkeys = self.world.parents(key, lambda pk: (get(pk).value if get(pk) else None))
        except KeyError:
            return None, t
        if not pkeys or any("(None)" in p for p in pkeys):
            return None, t
        parents = [get(p) for p in pkeys]
        if any(p is None for p in parents):
            return None, t
        value, lat = self.worker.derive(key, {p.key: p.value for p in parents}, t)
        t += lat
        k = self._store(SYSTEM_AGENT, key, value, "derived", 0.95 * min(p.confidence for p in parents), [],
                        [p.id for p in parents] if self.arch.dependencies else [], t)
        return k, t

    def validate_answer(self, agent_id: str, key: str, value: str, now: float) -> None:
        """Validate->Share for final answers to derived questions."""
        if not (self.arch.validate_answers and self.worker) or self.world.is_base(key):
            return
        k, t = self._ensure(key, max(now, self.busy_until))
        if k is None:
            return
        k.consumers.add(agent_id)   # the answer is now in the dependency graph: future changes reach it
        self.tracker.on_validation(agent_id, key, value, k.value)
        sent = self.__dict__.setdefault("_corrections", set())
        if k.value != value and (agent_id, key, k.value) not in sent:   # tell each branch a correction once
            sent.add((agent_id, key, k.value))
            self.notify(agent_id, key, k.value, t)

    def context_tokens(self, agent_id):
        return 0

    def snapshot(self) -> list[dict]:
        return [{"id": k.id, "key": k.key, "value": k.value, "status": k.status.value, "evidence": k.evidence,
                 "producer": k.produced_by, "sources": [(s.path, s.version) for s in k.sources],
                 "deps": k.dep_ids, "reuses": k.n_reuses, "tokens": count_tokens(k.render())}
                for k in self.k.values()]
