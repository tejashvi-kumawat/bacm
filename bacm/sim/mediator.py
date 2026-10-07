"""Transparent memory: the tool layer, not the LLM, talks to the shared memory.

With `ArchConfig.transparent` the model has no kb_query / kb_publish tools (those cost one full-context LLM
turn each). Instead this mediator, which sits between the agents and the memory:

  * publishes: every file an agent reads is turned into facts (symbol index, or an LLM note call) with
    provenance (file, version, supporting line) and published, attributed to the reader
  * serves reads: a file that some branch already read is returned as a short digest of only the facts relevant
    to the reader's open questions (the raw text is available with full=true)
  * resolves questions: when the facts a question needs are known, the manager computes the answer with the
    world's typed derivation rules - no LLM turn - and the agent is told so
  * plans: tool results say which facts a question still needs
  * maintains answers: when a source changes, the manager's incremental recompute updates the answers in
    place; only answers it cannot recompute (agent-derived ones) are handed back to the agent

The mediator never discovers new facts from the environment on its own: facts enter only from files an agent
read (and the manager re-reads, with provenance, to maintain them after a change).
"""
from __future__ import annotations

from ..agents.actions import Note
from ..core.knowledge import Claim, Level, Source
from ..env.tools import ToolResult
from ..util import count_tokens


class Mediator:
    def __init__(self, orch):
        self.o = orch
        self.arch = orch.cfg.arch
        self.world = orch.world
        self.memory = orch.memory
        self.tracker = orch.tracker
        self.qkeys: dict[str, list[str]] = {f"A{i}": [q.key for q in qs] for i, qs in enumerate(orch.subtasks)}
        self.digests: dict[str, tuple[int, list[tuple[str, str, str | None]]]] = {}
        self.verified: set[tuple[str, str]] = set()           # (agent, question) answered by the manager
        self.watch: dict[tuple[str, str], dict[str, str | None]] = {}   # unverified answers -> fact snapshot
        self.pending: dict[str, list[str]] = {a: [] for a in self.qkeys}
        self.last_hint: dict[str, str] = {}
        # focused model-based extraction (benchmark v2): structure from the outline, counts from one class body
        self.focused = bool(self.arch.llm_extraction and getattr(self.world, "bench", "") == "v2"
                            and hasattr(self.world, "class_block"))
        self.texts: dict[str, tuple[int, str]] = {}       # files agents have read (path -> (version, text))
        self.outline_notes: dict[tuple[str, int], list] = {}
        self.count_tried: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------ helpers
    @property
    def versioned(self) -> bool:
        return bool(getattr(self.memory, "can_detect_changes", False))

    def _agent(self, aid):
        return self.o.agents[aid]

    def open_questions(self, aid: str) -> list[str]:
        return [k for k in self.qkeys[aid] if k not in self._agent(aid).answers]

    def relevant_keys(self, aid: str, now: float, roots: list[str] | None = None) -> set[str]:
        """Facts the agent's open questions (or the given roots) depend on, following the values known so far."""
        keys: set[str] = set()
        frontier = list(roots if roots is not None else self.open_questions(aid))
        while frontier and len(keys) < 400:
            k = frontier.pop()
            if k in keys:
                continue
            keys.add(k)
            frontier += [p for p in self.world.parents(k, lambda x: self.memory.known_value(x, now))
                         if "(None)" not in p]
        return keys

    # ------------------------------------------------------------------ tools
    def tool(self, aid: str, action, now: float):
        """Execute a Tool action for an agent. Returns (ToolResult, latency)."""
        lat = self.o.cfg.cost.tool_latency
        if action.op == "read_file":
            res, dt = self._read(aid, action.args, now)
        else:
            res = self.o.toolbox.call(aid, action.op, action.args, now)
            dt = lat.get(action.op, 0.05)
        note = self.after_tool(aid, now)
        if note:
            res = ToolResult(res.text + "\n\n" + note, res.tokens + count_tokens(note), res.data)
        return res, dt

    def _read(self, aid: str, args: dict, now: float):
        w, lat = self.world, self.o.cfg.cost.tool_latency
        path = args.get("path", "")
        d = self.digests.get(path)
        fresh = d is not None and (not self.versioned or d[0] == w.version(path))
        if self.arch.serve_digest and fresh and not args.get("full"):
            facts = self._relevant_facts(aid, d[1], now)
            text = self._digest_text(path, d[0], facts, len(d[1]))
            toks = count_tokens(text)
            self.tracker.on_tool(aid, "kb_digest", {"path": path}, d[0], toks, now)
            self.tracker.digest_reads += 1
            return ToolResult(text, toks, {"path": path, "version": d[0], "facts": facts, "digest": True}), \
                lat["kb_query"]
        res = self.o.toolbox.call(aid, "read_file", args, now)
        dt = lat["read_file"]
        if res.data and path != getattr(w, "doc_file", None):
            ver = res.data["version"]
            if self.focused:
                self.texts[path] = (ver, res.text)
                facts, extra = self._note_outline(path, ver, res.text, now)
                dt += extra
            elif self.arch.llm_extraction:
                facts, extra = self.o._note(aid, Note(path, ver, res.text), now)
                dt += extra
            else:
                facts = w.harvest(path, res.text)
            self.digests[path] = (ver, facts)
            for key, value, span in facts:
                claim = Claim(key, value, 0.95, "code", sources=[Source(path, ver, span)], level=Level.L2)
                self.tracker.on_obtained(aid, claim)
                self.memory.publish(aid, claim, now)
            res = ToolResult(res.text, res.tokens, dict(res.data, facts=self._relevant_facts(aid, facts, now)))
        return res, dt

    # ------------------------------------------------- focused extraction
    def _effort(self) -> dict:
        """Manager calls see a small context, so a little reasoning there is cheap (agents keep the default)."""
        client = self.o.llm_client
        return {"effort": self.arch.manager_effort} if self.arch.manager_effort and hasattr(client, "reasoning_effort") else {}

    def _llm(self, prompt: str, now: float, purpose: str) -> tuple[str, float]:
        blocks, tin, tout, _ = self.o.llm_client.complete(
            "You extract facts from source code precisely.", [{"role": "user", "content": prompt}], None,
            **self._effort())
        self.tracker.on_llm("context-manager", tin, tout, now, purpose)
        text = " ".join(getattr(b, "text", "") or "" for b in blocks).strip()
        return text, getattr(self.o.llm_client, "last_latency", 0.0)

    def _note_outline(self, path: str, ver: int, text: str, now: float):
        """Structure facts (defined_in, base_of) from the file outline: one small call per file version."""
        key = (path, ver)
        if key in self.outline_notes:
            return self.outline_notes[key], 0.0
        from ..env.realrepo import NOTE_PROMPT_OUTLINE
        outline = self.world.outline(text)
        if not outline.strip():
            self.outline_notes[key] = []
            return [], 0.0
        out, dt = self._llm(f"{NOTE_PROMPT_OUTLINE}\n\nOutline of {path}:\n{outline}", now, "extract:outline")
        facts = [f for f in self.world.note_facts(out, path, text.splitlines()) if not f[0].startswith("n_methods(")]
        self.outline_notes[key] = facts
        return facts, dt

    def _extract_counts(self, now: float) -> None:
        """Method counts, extracted on demand from the body of one class, only for classes some open question
        needs and only from files an agent has read (the current version)."""
        if not self.focused:
            return
        needed = set()
        for aid in self.qkeys:
            needed |= {k for k in self.relevant_keys(aid, now) if k.startswith("n_methods(")}
        for key in sorted(needed):
            if self.memory.known_value(key, now) is not None:
                continue
            cls = key[len("n_methods("):-1]
            path = self.memory.known_value(f"defined_in({cls})", now)
            if path not in self.texts or self.texts[path][0] != self.world.version(path):
                continue
            ver, text = self.texts[path]
            if (key, ver) in self.count_tried:
                continue
            self.count_tried.add((key, ver))
            block = self.world.class_block(text, cls)
            if not block:
                continue
            out, _ = self._llm(self.world.count_prompt(cls, block), now, "extract:count")
            value = self.world.normalize(out.split()[0] if out.split() else "")
            if not value.isdigit():
                continue
            # the whole class body supports a method count (not just its header line)
            claim = Claim(key, value, 0.95, "code", sources=[Source(path, ver, block)],
                          level=Level.L2)
            self.memory.publish("context-manager", claim, now)

    def _relevant_facts(self, aid, facts, now):
        rel = self.relevant_keys(aid, now)
        return [f for f in facts if f[0] in rel]

    @staticmethod
    def _digest_text(path: str, version: int, facts: list, total: int) -> str:
        head = f"[memory digest of {path}@v{version}: another branch already read this file]"
        body = "\n".join(f"  {k} = {v}" for k, v, _ in facts) if facts else "  (no facts relevant to your open questions)"
        return (f"{head}\n{body}\n({len(facts)} of {total} facts shown. For the raw text call read_file with "
                f"full=true.)")

    # ----------------------------------------------------------- resolution
    def _set_answer(self, aid: str, key: str, value: str, now: float, verified: bool, revision: bool = False):
        ag = self._agent(aid)
        ag.answers[key] = value
        self.tracker.on_answer(aid, key, value, now)
        self.o.answered.add((aid, key))
        if verified:
            self.verified.add((aid, key))
        if revision:
            self.tracker.answer_revisions += 1

    def try_resolve_all(self, now: float) -> int:
        """Resolve every open question whose facts are known, for every agent (including unverified answers
        that the facts now contradict). Returns the number of newly resolved questions."""
        if not self.arch.auto_resolve:
            return 0
        n = 0
        for aid, keys in self.qkeys.items():
            ag = self._agent(aid)
            for key in keys:
                if (aid, key) in self.verified:
                    continue
                hit = self.memory.resolve_known(key, now)
                if hit is None:
                    continue
                prior = ag.answers.get(key)
                self._set_answer(aid, key, hit.value, now, verified=True, revision=prior is not None)
                self.watch.pop((aid, key), None)
                self.memory.subscribe(aid, key, now)
                if prior is None:
                    self.pending[aid].append(f"{key} = {hit.value}")
                    self.tracker.auto_resolved += 1
                    n += 1
                elif prior != hit.value:
                    self.pending[aid].append(f"{key} = {hit.value} (corrects your answer)")
                    self.tracker.answer_corrections += 1
        if n:
            self.o._maybe_apply_changes(now)
        return n

    def answer(self, aid: str, key: str, value: str, now: float):
        """The agent's `answer` call. Verified knowledge wins over the agent's own derivation."""
        hit = self.memory.resolve_known(key, now) if self.arch.auto_resolve else None
        if hit is not None:
            same = hit.value == value
            self._set_answer(aid, key, hit.value, now, verified=True)
            self.memory.subscribe(aid, key, now)
            self.o._maybe_apply_changes(now)
            if same:
                return hit.value, "recorded (confirmed by verified shared facts)"
            self.tracker.answer_corrections += 1
            return hit.value, (f"recorded, but verified shared facts give {key} = {hit.value}; that value was "
                               f"recorded instead of yours")
        self._set_answer(aid, key, value, now, verified=False)
        self.tracker.unverified_answers += 1
        self._watch(aid, key, now)
        self.o._maybe_apply_changes(now)
        return value, "recorded (not verifiable from shared facts yet)"

    def _watch(self, aid: str, key: str, now: float) -> None:
        """Track the facts an unverified answer rests on, so a change to them can send it back to the agent."""
        snap = {}
        for k in self.relevant_keys(aid, now) | {key}:
            if k != key:
                v = self.memory.known_value(k, now)
                if v is not None:
                    snap[k] = v
                    self.memory.subscribe(aid, k, now)
        if snap:
            self.watch[(aid, key)] = snap

    # --------------------------------------------------------------- notes
    def after_tool(self, aid: str, now: float) -> str:
        self._extract_counts(now)
        self.try_resolve_all(now)
        lines = []
        if self.pending[aid]:
            lines.append("[memory] Resolved from verified shared facts (no need to answer these): "
                         + "; ".join(self.pending[aid]))
            self.pending[aid] = []
        if self.arch.status_hints:
            hint = self._hint(aid, now)
            if hint and hint != self.last_hint.get(aid):
                lines.append(hint)
            self.last_hint[aid] = hint
        return "\n".join(lines)

    def _hint(self, aid: str, now: float) -> str:
        need = []
        for key in self.open_questions(aid)[:4]:
            # never send agents after Python built-ins (Exception, dict, ...): no repository file defines them
            is_builtin = getattr(self.world, "is_builtin", lambda n: False)
            m = [x for x in self.memory.missing_facts(key, now) if x != key
                 and not (x.startswith("defined_in(") and is_builtin(x[len("defined_in("):-1]))][:3]
            if m:
                need.append(f"{key} needs {', '.join(m)}")
        if not need:
            return ""
        return ("[memory] Still needed: " + "; ".join(need)
                + ". (A needed fact that no repository file defines belongs to an external class and ends the chain.)")

    # ------------------------------------------------------- notifications
    def reanswer(self, aid: str, key: str, now: float) -> bool:
        """Re-derive an answer in a small fresh context from the current verified facts (no agent wake-up)."""
        client = self.o.llm_client
        if client is None or not self.arch.reanswer_small or not hasattr(self.world, "reanswer_prompt"):
            return False
        facts = {k: v for k in self.relevant_keys(aid, now, roots=[key]) if k != key
                 and (v := self.memory.known_value(k, now)) is not None}
        if not facts:
            return False
        blocks, tin, tout, _ = client.complete(
            "You answer questions about a software repository from verified facts. Reply with only the value.",
            [{"role": "user", "content": self.world.reanswer_prompt(key, facts)}], None, **self._effort())
        self.tracker.on_llm("context-manager", tin, tout, now, "reanswer")
        text = " ".join(getattr(b, "text", "") or "" for b in blocks).strip()
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        value = self.world.normalize(first.split("=")[-1] if "=" in first else first)
        if not value:
            return False
        self._set_answer(aid, key, value, now, verified=False, revision=True)
        self.tracker.reanswers += 1
        self.verified.discard((aid, key))
        self._watch(aid, key, now)
        return True

    def on_updates(self, aid: str, updates: dict, now: float) -> tuple[dict, bool]:
        """Memory notifications for an agent. Returns (inbox updates for the agent, wake?)."""
        inbox: dict[str, str | None] = {}
        for key, val in updates.items():
            if key in self.qkeys[aid]:
                if val is not None and self.arch.maintain_answers:
                    self._set_answer(aid, key, val, now, verified=True, revision=True)
                    self.pending[aid].append(f"{key} = {val} (updated after a source change)")
                    continue
                if self.arch.maintain_answers and self.reanswer(aid, key, now):
                    continue
                self._agent(aid).answers.pop(key, None)
                self.verified.discard((aid, key))
                inbox[key] = val
            for (a, qk), snap in list(self.watch.items()):
                if a == aid and key in snap and snap[key] != val:
                    self.watch.pop((a, qk), None)
                    if self.arch.maintain_answers and self.reanswer(aid, qk, now):
                        continue
                    self._agent(aid).answers.pop(qk, None)
                    inbox[qk] = None
                    inbox[key] = val
        return inbox, bool(inbox)
