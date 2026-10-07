"""A real LLM agent (Claude, native tool use) running inside the same orchestrator,
environment, memory architectures and metrics as the simulated agent.

Provenance is captured by the tool layer, not trusted from the model: when the
agent publishes a base fact, the supporting file/version/line is located among
the files this branch actually read.
"""
from __future__ import annotations

import json

from ..core.knowledge import Claim, Level, Source
from .actions import Answer, KBQuery, LLMCall, Need, Note, Prefetch, Publish, Tool, UseShared

SYSTEM = """You are agent {aid}, one of several agents investigating the same software repository in parallel.
Answer each of your questions by inspecting the repository with the tools. {guide}
{sharing}
Call `answer` once per question with the bare value (e.g. {example}). If you receive an
invalidation notice, re-check the affected facts and call `answer` again for any question that depended on them.
Stop calling tools when every question is answered.

{c0}
"""

SHARING_TOOL = """A shared knowledge base holds facts other agents have verified. Before reading files for a fact,
call kb_query with its key. After you establish a fact (base or derived), call kb_publish so others can reuse it."""
FAST_HINTS = {
    "parallel_lookup": "Issue kb_query in the same turn as the search you would otherwise need (parallel tool calls).",
    "piggyback_publish": "Call kb_publish in the same turn as your next action, not in a separate turn.",
    "harvest": "When you read a code file, kb_publish every fact you can see in it, not only the one you need.",
}
SHARING_AUTO = """Shared memory works automatically; you have no memory tools. Tool results may end with [memory] lines:
facts other agents already established, questions the system has already resolved for you (do not answer those),
and which facts a question still needs. Reading a file another agent already read returns a short digest of the
facts relevant to you instead of the raw text; pass full=true to read_file if you need the raw text. A fact that no
repository file defines belongs to an external class (it ends a chain). Do not re-verify what [memory] reports;
spend your effort on the facts a question still needs."""
SHARING_TRANSCRIPT = """You can see the full shared log of every other agent's actions below; you may rely on it."""


def _tool(name, desc, props, required):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": required,
                             "additionalProperties": False}}


BASE_TOOLS = [
    _tool("list_files", "List all repository files.", {}, []),
    _tool("search", "Find files relevant to a query (returns paths).", {"query": {"type": "string"}}, ["query"]),
    _tool("read_file", "Read a repository file.", {"path": {"type": "string"}}, ["path"]),
    _tool("answer", "Submit the answer to one of your questions.",
          {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
]
AUTO_TOOLS = [
    BASE_TOOLS[0], BASE_TOOLS[1],
    _tool("read_file", "Read a repository file. Files another agent already read return a digest of the facts "
          "relevant to you; set full=true for the raw text.",
          {"path": {"type": "string"}, "full": {"type": "boolean"}}, ["path"]),
    BASE_TOOLS[3],
]
KB_TOOLS = [
    _tool("kb_query", "Look up a fact key in the shared knowledge base.", {"key": {"type": "string"}}, ["key"]),
    _tool("kb_publish", "Publish a fact you established.",
          {"key": {"type": "string"}, "value": {"type": "string"},
           "depends_on": {"type": "array", "items": {"type": "string"}},
           "confidence": {"type": "number"}}, ["key", "value"]),
]


class LLMAgent:
    def __init__(self, aid, questions, cfg, share_mode, c0, world):
        self.id = aid
        self.world = world   # used only by the tool layer (provenance), never shown to the model
        self.questions = questions
        self.cfg = cfg
        self.share_mode = share_mode
        self.arch = cfg.arch
        self.transparent = bool(self.arch.transparent)
        sharing = {"tool": SHARING_TOOL, "transcript": SHARING_TRANSCRIPT}.get(share_mode, "")
        if self.transparent:
            sharing = SHARING_AUTO
        elif share_mode == "tool":
            sharing += "\n" + "\n".join(h for f, h in FAST_HINTS.items() if getattr(self.arch, f))
        # dropping old file text from the history is unsafe with preserved-thinking models (Anthropic backend)
        self.evict = self.arch.evict and cfg.backend != "anthropic"
        self.read_ids: dict[str, str] = {}   # tool_use_id -> path, for eviction
        self.notes: dict[str, list] = {}     # path -> facts noted from it (kept when raw text is evicted)
        self.prefetched = False
        self.system = SYSTEM.format(aid=aid, sharing=sharing, c0=c0, guide=world.agent_guide,
                                    example=world.answer_example)
        self.tools = AUTO_TOOLS if self.transparent else BASE_TOOLS + (KB_TOOLS if share_mode == "tool" else [])
        qtext = "\n".join(f"- [{q.key}] {q.text}" for q in questions)
        self.messages = [{"role": "user", "content": f"Your questions:\n{qtext}"}]
        self.answers: dict[str, str] = {}
        self.reads: dict[str, tuple[int, str]] = {}
        self.inbox: dict[str, str | None] = {}
        self.steps = 0

    def reset(self) -> None:
        self.messages = self.messages[:1]
        self.answers.clear()
        self.reads.clear()
        self.inbox.clear()

    def context_tokens(self) -> int:
        return 0

    def add_context(self, tokens: int) -> None:
        pass

    def _all_answered(self) -> bool:
        return all(q.key in self.answers for q in self.questions)

    def run(self):
        if self.share_mode == "tool" and self.arch.prefetch and not self.prefetched and not self.transparent:
            self.prefetched = True
            keys = [q.key for q in self.questions]
            hits = yield Prefetch(keys + [f"route({q.key[q.key.index('(') + 1:-1]})" for q in self.questions
                                          if not q.key.startswith("feature")])
            if hits:
                for h in hits:
                    yield Need(h.key)
                    yield UseShared(h.key, h)
                self.messages[0]["content"] += ("\n\nAlready-verified shared knowledge relevant to your questions:\n"
                                                + "\n".join(h.text for h in hits))
        while self.steps < self.cfg.max_llm_steps_per_agent:
            if self.transparent and not self.inbox and self._all_answered():
                return   # the manager resolved everything: no further LLM turn is needed
            if self.inbox:
                items = sorted(self.inbox.items())
                self.inbox.clear()
                for q in self.questions:
                    if q.key in dict(items):
                        self.answers.pop(q.key, None)
                lines = [f"{k} = {v} (verified current value)" if v is not None else f"{k}: no longer valid, re-check"
                         for k, v in items]
                self._append_user("[L3 update notice] Shared knowledge you relied on changed:\n" + "\n".join(lines)
                                  + "\nRe-answer only the questions whose answer this changes; leave the others.")
            self.steps += 1
            content, stop = yield LLMCall(self.system, self.messages, self.tools)
            self.messages.append({"role": "assistant", "content": content})
            uses = [b for b in content if getattr(b, "type", None) == "tool_use"]
            if not uses:
                if all(q.key in self.answers for q in self.questions) or stop == "refusal":
                    return
                self._append_user("Some questions are still unanswered; continue.")
                continue
            results = []
            for b in uses:
                try:
                    out = yield from self._exec(b.name, b.input if isinstance(b.input, dict) else json.loads(b.input))
                except (KeyError, TypeError, ValueError) as e:   # malformed arguments from the model
                    out = f"error: bad arguments for {b.name}: {e}"
                if b.name == "read_file" and not out.startswith("error") and not out.startswith("[memory digest"):
                    self.read_ids[b.id] = b.input.get("path", "")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            if self.evict:
                self._evict_old_reads()
            self.messages.append({"role": "user", "content": results})

    def _evict_old_reads(self) -> None:
        """Replace raw file text from earlier turns with a stub; the facts live in shared/local knowledge
        and provenance says where to re-read if needed."""
        for m in self.messages:
            if m["role"] != "user" or not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result" and b["tool_use_id"] in self.read_ids:
                    path = self.read_ids.pop(b["tool_use_id"])
                    ver = self.reads.get(path, (None,))[0]
                    noted = ", ".join(f"{k} = {v}" for k, v, _ in self.notes.get(path, []))
                    b["content"] = (f"[raw text of {path}@v{ver} removed from context. Facts noted from it: "
                                    f"{noted or 'none'}. Re-read the file only if you need something else.]")

    def _append_user(self, text):
        last = self.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append({"type": "text", "text": text})
        else:
            self.messages.append({"role": "user", "content": text})

    def _exec(self, name, args):
        if name in ("list_files", "search", "read_file"):
            res = yield Tool(name, args)
            if name == "read_file" and res.data:
                path, ver = res.data["path"], res.data["version"]
                self.reads[path] = (ver, res.text)
                if self.transparent:
                    self.notes[path] = res.data.get("facts", [])
                elif self.arch.harvest and path != self.world.doc_file:
                    facts = yield Note(path, ver, res.text)
                    self.notes[path] = facts
                    for key, value, span in facts:
                        yield Publish(Claim(key, value, 0.95, "code", sources=[Source(path, ver, span)],
                                            level=Level.L2), piggyback=False)
            return res.text
        if name in ("kb_query", "kb_publish") and not self.world.valid_key(str(args.get("key", ""))):
            return (f"error: {args.get('key')!r} is not a canonical fact key. Use the key forms listed in your "
                    "instructions, e.g. the ones in your questions.")
        if name == "kb_query":
            yield Need(args["key"])
            hit = yield KBQuery(args["key"])
            return hit.text if hit else "no valid shared knowledge for this key"
        if name == "kb_publish":
            claim = self._claim(args["key"], self.world.normalize(args["value"]), args.get("depends_on") or [],
                                float(args.get("confidence", 0.9)))
            kid = yield Publish(claim)
            return f"published as {kid}" if kid else "not admitted to shared memory (kept local)"
        if name == "answer":
            key, value = args["key"], self.world.normalize(args["value"])
            if key not in {q.key for q in self.questions}:
                return f"error: {key!r} is not one of your questions; answer with the exact key in [brackets]"
            if self.share_mode != "tool":   # independent / transcript agents still feed the reuse oracle
                yield Publish(self._claim(key, value, [], 0.9))
            out = yield Answer(key, value)
            if self.transparent and out:   # the manager records the verified value, not necessarily ours
                final, message = out
                self.answers[key] = final
                return message
            self.answers[key] = value
            return "recorded"
        return f"unknown tool {name}"

    def _claim(self, key, value, deps, confidence) -> Claim:
        sources, evidence = [], "derived" if deps else "hypothesis"
        if self.world.is_base_key(key):
            doc = self.world.doc_file
            for path, (ver, text) in self.reads.items():
                parsed = self.world.parse(key, path, text) if path != doc else None
                if parsed and parsed[0] == value:
                    sources, evidence = [Source(path, ver, parsed[1])], "code"
                    break
            else:
                if doc and doc in self.reads and self.world.parse_doc(key, self.reads[doc][1]) == value:
                    sources, evidence = [Source(doc, self.reads[doc][0])], "doc"
        conf = min(confidence, 0.55) if evidence in ("doc", "hypothesis") else confidence
        return Claim(key, value, conf, evidence, sources=sources, deps=list(deps), level=Level.L2)
