"""Instrumented environment tools. Every call is recorded for redundancy analysis."""
from __future__ import annotations

from dataclasses import dataclass

from ..util import count_tokens
from .world import World


@dataclass
class ToolResult:
    text: str
    tokens: int
    data: object = None


class ToolBox:
    def __init__(self, world: World, tracker):
        self.world = world
        self.tracker = tracker

    def call(self, agent_id: str, op: str, args: dict, now: float) -> ToolResult:
        w = self.world
        required = {"search": "query", "read_file": "path"}.get(op)
        if required and required not in args:   # models sometimes omit a required argument
            msg = f"error: {op} needs the argument {required!r}"
            self.tracker.on_tool(agent_id, op, {}, None, count_tokens(msg), now)
            return ToolResult(msg, count_tokens(msg), None)
        for field in ("path", "query"):   # models sometimes send lists/numbers: answer with a clear error
            if field in args and not isinstance(args[field], str):
                msg = f"error: argument {field!r} must be a single string, got {type(args[field]).__name__}"
                self.tracker.on_tool(agent_id, op, {field: repr(args[field])}, None, count_tokens(msg), now)
                return ToolResult(msg, count_tokens(msg), None)
        if op == "list_files":
            paths = sorted(w.files)
            res = ToolResult("\n".join(paths), count_tokens("\n".join(paths)), paths)
            target_version = None
        elif op == "search":
            paths = w.search(args["query"])
            text = "\n".join(paths) if paths else "(no matches)"
            res = ToolResult(text, count_tokens(text), paths)
            target_version = None
        elif op == "read_file":
            path = args["path"]
            if path not in w.files:
                res = ToolResult(f"error: no such file {path}", 8, None)
                target_version = None
            else:
                fs = w.files[path]
                res = ToolResult(fs.content, count_tokens(fs.content), {"path": path, "version": fs.version})
                target_version = fs.version
        else:
            raise ValueError(f"unknown tool {op}")
        self.tracker.on_tool(agent_id, op, args, target_version, res.tokens, now)
        return res
