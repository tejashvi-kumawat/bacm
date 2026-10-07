"""Ground-truth facts about a real Python code base, from static analysis (ast).

Base facts (each supported by one source line):
  defined_in(Sym)  -> repo-relative path of the file defining top-level class/function Sym
  base_of(Cls)     -> name of the first base class of Cls ("object" if none)
Derived facts (the dependency structure agents must reconstruct):
  base_file(Cls)     = defined_in(base_of(Cls))
  grandbase(Cls)     = base_of(base_of(Cls))
  family_files(Cls)  = sorted unique defined_in over Cls and all its in-repo ancestors
Counting facts (benchmark v2; need the class body, error-prone for models):
  n_methods(Cls)     -> number of functions/async functions defined directly in the class body
  method_total(Cls)  =  n_methods summed over Cls and all its in-repo ancestors
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

EXCLUDE_DIRS = ("test", "tests", "testing", "docs", "doc", "examples", "example", "benchmarks", "scripts",
                "tools", "build", "dist", "site-packages", ".github")


def is_source(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    parts = path.split("/")
    return not any(p in EXCLUDE_DIRS or p.startswith(("test_", ".")) for p in parts[:-1]) and \
        not parts[-1].startswith("test_") and parts[-1] != "conftest.py" and parts[-1] != "setup.py"


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


@lru_cache(maxsize=4096)
def class_method_counts(text: str) -> dict[str, tuple[int, int]]:
    """{top-level class name: (methods defined directly in its body, line number)} for one file's text."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            n = sum(isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) for b in node.body)
            out[node.name] = (n, node.lineno)
    return out


class FactTable:
    """Top-level definitions and class bases of a set of files."""

    def __init__(self, files: dict[str, str]):
        defs: dict[str, list[str]] = {}
        self.bases: dict[str, str] = {}
        self.def_line: dict[tuple[str, str], str] = {}   # (sym, path) -> supporting line
        self.base_line: dict[str, str] = {}
        self.classes: set[str] = set()
        self.n_methods: dict[str, int] = {}
        for path, text in files.items():
            if not is_source(path):
                continue
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                continue
            lines = text.splitlines()
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    defs.setdefault(node.name, []).append(path)
                    line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                    self.def_line[(node.name, path)] = line
                    if isinstance(node, ast.ClassDef):
                        self.classes.add(node.name)
                        self.n_methods[node.name] = sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                                                        for x in node.body)
                        b = next((n for n in (_base_name(x) for x in node.bases) if n), None)
                        self.bases[node.name] = b or "object"
                        self.base_line[node.name] = line
        # only unambiguous names are facts (a name defined in two files has no single answer)
        self.defined_in = {s: ps[0] for s, ps in defs.items() if len(ps) == 1}
        self.classes = {c for c in self.classes if c in self.defined_in}

    def in_repo(self, cls: str | None) -> bool:
        return bool(cls) and cls in self.defined_in and cls in self.classes

    def ancestors(self, cls: str, limit: int = 12) -> list[str]:
        out, cur = [], cls
        while (self.in_repo(self.bases.get(cur)) and self.bases[cur] != cur and len(out) < limit
               and self.bases[cur] not in out):   # base with the class's own name = an external class
            cur = self.bases[cur]
            out.append(cur)
        return out

    # ------------------------------------------------------------------ truth
    def value(self, key: str) -> str | None:
        kind, arg = key[:key.index("(")], key[key.index("(") + 1:-1]
        if kind == "defined_in":
            return self.defined_in.get(arg)
        if kind == "base_of":
            return self.bases.get(arg) if arg in self.classes else None
        if kind == "base_file":
            b = self.bases.get(arg)
            return self.defined_in.get(b) if self.in_repo(b) and arg in self.classes else None
        if kind == "grandbase":
            b = self.bases.get(arg)
            return self.bases.get(b) if self.in_repo(b) and arg in self.classes else None
        if kind == "family_files":
            if arg not in self.classes:
                return None
            return ",".join(sorted({self.defined_in[c] for c in [arg] + self.ancestors(arg)}))
        if kind == "n_methods":
            return str(self.n_methods[arg]) if arg in self.classes else None
        if kind == "method_total":
            if arg not in self.classes:
                return None
            return str(sum(self.n_methods[c] for c in [arg] + self.ancestors(arg)))
        return None


KEY_RE = re.compile(r"(defined_in|base_of|base_file|grandbase|family_files|n_methods|method_total)"
                    r"\(([A-Za-z_][A-Za-z0-9_]*)\)")
