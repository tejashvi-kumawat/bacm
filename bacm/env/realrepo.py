"""Real open-source repositories as the environment ("RepoQA-Dyn").

The repository is checked out at a pinned release (rev_a). Agents answer questions about its
class hierarchy; ground truth comes from static analysis (env/pyfacts.py) of the *current* files.
In dynamic runs the repository is moved to a later real release (rev_b) mid-run: every source file
that changed between the two releases changes at once, exactly as a `git pull` would, and ground
truth is recomputed from the new code. Nothing is synthetic except the choice of questions.
"""
from __future__ import annotations

import builtins
import random
import re
import subprocess
from pathlib import Path

from .pyfacts import KEY_RE, FactTable, class_method_counts, is_source
from .world import Change, FileState

SCHEMA = """Facts use canonical keys (Sym = a top-level class or function name, Cls = a class name):
  defined_in(Sym)     -> repo-relative path of the file that defines Sym at top level
  base_of(Cls)        -> name of the first base class of Cls as written in its class statement ("object" if none)
  base_file(Cls)      =  defined_in(base_of(Cls))
  grandbase(Cls)      =  base_of(base_of(Cls))
  family_files(Cls)   =  sorted unique defined_in of Cls and of every ancestor class defined in this repository,
                         comma-separated"""
NOTE_PROMPT = ("List every fact in this file, one per line, exactly as `key = value`: defined_in(Name) = <this file's "
               "path> for every top-level class and function, and base_of(Class) = <first base class name, or "
               "object> for every top-level class. No other text.")
NOTE_RE = re.compile(r"(defined_in|base_of)\(([A-Za-z_][A-Za-z0-9_]*)\)\s*=\s*[`\"']?([A-Za-z0-9_./-]+)")

# ---- benchmark v2: adds counting facts, which need the class body (error-prone for models) ----
SCHEMA_V2 = SCHEMA.replace("""family_files(Cls)   =  sorted unique defined_in of Cls and of every ancestor class defined in this repository,
                         comma-separated""", """family_files(Cls)   =  sorted unique defined_in of Cls and of every ancestor class defined in this repository,
                         comma-separated
  n_methods(Cls)      -> number of `def` / `async def` statements exactly one indentation level inside the
                         top-level class Cls; assignments such as `alias = method`, nested classes' functions,
                         functions inside functions and inherited methods are NOT counted
  method_total(Cls)   =  n_methods summed over Cls and every ancestor class defined in this repository""")
NOTE_PROMPT_V2 = ("List every fact in this file, one per line, exactly as `key = value`: defined_in(Name) = <this "
                  "file's path> for every top-level class and function; base_of(Class) = <first base class name, or "
                  "object> for every top-level class; n_methods(Class) = <number of lines beginning with def or async "
                  "def exactly one indentation level inside the class; assignments like alias = method are not "
                  "counted> for every top-level class. No other text.")
# focused extraction (v2): structure from the file outline, counts from one class body at a time
NOTE_PROMPT_OUTLINE = ("Below is the outline of a Python file: its top-level class and def lines. List, one per line, "
                       "exactly as `key = value`: defined_in(Name) = <this file's path> for every top-level class and "
                       "function, and base_of(Class) = <first base class name as written, without any module prefix, or "
                       "object if it has no base> for every class. No other text.")
NOTE_RE_V2 = re.compile(r"(defined_in|base_of|n_methods)\(([A-Za-z_][A-Za-z0-9_]*)\)\s*=\s*[`\"']?([A-Za-z0-9_./-]+)")
DEF_RE = r"^(?:async\s+def|def|class)\s+{name}\b"
CLASS_RE = r"^class\s+{name}\s*(\((.*?)\))?\s*:"

# The benchmark: real releases with real refactors between them.
REPOS = {
    "httpx": ("https://github.com/encode/httpx.git", "0.18.0", "0.27.0"),
    "requests": ("https://github.com/psf/requests.git", "v2.25.0", "v2.32.0"),
    "flask": ("https://github.com/pallets/flask.git", "2.0.0", "3.0.0"),
    "click": ("https://github.com/pallets/click.git", "7.1", "8.1.0"),
    "rich": ("https://github.com/Textualize/rich.git", "v10.0.0", "v13.0.0"),
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          errors="replace").stdout


def ensure_clone(name: str, url: str, cache_dir: str) -> Path:
    path = Path(cache_dir) / name
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", url, str(path)], check=True)
    return path


_SNAPSHOTS: dict[tuple[str, str], dict[str, str]] = {}


def snapshot(repo: Path, rev: str) -> dict[str, str]:
    """Source files of the repository at a revision (cached per process)."""
    key = (str(repo), rev)
    if key not in _SNAPSHOTS:
        names = _git(repo, "ls-tree", "-r", "--name-only", rev).split("\n")
        _SNAPSHOTS[key] = {n: _git(repo, "show", f"{rev}:{n}") for n in names if n and is_source(n)}
    return _SNAPSHOTS[key]


def _first_base(bases: str | None) -> str:
    if not bases or not bases.strip():
        return "object"
    first = re.split(r"[,\[(]", bases.strip())[0].strip()
    first = first.split(".")[-1].strip()
    if not first or "=" in first:   # e.g. class C(metaclass=M):
        return "object"
    return first


class RealRepoWorld:
    doc_file = None

    def __init__(self, name: str, repo_path: Path, rev_a: str, rev_b: str | None, seed: int, bench: str = "v1"):
        self.bench = bench
        v2 = bench == "v2"
        self.schema = SCHEMA_V2 if v2 else SCHEMA
        self.agent_guide = "Read the source code to establish facts.\n" + self.schema
        self.answer_example = ('"src/pkg/models.py", "Base", "pkg/a.py,pkg/b.py" or "7"' if v2
                               else '"src/pkg/models.py", "Base" or "pkg/a.py,pkg/b.py"')
        self.note_prompt = NOTE_PROMPT_V2 if v2 else NOTE_PROMPT
        self.note_re = NOTE_RE_V2 if v2 else NOTE_RE
        self.name = f"real:{name}"
        self.repo_name, self.repo_path, self.rev_a, self.rev_b = name, repo_path, rev_a, rev_b
        self.rng = random.Random(seed)
        self.files = {p: FileState(p, t.splitlines()) for p, t in snapshot(repo_path, rev_a).items()}
        self.changes: list[Change] = []
        self._table: FactTable | None = None
        self._table_a = FactTable({p: f.content for p, f in self.files.items()})
        self._table_b = FactTable(snapshot(repo_path, rev_b)) if rev_b else None

    @classmethod
    def load(cls, repo_cfg, seed: int) -> "RealRepoWorld":
        path = ensure_clone(repo_cfg.name, repo_cfg.url, repo_cfg.cache_dir)
        return cls(repo_cfg.name, path, repo_cfg.rev_a, repo_cfg.rev_b, seed, repo_cfg.bench)

    # ------------------------------------------------------------ ground truth
    @property
    def table(self) -> FactTable:
        if self._table is None:
            self._table = FactTable({p: f.content for p, f in self.files.items()})
        return self._table

    def truth(self, key: str) -> str | None:
        return self.table.value(key) if self.valid_key(key) else None

    def is_base_key(self, key: str) -> bool:
        return key.startswith(("defined_in(", "base_of(", "n_methods("))

    def is_base(self, key: str) -> bool:
        return self.is_base_key(key) and self.truth(key) is not None

    def valid_key(self, key: str) -> bool:
        return bool(KEY_RE.fullmatch(key))

    def parents(self, key: str, get) -> list[str]:
        if not self.valid_key(key):
            return []
        kind, arg = key[:key.index("(")], key[key.index("(") + 1:-1]
        if kind in ("base_file", "grandbase"):
            b = get(f"base_of({arg})")
            if b is None:
                return [f"base_of({arg})", "base_of(None)"]
            return [f"base_of({arg})", f"defined_in({b})" if kind == "base_file" else f"base_of({b})"]
        if kind == "family_files":
            out = [f"defined_in({arg})", f"base_of({arg})"]
            b = get(f"base_of({arg})")
            if b and b not in ("object", arg):   # the chain continues (or leaves the repository) at b
                out += [f"defined_in({b})", f"family_files({b})"]
            return out
        if kind == "method_total":
            out = [f"n_methods({arg})", f"base_of({arg})"]
            b = get(f"base_of({arg})")
            if b and b not in ("object", arg):
                out += [f"defined_in({b})", f"method_total({b})"]
            return out
        return []

    def deps(self, key: str) -> list[str]:
        return [p for p in self.parents(key, self.truth) if "(None)" not in p]

    def dependency_closure(self, key: str, _depth: int = 0) -> set[str]:
        out = {key}
        if _depth < 15:
            for p in self.deps(key):
                out |= self.dependency_closure(p, _depth + 1)
        return out

    def combine(self, key: str, pv: dict[str, str]) -> str:
        if not self.valid_key(key):
            raise KeyError(key)
        kind, arg = key[:key.index("(")], key[key.index("(") + 1:-1]
        if kind == "base_file":
            return next(v for k, v in pv.items() if k.startswith("defined_in("))
        if kind == "grandbase":
            b = pv[f"base_of({arg})"]
            return pv[f"base_of({b})"]
        if kind == "family_files":
            files = {pv[f"defined_in({arg})"]}
            for k, v in pv.items():
                if k.startswith("family_files("):
                    files |= set(v.split(","))
            return ",".join(sorted(files))
        if kind == "method_total":
            return str(int(pv[f"n_methods({arg})"]) + sum(int(v) for k, v in pv.items()
                                                          if k.startswith("method_total(")))
        raise KeyError(key)

    def normalize(self, value: str) -> str:
        v = value.strip().strip("`\"'").strip()
        if v.replace(".", "", 1).isdigit():   # counts: "7", "7.0" -> "7"
            return str(int(float(v)))
        parts = [p.strip().strip("`\"'").removeprefix("./") for p in v.split(",") if p.strip()]
        return ",".join(sorted(set(parts))) if len(parts) > 1 else (parts[0] if parts else "")

    # ------------------------------------------------------------ source parsing (tool layer)
    def parse(self, key: str, path: str, text: str):
        m = KEY_RE.fullmatch(key)
        if not m or m.group(1) not in ("defined_in", "base_of", "n_methods"):
            return None
        kind, name = m.group(1), m.group(2)
        if kind == "n_methods":   # supporting span = the whole class body
            hit = class_method_counts(text).get(name)
            if hit is None:
                return None
            return str(hit[0]), (self.class_block(text, name) or "")
        if kind == "defined_in":
            pat = re.compile(DEF_RE.format(name=re.escape(name)))
            line = next((ln for ln in text.splitlines() if pat.match(ln)), None)
            return (path, line) if line else None
        pat = re.compile(CLASS_RE.format(name=re.escape(name)))
        for ln in text.splitlines():
            mm = pat.match(ln)
            if mm:
                return _first_base(mm.group(2)), ln
        return None

    def harvest(self, path: str, text: str):
        out = []
        for ln in text.splitlines():
            m = re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z][A-Za-z0-9_]*)", ln)
            if not m:
                continue
            out.append((f"defined_in({m.group(1)})", path, ln))
            if ln.startswith("class "):
                b = self.parse(f"base_of({m.group(1)})", path, ln)
                if b:
                    out.append((f"base_of({m.group(1)})", b[0], ln))
        if self.bench == "v2":
            lines = text.splitlines()
            for name, (n, lineno) in class_method_counts(text).items():
                out.append((f"n_methods({name})", str(n), self.class_block(text, name) or ""))
        return out

    def note_normalize(self, text: str) -> str:
        return text

    def note_facts(self, note: str, path: str, lines: list[str]) -> list[tuple[str, str, str | None]]:
        out, text = [], "\n".join(lines)
        for m in self.note_re.finditer(note):
            key, value = f"{m.group(1)}({m.group(2)})", self.normalize(m.group(3))
            out.append((key, value, self.note_span(key, value, path, lines) if text else None))
        return out

    def note_span(self, key: str, value: str, path: str, lines: list[str]):
        parsed = self.parse(key, path, "\n".join(lines))
        return parsed[1] if parsed and parsed[0] == value else None

    def locate(self, key: str) -> list[str]:
        m = KEY_RE.fullmatch(key)
        if not m:
            return []
        pat = re.compile(DEF_RE.format(name=re.escape(m.group(2))))
        return [p for p, f in sorted(self.files.items()) if any(pat.match(ln) for ln in f.lines)]

    def wrong_value(self, key: str, value: str, rng) -> str:
        if value.isdigit():
            return str(max(0, int(value) + rng.choice([-2, -1, 1, 2])))
        pool = sorted(set(self.table.defined_in.values()) if key.startswith("defined_in(")
                      else self.table.classes | {"object"})
        pool = [v for v in pool if v != value]
        return rng.choice(pool) if pool else value

    def extract_prompt(self, key: str, path: str, text: str) -> str:
        if self.bench == "v2":   # focused view: one class body or the outline, not the whole file
            view = self.focus_text(key, path, text)
            return f"{self.schema}\n\nFrom {path}:\n{view}\n\nWhat is {key}? Reply with only the value."
        return f"{SCHEMA}\n\nFile {path}:\n{text}\n\nWhat is {key}? Reply with only the value."

    # ---------------------------------------------- focused views (span-level provenance for extraction)
    @staticmethod
    def outline(text: str) -> str:
        """Top-level class/def lines of a file (what an editor's outline / ctags shows)."""
        return "\n".join(ln for ln in text.splitlines() if re.match(r"^(class|def|async\s+def)\s", ln))

    @staticmethod
    def class_block(text: str, name: str) -> str | None:
        """Source of one top-level class: its header line(s) up to the next top-level statement."""
        lines = text.splitlines()
        start = next((i for i, ln in enumerate(lines) if re.match(rf"^class\s+{re.escape(name)}\b", ln)), None)
        if start is None:
            return None
        i = start
        while i < len(lines) and not lines[i].rstrip().endswith(":"):   # multi-line class header
            i += 1
        j = i + 1
        while j < len(lines):
            ln = lines[j]
            if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith("#"):
                break
            j += 1
        return "\n".join(lines[start:j]).rstrip()

    def focus_text(self, key: str, path: str, text: str) -> str:
        """The smallest view of a file that supports `key` (used for model-based extraction)."""
        kind, arg = key[:key.index("(")], key[key.index("(") + 1:-1]
        if kind in ("n_methods", "base_of"):
            return self.class_block(text, arg) or self.outline(text)
        return self.outline(text)

    def count_prompt(self, cls: str, block: str) -> str:
        return (f"Here is the source of the Python class `{cls}`:\n\n{block}\n\nHow many methods does it define? "
                f"Rule: count only lines that begin with `def ` or `async def ` exactly one indentation level inside the class; assignments such as `alias = method`, properties created by assignment, nested classes and functions inside functions are not counted. Reply with only the number.")

    @staticmethod
    def is_builtin(name: str) -> bool:
        return hasattr(builtins, name) and isinstance(getattr(builtins, name), type)

    def reanswer_prompt(self, key: str, facts: dict[str, str]) -> str:
        known = "\n".join(f"{k} = {v}" for k, v in sorted(facts.items())) or "(none)"
        return (f"{self.schema}\n\nVerified facts about the current version of the repository:\n{known}\n\n"
                f"Question: {self.question_text(key)}\nA fact that is not listed and that no file defines belongs to "
                f"an external class and ends a chain. Reply with only the answer value.")

    def derive_prompt(self, key: str, parent_values: dict[str, str]) -> str:
        facts = "\n".join(f"{k} = {v}" for k, v in parent_values.items())
        return f"{SCHEMA}\n\nKnown facts:\n{facts}\n\nWhat is {key}? Reply with only the value."

    # ------------------------------------------------------------ files
    def lines(self, path: str) -> list[str]:
        fs = self.files.get(path)
        return fs.lines if fs else []

    def version(self, path: str) -> int:
        fs = self.files.get(path)
        return fs.version if fs else -1

    def search(self, query: str, limit: int = 5) -> list[str]:
        terms = [t for t in re.split(r"[^A-Za-z0-9_/.-]+", query) if t]
        scored = []
        for path, fs in self.files.items():
            text = fs.content
            score = sum(3 * (t.lower() in path.lower()) + min(text.count(t), 5) for t in terms)
            if score:
                scored.append((-score, path))
        scored.sort()
        return [p for _, p in scored[:limit]]

    def initial_context(self) -> str:
        return "\n".join([f"Repository: {self.repo_name} (release {self.rev_a})", "Python source files:"]
                         + [f"  {p}" for p in sorted(self.files)])

    # ------------------------------------------------------------ tasks
    def candidate_classes(self) -> list[str]:
        """Classes with an in-repo base class (so derived questions need >= 2 facts), present in both
        releases for dynamic runs."""
        a = self._table_a
        out = [c for c in sorted(a.classes) if a.in_repo(a.bases.get(c))]
        if self._table_b is not None:
            b = self._table_b
            out = [c for c in out if c in b.classes and b.in_repo(b.bases.get(c))]
        if self.bench == "v2":   # a base with the class's own name is an external class, not a repository chain
            out = [c for c in out if a.bases[c] != c and (self._table_b is None or self._table_b.bases.get(c) != c)]
        return out

    def question_types(self, task_cfg):
        if self.bench == "v2":
            return self._question_types_v2()
        keys = [f"{k}({c})" for c in self.candidate_classes() for k in ("base_file", "grandbase", "family_files")]
        keys = [k for k in keys if self._table_a.value(k) is not None]
        if self._table_b is None:
            return [(1.0, keys)]
        changed = [k for k in keys if self._table_b.value(k) not in (None, self._table_a.value(k))]
        same = [k for k in keys if k not in changed]
        return [(0.5, changed), (0.5, same)]   # half the questions are affected by the release change

    V2_KINDS = ("base_file", "grandbase", "family_files", "n_methods", "method_total")

    def _question_types_v2(self):
        """Five question kinds with equal weight. In dynamic runs half of the questions of each kind are about
        facts that change between the two releases and half are about facts that do not."""
        a, b = self._table_a, self._table_b
        out = []
        for kind in self.V2_KINDS:
            keys = [f"{kind}({c})" for c in self.candidate_classes()]
            keys = [k for k in keys if a.value(k) is not None]
            if b is None:
                out.append((1.0 / len(self.V2_KINDS), keys))
                continue
            changed = [k for k in keys if b.value(k) not in (None, a.value(k))]
            same = [k for k in keys if k not in changed]
            out += [(0.5 / len(self.V2_KINDS), changed), (0.5 / len(self.V2_KINDS), same)]
        return out

    def question_text(self, key: str) -> str:
        kind, arg = key[:key.index("(")], key[key.index("(") + 1:-1]
        return {
            "n_methods": f"How many methods does the class `{arg}` define? Rule: count only lines that begin with `def ` or `async def ` exactly one indentation level inside the class; assignments such as `alias = method`, properties created by assignment, nested classes and functions inside functions are not counted.",
            "method_total": f"What is the sum of the method counts of `{arg}` and of each of its ancestor classes "
                            f"that live in this repository? Method count rule for each class: count only lines that begin with `def ` or `async def ` exactly one indentation level inside the class; assignments such as `alias = method`, properties created by assignment, nested classes and functions inside functions are not counted.",
            "base_file": f"In which file is the first base class of `{arg}` defined?",
            "grandbase": f"What is the first base class of the first base class of `{arg}`?",
            "family_files": f"Which files (sorted, comma-separated) define `{arg}` and each of its ancestor "
                            f"classes that live in this repository?",
        }.get(kind, f"What is {key}?")

    # ------------------------------------------------------------ dynamics
    def plan_changes(self, subtasks, dyn, seed):
        return [("release", self.rev_b)] if (self.rev_b and dyn.n_changes) else []

    def apply_change(self, item) -> list[Change]:
        """Move the working tree to a later real release: every changed source file changes at once."""
        _, rev = item
        new = snapshot(self.repo_path, rev)
        out = []
        for path in sorted(set(self.files) | set(new)):
            old_fs = self.files.get(path)
            if path not in new:
                out.append(Change(path, old_fs.version, -1, None, None, None))
                del self.files[path]
            elif old_fs is None:
                self.files[path] = FileState(path, new[path].splitlines())
                out.append(Change(path, 0, 1, None, None, None))
            elif old_fs.content != new[path]:
                old_fs.lines = new[path].splitlines()
                out.append(Change(path, old_fs.version, old_fs.version + 1, None, None, None))
                old_fs.version += 1
        self._table = None
        self.changes += out
        return out
