"""Synthetic, versioned software repository with known ground truth.

The repository contains:
  services/<svc>/config.py   ->  DATASTORE = "<db>"          base fact  datastore(<svc>)
  services/<svc>/server.py   ->  PROTOCOL = "<proto>"        base fact  protocol(<svc>)
  api/routes_<i>.py          ->  "/<endpoint>": "<svc>",     base facts route(/<endpoint>)
  docs/ARCHITECTURE.md       ->  outdated prose; some values are wrong (distractor evidence)

Derived facts (the dependency graph the agents must reconstruct):
  endpoint_datastore(/e) = datastore(route(/e))
  endpoint_protocol(/e)  = protocol(route(/e))
  feature_datastores(f)  = sorted unique endpoint_datastore(e) for e in feature f

Every file has a version that increments on edit, so knowledge produced from an
old version can be detected as stale.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..config import WorldConfig

DATASTORES = ["postgresql", "mongodb", "mysql", "redis", "cassandra", "dynamodb", "sqlite"]
PROTOCOLS = ["rest", "grpc", "graphql", "websocket", "amqp"]
SERVICE_NAMES = ["auth", "billing", "catalog", "search", "notify", "users", "orders", "inventory",
                 "shipping", "reviews", "analytics", "payments", "gateway", "media", "chat", "reports"]
ENDPOINT_WORDS = ["login", "logout", "signup", "profile", "cart", "checkout", "invoice", "refund",
                  "products", "query", "suggest", "email", "sms", "push", "orders", "status", "stock",
                  "restock", "track", "label", "rate", "comment", "events", "metrics", "pay", "upload",
                  "thumbnail", "rooms", "messages", "export", "summary", "tokens", "session", "wishlist",
                  "coupons", "address", "returns", "audit", "health", "settings"]
FEATURE_WORDS = ["onboarding", "purchase", "account", "discovery", "fulfilment", "support", "insights",
                 "messaging", "returns", "promotions", "media-library", "admin"]

DOC_FILE = "docs/ARCHITECTURE.md"

# Fact schema shown to LLM agents, and the note-taking prompt / parser for real-LLM harvest.
SCHEMA = """Facts use canonical keys:
  route(/endpoint)            -> service that owns the endpoint (api/routes_*.py)
  datastore(service)          -> DATASTORE in services/<service>/config.py
  protocol(service)           -> PROTOCOL in services/<service>/server.py
  endpoint_datastore(/e)      =  datastore(route(/e))
  endpoint_protocol(/e)       =  protocol(route(/e))
  feature_datastores(feature) =  sorted unique endpoint_datastore of the feature's endpoints, comma-separated
docs/ARCHITECTURE.md may be outdated: prefer source code."""
NOTE_PROMPT = ("List every fact stated in this file, one per line, exactly as `key = value`, using only these "
               "keys: route(/endpoint) for each entry of a ROUTES table (value = service), datastore(<service>) "
               "for DATASTORE (service = the folder under services/), protocol(<service>) for PROTOCOL. "
               "No other text.")
NOTE_RE = re.compile(r"(route\(/[^)\s]+\)|datastore\([a-z0-9-]+\)|protocol\([a-z0-9-]+\))\s*=\s*`?\"?([a-z0-9-]+)")


@dataclass
class FileState:
    path: str
    lines: list[str]
    version: int = 1

    @property
    def content(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Change:
    path: str
    old_version: int
    new_version: int
    key: str | None          # fact key whose value changed (None for cosmetic edits)
    old_value: str | None
    new_value: str | None


@dataclass
class World:
    cfg: WorldConfig
    rng: random.Random
    services: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    features: dict[str, list[str]] = field(default_factory=dict)
    base: dict[str, str] = field(default_factory=dict)          # base fact key -> value
    base_path: dict[str, str] = field(default_factory=dict)     # base fact key -> file path
    doc_claims: dict[str, str] = field(default_factory=dict)    # what the docs (wrongly or rightly) say
    files: dict[str, FileState] = field(default_factory=dict)
    changes: list[Change] = field(default_factory=list)
    _route_file: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def generate(cls, cfg: WorldConfig, seed: int) -> "World":
        w = cls(cfg=cfg, rng=random.Random(seed))
        r = w.rng
        w.services = r.sample(SERVICE_NAMES, cfg.n_services)
        n_endpoints = cfg.n_route_files * cfg.endpoints_per_file
        w.endpoints = ["/" + e for e in r.sample(ENDPOINT_WORDS, n_endpoints)]
        for s in w.services:
            w._set_base(f"datastore({s})", r.choice(DATASTORES), f"services/{s}/config.py")
            w._set_base(f"protocol({s})", r.choice(PROTOCOLS), f"services/{s}/server.py")
        for i, e in enumerate(w.endpoints):
            path = f"api/routes_{i // cfg.endpoints_per_file}.py"
            w._route_file[e] = path
            w._set_base(f"route({e})", r.choice(w.services), path)
        for fname in r.sample(FEATURE_WORDS, cfg.n_features):
            w.features[fname] = r.sample(w.endpoints, cfg.endpoints_per_feature)
        # outdated documentation
        for key, val in w.base.items():
            if key.startswith("route("):
                continue
            pool = DATASTORES if key.startswith("datastore(") else PROTOCOLS
            if r.random() < cfg.doc_error_rate:
                w.doc_claims[key] = r.choice([v for v in pool if v != val])
            else:
                w.doc_claims[key] = val
        w._render_all()
        return w

    def _set_base(self, key: str, value: str, path: str) -> None:
        self.base[key] = value
        self.base_path[key] = path

    # ---------------------------------------------------------------- render
    def _filler(self, n: int, tag: str) -> list[str]:
        r = self.rng
        out = []
        for i in range(n):
            k = r.random()
            name = f"{tag}_{r.choice(['handle', 'load', 'build', 'check', 'sync', 'parse', 'emit'])}_{i}"
            if k < 0.15:
                out.append(f"def {name}(ctx, payload=None):")
            elif k < 0.3:
                out.append(f"    # TODO({r.choice(['ana', 'raj', 'li', 'sam'])}): revisit {name} edge cases")
            elif k < 0.6:
                out.append(f"    result = {name}_impl(ctx, retries={r.randint(1, 5)}, timeout={r.randint(1, 30)})")
            elif k < 0.8:
                out.append(f"    if not ctx.get('{name}'):\n        logger.debug('skip {name}')")
            else:
                out.append(f"    return normalize(result, mode='{r.choice(['fast', 'safe', 'strict'])}')")
        return out

    def fact_line(self, key: str) -> str:
        val = self.base[key]
        if key.startswith("datastore("):
            return f'DATASTORE = "{val}"  # primary persistence backend'
        if key.startswith("protocol("):
            return f'PROTOCOL = "{val}"  # wire protocol exposed by this service'
        ep = key[len("route("):-1]
        return f'    "{ep}": "{val}",  # route -> owning service'

    def _render_all(self) -> None:
        lo, hi = self.cfg.filler_lines
        for s in self.services:
            for kind, fname in (("datastore", "config.py"), ("protocol", "server.py")):
                key = f"{kind}({s})"
                pre = self._filler(self.rng.randint(lo, hi) // 2, s)
                post = self._filler(self.rng.randint(lo, hi) // 2, s)
                lines = [f'"""{s} service - {fname}"""', "import logging", "logger = logging.getLogger(__name__)", ""]
                lines += pre + ["", self.fact_line(key), ""] + post
                path = f"services/{s}/{fname}"
                self.files[path] = FileState(path, lines)
        for i in range(self.cfg.n_route_files):
            path = f"api/routes_{i}.py"
            eps = [e for e in self.endpoints if self._route_file[e] == path]
            lines = [f'"""HTTP routing table, part {i}"""', ""] + self._filler(self.rng.randint(lo, hi) // 3, f"r{i}")
            lines += ["", "ROUTES = {"] + [self.fact_line(f"route({e})") for e in eps] + ["}", ""]
            lines += self._filler(self.rng.randint(lo, hi) // 3, f"r{i}")
            self.files[path] = FileState(path, lines)
        doc = ["# Architecture overview (last updated two releases ago)", ""]
        for s in self.services:
            doc.append(f"- The {s} service stores its data in {self.doc_claims[f'datastore({s})']} "
                       f"and speaks {self.doc_claims[f'protocol({s})']}.")
        doc += ["", "Routing is defined under api/. Features:"]
        for f, eps in self.features.items():
            doc.append(f"- {f}: {', '.join(eps)}")
        self.files[DOC_FILE] = FileState(DOC_FILE, doc)

    # ---------------------------------------------------------- ground truth
    def is_base(self, key: str) -> bool:
        return key in self.base

    def deps(self, key: str) -> list[str]:
        """Parents of a derived fact *under the current world state*."""
        if key in self.base:
            return []
        if key.startswith("endpoint_datastore(") or key.startswith("endpoint_protocol("):
            ep = key[key.index("(") + 1:-1]
            svc = self.base[f"route({ep})"]
            kind = "datastore" if key.startswith("endpoint_datastore(") else "protocol"
            return [f"route({ep})", f"{kind}({svc})"]
        if key.startswith("feature_datastores("):
            f = key[len("feature_datastores("):-1]
            return [f"endpoint_datastore({e})" for e in self.features[f]]
        raise KeyError(key)

    def truth(self, key: str) -> str:
        if key in self.base:
            return self.base[key]
        parents = self.deps(key)
        return derive(key, {p: self.truth(p) for p in parents})

    def dependency_closure(self, key: str) -> set[str]:
        out = {key}
        for p in self.deps(key):
            out |= self.dependency_closure(p)
        return out


    # ------------------------------------------------------- environment interface
    # Everything outside env/ uses a world only through these members, so the synthetic repository
    # and real repositories (env/realrepo.py) are interchangeable.
    name = "synthetic"
    schema = SCHEMA
    note_prompt = NOTE_PROMPT
    note_re = NOTE_RE
    doc_file = DOC_FILE
    agent_guide = "Prefer source code over docs:\n" + SCHEMA.split("\ndocs/")[0].replace(
        "Facts use canonical keys:", "docs/ARCHITECTURE.md may be outdated. Facts use canonical keys:")
    answer_example = '"redis" or "mysql,redis"'

    def valid_key(self, key: str) -> bool:
        return bool(re.fullmatch(r"(route|datastore|protocol|endpoint_datastore|endpoint_protocol|feature_datastores)"
                                 r"\([^()\s]+\)", key))

    def is_base_key(self, key: str) -> bool:
        """Is this key of a base-fact kind (even if the entity does not exist)?"""
        return key.startswith(("datastore(", "protocol(", "route("))

    def note_normalize(self, text: str) -> str:
        return text.lower()

    def note_facts(self, note: str, path: str, lines: list[str]) -> list[tuple[str, str, str | None]]:
        """Parse a note-taking call's output into (key, value, provenance line)."""
        out = []
        for m in self.note_re.finditer(self.note_normalize(note)):
            key, value = m.group(1), m.group(2)
            out.append((key, value, self.note_span(key, value, path, lines)))
        return out

    def normalize(self, value: str) -> str:
        """Canonical form of an agent's answer (values here are lower-case; lists are sorted)."""
        v = value.strip().strip('"').lower()
        if "," in v:
            v = ",".join(sorted({x.strip() for x in v.split(",") if x.strip()}))
        return v

    def note_span(self, key: str, value: str, path: str, lines: list[str]):
        arg = key[key.index("(") + 1:-1]
        needle = arg if key.startswith("route(") else key.split("(")[0].upper()
        return next((ln for ln in lines if needle in ln and f'"{value}"' in ln), None)

    def parse_doc(self, key: str, text: str):
        return parse_doc(key, text) if not key.startswith("route(") else None

    def lines(self, path: str) -> list[str]:
        fs = self.files.get(path)
        return fs.lines if fs else []

    def parse(self, key: str, path: str, text: str):
        """(value, supporting line) of a base fact in a file's text, or None."""
        return parse_fact(key, text)

    def harvest(self, path: str, text: str):
        return harvest_facts(path, text)

    def locate(self, key: str) -> list[str]:
        """Where a base fact can live, by repository convention."""
        return code_path_for(key, self.cfg.n_route_files) if key in self.base else []

    def parents(self, key: str, get) -> list[str]:
        return parent_keys(key, get, self.features)

    def combine(self, key: str, parent_values: dict[str, str]) -> str:
        return derive(key, parent_values)

    def extract_prompt(self, key: str, path: str, text: str) -> str:
        """Prompt for the manager's real-LLM worker to read one fact from one file."""
        return (f"File {path}:\n{text}\n\nWhat is the value of {key}? "
                "(route(/x) = owning service; datastore(s) = DATASTORE; protocol(s) = PROTOCOL)")

    def derive_prompt(self, key: str, parent_values: dict[str, str]) -> str:
        facts = "\n".join(f"{k} = {v}" for k, v in parent_values.items())
        rule = ("endpoint_datastore(/e) = datastore(route(/e)); endpoint_protocol(/e) = protocol(route(/e)); "
                "feature_datastores(f) = sorted unique datastores of its endpoints, comma-separated, no spaces")
        return f"Rules: {rule}\nFacts:\n{facts}\n\nWhat is {key}?"

    def wrong_value(self, key: str, value: str, rng) -> str:
        kind = key.split("(")[0]
        pool = sorted({v for k, v in self.base.items() if k.split("(")[0] == kind} - {value})
        return rng.choice(pool) if pool else value

    def question_types(self, task_cfg):
        from .tasks import synthetic_question_types
        return synthetic_question_types(self, task_cfg)

    def question_text(self, key: str) -> str:
        from .tasks import question_text
        return question_text(key)

    def plan_changes(self, subtasks, dyn, seed):
        if not (dyn.n_changes or dyn.n_cosmetic):
            return []
        from .tasks import plan_changes
        return plan_changes(self, subtasks, dyn, seed)

    def apply_change(self, item) -> list[Change]:
        kind, target = item
        return [self.mutate_fact(target) if kind == "fact" else self.cosmetic_edit(target)]

    def initial_context(self) -> str:
        lines = ["Repository layout:"] + [f"  {p}" for p in sorted(self.files)]
        lines += ["Feature catalogue:"] + [f"  {f}: {', '.join(e)}" for f, e in self.features.items()]
        return "\n".join(lines)

    # --------------------------------------------------------------- mutation
    def version(self, path: str) -> int:
        fs = self.files.get(path)
        return fs.version if fs else -1   # -1: the file no longer exists

    def mutate_fact(self, key: str, new_value: str | None = None) -> Change:
        path = self.base_path[key]
        old_val = self.base[key]
        if new_value is None:
            if key.startswith("route("):
                pool = self.services
            elif key.startswith("datastore("):
                pool = DATASTORES
            else:
                pool = PROTOCOLS
            new_value = self.rng.choice([v for v in pool if v != old_val])
        fs = self.files[path]
        old_line = self.fact_line(key)
        idx = fs.lines.index(old_line)
        self.base[key] = new_value
        fs.lines[idx] = self.fact_line(key)
        ch = Change(path, fs.version, fs.version + 1, key, old_val, new_value)
        fs.version += 1
        self.changes.append(ch)
        return ch

    def cosmetic_edit(self, path: str) -> Change:
        fs = self.files[path]
        fs.lines.insert(self.rng.randint(4, 8), f"# refactor: tidy imports (rev {fs.version + 1})")
        ch = Change(path, fs.version, fs.version + 1, None, None, None)
        fs.version += 1
        self.changes.append(ch)
        return ch

    # ---------------------------------------------------------------- search
    def search(self, query: str, limit: int = 5) -> list[str]:
        terms = [t for t in re.split(r"[^a-z0-9/_-]+", query.lower()) if t]
        scored = []
        for path, fs in self.files.items():
            text = fs.content.lower()
            score = sum(3 * (t in path) + min(text.count(t), 5) for t in terms)
            if score:
                scored.append((-score, path))
        scored.sort()
        return [p for _, p in scored[:limit]]


def derive(key: str, parent_values: dict[str, str]) -> str:
    """The reasoning rule that produces a derived fact from its parents' values."""
    if key.startswith("endpoint_datastore(") or key.startswith("endpoint_protocol("):
        kind = "datastore(" if key.startswith("endpoint_datastore(") else "protocol("
        return next(v for k, v in parent_values.items() if k.startswith(kind))
    if key.startswith("feature_datastores("):
        return ",".join(sorted(set(parent_values.values())))
    raise KeyError(key)


FACT_PATTERNS = {
    "datastore": re.compile(r'^DATASTORE = "([a-z0-9]+)"'),
    "protocol": re.compile(r'^PROTOCOL = "([a-z0-9]+)"'),
}


def parse_fact(key: str, content: str) -> tuple[str, str] | None:
    """Extract a base fact's value from file content. Returns (value, supporting_line)."""
    if key.startswith("route("):
        ep = key[len("route("):-1]
        pat = re.compile(r'^\s+"' + re.escape(ep) + r'": "([a-z0-9-]+)",')
    else:
        pat = FACT_PATTERNS[key[:key.index("(")]]
    for line in content.splitlines():
        m = pat.match(line)
        if m:
            return m.group(1), line
    return None


def parse_doc(key: str, content: str) -> str | None:
    svc = key[key.index("(") + 1:-1]
    m = re.search(rf"The {re.escape(svc)} service stores its data in ([a-z0-9]+) and speaks ([a-z0-9]+)", content)
    if not m:
        return None
    return m.group(1) if key.startswith("datastore(") else m.group(2)


ROUTE_LINE = re.compile(r'^\s+"(/[^"]+)": "([a-z0-9-]+)",')


def harvest_facts(path: str, content: str) -> list[tuple[str, str, str]]:
    """Every base fact visible in a code file: [(key, value, supporting_line)].
    Models an agent that, having read a file, notes all the facts it saw ("read once, harvest all")."""
    out = []
    if path.startswith("api/routes_"):
        for line in content.splitlines():
            m = ROUTE_LINE.match(line)
            if m:
                out.append((f"route({m.group(1)})", m.group(2), line))
    elif path.startswith("services/"):
        svc = path.split("/")[1]
        kind = "datastore" if path.endswith("config.py") else "protocol"
        for line in content.splitlines():
            m = FACT_PATTERNS[kind].match(line)
            if m:
                out.append((f"{kind}({svc})", m.group(1), line))
    return out


def key_arg(key: str) -> str:
    return key[key.index("(") + 1:-1]


def parent_keys(key: str, get, features: dict[str, list[str]]) -> list[str]:
    """The derivation rules (stated in every agent's task description): which facts a
    derived fact is computed from. get(key) returns the current value of a parent."""
    if key.startswith(("endpoint_datastore(", "endpoint_protocol(")):
        ep = key_arg(key)
        kind = "datastore" if key.startswith("endpoint_datastore(") else "protocol"
        return [f"route({ep})", f"{kind}({get(f'route({ep})')})"]
    if key.startswith("feature_datastores("):
        return [f"endpoint_datastore({e})" for e in features[key_arg(key)]]
    return []


def code_path_for(key: str, n_route_files: int) -> list[str]:
    """Where a base fact can live, by repository convention."""
    kind, arg = key[:key.index("(")], key_arg(key)
    if kind == "datastore":
        return [f"services/{arg}/config.py"]
    if kind == "protocol":
        return [f"services/{arg}/server.py"]
    return [f"api/routes_{i}.py" for i in range(n_route_files)]

