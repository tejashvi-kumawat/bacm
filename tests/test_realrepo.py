"""Real-repository environment, end to end, on a tiny local git repo (hermetic: no network)."""
import json
import subprocess
import threading
from http.server import HTTPServer

from bacm.config import DynamicsConfig, RepoConfig, RunConfig, TaskConfig, get_arch
from bacm.env.realrepo import RealRepoWorld
from bacm.llm.openai_compat import OpenAICompatClient
from bacm.sim.orchestrator import Orchestrator

from test_openai_compat import Fake

V1 = {
    "pkg/base.py": "class Base:\n    pass\n\nclass Mid(Base):\n    pass\n",
    "pkg/models.py": "from .base import Mid\n\nclass Leaf(Mid):\n    pass\n\nclass Other(Mid):\n    x = 1\n",
}
V2 = {   # a real-looking refactor: Mid moves to its own module and gains a new base
    "pkg/base.py": "class Base:\n    pass\n\nclass Root:\n    pass\n",
    "pkg/mid.py": "from .base import Root\n\nclass Mid(Root):\n    pass\n",
    "pkg/models.py": "from .mid import Mid\n\nclass Leaf(Mid):\n    pass\n\nclass Other(Mid):\n    x = 1\n",
}


def make_repo(tmp_path):
    repo = tmp_path / "toy"
    repo.mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    for tag, files in (("v1", V1), ("v2", V2)):
        for p in list(repo.glob("pkg/*.py")):
            p.unlink()
        for p, text in files.items():
            (repo / p).parent.mkdir(exist_ok=True)
            (repo / p).write_text(text)
        run("add", "-A")
        run("commit", "-q", "-m", tag)
        run("tag", tag)
    return RepoConfig("toy", str(repo), "v1", "v2", cache_dir=str(tmp_path / "cache"))


def test_truth_parse_and_release_change(tmp_path):
    w = RealRepoWorld.load(make_repo(tmp_path), 0)
    assert w.truth("base_file(Leaf)") == "pkg/base.py"
    assert w.truth("family_files(Leaf)") == "pkg/base.py,pkg/models.py"
    assert w.truth("grandbase(Leaf)") == "Base"
    assert w.parse("base_of(Leaf)", "pkg/models.py", V1["pkg/models.py"])[0] == "Mid"
    assert {k for k, _, _ in w.harvest("pkg/base.py", V1["pkg/base.py"])} == \
        {"defined_in(Base)", "defined_in(Mid)", "base_of(Base)", "base_of(Mid)"}
    changes = w.apply_change(("release", "v2"))
    assert {c.path for c in changes} == {"pkg/base.py", "pkg/mid.py", "pkg/models.py"}
    assert w.truth("base_file(Leaf)") == "pkg/mid.py" and w.truth("grandbase(Leaf)") == "Root"
    assert w.version("pkg/mid.py") == 1


def test_agents_run_on_a_real_repo(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for arch in ("independent", "bacm"):
            client = OpenAICompatClient("custom", "fake", f"http://127.0.0.1:{srv.server_port}/v1",
                                        cache_dir=str(tmp_path / "llm"))
            cfg = RunConfig(arch=get_arch(arch), seed=0, backend="openai_compat", model="fake",
                            repo=make_repo(tmp_path / arch), task=TaskConfig(n_agents=2, questions_per_agent=2),
                            dynamics=DynamicsConfig(n_changes=1))
            o = Orchestrator(cfg, client)
            m = o.run()
            assert m["llm_calls"] > 0 and m["aborted"] == 0.0
            assert o.world.changes, "the release change was applied mid-run"
    finally:
        srv.shutdown()


def test_note_parsing_on_real_repo(tmp_path):
    w = RealRepoWorld.load(make_repo(tmp_path), 0)
    note = "defined_in(Leaf) = pkg/models.py\nbase_of(Leaf) = `Mid`\nnot a fact"
    facts = w.note_facts(note, "pkg/models.py", V1["pkg/models.py"].splitlines())
    assert [(k, v) for k, v, _ in facts] == [("defined_in(Leaf)", "pkg/models.py"), ("base_of(Leaf)", "Mid")]
    assert all(span for _, _, span in facts)
    assert not w.valid_key("Leaf") and w.parents("Leaf", lambda k: None) == []


class ScriptedAuto:
    """Scripted 'LLM' for transparent memory: each agent reads the files it is told to, then stops."""

    def __init__(self, plan):
        self.plan, self.calls = plan, {}
        self.last_latency = 0.0

    def complete(self, system, messages, tools=None):
        from test_llm_plumbing import Block
        who = system.split("agent ")[1].split(",")[0]
        n = self.calls.get(who, 0)
        self.calls[who] = n + 1
        steps = self.plan[who]
        if n < len(steps):
            return [Block("tool_use", name="read_file", input={"path": steps[n]}, id=f"{who}{n}")], 100, 10, "tool_use"
        return [Block("text", text="done")], 100, 5, "end_turn"


def test_transparent_memory_digest_resolve_and_maintain(tmp_path):
    cfgrepo = make_repo(tmp_path)
    for arch in ("bacm_auto", "selective_auto"):
        cfg = RunConfig(arch=get_arch(arch), seed=0, backend="openai_compat", model="fake", repo=cfgrepo,
                        task=TaskConfig(n_agents=2, questions_per_agent=2), dynamics=DynamicsConfig())
        o = Orchestrator(cfg, ScriptedAuto({"A0": ["pkg/models.py", "pkg/base.py"], "A1": ["pkg/models.py"]}))
        m = o.run()
        # the second reader of pkg/models.py got a digest, not raw text
        assert m["digest_reads"] >= 1 and m["file_reads"] == 2 + 0
        # questions about Leaf/Other are answered without the model ever calling `answer`
        assert m["auto_resolved"] >= 1 and m["question_accuracy"] == 1.0
        assert not any(isinstance(b, dict) is False and getattr(b, "name", "") == "kb_query"
                       for ag in o.agents.values() for msg in ag.messages if msg["role"] == "assistant"
                       and not isinstance(msg["content"], str) for b in msg["content"])


def test_maintained_answers_follow_a_release_change_without_llm_calls(tmp_path):
    cfgrepo = make_repo(tmp_path)
    cfg = RunConfig(arch=get_arch("bacm_auto"), seed=0, backend="openai_compat", model="fake", repo=cfgrepo,
                    task=TaskConfig(n_agents=1, questions_per_agent=4, overlap=0.0),
                    dynamics=DynamicsConfig(n_changes=1, trigger_progress=0.25))
    o = Orchestrator(cfg, ScriptedAuto({"A0": ["pkg/models.py", "pkg/base.py", "pkg/models.py"]}))
    before = o.run()
    assert o.world.changes, "the release change was applied mid-run"
    truth = {q.key: o.world.truth(q.key) for q in o.subtasks[0]}
    final = {}
    for a, k, v, t, ok in o.tracker.answers:
        final[k] = v
    assert all(final.get(k) == v for k, v in truth.items() if v is not None), (final, truth)
    assert before["answer_revisions"] >= 1 and before["system_tokens"] == 0


def test_no_leak_nothing_is_resolved_without_reading(tmp_path):
    """Control: agents that read nothing get no facts, so the manager resolves nothing (accuracy cannot come
    from ground truth leaking into the memory)."""
    cfg = RunConfig(arch=get_arch("bacm_auto"), seed=0, backend="openai_compat", model="fake",
                    repo=make_repo(tmp_path), task=TaskConfig(n_agents=2, questions_per_agent=2),
                    dynamics=DynamicsConfig())
    o = Orchestrator(cfg, ScriptedAuto({"A0": [], "A1": []}))
    m = o.run()
    assert m["auto_resolved"] == 0 and m["question_accuracy"] == 0.0 and not o.tracker.answers


class ScriptedExtract(ScriptedAuto):
    """Also answers the manager's extraction prompts: outline -> structure facts, class body -> method count."""

    def complete(self, system, messages, tools=None):
        import re
        from test_llm_plumbing import Block
        if tools:
            return super().complete(system, messages, tools)
        prompt = messages[-1]["content"]
        self.extract_calls = getattr(self, "extract_calls", 0) + 1
        if "outline" in prompt.lower():
            path = re.search(r"Outline of (\S+):", prompt).group(1)
            out = []
            for ln in prompt.split(":\n", 1)[1].splitlines():
                m = re.match(r"^class (\w+)(?:\((\w+)\))?", ln)
                if m:
                    out += [f"defined_in({m.group(1)}) = {path}", f"base_of({m.group(1)}) = {m.group(2) or 'object'}"]
            return [Block("text", text="\n".join(out))], 50, 10, "end_turn"
        if "How many methods" in prompt:
            n = len(re.findall(r"^    (?:async )?def ", prompt, re.M))
            return [Block("text", text=str(n))], 30, 2, "end_turn"
        return [Block("text", text="")], 10, 1, "end_turn"


V2_REPO = {
    "pkg/base.py": "class Base:\n    def a(self):\n        pass\n\nclass Mid(Base):\n    def b(self):\n        pass\n"
                   "    def c(self):\n        def inner():\n            pass\n",
    "pkg/models.py": "from .base import Mid\n\nclass Leaf(Mid):\n    def d(self):\n        pass\n",
}


def test_focused_model_extraction_v2(tmp_path):
    import subprocess
    repo = tmp_path / "toy2"
    repo.mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    for p, text in V2_REPO.items():
        (repo / p).parent.mkdir(exist_ok=True)
        (repo / p).write_text(text)
    run("add", "-A"); run("commit", "-q", "-m", "v1"); run("tag", "v1")
    rc = RepoConfig("toy2", str(repo), "v1", None, cache_dir=str(tmp_path / "c"), bench="v2")
    w = RealRepoWorld.load(rc, 0)
    assert w.truth("method_total(Leaf)") == "4" and w.truth("n_methods(Mid)") == "2"
    cfg = RunConfig(arch=get_arch("bacm_llm"), seed=0, backend="openai_compat", model="fake", repo=rc,
                    task=TaskConfig(n_agents=1, questions_per_agent=5, overlap=0.0), dynamics=DynamicsConfig())
    client = ScriptedExtract({"A0": ["pkg/models.py", "pkg/base.py"]})
    o = Orchestrator(cfg, client)
    m = o.run()
    assert m["question_accuracy"] == 1.0, [(k, v, o.world.truth(k)) for _, k, v, *_ in o.tracker.answers]
    assert client.extract_calls >= 3   # outline notes + on-demand counts, no whole-file counting


def test_method_count_goes_stale_when_a_method_is_added(tmp_path):
    """Regression: a method count is supported by the whole class body, so adding a method (header unchanged)
    must invalidate it."""
    import subprocess
    repo = tmp_path / "toy3"
    repo.mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (repo / "pkg").mkdir()
    base = "class Base:\n    def a(self):\n        pass\n\nclass Leaf(Base):\n    def b(self):\n        pass\n"
    (repo / "pkg/m.py").write_text(base); run("add", "-A"); run("commit", "-q", "-m", "1"); run("tag", "v1")
    (repo / "pkg/m.py").write_text(base + "    def c(self):\n        pass\n")
    run("add", "-A"); run("commit", "-q", "-m", "2"); run("tag", "v2")
    rc = RepoConfig("toy3", str(repo), "v1", "v2", cache_dir=str(tmp_path / "c"), bench="v2")
    from bacm.core.manager import ContextManager
    from bacm.env.tools import ToolBox
    from bacm.metrics.tracker import Tracker
    w = RealRepoWorld.load(rc, 0)
    tr = Tracker(w, RunConfig(arch=get_arch("bacm_auto")).cost)
    cm = ContextManager(get_arch("bacm_auto"), w, tr, ToolBox(w, tr))
    v, span = w.parse("n_methods(Leaf)", "pkg/m.py", w.files["pkg/m.py"].content)
    from bacm.core.knowledge import Claim, Level, Source
    cm.publish("A0", Claim("n_methods(Leaf)", v, 0.95, "code", [Source("pkg/m.py", 1, span)], level=Level.L2), 0)
    assert cm.query("A1", "n_methods(Leaf)", 1).value == "1"
    for ch in w.apply_change(("release", "v2")):
        cm.on_change(ch, 2)
    assert cm.query("A1", "n_methods(Leaf)", 3) is None    # header line unchanged, body changed -> stale


def test_derived_chain_is_rebuilt_after_a_change_deep_in_it(tmp_path):
    """Regression: when a release changes an ancestor's body, the answer's derived parent (the ancestor's total)
    is itself stale; recomputation must rebuild it recursively instead of giving up on the answer."""
    import subprocess
    repo = tmp_path / "toy4"
    repo.mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (repo / "pkg").mkdir()
    root = "class Root:\n    def a(self):\n        pass\n"
    mid = "from pkg.r import Root\n\nclass Mid(Root):\n    def b(self):\n        pass\n"
    leaf = "from pkg.m import Mid\n\nclass Leaf(Mid):\n    def c(self):\n        pass\n"
    (repo / "pkg/r.py").write_text(root); (repo / "pkg/m.py").write_text(mid); (repo / "pkg/l.py").write_text(leaf)
    run("add", "-A"); run("commit", "-q", "-m", "1"); run("tag", "v1")
    (repo / "pkg/m.py").write_text(mid + "    def d(self):\n        pass\n")   # Mid gains a method
    run("add", "-A"); run("commit", "-q", "-m", "2"); run("tag", "v2")
    rc = RepoConfig("toy4", str(repo), "v1", "v2", cache_dir=str(tmp_path / "c"), bench="v2")
    from bacm.core.knowledge import Claim, Level, Source
    from bacm.core.manager import ContextManager
    from bacm.env.tools import ToolBox
    from bacm.metrics.tracker import Tracker
    w = RealRepoWorld.load(rc, 0)
    cfg = RunConfig(arch=get_arch("bacm_auto"))
    tr = Tracker(w, cfg.cost)
    cm = ContextManager(get_arch("bacm_auto"), w, tr, ToolBox(w, tr))
    cm.configure(cfg.agent, 0)
    for path in ("pkg/r.py", "pkg/m.py", "pkg/l.py"):
        for key, value, span in w.harvest(path, w.files[path].content):
            cm.publish("A0", Claim(key, value, 0.95, "code", [Source(path, 1, span)], level=Level.L2), 0)
    cm.set_questions({"A1": ["method_total(Leaf)"]})
    assert cm.resolve_known("method_total(Leaf)", 1).value == "3"
    cm.subscribe("A1", "method_total(Leaf)", 1)
    for ch in w.apply_change(("release", "v2")):
        cm.on_change(ch, 2)
    hit = cm.query("A1", "method_total(Leaf)", 10)
    assert hit is not None and hit.value == "4"
