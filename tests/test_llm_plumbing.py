"""Exercise the real-LLM agent loop with a scripted fake client (no API calls)."""
from dataclasses import dataclass, field

from bacm.config import RunConfig, TaskConfig, get_arch
from bacm.sim.orchestrator import Orchestrator


@dataclass
class Block:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = ""


class ScriptedClient:
    """Reads routes + config for the first question, publishes, then answers everything."""

    def __init__(self):
        self.n = 0

    def complete(self, system, messages, tools):
        self.n += 1
        if not tools:   # note-taking / worker call
            return [Block("text", text="datastore(auth) = redis")], 50, 5, "end_turn"
        first = messages[0]["content"]
        keys = [line.split("]")[0].split("[")[1] for line in first.splitlines() if line.startswith("- [")]
        answered = sum(1 for m in messages if m["role"] == "assistant"
                       for b in m["content"] if getattr(b, "name", "") == "answer")
        turns = sum(1 for m in messages if m["role"] == "assistant")
        names = {t["name"] for t in tools}
        if turns == 0:
            return [Block("tool_use", name="read_file", input={"path": "api/routes_0.py"}, id=f"t{self.n}")], 100, 20, "tool_use"
        if turns == 1 and "kb_publish" in names:
            return [Block("tool_use", name="kb_query", input={"key": keys[0]}, id=f"t{self.n}")], 100, 20, "tool_use"
        if answered < len(keys):
            return [Block("tool_use", name="answer", input={"key": keys[answered], "value": "redis"},
                          id=f"t{self.n}")], 120, 20, "tool_use"
        return [Block("text", text="done")], 120, 5, "end_turn"


def test_llm_agent_loop_with_fake_client():
    for arch in ("independent", "bacm", "fully_shared"):
        cfg = RunConfig(arch=get_arch(arch), seed=1, backend="anthropic", task=TaskConfig(n_agents=2))
        o = Orchestrator(cfg, ScriptedClient())
        m = o.run()
        assert m["llm_calls"] > 0 and m["file_reads"] == 2
        assert len({(a, k) for a, k, *_ in o.tracker.answers}) == 2 * cfg.task.questions_per_agent
