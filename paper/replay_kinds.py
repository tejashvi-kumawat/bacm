"""Per-question-kind accuracy and token anatomy of the final evaluation, replayed from the response cache
(no API calls: a cache miss raises). Writes paper/generated/kind_accuracy.json and anatomy.json.

  .venv/bin/python paper/replay_kinds.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bacm.__main__ import load_dotenv  # noqa: E402
from bacm.experiments.suites import repobench2  # noqa: E402
from bacm.llm.openai_compat import LLMError, OpenAICompatClient  # noqa: E402
from bacm.sim.orchestrator import Orchestrator  # noqa: E402

load_dotenv(ROOT / ".env")
# Replay never calls the API (CacheOnly raises on a miss), but the client insists on a key being configured.
os.environ.setdefault("OPENAI_API_KEY", "replay-only-placeholder")


class CacheOnly(OpenAICompatClient):
    def _post(self, raw):
        raise LLMError("cache miss")


def main(seeds=range(200, 206), archs=("independent_evict", "selective_llm", "bacm_llm", "bacm_auto")):
    kinds = defaultdict(lambda: defaultdict(list))
    anatomy = defaultdict(lambda: defaultdict(list))
    growth = defaultdict(list)   # cell -> per-branch list of input tokens per agent turn (tests Eq. cost)
    missing = 0
    import os
    os.chdir(ROOT)
    for g, cfg in repobench2(seeds):
        if g["arch"] not in archs:
            continue
        cfg = cfg.with_(backend="openai_compat", model="gpt-5-mini")
        try:
            o = Orchestrator(cfg, CacheOnly("openai", "gpt-5-mini"))
            o.run()
        except LLMError:
            missing += 1
            continue
        final = {}
        for a, k, v, t, ok in o.tracker.answers:
            final[(a, k)] = v
        for (a, k), v in final.items():
            kinds[f"{g['env']}|{g['arch']}"][k.split("(")[0]].append(v == o.world.truth(k))
        turns = defaultdict(list)
        for ag, tin, tout, t, p in o.tracker.llm:
            if ag != "context-manager" and p != "note":
                turns[ag].append(tin)
        growth[f"{g['env']}|{g['arch']}"] += list(turns.values())
        by = Counter()
        for ag, tin, tout, t, p in o.tracker.llm:
            by["manager" if ag == "context-manager" else ("note" if p == "note" else "agent")] += tin + tout
        for k in ("agent", "note", "manager"):
            anatomy[f"{g['env']}|{g['arch']}"][k].append(by[k])
    out = {}
    for cell, d in kinds.items():
        env, arch = cell.split("|")
        if env == "static":
            out[arch] = {k: statistics.fmean(v) for k, v in d.items()}
    gen = ROOT / "paper" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "kind_accuracy.json").write_text(json.dumps(out, indent=1))
    (gen / "kind_accuracy_all.json").write_text(json.dumps(
        {c: {k: [statistics.fmean(v), len(v)] for k, v in d.items()} for c, d in kinds.items()}, indent=1))
    (gen / "anatomy.json").write_text(json.dumps(
        {c: {k: statistics.fmean(v) for k, v in d.items()} for c, d in anatomy.items()}, indent=1))
    (gen / "growth.json").write_text(json.dumps(growth))
    print(f"replayed; {missing} runs not fully cached")


if __name__ == "__main__":
    main()
