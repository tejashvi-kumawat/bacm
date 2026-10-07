"""Command line interface.

  python -m bacm demo                      # one seed, all architectures side by side
  python -m bacm demo --dynamic --trace    # with mid-run source changes + knowledge graph dump
  python -m bacm run --suite pilot --seeds 10
  python -m bacm run --suite all --seeds 20 --workers 16
  python -m bacm run --suite llm_pilot --seeds 3 --backend openai_compat --provider gemini --model gemini-3.6-flash
  python -m bacm report results_real/<dir>  # tables with 95% CIs and paired comparisons
  python -m bacm doctor --ping              # check Python, keys, providers, Ollama
  python -m bacm status                     # everything so far, as tables

API keys can live in a `.env` file in the working directory (never commit it), e.g.
  GEMINI_API_KEYS=key1,key2
  GROQ_API_KEY=...
Works the same on Windows, macOS and Linux.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .config import ABLATIONS, ARCHITECTURES, DynamicsConfig, RunConfig, TaskConfig, get_arch
from .experiments.runner import run_suite
from .experiments.suites import ALL_MAIN, DYNAMIC, SIM_SUITES, SUITES
from .sim.orchestrator import Orchestrator

SHOW = ["question_accuracy", "tokens_total", "cost_usd", "llm_calls", "tool_calls", "file_reads",
        "redundant_file_reads", "reuse_rate", "false_reuse_rate", "ctx_avg", "ctx_max", "latency_sim_s",
        "stale_answer_rate", "invalidations", "unnecessary_invalidations", "conflicts", "correlated_error_rate"]


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def _quota_wait(failed_path: Path) -> float | None:
    """Seconds until the earliest quota reset named in failed.json, or None if nothing hit a quota."""
    import json
    import re
    if not failed_path.exists():
        return None
    errs = [f["error"] for f in json.loads(failed_path.read_text(encoding="utf-8")) if "QuotaExhausted" in f["error"]]
    if not errs:
        return None
    secs = []
    for e in errs:
        for h, m, sec in re.findall(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)?s?", e):
            if h or m or sec:
                secs.append(int(h or 0) * 3600 + int(m or 0) * 60 + float(sec or 0))
    return min(max(secs), 6 * 3600) + 120 if secs else 3600.0


def cmd_doctor(a):
    import platform
    import urllib.request

    from .llm.openai_compat import PRESETS, LLMError, OpenAICompatClient
    ok = True
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 10):
        ok = False
        print("  !! Python 3.10+ is required")
    try:
        import matplotlib  # noqa: F401
        print("matplotlib: installed (plots available)")
    except ImportError:
        print("matplotlib: not installed (optional, only for plots)")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            import json
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        print(f"Ollama: running, models = {models or 'none pulled yet (ollama pull qwen2.5:7b)'}")
    except Exception:
        print("Ollama: not running (optional; only needed for --provider ollama)")
    for name, (url, env, model) in PRESETS.items():
        if env is None:
            continue
        keys = os.environ.get(env + "S") or os.environ.get(env)
        n = len([k for k in (keys or "").split(",") if k.strip()])
        line = f"{name:<11} {env}: {n} key(s)" if n else f"{name:<11} {env}: not set"
        if n and a.ping:
            for i, k in enumerate(x.strip() for x in keys.split(",") if x.strip()):
                c = OpenAICompatClient(name, a.model if a.model else model or "", api_key=k, cache_dir=None)
                c.max_tokens = 5
                try:
                    c.complete("Reply ok.", [{"role": "user", "content": "ok?"}], None)
                    line += f"\n    key {i + 1} ({k[:8]}...): OK with {c.model}"
                except LLMError as e:
                    ok = False
                    line += f"\n    key {i + 1} ({k[:8]}...): {str(e)[:160]}"
        print(line)
    print("All good." if ok else "Some checks failed (see above).")


def cmd_status(a):
    from .experiments.status import status
    print(status(Path(a.root)))


def cmd_report(a):
    from .experiments.report import report
    for d in a.dirs:
        text = report(Path(d))
        (Path(d) / "report.md").write_text(text, encoding="utf-8")
        print(text)


def cmd_demo(a):
    dyn = DYNAMIC if a.dynamic else DynamicsConfig()
    archs = a.archs.split(",") if a.archs else ALL_MAIN
    results = {}
    last = None
    for name in archs:
        cfg = RunConfig(arch=get_arch(name), seed=a.seed, dynamics=dyn, task=TaskConfig(n_agents=a.agents))
        o = Orchestrator(cfg)
        results[name] = o.run()
        last = o
        if a.trace and cfg.arch.memory == "bacm":
            print(f"\n=== {name}: shared knowledge graph ===")
            for k in o.memory.snapshot():
                print(f"  {k['id']:>4} {k['status']:<10} {k['key']:<32} = {k['value']:<28} "
                      f"by {k['producer']} src={k['sources']} deps={k['deps']} reuses={k['reuses']}")
    print("\nSubtasks:")
    for i, qs in enumerate(last.subtasks):
        print(f"  A{i}: " + ", ".join(q.key for q in qs))
    if last.tracker.change_events:
        print("Source changes:")
        for t, c in last.tracker.change_events:
            what = f"{c.key}: {c.old_value} -> {c.new_value}" if c.key else "cosmetic edit"
            print(f"  t={t:7.1f}s  {c.path} v{c.old_version}->v{c.new_version}  {what}")
    w = max(len(n) for n in archs) + 2
    print("\n" + "metric".ljust(26) + "".join(n.rjust(w) for n in archs))
    for k in SHOW:
        print(k.ljust(26) + "".join(_fmt(results[n][k]).rjust(w) for n in archs))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"
    return f"{v:,}"


def cmd_run(a):
    if a.backend == "anthropic" and not a.yes_bill_api:
        sys.exit("--backend anthropic makes real API calls billed to your Anthropic API key/credits "
                 "(not a Claude Pro/Max plan). Re-run with --yes-bill-api to confirm.")
    suites = SIM_SUITES if a.suite == "all" else a.suite.split(",")
    seeds = range(a.seed0, a.seed0 + a.seeds)
    llm = {"provider": a.provider, "base_url": a.base_url, "rpm": a.rpm, "max_usd": a.max_usd}
    if a.provider == "openai" and a.backend == "openai_compat" and a.max_usd is None:
        sys.exit("OpenAI is a paid API: pass --max-usd N (a hard spending limit across all workers).")
    archs = a.archs.split(",") if a.archs else None
    repos = a.repos.split(",") if a.repos else None
    for s in suites:
        for round_ in range(1, 31):
            t = time.time()
            out = run_suite(s, seeds, Path(a.out), a.workers, a.backend, a.model, a.effort, llm=llm, archs=archs,
                            repos=repos)
            print(f"{s}: wrote {out}/runs.jsonl and summary.csv in {time.time() - t:.1f}s")
            wait = _quota_wait(out / "failed.json")
            if wait is None or not a.until_done:
                break
            print(f"{s}: free quota used up; waiting {wait / 3600:.1f} h for the reset, then resuming from cache "
                  f"(round {round_}). Ctrl+C to stop; rerunning the same command later also resumes.", flush=True)
            time.sleep(wait)
        if a.backend != "sim":
            from .experiments.report import report
            (out / "report.md").write_text(report(out), encoding="utf-8")
            print(f"{s}: report -> {out / 'report.md'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bacm")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="single-seed side-by-side comparison")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--agents", type=int, default=4)
    d.add_argument("--dynamic", action="store_true", help="inject source changes mid-run")
    d.add_argument("--trace", action="store_true", help="dump the shared knowledge graph")
    d.add_argument("--archs", help=f"comma list from {sorted({**ARCHITECTURES, **ABLATIONS})}")
    d.set_defaults(fn=cmd_demo)
    r = sub.add_parser("run", help="run experiment suites")
    r.add_argument("--suite", default="pilot", help=f"{','.join(SUITES)} or all (comma-separated ok)")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--seed0", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", default="results")
    r.add_argument("--backend", default="sim", choices=["sim", "openai_compat", "anthropic"],
                   help="sim (free, default) | openai_compat (Ollama / free-tier APIs) | anthropic (paid)")
    r.add_argument("--provider", default="ollama", help="openai_compat preset: ollama, groq, gemini, openrouter, cerebras")
    r.add_argument("--base-url", default=None, help="custom OpenAI-compatible endpoint")
    r.add_argument("--rpm", type=float, default=None, help="throttle to N requests/minute (free-tier limits)")
    r.add_argument("--archs", default=None, help="only run these architectures (comma list)")
    r.add_argument("--repos", default=None, help="repobench: only these repositories (comma list)")
    r.add_argument("--model", default=None, help="model id (defaults per provider; claude-opus-5-5 for anthropic)")
    r.add_argument("--effort", default=None, choices=["minimal", "low", "medium", "high", "xhigh", "max"],
                   help="Anthropic effort, or OpenAI GPT-5 reasoning_effort (default minimal)")
    r.add_argument("--max-usd", type=float, default=None,
                   help="hard spending limit for paid providers (checked on every request, all workers)")
    r.add_argument("--yes-bill-api", action="store_true")
    r.add_argument("--until-done", action="store_true",
                   help="when every key is out of free quota, wait for the reset and resume (from cache)")
    r.set_defaults(fn=cmd_run)
    doc = sub.add_parser("doctor", help="check the environment, API keys and local models")
    doc.add_argument("--ping", action="store_true", help="send one tiny request per key (uses 1 request each)")
    doc.add_argument("--model", default=None, help="model to ping with (default: provider default)")
    doc.set_defaults(fn=cmd_doctor)
    st = sub.add_parser("status", help="tables of everything so far: result sets, per-architecture results, spend")
    st.add_argument("root", nargs="?", default=".")
    st.set_defaults(fn=cmd_status)
    rp = sub.add_parser("report", help="summarise result directories (95%% CIs, paired comparisons)")
    rp.add_argument("dirs", nargs="+")
    rp.set_defaults(fn=cmd_report)
    load_dotenv()
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
