"""Run a suite (in parallel), write raw per-run rows and an aggregated summary."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..sim.orchestrator import run_once
from .suites import SUITES

# Work measures used for Avoided Redundant Work: ARW = 1 - W_system / W_independent (section 18)
ARW_MEASURES = ["tokens_total", "llm_calls", "tool_calls", "file_reads", "cost_usd", "latency_sim_s"]


def make_client(cfg, llm: dict):
    if cfg.backend == "anthropic":
        from ..llm.clients import AnthropicClient
        return AnthropicClient(cfg.model, llm.get("effort"))
    if cfg.backend == "openai_compat":
        from ..llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(llm.get("provider", "ollama"), cfg.model, llm.get("base_url"),
                                  rpm=llm.get("rpm"), max_usd=llm.get("max_usd"),
                                  reasoning_effort=llm.get("effort"))
    return None


def _run(job):
    group, cfg, llm = job
    try:
        m = run_once(cfg, make_client(cfg, llm))
    except Exception as e:   # one failed run must not lose the others
        from ..llm.openai_compat import AuthError, BudgetExceeded, ConfigError, QuotaExhausted
        fatal = isinstance(e, (AuthError, ConfigError, QuotaExhausted, BudgetExceeded))
        return {"group": group, "seed": cfg.seed, "error": f"{type(e).__name__}: {e}", "fatal": fatal}
    return {"group": group, "seed": cfg.seed, "metrics": m}


def run_suite(name: str, seeds, out_dir: Path, workers: int = 8, backend: str = "sim",
              model: str | None = None, effort: str | None = None, progress=print, llm: dict | None = None,
              archs: list[str] | None = None, repos: list[str] | None = None) -> Path:
    """Runs every job; failed runs are recorded in failed.json and the rest are still saved.
    On a fatal error (all keys out of quota / rejected) remaining jobs are skipped - rerun the same
    command later: finished LLM calls are replayed from the cache, so nothing is paid twice."""
    from ..llm.openai_compat import safe_name
    llm = dict(llm or {}, effort=effort)
    jobs = [(g, c.with_(backend=backend, model=model), llm) for g, c in SUITES[name](seeds)
            if (not archs or g["arch"] in archs) and (not repos or g.get("repo") in repos)]
    tag = name if backend == "sim" else f"{name}-{backend}-{safe_name(model or llm.get('provider') or '')}"
    out = out_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []

    def take(r) -> bool:
        (failed if "error" in r else rows).append(r)
        if "error" in r:
            progress(f"[{name}] FAILED {r['group']} seed={r['seed']}: {r['error'][:300]}")
        return bool(r.get("fatal"))

    stop = False
    if workers <= 1:
        for i, j in enumerate(jobs):
            stop = take(_run(j))
            progress(f"[{name}] {i + 1}/{len(jobs)} {j[0]} seed={j[1].seed}")
            if stop:
                break
    else:
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(_run, j) for j in jobs]
            for i, f in enumerate(as_completed(futs)):
                if take(f.result()):
                    stop = True
                    for other in futs:
                        other.cancel()
                    break
                if (i + 1) % max(1, len(jobs) // 10) == 0 or i + 1 == len(jobs):
                    progress(f"[{name}] {i + 1}/{len(jobs)} runs done")
            if stop:   # collect runs that were already in flight
                for f in futs:
                    if not f.cancelled():
                        r = f.result()
                        if r not in rows and r not in failed:
                            take(r)
    rows.sort(key=lambda r: (r["seed"], sorted(r["group"].items())))
    add_arw(rows)
    with open(out / "runs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_jsonable) + "\n")
    (out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
    if rows:
        summarize(rows, out / "summary.csv")
    skipped = len(jobs) - len(rows) - len(failed)
    progress(f"[{name}] saved {len(rows)} runs, {len(failed)} failed, {skipped} skipped -> {out}"
             + (" | stopped early: rerun the same command later to resume from cache" if stop else ""))
    return out


def _jsonable(x):
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return str(x)


def _gkey(group: dict, drop=("arch",)) -> tuple:
    return tuple(sorted((k, v) for k, v in group.items() if k not in drop))


def add_arw(rows: list[dict]) -> None:
    """Attach ARW per measure, relative to the independent run with the same seed and conditions."""
    base = {}
    for r in rows:
        if r["group"]["arch"] == "independent":
            base[(_gkey(r["group"]), r["seed"])] = r["metrics"]
    for r in rows:
        b = base.get((_gkey(r["group"]), r["seed"]))
        if not b:
            continue
        for m in ARW_MEASURES:
            r["metrics"][f"ARW_{m}"] = 1 - r["metrics"][m] / b[m] if b[m] else math.nan


def summarize(rows: list[dict], path: Path) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(sorted(r["group"].items()))].append(r["metrics"])
    out = []
    for cell, ms in cells.items():
        row = dict(cell)
        row["n_runs"] = len(ms)
        for k in ms[0]:
            vals = [m[k] for m in ms if isinstance(m.get(k), (int, float)) and math.isfinite(m[k])]
            n_inf = sum(1 for m in ms if isinstance(m.get(k), float) and math.isinf(m[k]))
            row[f"{k}_mean"] = statistics.fmean(vals) if vals else math.nan
            row[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[f"{k}_ci95"] = 1.96 * row[f"{k}_std"] / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            if n_inf:
                row[f"{k}_n_inf"] = n_inf
        out.append(row)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    return out
