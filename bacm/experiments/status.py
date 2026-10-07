"""One-screen status of everything: result sets, per-architecture tables, spend, failures.

  python -m bacm status                 # everything under results*/ and .llm_cache/
  python -m bacm status results_real    # only that tree
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARCH_ORDER = ["independent", "independent_evict", "fully_shared", "selective", "selective_fast", "selective_auto",
              "selective_global_recompute", "bacm_v1", "bacm", "bacm_validate", "bacm_auto"]


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    return "\n".join([line(header), sep] + [line(r) for r in rows])


def _fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.0f}" if kind == "tok" else f"{v:.2f}"


def _load(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln, parse_constant=lambda c: float("nan")))
            except json.JSONDecodeError:
                pass
    return rows


def result_sets(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("runs.jsonl") if ".venv" not in p.parts)


def status(root: Path = Path(".")) -> str:
    out: list[str] = []
    sets = [d for d in result_sets(root) if "results" in str(d)]
    # ---- 1. inventory
    inv = []
    for d in sets:
        rows = _load(d / "runs.jsonl")
        failed = []
        if (d / "failed.json").exists():
            try:
                failed = json.loads((d / "failed.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        kinds = defaultdict(int)
        for f in failed:
            kinds[f["error"].split(":")[0]] += 1
        inv.append([str(d), len(rows), len(failed), ", ".join(f"{k}x{v}" for k, v in kinds.items()) or "-",
                    ",".join(sorted({str(r["group"].get("repo", "synthetic")) for r in rows})) or "-",
                    f"{min((r['seed'] for r in rows), default='-')}..{max((r['seed'] for r in rows), default='-')}"])
    out += ["## 1. Result sets", "", _table(inv, ["directory", "runs", "failed", "failure types", "repos", "seeds"]), ""]

    # ---- 2. per-architecture tables (real-LLM sets only: they have repo or llm model in the name)
    for d in sets:
        rows = _load(d / "runs.jsonl")
        if not rows or not any(k in str(d) for k in ("real", "ollama")):
            continue
        by = defaultdict(list)
        for r in rows:
            by[(r["group"].get("env", "-"), r["group"]["arch"])].append(r)
        tbl = []
        for env in sorted({k[0] for k in by}):
            for arch in sorted({k[1] for k in by if k[0] == env}, key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else 99):
                rs = by[(env, arch)]
                m = lambda key: [r["metrics"][key] for r in rs if isinstance(r["metrics"].get(key), (int, float))
                                 and not math.isnan(r["metrics"][key])]
                acc, tok, calls, reads = m("question_accuracy"), m("tokens_total"), m("llm_calls"), m("file_reads")
                stale = m("stale_answer_rate")
                tbl.append([env, arch, len(rs),
                            f"{statistics.fmean(acc):.2f}" + (f" ±{1.96 * statistics.stdev(acc) / math.sqrt(len(acc)):.2f}"
                                                              if len(acc) > 1 else ""),
                            _fmt(statistics.fmean(tok), "tok") if tok else "-",
                            _fmt(statistics.fmean(calls), "tok") if calls else "-",
                            _fmt(statistics.fmean(reads), "tok") if reads else "-",
                            _fmt(statistics.fmean(stale), "acc") if stale else "-",
                            _fmt(statistics.fmean(m("auto_resolved")), "acc") if m("auto_resolved") else "-",
                            _fmt(statistics.fmean(m("digest_reads")), "acc") if m("digest_reads") else "-"])
        out += [f"## 2. {d}", "", _table(tbl, ["env", "arch", "runs", "accuracy (±95%CI)", "tokens", "LLM calls",
                                              "file reads", "stale", "auto-resolved", "digest reads"]), ""]

    # ---- 3. spend and cache
    cache = root / ".llm_cache"
    sp = []
    if cache.exists():
        for m in sorted(p for p in cache.iterdir() if p.is_dir()):
            n = sum(1 for _ in m.glob("*.json"))
            led = m / "_spend.log"
            usd = sum(float(x) for x in led.read_text(encoding="utf-8").split()) if led.exists() else 0.0
            sp.append([m.name, n, f"${usd:.2f}" if led.exists() else "free tier"])
    out += ["## 3. LLM calls cached and spend (replays from the cache are free)", "",
            _table(sp, ["model", "cached calls", "spend"]), ""]
    return "\n".join(out)
