"""Summarise a results directory for a paper.

* per-cell tables (mean and 95% bootstrap CI) for every architecture
* paired comparisons of `bacm` (and `bacm_validate`) against each baseline, pairing runs that share
  everything except the architecture (same seed, same repo, same environment)
* exact sign-flip permutation tests (Monte-Carlo when there are many pairs) with Holm correction over
  all comparisons in the report, plus the number of pairs needed for 80% power

  python -m bacm report results_real/<dir>
"""
from __future__ import annotations

import itertools
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["question_accuracy", "tokens_total", "llm_calls", "file_reads", "kb_hits", "stale_answer_rate",
           "latency_sim_s"]
PROPOSED = ("bacm", "bacm_validate", "bacm_auto", "bacm_auto-llm_extraction")
BASELINES = ("independent", "independent_evict", "fully_shared", "selective", "selective_fast",
             "selective_global_recompute", "bacm_v1", "selective_auto", "selective_auto-llm_extraction")


def load(d: Path) -> list[dict]:
    rows = []
    for line in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line, parse_constant=lambda c: float(c)))
    return rows


def boot_ci(xs: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (math.nan, math.nan)
    r = random.Random(seed)
    means = sorted(statistics.fmean(r.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def signflip_p(d: list[float], mc: int = 20000, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(d) == 0 (exact up to 16 pairs)."""
    d = [x for x in d if x == x]
    if not d or all(x == 0 for x in d):
        return 1.0
    obs = abs(sum(d))
    if len(d) <= 16:
        hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
                   for signs in itertools.product((1, -1), repeat=len(d)))
        return hits / 2 ** len(d)
    r = random.Random(seed)
    hits = sum(abs(sum(x if r.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(mc))
    return (hits + 1) / (mc + 1)


def pairs_for_power(d: list[float], alpha: float = 0.05, power: float = 0.8) -> float:
    """Normal-approximation number of pairs for a paired test to detect the observed mean effect."""
    if len(d) < 2:
        return math.nan
    sd = statistics.stdev(d)
    mu = abs(statistics.fmean(d))
    if mu == 0:
        return math.inf
    if sd == 0:
        return float(len(d))
    z = 1.959964 + 0.841621   # alpha=0.05 two-sided, power=0.8
    return math.ceil((z * sd / mu) ** 2)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"


def report(d: Path) -> str:
    rows = load(d)
    gkeys = sorted({k for r in rows for k in r["group"] if k != "arch"})
    cells: dict[tuple, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cell = tuple((k, r["group"].get(k)) for k in gkeys)
        cells[cell][r["group"]["arch"]][r["seed"]] = r["metrics"]
    archs = list(dict.fromkeys(r["group"]["arch"] for r in rows))
    out = [f"# Results: {d}", "", f"{len(rows)} runs."]

    for cell in sorted(cells):
        title = ", ".join(f"{k}={v}" for k, v in cell) or "all"
        out += ["", f"## {title}", "", "| arch | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
        for a in archs:
            runs = cells[cell].get(a, {})
            if not runs:
                continue
            vals = []
            for m in METRICS:
                xs = [x[m] for x in runs.values() if isinstance(x.get(m), (int, float)) and math.isfinite(x[m])]
                if not xs:
                    vals.append("-")
                    continue
                lo, hi = boot_ci(xs)
                vals.append(f"{fmt(statistics.fmean(xs))} [{fmt(lo)}, {fmt(hi)}]" if len(xs) > 1 else fmt(xs[0]))
            out.append(f"| {a} | {len(runs)} | " + " | ".join(vals) + " |")

    # pooled paired comparisons per environment (pairs = same seed and same other conditions)
    envkey = "env" if "env" in gkeys else None
    envs = sorted({dict(c).get(envkey) for c in cells}) if envkey else [None]
    tests = []
    for env in envs:
        for prop in PROPOSED:
            for base in BASELINES:
                acc, tok, n, tot_p, tot_b = [], [], 0, 0.0, 0.0
                for cell, by_arch in cells.items():
                    if envkey and dict(cell).get(envkey) != env:
                        continue
                    if prop not in by_arch or base not in by_arch:
                        continue
                    for s in sorted(set(by_arch[prop]) & set(by_arch[base])):
                        p, b = by_arch[prop][s], by_arch[base][s]
                        acc.append(p["question_accuracy"] - b["question_accuracy"])
                        if b["tokens_total"]:
                            tok.append(p["tokens_total"] / b["tokens_total"] - 1)
                            tot_p += p["tokens_total"]
                            tot_b += b["tokens_total"]
                        n += 1
                if n:
                    tests.append((env, prop, base, "accuracy", acc, None))
                    tests.append((env, prop, base, "tokens", tok, tot_p / tot_b - 1 if tot_b else math.nan))
    if tests:
        pvals = [signflip_p(t[4]) for t in tests]
        adj = holm(pvals)
        out += ["", "## Paired comparisons (pooled over repos/conditions; Holm-corrected over all rows)", "",
                "| env | proposed | baseline | metric | pairs | mean diff [95% CI] | pooled | wins/losses | p | p (Holm) | "
                "pairs for 80% power |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (env, prop, base, metric, diffs, pooled), p, pa in zip(tests, pvals, adj):
            lo, hi = boot_ci(diffs)
            unit = "%" if metric == "tokens" else ""
            f = (lambda v: f"{v:+.1%}") if metric == "tokens" else (lambda v: f"{v:+.3f}")
            wins = sum(x > 0 for x in diffs) if metric == "accuracy" else sum(x < 0 for x in diffs)
            losses = sum(x < 0 for x in diffs) if metric == "accuracy" else sum(x > 0 for x in diffs)
            mean = statistics.fmean(diffs) if diffs else math.nan
            ci = f"[{f(lo)}, {f(hi)}]" if lo == lo else ""
            sig = " **" if pa < 0.05 else ""
            pooled_s = f(pooled) if pooled is not None and pooled == pooled else ""
            out.append(f"| {env} | {prop} | {base} | {metric}{unit and ' change'} | {len(diffs)} | {f(mean)} {ci} | "
                       f"{pooled_s} | {wins}/{losses} | {p:.4f} | {pa:.4f}{sig} | {fmt(pairs_for_power(diffs))} |")
        out += ["", "`**` = significant at 0.05 after Holm correction. Accuracy diff > 0 and token change < 0 "
                "favour the proposed system. 'pooled' token change = total tokens of the proposed system over "
                "total tokens of the baseline across all pairs (robust to cheap outlier baselines); the test uses "
                "the per-pair ratios."]
    return "\n".join(out)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        text = report(Path(arg))
        (Path(arg) / "report.md").write_text(text, encoding="utf-8")
        print(text)
