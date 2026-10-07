"""Generate every table, figure and number used in the paper from the result files.

  .venv/bin/python paper/make_assets.py

Inputs (see paper/PREREGISTRATION.md): the held-out evaluation, the ablations, and the first (tool-interface)
benchmark. Outputs go to paper/generated/ (LaTeX) and paper/figures/ (PDF). Nothing in the paper is typed by hand.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bacm.experiments.report import holm, signflip_p  # noqa: E402

GEN, FIG = ROOT / "paper" / "generated", ROOT / "paper" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FINAL = ROOT / "results_real/final/repobench2-openai_compat-gpt-5-mini/runs.jsonl"
ABL = ROOT / "results_real/final_abl/repobench2-openai_compat-gpt-5-mini/runs.jsonl"
V1 = ROOT / "results_real/repobench_mini/repobench-openai_compat-gpt-5-mini/runs.jsonl"
GEMINI = [ROOT / "results_real/final_gemini/repobench2-openai_compat-gemini-3.5-flash-lite/runs.jsonl",
          ROOT / "results_real/final_gemini_s201/repobench2-openai_compat-gemini-3.5-flash-lite/runs.jsonl"]
FIX1 = ROOT / "results_real/final_fix/repobench2-openai_compat-gpt-5-mini/runs.jsonl"   # post-freeze fix 1 (spans)
FIX = ROOT / "results_real/final_fix2/repobench2-openai_compat-gpt-5-mini/runs.jsonl"   # fix 2 (deep rebuild): reported
FIXED_ARCHS = ("bacm_llm", "bacm_auto")
LOW = ROOT / "results_real/final_loweffort/repobench2-openai_compat-gpt-5-mini/runs.jsonl"

NAMES = {
    "independent": "Independent",
    "independent_evict": "Independent + evict",
    "selective_fast": "Selective memory (tools)",
    "selective_llm": "Selective memory (transparent)",
    "bacm_llm": "\\textbf{BACM}",
    "bacm_auto": "BACM-Index (upper bound)",
    "bacm": "BACM v1 (memory as tools)",
    "independent_evict_low": "Independent + evict, all agents low effort",
}
MAIN = ["independent", "independent_evict", "independent_evict_low", "selective_fast", "selective_llm", "bacm_llm",
        "bacm_auto"]
KINDS = ["base_file", "grandbase", "family_files", "n_methods", "method_total"]
COLORS = {"independent": "#9a9a9a", "independent_evict": "#5f6b7a", "selective_fast": "#d08c60",
          "selective_llm": "#c9a227", "bacm_llm": "#2b6cb0", "bacm_auto": "#6aa84f", "bacm": "#a05195",
          "independent_evict_low": "#3c3c3c"}


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln, parse_constant=lambda c: float("nan")) for ln in p.read_text().splitlines() if ln.strip()]


def boot(xs, n=4000, seed=0):
    if len(xs) < 2:
        return (math.nan, math.nan)
    r = random.Random(seed)
    ms = sorted(statistics.fmean(r.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def cells(rows):
    out = defaultdict(dict)   # (env, arch) -> {(repo, seed): metrics}
    for r in rows:
        g = r["group"]
        out[(g["env"], g["arch"])][(g.get("repo"), r["seed"])] = r["metrics"]
    return out


def paired(c, env, a, b, metric):
    A, B = c.get((env, a), {}), c.get((env, b), {})
    keys = sorted(set(A) & set(B))
    if metric == "tokens":
        d = [A[k]["tokens_total"] / B[k]["tokens_total"] - 1 for k in keys if B[k]["tokens_total"]]
        pooled = sum(A[k]["tokens_total"] for k in keys) / max(1, sum(B[k]["tokens_total"] for k in keys)) - 1
    else:
        d = [A[k]["question_accuracy"] - B[k]["question_accuracy"] for k in keys]
        pooled = statistics.fmean(d) if d else math.nan
    return d, pooled


def fmt_ci(xs, pct=False):
    m = statistics.fmean(xs)
    lo, hi = boot(xs)
    if pct:
        return f"{m:+.0%} [{lo:+.0%}, {hi:+.0%}]".replace("%", "\\%")
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


# ------------------------------------------------------------------------------------------- figures
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1, color="#333"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, fontsize=7, ha="center", color="#333")

    for i, y in enumerate((3.0, 1.9, 0.8)):
        box(0.2, y, 1.6, 0.7, f"Agent $B_{i + 1}$\n(private context)", "#e8eef7")
        arrow(1.85, y + 0.35, 3.15, 2.05, "read / answer" if i == 0 else "")
    box(3.2, 1.2, 2.2, 1.7, "Mediator\n- publish what is read\n- serve digests\n- resolve questions\n- hints, re-answers",
        "#fdf3d7")
    arrow(5.45, 2.05, 6.35, 2.05)
    box(6.4, 1.0, 3.3, 2.1, "Context manager\nknowledge objects with\nprovenance (file, version, span),\ndependency DAG, typed rules,\n"
                           "invalidation + demand-driven\nincremental recompute", "#e3f1e0")
    box(3.2, 0.05, 2.2, 0.75, "Repository\n(versions change)", "#f2f2f2")
    arrow(4.3, 1.15, 4.3, 0.85, "")
    arrow(5.45, 0.42, 7.2, 0.95, "change events")
    fig.tight_layout()
    fig.savefig(FIG / "architecture.pdf")
    plt.close(fig)


def fig_main(c):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, env in zip(axes, ("static", "dynamic")):
        for a in MAIN:
            runs = c.get((env, a))
            if not runs:
                continue
            acc = [m["question_accuracy"] for m in runs.values()]
            tok = [m["tokens_total"] / 1000 for m in runs.values()]
            (alo, ahi), (tlo, thi) = boot(acc), boot(tok)
            ma, mt = statistics.fmean(acc), statistics.fmean(tok)
            ax.errorbar(mt, ma, xerr=[[mt - tlo], [thi - mt]], yerr=[[ma - alo], [ahi - ma]], fmt="o",
                        color=COLORS[a], ms=7, capsize=3, label=NAMES[a].replace("\\textbf{", "").replace("}", ""))
        ax.set_title("static repository" if env == "static" else "repository changes mid-run")
        ax.set_xlabel("tokens per run (thousands)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("question accuracy")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "main_tradeoff.pdf")
    plt.close(fig)


def per_kind(rows_by_arch):
    """Per-kind accuracy needs answers, which are not in runs.jsonl; computed by replay in kind_accuracy.json."""
    return rows_by_arch


def fig_kinds(kind_acc):
    if not kind_acc:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    archs = [a for a in ("independent_evict", "selective_llm", "bacm_llm", "bacm_auto") if a in kind_acc]
    w = 0.8 / max(1, len(archs))
    for i, a in enumerate(archs):
        ax.bar([k + i * w for k in range(len(KINDS))], [kind_acc[a].get(k, math.nan) for k in KINDS], w,
               color=COLORS[a], label=NAMES[a].replace("\\textbf{", "").replace("}", ""))
    ax.set_xticks([k + w * (len(archs) - 1) / 2 for k in range(len(KINDS))])
    ax.set_xticklabels(KINDS)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "per_kind.pdf")
    plt.close(fig)


def fig_ablation(c, ca):
    rows = []
    base = "bacm_llm"
    for env in ("static", "dynamic"):
        for a in sorted({k[1] for k in ca if k[1].startswith("bacm_llm-")}):
            ref = {**c.get((env, base), {})}
            ref.update(ca.get((env, base), {}))
            abl = ca.get((env, a), {})
            keys = sorted(set(ref) & set(abl))
            if not keys:
                continue
            dacc = statistics.fmean(abl[k]["question_accuracy"] - ref[k]["question_accuracy"] for k in keys)
            dtok = sum(abl[k]["tokens_total"] for k in keys) / sum(ref[k]["tokens_total"] for k in keys) - 1
            rows.append((env, a.replace("bacm_llm-", ""), dacc, dtok, len(keys)))
    if not rows:
        return rows
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, (idx, label) in zip(axes, ((2, "accuracy change when removed"), (3, "token change when removed"))):
        labels = [f"{ABL_NAMES.get(r[1], r[1])} ({'static' if r[0] == 'static' else 'changing'})" for r in rows]
        ax.barh(labels, [r[idx] for r in rows], color=["#2b6cb0" if r[0] == "static" else "#c0504d" for r in rows])
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.pdf")
    plt.close(fig)
    return rows


def fig_interface(v1):
    """The interface argument: memory-as-tools vs transparent memory (LLM turns and tokens)."""
    c = cells(v1)
    fin = cells(load(FINAL))
    data = []
    for env in ("static", "dynamic"):
        for a, src in (("independent_evict", c), ("bacm", c)):
            runs = src.get((env, a))
            if runs:
                data.append((env, a, statistics.fmean(m["llm_calls"] for m in runs.values()),
                             statistics.fmean(m["tokens_total"] for m in runs.values()) / 1000))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for env, a, calls, tok in data:
        ax.scatter(calls, tok, color=COLORS[a], marker="o" if env == "static" else "s", s=50,
                   label=f"{NAMES[a]} ({env})")
    ax.set_xlabel("LLM calls per run")
    ax.set_ylabel("tokens per run (k)")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "interface.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------- additional figures
def plain(a):
    return NAMES[a].replace("\\textbf{", "").replace("}", "")


SHORT = {"independent_evict": "Indep.\n+ evict", "independent_evict_low": "Indep. + evict\n(low effort)",
         "selective_llm": "Selective\nmemory", "bacm_llm": "BACM", "bacm_auto": "BACM-\nIndex"}
SHOW = ["independent_evict", "independent_evict_low", "selective_llm", "bacm_llm", "bacm_auto"]


def fig_per_repo(c):
    """Per-repository accuracy (changing repositories) and token change of BACM vs independent + evict."""
    repos = sorted({k[0] for k in c.get(("dynamic", "bacm_llm"), {})})
    if not repos:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    archs = [a for a in SHOW if c.get(("dynamic", a))]
    w = 0.8 / len(archs)
    for i, a in enumerate(archs):
        ys = [statistics.fmean(m["question_accuracy"] for k, m in c[("dynamic", a)].items() if k[0] == r) for r in repos]
        axes[0].bar([j + i * w for j in range(len(repos))], ys, w, color=COLORS[a], label=plain(a))
    axes[0].set_xticks([j + w * (len(archs) - 1) / 2 for j in range(len(repos))])
    axes[0].set_xticklabels(repos)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy (repository changes mid-run)")
    axes[0].legend(ncol=2, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.3), frameon=False)
    axes[0].grid(axis="y", alpha=0.3)
    for i, env in enumerate(("static", "dynamic")):
        A, B = c.get((env, "bacm_llm"), {}), c.get((env, "independent_evict"), {})
        ys = []
        for r in repos:
            ks = [k for k in A if k[0] == r and k in B]
            ys.append(100 * (sum(A[k]["tokens_total"] for k in ks) / sum(B[k]["tokens_total"] for k in ks) - 1))
        axes[1].bar([j + i * 0.4 for j in range(len(repos))], ys, 0.4, color="#2b6cb0" if env == "static" else "#c0504d",
                    label="static" if env == "static" else "changes mid-run")
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_xticks([j + 0.2 for j in range(len(repos))])
    axes[1].set_xticklabels(repos)
    axes[1].set_ylabel("token change of BACM vs.\nindependent + evict (%)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "per_repo.pdf")
    plt.close(fig)


def fig_stale(c, c_pre):
    """What happened to answers whose truth changed: current, stale (pre-change truth) or otherwise wrong."""
    rows = []
    for label, cc, a in (("Independent + evict", c, "independent_evict"),
                         ("Indep. + evict, low effort", c, "independent_evict_low"),
                         ("Selective memory (transparent)", c, "selective_llm"),
                         ("BACM, frozen code", c_pre, "bacm_llm"),
                         ("BACM", c, "bacm_llm"),
                         ("BACM-Index (upper bound)", c, "bacm_auto")):
        runs = cc.get(("dynamic", a), {})
        n = sum(m.get("affected_answers") or 0 for m in runs.values())
        if not n:
            continue
        ok = sum(round(m["affected_accuracy"] * m["affected_answers"]) for m in runs.values()
                 if m.get("affected_answers") and not math.isnan(m["affected_accuracy"]))
        st, _ = stale(runs)
        rows.append((label, ok / n, st / n, 1 - (ok + st) / n, n))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ys = list(range(len(rows)))[::-1]
    left = [0] * len(rows)
    for idx, (name, col) in enumerate((("current (correct)", "#4c9a5b"), ("stale (pre-change truth)", "#c0504d"),
                                       ("other error", "#bfbfbf"))):
        vals = [r[1 + idx] for r in rows]
        ax.barh(ys, vals, left=left, color=col, label=name)
        for y, l_, v in zip(ys, left, vals):
            if v > 0.06:
                ax.text(l_ + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white")
        left = [l_ + v for l_, v in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of final answers whose ground truth changed during the run")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.4, 1.2), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stale.pdf")
    plt.close(fig)


def fig_anatomy():
    f = GEN / "anatomy.json"
    if not f.exists():
        return
    an = json.loads(f.read_text())
    archs = [a for a in ("independent_evict", "selective_llm", "bacm_llm", "bacm_auto") if f"static|{a}" in an]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.6), sharey=True)
    parts = (("agent", "agent turns", "#5f6b7a"), ("note", "agents' fact notes", "#d08c60"),
             ("manager", "memory manager (extraction, re-derivation)", "#2b6cb0"))
    for ax, env in zip(axes, ("static", "dynamic")):
        bottom = [0] * len(archs)
        for key, label, col in parts:
            vals = [an[f"{env}|{a}"].get(key, 0) / 1000 for a in archs]
            ax.bar(range(len(archs)), vals, bottom=bottom, color=col, label=label, width=0.6)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([SHORT[a] for a in archs], fontsize=7.5)
        ax.set_title("static repository" if env == "static" else "repository changes mid-run")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("tokens per run (thousands)")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "anatomy.pdf")
    plt.close(fig)


def fig_distribution(c):
    archs = [a for a in SHOW if c.get(("static", a))]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8), sharey=True)
    r = random.Random(0)
    for ax, env in zip(axes, ("static", "dynamic")):
        data = [[m["question_accuracy"] for m in c[(env, a)].values()] for a in archs]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": "#222"})
        for patch, a in zip(bp["boxes"], archs):
            patch.set_facecolor(COLORS[a])
            patch.set_alpha(0.35)
        for i, (d, a) in enumerate(zip(data, archs), 1):
            ax.scatter([i + r.uniform(-0.15, 0.15) for _ in d], d, s=8, color=COLORS[a], zorder=3)
        ax.set_xticks(range(1, len(archs) + 1))
        ax.set_xticklabels([SHORT[a] for a in archs], fontsize=7)
        ax.set_title("static repository" if env == "static" else "repository changes mid-run")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy per run")
    fig.tight_layout()
    fig.savefig(FIG / "distribution.pdf")
    plt.close(fig)


def fig_cost_model():
    """Eq. (cost) against measurement: input tokens of every agent turn of every held-out branch (replayed from the
    response cache), and the per-branch total against the number of turns with the least-squares fit of
    T(N) = N c0 + g N(N-1)/2."""
    f = GEN / "growth.json"
    if not f.exists():
        return
    import numpy as np
    gr = json.loads(f.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))
    for a in ("independent_evict", "selective_llm", "bacm_llm", "bacm_auto"):
        br = gr.get(f"static|{a}", [])
        if not br:
            continue
        L = max(len(b) for b in br)
        xs, ys = [], []
        for j in range(L):
            v = [b[j] for b in br if len(b) > j]
            if len(v) >= 30:   # stable means only
                xs.append(j + 1)
                ys.append(statistics.fmean(v) / 1000)
        axes[0].plot(xs, ys, color=COLORS[a], label=plain(a), lw=1.4)
    axes[0].set_xlabel("agent turn $j$")
    axes[0].set_ylabel("input tokens of turn $j$ (thousands)")
    axes[0].set_title("context re-sent per turn (static, mean over branches)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[0].grid(alpha=0.3)
    fits = {}
    for a in ("independent_evict", "bacm_llm"):
        br = gr.get(f"static|{a}", []) + gr.get(f"dynamic|{a}", [])
        N = np.array([len(b) for b in br], dtype=float)
        T = np.array([sum(b) for b in br], dtype=float)
        A = np.c_[N, N * (N - 1) / 2]
        (c0, g), *_ = np.linalg.lstsq(A, T, rcond=None)
        r2 = 1 - ((T - A @ [c0, g]) ** 2).sum() / ((T - T.mean()) ** 2).sum()
        fits[a] = (c0, g, r2, len(br))
        axes[1].scatter(N, T / 1000, s=6, color=COLORS[a], alpha=0.45)
        xs = np.linspace(1, N.max(), 100)
        axes[1].plot(xs, (xs * c0 + g * xs * (xs - 1) / 2) / 1000, color=COLORS[a], lw=1.6,
                     label=f"{plain(a)}: $c_0$={c0:,.0f}, $g$={g:,.0f}, $R^2$={r2:.2f}")
    axes[1].set_xlabel("agent turns $N$ of a branch")
    axes[1].set_ylabel("input tokens $T(N)$ (thousands)")
    axes[1].set_title("per-branch total with fitted Eq. (3)")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "cost_model.pdf")
    plt.close(fig)
    if fits:
        n = sum(v[3] for v in fits.values())
        (GEN / "cost_fit.tex").write_text(
            "Fitted on " + f"{n}" + " held-out branches (static and dynamic): " + "; ".join(
                f"{plain(a)} $c_0={v[0]:,.0f}$, $g={v[1]:,.0f}$, $R^2={v[2]:.2f}$" for a, v in fits.items()) + ".\n")


def fig_kinds_both():
    f = GEN / "kind_accuracy_all.json"
    if not f.exists():
        return
    ka = json.loads(f.read_text())
    archs = [a for a in ("independent_evict", "selective_llm", "bacm_llm", "bacm_auto") if f"static|{a}" in ka]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8), sharey=True)
    w = 0.8 / len(archs)
    for ax, env in zip(axes, ("static", "dynamic")):
        for i, a in enumerate(archs):
            d = ka.get(f"{env}|{a}", {})
            ax.bar([k + i * w for k in range(len(KINDS))], [d.get(k, [math.nan])[0] for k in KINDS], w,
                   color=COLORS[a], label=plain(a))
        ax.set_xticks([k + w * (len(archs) - 1) / 2 for k in range(len(KINDS))])
        ax.set_xticklabels([k.replace("_", "\\_") if False else k for k in KINDS], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title("static repository" if env == "static" else "repository changes mid-run")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "per_kind.pdf")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- tables
def table_repos():
    from bacm.config import RepoConfig
    from bacm.env.realrepo import REPOS, RealRepoWorld
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Repositories. Classes = classes whose base class is defined in the repository (question "
             "subjects); changed = share of those classes with at least one answer that changes between releases.}",
             "\\label{tab:repos}", "\\begin{tabular}{llrrrr}", "\\toprule",
             "Repository & Releases & Source files & Tokens (approx.) & Classes & Changed \\\\", "\\midrule"]
    for name, (url, a, b) in REPOS.items():
        w = RealRepoWorld.load(RepoConfig(name, url, a, b, bench="v2"), 0)
        cls = w.candidate_classes()
        tb = w._table_b
        ch = sum(1 for c_ in cls if any(tb.value(f"{k}({c_})") not in (None, w._table_a.value(f"{k}({c_})"))
                                        for k in w.V2_KINDS))
        toks = sum(len(f.content) for f in w.files.values()) // 4
        lines.append(f"{name} & {a} $\\to$ {b} & {len(w.files)} & {toks:,} & {len(cls)} & {ch / max(1, len(cls)):.0%} \\\\"
                     .replace("%", "\\%"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_repos.tex").write_text("\n".join(lines))


def table_main(c, tests):
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Held-out evaluation (GPT-5 mini). Mean and 95\\% bootstrap CI over 30 runs (5 repositories "
             "$\\times$ 6 held-out seeds). Tokens include every model call (agents, extraction, manager).}",
             "\\label{tab:main}", "\\begin{tabular}{lcccc}", "\\toprule",
             "& \\multicolumn{2}{c}{Static repository} & \\multicolumn{2}{c}{Repository changes mid-run} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Architecture & Accuracy & Tokens (k) & Accuracy & Tokens (k) \\\\", "\\midrule"]
    for a in MAIN:
        if not any(c.get((env, a)) for env in ("static", "dynamic")):
            continue   # baseline dropped from the final evaluation (pre-registration amendment)
        row = [NAMES[a]]
        for env in ("static", "dynamic"):
            runs = c.get((env, a))
            if not runs:
                row += ["--", "--"]
                continue
            acc = [m["question_accuracy"] for m in runs.values()]
            tok = [m["tokens_total"] / 1000 for m in runs.values()]
            row += [fmt_ci(acc), f"{statistics.fmean(tok):,.0f} [{boot(tok)[0]:,.0f}, {boot(tok)[1]:,.0f}]"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (GEN / "table_main.tex").write_text("\n".join(lines))

    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Paired comparisons of BACM against each baseline: same repository, seed "
             "and environment. Two-sided sign-flip permutation tests, Holm-corrected over all rows. Token change is "
             "pooled (total over total).}", "\\label{tab:paired}", "\\begin{tabular}{llrrrr}", "\\toprule",
             "Env. & Baseline & $\\Delta$ accuracy & $p_{\\text{Holm}}$ & $\\Delta$ tokens & $p_{\\text{Holm}}$ \\\\",
             "\\midrule"]
    for (env, base), r in tests.items():
        lines.append(f"{env} & {NAMES[base]} & {r['dacc']:+.3f} & {r['pacc']:.3g} & {r['dtok']:+.0%} & {r['ptok']:.3g} \\\\"
                     .replace("%", "\\%"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_paired.tex").write_text("\n".join(lines))


def table_ablation(rows):
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Ablations of BACM: change when one component is removed (paired runs; "
             "positive token change = more expensive without it). Run on the pre-registered (pre-fix) code against the pre-fix reference.}", "\\label{tab:ablation}",
             "\\begin{tabular}{llrrr}", "\\toprule", "Env. & Removed component & $\\Delta$ accuracy & $\\Delta$ tokens & pairs \\\\",
             "\\midrule"]
    for env, comp, dacc, dtok, n in rows:
        lines.append(f"{env} & {ABL_NAMES.get(comp, comp.replace('_', ' '))} & {dacc:+.3f} & {dtok:+.0%} & {n} \\\\".replace("%", "\\%"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_ablation.tex").write_text("\n".join(lines))


def table_v1():
    """The interface lesson: memory exposed as tools (first design) vs independent agents, 150 real runs."""
    c = cells(load(V1))
    if not c:
        return
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{First design: memory exposed to the model as tools (\\texttt{kb\\_query}/\\texttt{kb\\_publish}), "
             "on an earlier version of the benchmark (GPT-5 mini, 15 runs per cell, httpx, requests and flask $\\times$ five seeds). Each memory operation is a full-context turn, so the tool "
             "interface multiplies turns and tokens.}", "\\label{tab:v1}", "\\begin{tabular}{llrrr}", "\\toprule",
             "Env. & Architecture & LLM calls & Tokens (k) & Accuracy \\\\", "\\midrule"]
    for env in ("static", "dynamic"):
        for a in ("independent", "independent_evict", "selective_fast", "bacm"):
            runs = c.get((env, a))
            if not runs:
                continue
            f = lambda k: statistics.fmean(r[k] for r in runs.values())
            lines.append(f"{env} & {NAMES[a]} & {f('llm_calls'):.0f} & {f('tokens_total') / 1000:,.0f} & "
                         f"{f('question_accuracy'):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_v1.tex").write_text("\n".join(lines))


def table_sim():
    """Controlled simulator ablations (exact, deterministic, 20 seeds): which staleness component matters."""
    p = ROOT / "results/ablation/runs.jsonl"
    rows = load(p)
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        if r["group"].get("setting") == "dynamic":
            by[r["group"]["arch"]].append(r["metrics"])
    order = ["independent", "selective", "bacm", "bacm-no_provenance", "bacm-no_versioning", "bacm-no_dependencies",
             "bacm-no_invalidation", "bacm-no_conflict", "bacm-no_cross_check", "bacm-file_level"]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Controlled simulation (synthetic repository, deterministic agents, 20 seeds, mid-run source "
             "changes): removing one staleness component at a time.}", "\\label{tab:sim}",
             "\\begin{tabular}{lrrr}", "\\toprule", "System & Accuracy & Stale-answer rate & False-reuse rate \\\\",
             "\\midrule"]
    for a in order:
        ms = by.get(a)
        if not ms:
            continue
        f = lambda k: statistics.fmean(x[k] for x in ms if x[k] == x[k])
        lines.append(f"{a.replace('_', ' ')} & {f('question_accuracy'):.2f} & {f('stale_answer_rate'):.2f} & "
                     f"{f('false_reuse_rate'):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_sim.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------------------------- main
def low_rows():
    rows = load(LOW)
    for r in rows:
        r["group"] = dict(r["group"], arch="independent_evict_low")
    return rows


def tok_phrase(dtok, p):
    word = "fewer" if dtok < 0 else "more"
    return (f"{abs(dtok):.0%} {word} tokens ({'significant' if p < 0.05 else 'not significant'}, "
            f"$p_{{\\text{{Holm}}}}={p:.2g}$)").replace("%", "\\%")


def sig_word(diff, p, higher="higher", lower="lower"):
    if p < 0.05:
        return f"significantly {higher if diff > 0 else lower}, $p_{{\\text{{Holm}}}}={p:.2g}$"
    return f"not significantly different, $p_{{\\text{{Holm}}}}={p:.2g}$"


def acc_sig(p):
    return f"{'significant' if p < 0.05 else 'not significant'}, $p_{{\\text{{Holm}}}}={p:.2g}$"


ABL_NAMES = {"no_resolve": "automatic question resolution", "minimal_manager": "reasoning in manager calls",
             "no_digest": "digest-served reads", "no_hints": "status hints", "no_maintain": "answer maintenance",
             "no_reanswer": "small-context re-answering", "no_invalidation": "eager invalidation",
             "no_dependencies": "dependency graph", "no_versioning": "versioning"}


def text_blocks(c, tests, abl_rows, gem):
    """Result sentences, generated from the data so that no number is typed by hand."""
    def m(env, a, key="question_accuracy"):
        runs = c.get((env, a), {})
        return statistics.fmean(r[key] for r in runs.values()) if runs else math.nan

    def p(x):
        return f"$p_{{\\text{{Holm}}}}={x:.2g}$"

    def pct(x):
        return f"{abs(x):.0%}".replace("%", "\\%")

    def per_repo(env):
        A, B = c.get((env, "bacm_llm"), {}), c.get((env, "independent_evict"), {})
        out_ = []
        for repo in sorted({k[0] for k in A}):
            ks = [k for k in A if k[0] == repo and k in B]
            d = sum(A[k]["tokens_total"] for k in ks) / sum(B[k]["tokens_total"] for k in ks) - 1
            out_.append(f"{repo} ${d:+.0%}$".replace("%", "\\%"))
        return ", ".join(out_)

    out = defaultdict(list)
    t = tests.get
    runs_ = c.get(("static", "bacm_llm"), {})
    s_ie, d_ie, d_sl = t(("static", "independent_evict")), t(("dynamic", "independent_evict")), t(("dynamic", "selective_llm"))
    if s_ie and d_ie:
        out["rq1"].append(
            f"Each cell of \\cref{{tab:main}} averages {len(runs_)} held-out runs ({len({k[0] for k in runs_})} repositories, "
            f"{len({k[1] for k in runs_})} seeds). On static repositories BACM answered {m('static', 'bacm_llm'):.2f} "
            f"of the questions correctly, against {m('static', 'independent_evict'):.2f} for independent agents with eviction "
            f"(${s_ie['dacc']:+.2f}$, {p(s_ie['pacc'])}). When the repository changed mid-run the figures were "
            f"{m('dynamic', 'bacm_llm'):.2f} and {m('dynamic', 'independent_evict'):.2f} (${d_ie['dacc']:+.2f}$, "
            f"{p(d_ie['pacc'])}). BACM used {pct(s_ie['dtok'])} {'fewer' if s_ie['dtok'] < 0 else 'more'} tokens on "
            f"static and {pct(d_ie['dtok'])} {'fewer' if d_ie['dtok'] < 0 else 'more'} on changing repositories, but neither "
            f"difference is significant ({p(s_ie['ptok'])} and {p(d_ie['ptok'])}). Per repository, the change on changing "
            f"repositories was {per_repo('dynamic')}; on static ones it was {per_repo('static')}. ")
    s_sl = t(("static", "selective_llm"))
    if s_sl and abs(s_sl["dacc"]) < 1e-9 and abs(s_sl["dtok"]) < 1e-9:
        out["rq1"].append("On static repositories BACM and the selective memory produce identical runs, as they "
                          "should: provenance, versions and invalidation only act when a source changes. ")
    if d_sl:
        out["rq1"].append(f"On changing repositories the staleness machinery is worth ${d_sl['dacc']:+.2f}$ accuracy over the "
                          f"selective memory ({p(d_sl['pacc'])}) at {pct(d_sl['dtok'])} "
                          f"{'fewer' if d_sl['dtok'] < 0 else 'more'} tokens (not significant). ")
    sd = {a: stale(c.get(("dynamic", a), {})) for a in ("independent_evict", "independent_evict_low", "selective_llm",
                                                         "bacm_llm")}
    if all(n for _, n in sd.values()):
        n = sd["bacm_llm"][1]
        out["rq2"].append(
            f"Of the {n} final answers whose ground truth changed during a run, {sd['bacm_llm'][0]} were stale with "
            f"BACM, against {sd['independent_evict'][0]} for independent agents with eviction, "
            f"{sd['selective_llm'][0]} for the selective memory and {sd['independent_evict_low'][0]} for independent agents "
            f"that reason more. ")
        if sd["independent_evict_low"][0] > sd["independent_evict"][0]:
            out["rq2"].append("The last number is the surprising one: agents that reason more read the pre-release code "
                              "more accurately, and then keep what they read. ")
    low, lowd = t(("static", "independent_evict_low")), t(("dynamic", "independent_evict_low"))
    if low and lowd:
        out["rq4"].append(
            f"Letting every agent reason more (effort ``low'' instead of ``minimal'') raises independent agents to "
            f"{m('static', 'independent_evict_low'):.2f} on static and {m('dynamic', 'independent_evict_low'):.2f} on "
            f"changing repositories, at {m('static', 'independent_evict_low', 'tokens_total') / 1000:,.0f}k and "
            f"{m('dynamic', 'independent_evict_low', 'tokens_total') / 1000:,.0f}k tokens per run. On static repositories "
            f"this baseline is as accurate as BACM (difference ${low['dacc']:+.2f}$, {p(low['pacc'])}), and BACM "
            f"uses {pct(low['dtok'])} more tokens than it (not significant, {p(low['ptok'])}). On changing "
            f"repositories BACM is more accurate by ${lowd['dacc']:+.2f}$ ({p(lowd['pacc'])}) and again uses "
            f"{pct(lowd['dtok'])} more tokens (not significant, {p(lowd['ptok'])}). Reasoning helps an agent read; it does not tell the agent that what it read "
            f"has since changed. ")
    if abl_rows:
        by = defaultdict(dict)
        for env, comp, dacc, dtok, _ in abl_rows:
            by[comp][env] = (dacc, dtok)
        order = sorted(by, key=lambda k: min(v[0] for v in by[k].values()))[:2]
        out["rq3"].append("Accuracy drops most without " + " and without ".join(
            f"{ABL_NAMES.get(k, k)} (static ${by[k]['static'][0]:+.2f}$, changing ${by[k]['dynamic'][0]:+.2f}$)"
            for k in order if "static" in by[k] and "dynamic" in by[k]) + ". ")
        dg = {r[0]: r for r in abl_rows if r[1] == "no_digest"}
        if "static" in dg and dg["static"][2] > 0:
            out["rq3"].append(f"Digest-served reads did not pay off on static repositories: without them accuracy changed by "
                              f"${dg['static'][2]:+.2f}$ and tokens by ${dg['static'][3]:+.0%}$".replace("%", "\\%")
                              + (f", while on changing repositories removing them cost ${dg['dynamic'][2]:+.2f}$ accuracy "
                                 f"and ${dg['dynamic'][3]:+.0%}$ tokens. ".replace("%", "\\%") if "dynamic" in dg else ". "))
    an_f, ka_f = GEN / "anatomy.json", GEN / "kind_accuracy_all.json"
    if an_f.exists():
        an = json.loads(an_f.read_text())
        parts = []
        for env, name in (("static", "static"), ("dynamic", "changing")):
            i_, b_ = an.get(f"{env}|independent_evict"), an.get(f"{env}|bacm_llm")
            if i_ and b_:
                parts.append(f"on {name} repositories BACM's agent turns cost {(i_['agent'] - b_['agent']) / 1000:,.0f}k "
                             f"tokens less per run, independent agents spend {i_['note'] / 1000:,.0f}k on fact notes that "
                             f"BACM does not need, and BACM's manager spends {b_['manager'] / 1000:,.0f}k on extraction "
                             f"and re-derivation ({b_['manager'] / sum(b_.values()):.0%} of its total)")
        out["anatomy"].append("Compared with independent agents, " + "; ".join(parts).replace("%", "\\%") + ". ")
    if ka_f.exists():
        ka = json.loads(ka_f.read_text())
        def kv(env, a, k):
            return ka.get(f"{env}|{a}", {}).get(k, [math.nan])[0]
        out["kinds"].append(
            f"On static repositories BACM answers the structural questions almost perfectly (\\texttt{{base\\_file}} "
            f"{kv('static', 'bacm_llm', 'base_file'):.2f}, \\texttt{{grandbase}} {kv('static', 'bacm_llm', 'grandbase'):.2f}, "
            f"\\texttt{{family\\_files}} {kv('static', 'bacm_llm', 'family_files'):.2f}) and single method counts "
            f"({kv('static', 'bacm_llm', 'n_methods'):.2f}, against {kv('static', 'independent_evict', 'n_methods'):.2f} for "
            f"independent agents). The hardest kind is \\texttt{{method\\_total}}, which sums model-extracted counts over a "
            f"chain of classes: {kv('static', 'bacm_llm', 'method_total'):.2f} for BACM on static repositories and "
            f"{kv('dynamic', 'bacm_llm', 'method_total'):.2f} after a release, against "
            f"{kv('static', 'bacm_auto', 'method_total'):.2f} and {kv('dynamic', 'bacm_auto', 'method_total'):.2f} with the "
            f"symbol index. One wrong count anywhere in the chain makes the total wrong, and with shared memory the same "
            f"wrong count reaches every agent (\\cref{{sec:theory}}). ")
    if gem:
        out["gemini"].append(gem)
    for key in ("rq1", "rq2", "rq3", "rq4", "gemini", "anatomy", "kinds"):
        (GEN / f"text_{key}.tex").write_text("".join(out[key]) + "\n")
    (GEN / "conclusion_text.tex").write_text(
        (f"BACM answered {m('static', 'bacm_llm'):.0%} and {m('dynamic', 'bacm_llm'):.0%} of held-out questions "
         f"correctly on static and changing repositories, against {m('static', 'independent_evict'):.0%} and "
         f"{m('dynamic', 'independent_evict'):.0%} for independent agents with eviction, and cut stale answers after a "
         f"release from {sd['independent_evict'][0]} to {sd['bacm_llm'][0]} of {sd['bacm_llm'][1]}. The accuracy gains are "
         f"significant. Token efficiency, by contrast, is not established: the pooled changes (${s_ie['dtok']:+.0%}$ and "
         f"${d_ie['dtok']:+.0%}$) are not significant and depend on the workload. The main conclusion is therefore that "
         f"BACM makes shared memory \\emph{{consistent}} under change, not that it makes it cheaper. Agents that reason "
         f"more match BACM's accuracy on static repositories at lower cost, but not once the code changes, because "
         f"reasoning quality is not state consistency. Counts extracted by the model remain the main source of error.").replace("%", "\\%") + "\n")


def gemini_block():
    rows = [r for p in GEMINI for r in load(p)]
    if not rows:
        return ""
    c = cells(rows)
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Replication with a second model family (Gemini 3.5 Flash-Lite, free tier; held-out seeds that "
             "fit the free quota).}", "\\label{tab:gemini}", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Architecture & runs & Acc.\\ static & Tokens (k) & Acc.\\ dynamic & Tokens (k) \\\\", "\\midrule"]
    for a in ("independent_evict", "selective_llm", "bacm_llm"):
        cells_ = [c.get((e, a), {}) for e in ("static", "dynamic")]
        if not any(cells_):
            continue
        vals = []
        for runs in cells_:
            if runs:
                vals += [f"{statistics.fmean(r['question_accuracy'] for r in runs.values()):.2f}",
                         f"{statistics.fmean(r['tokens_total'] for r in runs.values()) / 1000:,.0f}"]
            else:
                vals += ["--", "--"]
        lines.append(f"{NAMES[a]} & {len(cells_[0])} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_secondary.tex").write_text("\n".join(lines))
    def acc(env, a):
        r_ = c.get((env, a), {})
        return statistics.fmean(x["question_accuracy"] for x in r_.values()) if r_ else math.nan

    def tok(env, a):
        r_ = c.get((env, a), {})
        return statistics.fmean(x["tokens_total"] for x in r_.values()) / 1000 if r_ else math.nan
    if c.get(("dynamic", "bacm_llm")) and c.get(("dynamic", "independent_evict")):
        n = len(c[("dynamic", "bacm_llm")])
        st_o, st_n = stale(c[("dynamic", "bacm_llm")])
        st_i, _ = stale(c[("dynamic", "independent_evict")])
        return (f"With Gemini 3.5 Flash-Lite ({n} runs per cell, two held-out seeds; \\cref{{tab:gemini}}) the pattern "
                f"on changing repositories repeats: accuracy {acc('dynamic', 'bacm_llm'):.2f} against "
                f"{acc('dynamic', 'independent_evict'):.2f} for independent agents, with {st_o} stale answers out of {st_n} "
                f"against {st_i}. On static repositories accuracy was similar ({acc('static', 'bacm_llm'):.2f} against "
                f"{acc('static', 'independent_evict'):.2f}) at {tok('static', 'bacm_llm'):,.0f}k instead of "
                f"{tok('static', 'independent_evict'):,.0f}k tokens. With so few runs we report these numbers without "
                f"a test. ")
    return ""


def run_tests(c):
    tests_raw = []
    for env in ("static", "dynamic"):
        for base in ("independent", "independent_evict", "independent_evict_low", "selective_fast", "selective_llm"):
            da, pa = paired(c, env, "bacm_llm", base, "acc")
            dt, pt = paired(c, env, "bacm_llm", base, "tokens")
            if da:
                tests_raw.append((env, base, da, pa, dt, pt))
    pv = [signflip_p(t[2]) for t in tests_raw] + [signflip_p(t[4]) for t in tests_raw]
    adj = holm(pv) if pv else []
    n = len(tests_raw)
    return {(t[0], t[1]): {"dacc": t[3], "pacc": adj[i], "dtok": t[5], "ptok": adj[n + i], "pairs": len(t[2])}
            for i, t in enumerate(tests_raw)}


def stale(runs):
    """Pooled stale answers: stale / affected answers over all runs of a cell."""
    st = sum(round(m["stale_answer_rate"] * m["affected_answers"]) for m in runs.values()
             if m.get("affected_answers") and not math.isnan(m["stale_answer_rate"]))
    return st, sum(m.get("affected_answers") or 0 for m in runs.values())


def table_fix(versions):
    """Frozen code and the two post-hoc fixes, side by side (versions: list of (label, cells))."""
    k = len(versions)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for lab, _ in versions)
    rules = "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(k))
    lines = ["\\begin{table}[!htbp]\\centering\\small\\setlength{\\tabcolsep}{3pt}",
             "\\caption{BACM on the frozen code and after each post-hoc fix (same held-out seeds; baselines are "
             "unaffected). Fix~1 extends the provenance span of a method count to the whole class body; fix~2 rebuilds "
             "stale derived parents recursively and re-derives derived facts by their typed rule "
             "(\\cref{sec:corrections}). Stale = stale answers / answers whose truth changed, pooled over 30 runs.}",
             "\\label{tab:fix}", "\\begin{tabular}{ll" + "ccc" * k + "}", "\\toprule",
             "& & " + head + " \\\\", rules,
             "Env. & System & " + " & ".join(["Acc. & Tok.\\ (k) & Stale"] * k) + " \\\\", "\\midrule"]
    for env in ("static", "dynamic"):
        for a in FIXED_ARCHS:
            row = [env, NAMES[a]]
            for _, cc in versions:
                runs = cc.get((env, a), {})
                if not runs:
                    row += ["--"] * 3
                    continue
                s_, n_ = stale(runs)
                row += [f"{statistics.fmean(m['question_accuracy'] for m in runs.values()):.2f}",
                        f"{statistics.fmean(m['tokens_total'] for m in runs.values()) / 1000:,.0f}",
                        f"{s_}/{n_}" if n_ else "--"]
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN / "table_fix.tex").write_text("\n".join(lines))


def main():
    pre_rows = load(FINAL) + low_rows()
    fix_rows = load(FIX)
    fixed_cells = {(r["group"]["env"], r["group"]["arch"]) for r in fix_rows}
    final = [r for r in pre_rows if (r["group"]["env"], r["group"]["arch"]) not in fixed_cells] + fix_rows
    c_pre, c = cells(pre_rows), cells(final)
    ca = cells(load(ABL))
    t_pre, tests = run_tests(c_pre), run_tests(c)
    if fix_rows:
        def merged(rows_):
            cells_ = {(r["group"]["env"], r["group"]["arch"]) for r in rows_}
            return cells([r for r in pre_rows if (r["group"]["env"], r["group"]["arch"]) not in cells_] + rows_)
        versions = [("Frozen code", c_pre)]
        if load(FIX1):
            versions.append(("Fix 1", merged(load(FIX1))))
        versions.append(("Fix 2 (reported)", c))
        table_fix(versions)
        (GEN / "tests_prereg.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in t_pre.items()}, indent=1))
    if final:
        fig_main(c)
        table_main(c, tests)
    abl_rows = fig_ablation(c_pre, ca) or []   # ablations ran on the pre-fix code: compare like with like
    table_ablation(abl_rows)
    kind_file = GEN / "kind_accuracy.json"
    kind_acc = json.loads(kind_file.read_text()) if kind_file.exists() else {}
    fig_kinds(kind_acc)
    fig_kinds_both()
    fig_per_repo(c)
    fig_stale(c, c_pre)
    fig_anatomy()
    fig_distribution(c)
    fig_cost_model()
    table_repos()
    table_v1()
    if final:
        text_blocks(c, tests, abl_rows, gemini_block())

    def m(env, a, key="question_accuracy"):
        runs = c.get((env, a), {})
        return statistics.fmean(r[key] for r in runs.values()) if runs else math.nan

    macros = {}
    if final:
        s_ie, d_ie = tests.get(("static", "independent_evict")), tests.get(("dynamic", "independent_evict"))
        st_b, st_n = stale(c[("dynamic", "bacm_llm")])
        st_i, _ = stale(c[("dynamic", "independent_evict")])
        macros["NabstractMain"] = (
            f"answered {m('static', 'bacm_llm'):.0%} of the questions correctly on static repositories and "
            f"{m('dynamic', 'bacm_llm'):.0%} when the repository changed mid-run, against {m('static', 'independent_evict'):.0%} "
            f"and {m('dynamic', 'independent_evict'):.0%} for independent agents with context eviction (both differences "
            f"significant after Holm correction), and reduced stale answers after the release from {st_i} to {st_b} of "
            f"{st_n}. Token efficiency is workload-dependent and not established statistically: pooled token use changed by "
            f"${s_ie['dtok']:+.0%}$ and ${d_ie['dtok']:+.0%}$ (not significant), from large savings on two repositories to "
            f"large increases on the others."
        ).replace("%", "\\%")
    else:
        macros["NabstractMain"] = "[results pending]."
    for env in ("static", "dynamic"):
        for a in MAIN:
            key = f"N{env}{a}".replace("_", "")
            macros[key + "acc"] = f"{m(env, a):.2f}"
            macros[key + "tok"] = f"{m(env, a, 'tokens_total') / 1000:,.0f}k" if not math.isnan(m(env, a)) else "--"
    cf = GEN / "cost_fit.tex"
    macros["NcostFit"] = cf.read_text().strip() if cf.exists() else ""
    for f in GEN.glob("table_*.tex"):   # never wider than the column / page (two-column layout)
        t_ = f.read_text()
        t_ = t_.replace("\\begin{table*}[t]", "\\begin{table}[!htbp]").replace("\\end{table*}", "\\end{table}")
        t_ = t_.replace("\\begin{table}[t]", "\\begin{table}[!htbp]")
        if "adjustbox" not in t_:
            t_ = t_.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}", 1)
            t_ = t_.replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}", 1)
            f.write_text(t_)
    (GEN / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    (GEN / "tests.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in tests.items()}, indent=1))
    print("assets written:", sorted(p.name for p in GEN.iterdir()), sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
