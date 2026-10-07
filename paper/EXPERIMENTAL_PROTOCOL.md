# Experimental protocol

Fixed on 2026-10-07, before the real-repository results (`results_real/repobench/`) were analysed.
Everything here is implemented in the repository. Commands are given so anyone can rerun them.

## Hypotheses

- **H1 (accuracy under change).** When the repository changes mid-task, the proposed context manager
  (`bacm`, `bacm_validate`) gives a higher per-question accuracy than independent agents, agents with
  the same context hygiene (`independent_evict`), and selective shared memory with the same interface
  optimisations (`selective_fast`).
- **H2 (no loss when nothing changes).** On a static repository, `bacm` is at least as accurate as
  every baseline.
- **H3 (efficiency).** `bacm` uses fewer total tokens than independent agents. Against
  `independent_evict` and `selective_fast`, token use is reported as a two-sided comparison, not
  claimed in advance.
- **H4 (components).** In the simulator, removing provenance, versioning, dependency tracking or
  invalidation raises the stale-answer rate (ablation suite).

## Benchmarks

1. **Synthetic repository** (simulator, `results/`): every suite × 20 seeds, fully deterministic.
2. **RepoQA-Dyn, real repositories** (`repobench` suite). Pinned releases of real open-source projects:

   | repo | from | to (dynamic) | effect of the change |
   |---|---|---|---|
   | encode/httpx | 0.18.0 | 0.27.0 | refactor; about half the sampled answers change |
   | psf/requests | v2.25.0 | v2.32.0 | move to `src/` layout; most answers change |
   | pallets/flask | 2.0.0 | 3.0.0 | some answers change |
   | pallets/click | 7.1 | 8.1.0 | many files change, no answers change |
   | Textualize/rich | v10.0.0 | v13.0.0 | many files change, no answers change |

   Questions ask about class hierarchies (`base_file`, `grandbase`, `family_files`). Each needs 2+
   facts read from source. Ground truth comes from Python's `ast` on the files as they are at
   answer time. In dynamic runs the working tree moves to the later release once half of all
   questions are answered, and answers are scored against the code after the change. 3 agents ×
   4 questions per run. Seed = question sample and agent split. Static and dynamic runs of a seed
   share the same questions, so they can be compared as pairs.

## Models

Primary: Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`, temperature 0, OpenAI-compatible endpoint).
All responses are cached in `.llm_cache/`, so analyses can be re-derived exactly from the cache.
Run-to-run API nondeterminism is reported as a limitation.

## Metrics

- Primary: per-run question accuracy, and total tokens (agents plus manager, all LLM calls).
- Secondary: stale-answer rate, LLM calls, file reads, simulated latency.

## Statistics

- Unit = one run. Pairs = same seed, repository and environment, differing only in architecture.
- Test: two-sided paired sign-flip permutation test (exact up to 16 pairs, otherwise 20,000 Monte
  Carlo draws).
- Holm correction over every comparison in a report. Significance level 0.05.
- 95% bootstrap confidence intervals for means and paired differences. `report.md` also gives the
  number of pairs needed for 80% power.
- Exclusions: none. Failed runs (API errors) are re-run from the cache; runs that cannot complete
  are listed in `failed.json` and reported.

## Reproduce

```bash
python -m bacm run --suite all --seeds 20                       # simulator
python -m bacm run --suite repobench --seeds 5 --backend openai_compat --provider gemini \
       --model gemini-3.5-flash-lite --out results_real/repobench
python -m bacm report results/main results_real/repobench/repobench-openai_compat-gemini-3.5-flash-lite
```
