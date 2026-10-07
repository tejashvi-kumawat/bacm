# Pre-registration of the final evaluation

Frozen on 2026-10-07 13:23 IST, after development on seeds 100-101 and before any run on held-out seeds.
Code fingerprint (sha256 over `bacm/**/*.py`): `3dbaa3cb73746c4b`. No change to `bacm/` is allowed after this point
except bug fixes that are reported in the paper.

## Benchmark
RepoQA-Dyn v2 (`--suite repobench2`): httpx, requests, flask, click, rich; 3 agents x 6 questions; five question
kinds; static and dynamic (later real release applied once half of the questions are answered).
**Held-out seeds: 200-205** (never used during development).

## Systems
- Proposed: `bacm_llm` (transparent memory, focused model-based extraction, manager calls at low reasoning effort).
- Baselines: `independent`, `independent_evict`, `selective_fast` (tool-based selective memory),
  `selective_llm` (same transparent interface and extraction, no provenance/versions/dependencies/invalidation).
- Effort control: `independent_evict` with every agent at low reasoning effort (does simply thinking harder
  everywhere give the same result?).
- Upper bound: `bacm_auto` (symbol-index extraction).
- Ablations of `bacm_llm` on seeds 200-202: no_resolve, no_digest, no_maintain, no_invalidation, minimal_manager.
- Second model family: Gemini 3.5 Flash-Lite (free tier) on `independent_evict`, `selective_llm`, `bacm_llm`, as many
  held-out seeds as the free quota allows; reported as a replication.

## Model
GPT-5 mini, reasoning effort minimal for agents (all systems), cache of every response kept.

## Hypotheses (two-sided tests)
- H1: `bacm_llm` accuracy > `independent_evict` accuracy (static and dynamic).
- H2: `bacm_llm` tokens < `independent_evict` tokens on dynamic repositories.
- H3: `bacm_llm` accuracy > `selective_llm` accuracy on dynamic repositories (value of the staleness machinery).
- H4: `bacm_llm` vs `independent_evict` at low effort: reported, no direction assumed.

## Analysis
Unit = run; pairs share repository, seed and environment. Two-sided paired sign-flip permutation tests (exact up to
16 pairs, else 20,000 Monte Carlo draws), Holm correction across all reported comparisons, alpha = 0.05, 95% bootstrap
CIs. Token change reported pooled (total/total) and per pair. No exclusions; failed runs are re-run from the cache
and any that cannot complete are reported.

## Amendments
- 2026-10-07 13:24: added `bacm_llm-minimal_manager` to the repobench2 architecture list so the
  pre-registered ablation can be selected (harness-only; no behavior change). New fingerprint: `4f80a7eefee8050e`.
- 2026-10-07 13:38: budget amendment. Measured cost (~$0.13/run) was about twice the estimate, so the plain
  `independent` and tool-based `selective_fast` baselines were dropped from the held-out evaluation before any
  held-out result was inspected (36 of 360 runs had finished; none were analysed). Both are dominated baselines whose
  cost profile is already documented on 150 runs of benchmark v1 (Table v1). Hypotheses H1-H4, the low-effort
  control and the ablations are unchanged.
- 2026-10-07 13:41: bug fix. A model call to `search` without its `query` argument crashed the run instead
  of returning an error to the model (found in one Gemini held-out run). The tool layer now returns an error message.
  Behaviour is unchanged for well-formed calls. Affected runs are re-run. New fingerprint: `3613dfba8979a506`.
- 2026-10-07 (after the held-out runs finished): bug fix found while analysing stale answers. The provenance span
  recorded for a method count (`n_methods`) was only the class header line, so a release that added or removed
  methods without touching the header left the count marked valid; most of `bacm_llm`'s dynamic stale answers were
  such counts. The span is now the whole class body, and a multi-line span is accepted only if the body is still
  complete (regression test `test_method_count_goes_stale_when_a_method_is_added`). Only systems with a context
  manager (`bacm_llm`, `bacm_auto`) are affected; baselines are not. Both cells are re-run on the same held-out seeds
  into `results_real/final_fix/`; static re-runs must reproduce the frozen numbers (no change point). The paper
  reports the pre-registered (buggy) numbers and the corrected numbers side by side; ablations stay on the
  pre-fix code (internally consistent). New fingerprint: `26275e0432101369`.
- 2026-10-07 (second post-hoc fix, found while explaining the low accuracy of `method_total` after a release):
  (a) when a release made a derived parent stale (e.g. an ancestor's `method_total`), recomputation rebuilt only
  stale *base* parents and gave up on the answer, which was then re-derived from incomplete facts; it now rebuilds
  stale parents recursively (regression test `test_derived_chain_is_rebuilt_after_a_change_deep_in_it`);
  (b) with model-based extraction, recomputed derived facts were re-derived by a model call instead of the typed rule
  used everywhere else; they now use the rule (benchmark v2 only). Both systems with a context manager are re-run on
  the same held-out seeds into `results_real/final_fix2/`. The paper reports frozen, first-fix and second-fix
  numbers side by side. New fingerprint: `a5a727f1d4db16a2`.
