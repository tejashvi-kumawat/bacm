# BACM: Branch-Aware Context Manager

Code, benchmark and analysis for the paper

> **BACM: Transparent, Provenance-Aware Shared Memory for Parallel LLM Agents on Changing Codebases**
> Tejashvi Kumawat.

BACM is a shared context manager for parallel LLM agents. It stores what agents read as knowledge objects with
provenance (file, version, supporting span), dependency edges and typed derivation rules; when the repository
changes it invalidates exactly the affected facts and recomputes only those that an open question needs. The model
never calls the memory: a mediator in the tool layer publishes what agents read, serves already-read files as
digests, and resolves questions from known facts.

`RepoQA-Dyn`, the benchmark, uses five real Python repositories (httpx, requests, flask, click, rich) in which a real
release lands in the middle of a run, with ground truth from static analysis (`bacm/env/realrepo.py`,
`bacm/env/pyfacts.py`).

**Main findings (GPT-5 mini, 30 held-out runs per cell).** Accuracy 0.89 (static) and 0.88 (repository changes
mid-run) against 0.58 and 0.49 for independent agents with context eviction, both significant after Holm
correction; 7 stale answers out of 186 changed answers against 61. Token savings (-22% and -26% pooled) are **not**
statistically significant and depend on the repository. See the paper for the full evaluation, the ablations, the
reasoning control and two post-registration corrections, which are reported next to the frozen results.

## Reproducing the paper

Every model response of the evaluation is cached, so all tables, figures and numbers can be regenerated **without any
API call or key**. The cache (about 210 MB) is attached to the
[latest release](https://github.com/tejashvi-kumawat/bacm/releases/latest) as `bacm-llm-cache.tar.gz`.

```bash
python3 -m venv .venv && .venv/bin/pip install pytest matplotlib numpy
tar -xzf bacm-llm-cache.tar.gz                      # creates .llm_cache/ (gpt-5-mini, gemini-3.5-flash-lite)
.venv/bin/python -m pytest                          # unit tests
.venv/bin/python paper/replay_kinds.py              # per-kind accuracy and token anatomy, replayed from the cache
.venv/bin/python paper/make_assets.py               # every table, figure and number of the paper
```

The two scripts write the paper's LaTeX tables and PDF figures to `paper/generated/` and `paper/figures/`
(not tracked).

The first run clones the five benchmark repositories at their pinned releases (network access needed once).
`paper/PREREGISTRATION.md` records the hypotheses, the frozen-code fingerprint and every later amendment;
`paper/EXPERIMENTAL_PROTOCOL.md` describes the protocol.
Result files of the runs are in `results_real/`.

## Citation

```bibtex
@misc{kumawat2026bacm,
  title  = {{BACM}: Transparent, Provenance-Aware Shared Memory for Parallel {LLM} Agents on Changing Codebases},
  author = {Kumawat, Tejashvi},
  year   = {2026},
  note   = {Code: https://github.com/tejashvi-kumawat/bacm}
}
```

Contact: tejashvikumawat@gmail.com, ce1230237@iitd.ac.in. Licence: MIT (see [LICENSE](LICENSE)).

## Development notes

The sections below describe the research prototype itself: the simulator, the architectures and ablations, and how
to run the experiments.

## Quick start

Linux / macOS:
```bash
python3 -m venv .venv && .venv/bin/pip install pytest matplotlib numpy   # or: uv venv .venv && uv pip install ...
.venv/bin/python -m pytest                          # unit tests
.venv/bin/python -m bacm demo                       # one seed, 4 architectures side by side
.venv/bin/python -m bacm demo --dynamic --trace     # with mid-run source changes + knowledge graph dump
.venv/bin/python -m bacm run --suite pilot --seeds 10
.venv/bin/python -m bacm run --suite all --seeds 20 --workers 16   # every experiment
.venv/bin/python -m bacm report results/main            # tables with 95% CIs + paired comparisons
.venv/bin/python -m bacm doctor --ping                  # check Python and API keys
```

Windows (PowerShell). The project is plain Python, so no `.exe` and nothing to sign:
```powershell
py -3.12 -m venv .venv            # Python from python.org or: winget install Python.Python.3.12
.venv\Scripts\python -m pip install pytest matplotlib numpy
.venv\Scripts\python -m pytest
.venv\Scripts\python -m bacm demo --dynamic
.venv\Scripts\python -m bacm run --suite llm_pilot --seeds 3 --backend openai_compat --provider gemini --model gemini-3.5-flash-lite
```

API keys go in a `.env` file in the project folder (copy `.env.example`; `.env` is git-ignored), e.g.
`GEMINI_API_KEYS=key1,key2`. Several keys are used round-robin. A revoked key (401) or a key that
used up its daily free quota is dropped automatically. If every key is exhausted, the run stops,
saves every finished run, and lists the rest in `failed.json`. Re-running the same command later
resumes for free, because every finished LLM call is cached in `.llm_cache/`.

Results go to `results/<suite>/runs.jsonl` (one row per run) and `results/<suite>/summary.csv`
(mean, std and 95% CI per cell, plus ARW relative to the independent baseline).

**Cost:** the default `sim` backend makes no API calls; its `cost_usd` metric is a *modelled* cost computed from
token counts. Real-model runs use `--backend openai_compat` (any OpenAI-compatible endpoint) or
`--backend anthropic`, both of which bill the corresponding API account. `--max-usd` aborts a run when the spend
ledger exceeds a limit and is required for the `openai` provider.

## Architecture

```
                    Orchestrator (discrete-event, agents race in simulated time)
                 C0 ──► branch A0   branch A1   branch A2   branch A3      (local context L_i each)
                           │           │           │           │
                           └───────────┴─────┬─────┴───────────┘
                                             ▼
                                      Context Manager             bacm/core/manager.py
     ┌───────────────┬───────────────┬───────┴───────┬────────────────┬──────────────┐
  Query/Reuse    Admission S(K)   Provenance +    Dependency      Incremental       Conflict
  + lazy         (L0 private,     versions +      graph G=(K,E)   invalidation +    detection /
  validation     L1/L2 shared)    span hashes                     L3 notifications  resolution
```

| Design concept | Where |
|---|---|
| Knowledge object K_i (content, source, version, branch, deps, confidence, status) | `bacm/core/knowledge.py` |
| Query & reuse engine, lazy validation on reuse | `ContextManager.query`, `_validate` |
| Provenance P(K_i) (agent, branch, source@version, supporting line) | `Knowledge.produced_by/branch/sources` |
| Dependency graph and incremental invalidation (affected subgraph only) | `_mark_stale`, `on_change` |
| Selective synchronization L0–L3 | `Level`; L0 never leaves the branch; L3 = `outbox` notifications |
| Admission S(K) = w1R + w2C + w3V + w4D − w5U > τ | `admission_score` |
| Conflict resolution (evidence strength, then a majority of independent observations plus fallible verification reads) | `_resolve_conflict`, `_vote` |
| Incremental recompute in a small fresh context, pushing corrected values (L3 update) | `core/worker.py`, `_recompute` |
| Verification of refused weak claims, and cross-checking a fact on its first reuse | `_verify_weak`, `_cross_check`, `_extract` |
| Context/interface efficiency: prefetch, parallel lookup, piggyback publish, harvest, evict | `memory/base.py:prefetch`, `agents/simulated.py` |
| Agent loop Observe→Query→Reuse/Read→Reason→Validate→Share | `bacm/agents/simulated.py` |
| Change loop Source change→Dependency lookup→Invalidate→Notify→Recompute | `on_change` + agent `_process_inbox` + idle-agent wake-up |

### Architectures compared (`bacm/config.py`)

| name | what it is |
|---|---|
| `independent` | Baseline 1: A_i(C_0). Nothing is shared |
| `fully_shared` | Baseline 2: A_i(C). Every observation, tool output and thought (including unverified guesses) goes into every agent's prompt |
| `selective` | Baseline 3: gated key→value memory bank (confidence gate, last write wins, no provenance, versions or invalidation) |
| `independent_evict` | Fair baseline: independent agents with the same context hygiene (harvest + evict), no sharing |
| `selective_fast` | Fair baseline: selective memory with every interface/context optimisation the proposed system has |
| `selective_global_recompute` | Strong dynamic baseline: knows *that* something changed, not *what*, so it discards everything and recomputes globally |
| `bacm_v1` | First version of the proposed system: knowledge layer only |
| `bacm` | Proposed system: knowledge layer + worker (recompute, weak-claim verification, cross-check) + efficient interface |
| `bacm-no_<component>` (15 of them) | Ablations: one component off each (§19) |

### What makes `bacm` efficient (and why it stays correct)

1. **Small-context worker.** The context manager owns a worker whose context holds only one file or
   a few parent facts. After a source change it re-derives exactly the affected subgraph and
   **pushes corrected values** to the branches that used them. Branches patch the value in place and
   re-answer, without re-reading anything inside their large contexts.
2. **Context hygiene (harvest + evict).** Read a file once, note *every* fact in it as a compact
   knowledge object, then drop the raw text from the branch's context. This is safe because
   provenance records exactly where to re-read and versions say when that is needed.
3. **Efficient interface.** Relevant valid knowledge is prefetched when a question starts (multi-hop
   associative retrieval: `route(/x)=auth` leads to `datastore(auth)`). A lookup rides in the same
   LLM turn as the search it would replace, and publishing rides alongside the next action, so
   sharing adds no extra LLM turns.
4. **Stopping errors from spreading.** A fact is re-read independently the first time another
   branch reuses it. Conflicts are settled by a majority of independent observations. A
   doc/guess-based claim that is refused gets verified against code, not silently dropped.

Points 2 and 3 are generic techniques, so the `independent_evict` and `selective_fast` baselines get
them too. That separates the benefit of the knowledge layer from the benefit of context hygiene.

### Environment (`bacm/env/world.py`)

A procedurally generated, versioned repository with known ground truth:

- base facts: `route(/endpoint)` in `api/routes_*.py`, and `datastore(svc)` / `protocol(svc)` in
  `services/<svc>/{config,server}.py`
- derived facts: `endpoint_datastore(/e) = datastore(route(/e))`, `endpoint_protocol`, and
  `feature_datastores(f)` (2-level dependency chains)
- an outdated `docs/ARCHITECTURE.md` that reports about 35% of facts wrongly (distractor evidence)
- mid-run fact-changing edits and cosmetic edits, triggered at a fraction of task progress

Each agent gets a subtask of questions. The `overlap` setting controls how much the agents'
information needs overlap.

### Agents

- **`sim` (default):** a seeded rule-based ReAct-style policy. It really searches, reads and parses
  files, so staleness and redundancy come from the environment. Its *imperfections* (trusting docs,
  misreading, speculating, adopting other agents' unverified claims) are probabilities in
  `AgentConfig`. Every action is preceded by an LLM step whose input is the agent's whole current
  context, so token use grows as it does for a real agent.
- **`anthropic`:** a real Claude agent with native tool use (`list_files`, `search`, `read_file`,
  `kb_query`, `kb_publish`, `answer`), running in the same orchestrator, memory and metrics.
  Provenance is captured by the tool layer, not trusted from the model.

## Metrics (`bacm/metrics/tracker.py`)

| Design section | metric key(s) |
|---|---|
| A. task success | `question_accuracy`, `agent_success_rate`, `team_success` |
| B. tokens | `tokens_in`, `tokens_out`, `tokens_total`, `llm_calls` |
| C. redundant tool calls | `redundant_ops`, `redundant_cross_agent_ops`, `redundancy_rate`, `redundant_file_reads` |
| D. context size | `ctx_avg`, `ctx_max` (prompt tokens per LLM call) |
| E. latency | `latency_sim_s` (makespan under the latency model), `wall_clock_s` |
| F. cost | `cost_usd` (from `CostModel`) |
| G. knowledge reuse | `reuse_rate` = needs served from shared memory ÷ needs another branch had already resolved |
| ARW (§18) | `ARW_<measure>` = 1 − W_system / W_independent for tokens, LLM calls, tool calls, reads, cost, latency |
| §20 dynamic change | `stale_answer_rate`, `affected_accuracy`, `time_to_consistency_s`, `invalidations`, `unnecessary_invalidations`, `revalidations`, `l3_events` |
| §21 conflicts | `conflicts`, `conflict_accuracy`, `false_reuse_rate`, `wrong_claims_admitted` |
| §13.6 branch diversity | `correlated_error_rate` (agents sharing the same wrong answer), `read_set_diversity` |

## Experiment suites (`bacm/experiments/suites.py`)

| suite | Design section | varies |
|---|---|---|
| `pilot` | §29 minimum viable experiment | baselines, static repo |
| `main` | §16 | all baselines × {static, dynamic} |
| `ablation` | §19 | each component removed × {dynamic, dynamic+conflict-prone} |
| `dynamic` | §20 | number of mid-run source changes 0/1/3/5/8 |
| `conflict` | §21 | misread rate 0.05/0.15/0.3 with doc-trusting agents |
| `scaling` | §22 | N = 2, 4, 8, 16 agents |
| `overlap` | when does sharing pay off | overlap 0 … 1 |

## Running with real models

The `openai_compat` backend talks to any OpenAI-compatible endpoint. It throttles with `--rpm`, retries on rate
limits, drops revoked or exhausted keys, and writes a first-writer-wins on-disk response cache (`.llm_cache/`), so an
interrupted suite resumes without paying twice and every analysis can be replayed offline.

```bash
export OPENAI_API_KEY=...    # keys go in .env (see .env.example); several keys: OPENAI_API_KEYS=k1,k2
.venv/bin/python -m bacm run --suite repobench2 --seed0 200 --seeds 6 \
    --archs independent_evict,selective_llm,bacm_llm,bacm_auto \
    --backend openai_compat --provider openai --model gpt-5-mini --max-usd 20 --out results_real/my_run
.venv/bin/python -m bacm status                     # progress and spend of every result directory
```

`repobench2` is the paper's evaluation (five repositories, static and dynamic, three agents with six questions each).
`llm_pilot` and `repobench` are smaller suites used during development; `--archs` selects a subset. Results go to
`<out>/<suite>-<backend>-<model>/` (`runs.jsonl`, `summary.csv`, `report.md`).

## Simulator limitations

- Simulator results depend on modelled agent behaviour, latency and prices; they show the *algorithmic* behaviour of
  the architecture (§28), not the behaviour of real models. The claims of the paper rest on the real-model
  evaluation (`repobench2`).
- The size of the token savings depends on how expensive context is. Most of the saving comes
  from harvest + evict, which is why the fair baselines exist. Report `bacm` against
  `independent_evict` and `selective_fast`, not only against `independent`.
- In the simulator, cross-check and conflict verification update shared state at once and deliver
  corrections after the worker's modelled latency. That is slightly optimistic for agents who
  query in between.
- The synthetic repository has unambiguous facts and an exact grader. Real repositories and web
  tasks (§23) will need fuzzier knowledge keys and retrieval.
- `stale_answer_rate` counts any affected answer that still equals the pre-change value. That
  includes values taken from the outdated docs, which are also stale sources.
