# Task breakdown (derived from SPEC.md §14)

Legend: ✅ built + tested · 🟡 built, needs real compute to prove · ⏳ deferred to compute phase

## Phase 0 — Foundations
- ✅ T0.1 Repo layout, pyproject, config, .gitignore/.dockerignore, SECURITY/DISCLOSURE/README
- ✅ T0.2 Sega design tokens (from awesome-design-skills `sega`) → `frontend/src/theme`

## Phase 1 — Backend core (MVP)
- ✅ T1.1 Telemetry DB (SQLModel) + corpus DB (separate engine, 0600/0700, view + indexes)
- ✅ T1.2 Cipher registry: Standard/Custom Base64, XOR, AES-128-GCM, ROT-N, none
- ✅ T1.3 MetricCapture: timing, avalanche, χ², diffusion, entropy
- ✅ T1.4 Targets: Mock (deterministic simulated guardrail), OpenAI-compatible HTTP (Ollama/vLLM/GPU), LiteLLM (optional)
- ✅ T1.5 Scorers: regex canary, refusal heuristic, two-axis composite, Llama-Guard hook
- ✅ T1.6 Orchestrator FSM (GENERATE→MUTATE→DISPATCH→EVALUATE→PERSIST→ADAPT), budget + timeout, strategies
- ✅ T1.7 API: providers, runs, explorer, analytics, WS hub
- 🟡 T1.8 PyRIT bridge (import-guarded, confined to `engine/`)

## Phase 2 — V1 differentiators
- ✅ T2.1 BYOK key manager (memory-only), log redaction filter, key-leak test
- ✅ T2.2 Corpus: promote / verify / verify-all / patch / delete / by-model view
- ✅ T2.3 Vendor disclosure: signed JSON + HTML, audit row, hold_until
- ✅ T2.4 Analytics: Spearman ρ, logistic fit, 95% CI, judge agreement, Pareto, OWASP/ATLAS grouping, leaderboard
- ✅ T2.5 Defenses: perplexity-ish filter, keyword gate, with ablation flag
- ✅ T2.6 Dataset loaders: canary (shipped) + JBB/HarmBench/StrongReject file loaders (BYO)
- ✅ T2.7 **Compute nodes** (GPU connectivity): register, probe, list models, one-click "use as provider"

## Phase 3 — Frontend (Sega arcade theme)
- ✅ T3.1 Shell, theme, pixel components (press buttons, cabinet-trim cards)
- ✅ T3.2 Pages: Dashboard, NewExperiment, LiveConsole, Analytics, DBExplorer, Corpus, Leaderboard, Compute, Settings (+ consent gate)

## Phase 4 — Ops
- ✅ T4.1 docker-compose (cpu) + `compute/docker-compose.gpu.yml`
- ✅ T4.2 Tests: ciphers, metrics, key-leak, corpus, full-run integration
- ⏳ T4.3 Playwright e2e (needs a browser install; scaffolded config only)
- ⏳ T4.4 Alembic migrations (schema is `create_all` for now; baseline revision is a TODO)

## Phase 5 — FINAL EXECUTION (left for high-end compute)
- ⏳ T5.1 Bring up GPU node, pull models (see REQUIREMENTS_FROM_YOU §A)
- ⏳ T5.2 Real runs: Attacker × Target × Judge arena, canary objectives first, then BYO datasets
- ⏳ T5.3 Fill Analytics with real data, re-verify loop, first disclosure report

## Done in round 2 (this pass)
- ✅ **PyRIT 1.1.0 integrated** (`engine/pyrit_native.py`): cipher `Converter` (+MetricCapture), `PromptTarget` adapter, canary + regex-refusal + float-threshold scorers, PromptSending / **Crescendo / TAP / PAIR** executors, `engine: native|pyrit` switch. Pinned `pyrit==1.1.0`.
- ✅ Adaptive strategies now receive the previous response (refusal feedback).
- ✅ Every model role (target/attacker/judge/guard) is metered: shared semaphore, hard 30 s deadline, budget gate, cost accounting; per-provider token pricing for any backend.
- ✅ DB connection no longer held across model calls (fixed pool exhaustion under concurrency).
- ✅ Analytics on a **DuckDB read replica**; shared filters (model/cipher/strategy/run/defense/date) on every endpoint; new: metric table, scaling fits, Pareto frontier, 3-signal judge agreement + kappa, defense ablation, model-size curve, strategy×cipher heatmap, timeline, **Guardrail Robustness Report (HTML/JSON, aggregate-only)**.
- ✅ DB Explorer: server-side pagination/sort/search, promote action, live ticker, redacted + provenance-stamped CSV/JSON export.
- ✅ Re-verify supports a judge for BYO objectives; refuses (409) instead of mis-marking `patched`.
- ✅ Codex review fixes: PyRIT init race, run finalisation on cancel, corpus connection across awaits, PAIR depth, judge deadline, multi-piece capability claim, PyRIT row cost/tokens.

## Follow-up tasks (next)
1. **PyRIT fidelity (Codex P1):** pass role-preserving multi-turn history through the Target boundary (currently flattened into one user message); persist the *selected branch's* request for TAP (currently last converter call); store the full conversation per attempt and replay it in corpus re-verify.
2. Use one success predicate (two-axis) for PyRIT's stopping criterion and the persisted judgment (a refusal quoting the canary can stop adaptation early).
3. Explicit attack deadline that separates semaphore queueing from per-call deadlines for multi-turn runs.
4. PyRIT-engine support for BYO datasets (judge-based objective scorer) and for defenses.
5. Real-LLM attacker/judge prompt validation on the GPU host (scripted attacker is mock-only); Llama Guard 3 wiring test.
6. Alembic baseline migrations for both DBs (columns were added: provider pricing — existing DBs need migration).
7. Frontend: vitest unit tests + Playwright e2e (Live Console, promote, corpus wizard); a11y audit (contrast of VT323 at small sizes, chart text alternatives).
8. Garak-compatible export; StrongReject third-axis scorer; Docker GPU profile smoke test.
9. Authentication / per-session BYOK ownership (deferred by request).
10. Compute phase: run real Attacker×Target×Judge arena (see REQUIREMENTS_FROM_YOU).

## Codex review round 2 (analytics/explorer) — status
Fixed: defense `transformed_payload` now redacted in exports; replica staleness (fingerprint covers db/WAL identity + mutable verdict/cost/run columns); NaN R² no longer breaks scaling JSON; stale/out-of-order responses ignored and cleared per query; pagination stays visible on empty pages (+clamp); metric selector stays visible when a metric has no data; overview `total_runs` is filter-scoped.
Open (follow-ups): (a) report lacks scaling/Pareto in JSON and defense-ablation/model-size/judge/standards in HTML, no PDF; (b) defense-ablation pools unmatched cohorts — add matched-cohort comparison and record scope in provenance; (c) build report from one snapshot; (d) incremental replica refresh instead of full rebuild; (e) SQL console execution deadline/progress limit; (f) export provenance: models/datasets/seeds/truncation flag (100k cap); (g) per-panel error states on secondary endpoints; (h) accessible data tables for charts.

## Schedule (round 3) — each item gets its own Codex review (R-xx)
| # | Task | Review |
|---|---|---|
| A1 | Complete report from ONE snapshot (+scaling/Pareto/ablation/model-size/judge/standards), printable to PDF | R-A1 |
| A2 | Matched-cohort defense ablation; filter scope recorded | R-A2 |
| A3 | Incremental replica refresh (append new ids; full rebuild only on in-place edits) | R-A3 |
| A4 | SQL console execution deadline; export provenance (models/datasets/seeds/truncation) | R-A4 |
| A5 | Per-panel error states; accessible data tables for every chart | R-A5 |
| B1 | Role-preserving multi-turn history through the Target boundary | R-B1 |
| B2 | TAP: persist the selected branch's payload; one success predicate for PyRIT stop + verdict | R-B2 |
| B3 | Persist the full conversation; replay it in corpus re-verify | R-B3 |
| B4 | Attack deadline separate from queueing; lightweight column migration for existing DBs | R-B4 |
