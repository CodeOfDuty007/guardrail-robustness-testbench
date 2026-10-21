# Sprint Plan — Attack Families, Prompt Injection, REX Defense, MCP

Source of truth for requirements: `ATTACK_FAMILIES_AND_DEFENSE_PLAN.md` (the "plan"). This file is the
**execution** plan: who builds what, in which order, against which frozen contract, and how work is
accepted. Branch: `feat/attack-families-rex-mcp`.

## Why (the gap this closes)

Before this work, every "attack" in the harness was the same fixed prompt under a different **encoding**
(base64 / custom base64 / XOR / AES / ROT-N) plus a handful of static string prefixes. That measures one
family (obfuscation) and calls it guardrail robustness. Encoding a prompt is not a prompt-injection test:
injection is about *untrusted content overriding developer instructions* (OWASP LLM01), which needs
its own axis (direct vs indirect, delivery carrier, tool-call side effects). The deliverable is a
per-family **Robustness Profile**, an injection profile, and a defense (REX, toy scale) evaluated
against both.

## Scrum roles

| Role | Who | Responsibility |
|---|---|---|
| Product owner / Scrum master | main Claude session | backlog, sprint 0 contract, merges, advisor review at each gate, human-review gate, PR |
| Dev agents | one subagent per story (git worktree) | implement the story; `Load gstack` and use `/autoplan`-style planning then implement, `/review` before hand-off |
| QA agent | one persistent subagent | runs `scripts/gate.sh`, checks every story's acceptance criteria (AC) line by line, reports defects back; loop until clean |
| Review agent | Codex (`codex:codex-rescue` / companion CLI) | independent second-model code review of each sprint's merged diff |

**Definition of Ready:** story has files, interface, AC. **Definition of Done:** AC met, `scripts/gate.sh`
green (ruff + mypy-no-new-errors + pytest + vitest + tsc build), QA sign-off, Codex review findings
resolved or explicitly waived with a reason.

## Sprint 0 — Frozen contract (DONE by scrum master)

- `backend/app/engine/families/base.py` — `AttackFamily` protocol, `CandidatePrompt`, `Turn`,
  `FAMILY_REGISTRY` + `@register_family`, `family_rng()` (sha256-derived, scheduling-independent),
  `ask_attacker()` (attacker LLM applies the technique; offline structural fallback).
- **Contract corrections vs the plan:** attacker is our `Target` protocol (not PyRIT `PromptChatTarget`
  — PyRIT stays inside `pyrit_native.py`/`pyrit_bridge.py`); PyRIT 1.1 exposes `CrescendoAttack` /
  `TAPAttack` (not `*Orchestrator`) — extend `pyrit_native.run_attack`.
- `family` reference implementation: `families/encoding.py` — golden-tested byte-identical to legacy
  `DirectCipherWrap` (`tests/golden/encoding_direct_cipher_wrap.json`).
- Schema: `attempts.family/category/channel/delivery` + idempotent `ALTER TABLE` migration
  (`db/telemetry/session.py`); mirrored in the DuckDB `flat` view; `Filters.family/category`.
- Orchestrator: `RunConfig.families`, `RunConfig.wrap_with_cipher`; `run_family_attempt`, `apply_cipher`,
  `family_cells` (§A.3 cross-product), `_dispatch` (indirect families call `target.send_document`).
  Legacy `strategies` remain registered (corpus replay depends on the names).
- Mock: `SIM_FAMILY_BOOST` + `sim_family` contextvar → simulated per-family susceptibility, so the
  profile is non-flat offline. Numbers are invented; documented as such.
- API: `GET /api/families`, `families` in `/api/meta`.
- Gate: `scripts/gate.sh`; `backend/mypy-baseline.txt` (58 pre-existing errors; gate = no NEW errors);
  vitest smoke test. Optional extras `[rex]` (torch) and `[mcp]`.
- REX license check: `github.com/AI45Lab/REX` is **Apache-2.0**. Its code targets 26–35B MoE on
  multi-GPU, so we **reimplement** the three mechanisms at toy scale (no vendoring) and cite paper +
  repo in README/NOTICE.

## Sprint 1 (parallel)

| Story | Plan phase | Owner | Key files |
|---|---|---|---|
| S1-A1 single-turn jailbreak families | A1 | dev agent A1 | `engine/families/{roleplay_persona,authority_context,persuasion,refusal_suppression,hypothetical_framing,many_shot,low_resource_lang,payload_splitting}.py`, `tests/unit/families/` |
| S1-A4 direct prompt-injection families | A4 | dev agent A4 | `engine/families/injection/*.py`, `datasets/loaders.py` (`injection_canary`), mock injection behaviour |
| S1-B REX toy MoE + relocation + excision | B1+B2 | dev agent B | `defenses/rex/{toy_moe,data,relocate,evaluate}.py`, `tests/unit/rex/`, `NOTICE` |

## Sprint 2

| Story | Plan phase | Key files |
|---|---|---|
| S2-A2 meta-families (Crescendo, TAP) via PyRIT, any single-turn family as leaf | A2 | `families/multi_turn.py`, `families/adaptive_tree.py`, `pyrit_native.py`, orchestrator meta runner |
| S2-A5 toy agent + indirect injection (`visible`, `html_comment`, `structured_data`) | A5 | `engine/targets/toy_agent_target.py`, `engine/tools/*.py`, `families/injection/{tool_hijack,data_exfil}.py` |
| S2-A3 Robustness Profile analytics + UI | A3 (+D.5) | `api/analytics.py` (`family-heatmap`, `family-radar`, `compounding`, `injection-breakdown`), `frontend/src/pages/Analytics.tsx` |

## Sprint 3

| Story | Plan phase | Key files |
|---|---|---|
| S3-B3 REX-in-the-loop: `ToyMoETarget`, full-vs-excised profile, Defenses chart | B3 | `engine/targets/toy_moe_target.py`, Analytics "Defenses" section |
| S3-C1 MCP server (7 tools + `get_injection_profile`; no key/delete/export) | C1 | `backend/app/mcp_server/*`, README `.mcp.json` |
| S3-A6 rendered-HTML deliveries (`css_hidden`, `tiny_text`, `color_match`, `offscreen`) | A6 (optional) | delivery wrappers + detector signals |
| S3-DOC README / NOTICE: taxonomy + citations, OWASP/ATLAS map, REX limitations, CC BY 4.0 + Apache-2.0 attribution | DoD | `README.md`, `NOTICE` |

## Safety invariants (QA checks these on every story)

1. No tuned adversarial payload library: families describe a published technique and generate at
   runtime; offline fallbacks are minimal structural templates.
2. Default objectives are benign canaries (`CANARY-…`, `SENTINEL-OK-…`, `INJECTION-OK-…`); real-harm
   datasets are BYO, never shipped.
3. REX forget set is synthetic (invented entities + random tokens); no real bio/chem/cyber/weapons text.
4. Mock tools are local and side-effect free; `send_canary` writes to a local table only.
5. MCP exposes read/compute tools only.
6. Determinism: config + seed reproduces a run byte-for-byte.
