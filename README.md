# Guardrail Robustness Testbench

An open-source research platform for evaluating the robustness of large language model (LLM)
safety guardrails against adversarial prompting, built on top of
[PyRIT](https://github.com/Azure/PyRIT) (Python Risk Identification Toolkit). The system
orchestrates attacker, target, and judge language models in a configurable pipeline, scores
outcomes along two independent axes (filter evasion and semantic compliance), and persists
every attempt with full provenance for later analysis, replication, and disclosure.

The project is intended for authorized security research and red-teaming of models the
operator owns or is explicitly permitted to test.

## Table of Contents

- [Motivation](#motivation)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation and Local Execution](#installation-and-local-execution)
- [Configuring Model Providers](#configuring-model-providers)
- [Running an Evaluation](#running-an-evaluation)
- [Data Model and Reproducibility](#data-model-and-reproducibility)
- [Deployment](#deployment)
- [Security and Key Handling](#security-and-key-handling)
- [Ethics and Responsible Use](#ethics-and-responsible-use)
- [Related Work](#related-work)
- [License](#license)

## Motivation

Evaluating whether an LLM's safety guardrails have been bypassed is harder than checking
whether a model produced a particular string. A model may emit a benign marker string for
reasons unrelated to any guardrail failure (for example, because the string itself carries no
policy weight), which produces a false signal of "jailbreak" when none has occurred. This
testbench instead scores every attempt on two independent axes:

1. **Filter evasion** (`filter_evaded`): whether a deterministic, pre-registered detector
   (regular expression or substring match) fires against the model's response.
2. **Semantic compliance** (`semantic_complied`): whether an independent judge model
   determines that the response constitutes a genuine violation of the stated objective,
   rather than incidental or decorative text.

An attempt is only treated as a confirmed bypass when both axes agree
(`filter_evaded ∧ semantic_complied`). This conjunctive criterion is the basis for every
summary statistic, corpus promotion, and vendor disclosure the system produces.

## Method

The pipeline composes three independent roles, each backed by a configurable model provider:

- **Target**: the model under evaluation.
- **Attacker**: an auxiliary model used by adaptive strategies (PAIR, TAP, Crescendo) to
  iteratively rewrite the objective in response to the target's prior refusals.
- **Judge**: an auxiliary model used to assess semantic compliance independently of the
  deterministic detector.

Attack construction separates two orthogonal dimensions:

- **Strategy**: the wording and structural framing of the request (for example, persona
  roleplay, instruction override, many-shot priming at varying context lengths, refusal
  suppression, prefix injection, hypothetical framing, payload splitting, or iterative
  LLM-driven refinement via PAIR/TAP/Crescendo).
- **Cipher**: an optional encoding layer applied to the payload before transmission
  (Base64, ROT-N, XOR masking, AES-128-GCM, or no encoding), used to study whether
  obfuscation of the channel independently affects guardrail robustness.

Every attempt is scored, timed, and persisted together with cryptographic/information-theoretic
metrics on the payload (avalanche effect, chi-squared confusion, diffusion score, Shannon
entropy) to support correlational analysis between payload properties and bypass rate.

## Repository Structure

```
.
├── backend/                     FastAPI application and evaluation engine
│   ├── app/
│   │   ├── api/                 HTTP and WebSocket route handlers
│   │   ├── datasets/            Objective loaders (shipped canaries; BYO benchmark datasets)
│   │   ├── db/
│   │   │   ├── telemetry/       Primary run/attempt/payload/response/judgment schema
│   │   │   └── corpus/          Curated, promoted-attack schema (disclosure pipeline)
│   │   ├── defenses/            Optional input/output defense transforms for ablation studies
│   │   ├── engine/
│   │   │   ├── converters/      Cipher implementations
│   │   │   ├── scorers/         Regex and LLM-judge scoring
│   │   │   ├── targets/         Provider-agnostic target abstraction (OpenAI-compatible, LiteLLM)
│   │   │   ├── orchestrator.py  Run lifecycle, persistence, replay
│   │   │   ├── pyrit_bridge.py  Optional native PyRIT execution path
│   │   │   ├── pyrit_native.py  PyRIT PromptTarget adapter
│   │   │   └── strategies.py    Prompting strategy registry
│   │   ├── providers/           BYOK key manager (process-memory only) and redaction utilities
│   │   └── corpus/               Promotion, re-verification, and signed disclosure generation
│   ├── scripts/                 Standalone evaluation batteries (see below)
│   ├── tests/                   Unit, integration, and end-to-end tests
│   └── pyproject.toml
├── frontend/                     React/TypeScript single-page application
│   └── src/
│       ├── pages/                Dashboard, New Run, Live Console, Analytics, DB Explorer, Corpus, Settings
│       ├── components/
│       ├── hooks/
│       └── lib/
├── compute/                       GPU-node compose files and model-pull scripts for self-hosted targets
├── configs/                        Example run configurations
├── datasets/                       BYO benchmark dataset drop directory (git-ignored; not shipped)
├── corpus/                          Local attack corpus database (git-ignored; generated at runtime)
├── docker-compose.yml
├── SPEC.md                         Full technical specification
├── SECURITY.md                     Vulnerability disclosure contact
└── DISCLOSURE.md                   Template for vendor disclosure reports
```

No attack corpus, benchmark dataset, or credential is shipped in this repository. The
`corpus/` and `datasets/` directories are populated only by the operator's own runs and are
excluded from version control.

## Requirements

- Python 3.11–3.13 (3.14 is not yet supported by the pinned PyRIT release)
- Node.js 18 or later
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- Optionally, Docker and Docker Compose for containerized execution
- Optionally, API credentials for one or more LLM providers (OpenAI-compatible endpoints,
  Google Gemini, or a self-hosted Ollama/vLLM instance). No credentials are required to
  run the system against the built-in simulated target.

## Installation and Local Execution

Clone the repository and start the backend:

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>/backend
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/uvicorn app.main:app --port 8000
```

In a second terminal, start the frontend:

```bash
cd <repo>/frontend
npm install
npm run dev
```

The application is served at `http://localhost:5173` and proxies API and WebSocket traffic
to the backend at `http://localhost:8000`.

To run the automated test suite:

```bash
cd backend
.venv/bin/pytest
```

To run the full stack under Docker Compose instead:

```bash
docker compose up
```

By default, the system runs against a simulated target model that requires no credentials.
This is sufficient to exercise the full pipeline, user interface, and database schema, but
produces no scientifically meaningful robustness results. Meaningful evaluation requires
configuring at least one real model provider, as described below.

## Configuring Model Providers

Model providers are registered and credentialed entirely at runtime through the API or user
interface; no credential is ever written to a configuration file, environment file tracked by
version control, or source file.

To register a provider:

```bash
curl -X POST http://localhost:8000/api/providers \
  -H "Content-Type: application/json" \
  -d '{"label": "<label>", "kind": "openai_compat", "model_name": "<model>",
       "base_url": "<endpoint>", "role_hint": "<target|attacker|judge>"}'
```

To supply a credential for a registered provider (held in server process memory only,
never persisted):

```bash
curl -X PUT http://localhost:8000/api/providers/<id>/key \
  -H "Content-Type: application/json" \
  -d '{"key": "<credential>", "remember": false}'
```

To confirm connectivity:

```bash
curl -X POST http://localhost:8000/api/providers/<id>/validate
```

A self-hosted target reachable over an OpenAI-compatible API (Ollama, vLLM, or an
equivalent server) can instead be attached through the **GPU Nodes** page of the user
interface, which probes the node and offers its served models as selectable providers.

## Running an Evaluation

An evaluation run specifies a dataset of objectives, a set of prompting strategies, a set of
ciphers, and the provider roles to use:

```bash
curl -X POST http://localhost:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"name": "<run-name>", "target_provider_id": <id>, "judge_provider_id": <id>,
       "attacker_provider_id": <id>, "dataset": "<dataset>", "limit": <n>,
       "ciphers": ["<cipher>", ...], "strategies": ["<strategy>", ...],
       "max_iterations": <n>, "concurrency": <n>}'
```

Run progress can be observed in real time through the Live Console page of the user
interface, which subscribes to the run over a WebSocket, or polled via
`GET /api/runs/{id}` and `GET /api/explorer/attempts`.

Two objective datasets are shipped with the repository:

- `canary`: benign, operator-planted marker strings used to validate that the pipeline and
  scoring are functioning correctly. A canary bypass demonstrates that an encoded or
  reframed instruction reached the model and was acted upon; it does not by itself
  demonstrate a safety-relevant guardrail failure, because the marker string carries no
  policy weight of its own.
- `policy_violation`: objectives whose success criterion is an affirmative statement that
  itself constitutes a policy violation (for example, a false claim about the model's own
  identity, or an explicit disclaimer of safety constraints). Because a well-aligned model
  has a standing reason to refuse these regardless of framing, a confirmed bypass on this
  dataset is stronger evidence of an actual guardrail failure than a canary bypass.

Additional benchmark datasets (for example JailbreakBench, HarmBench, or StrongREJECT) are
not shipped with this repository and must be obtained independently by the operator under
the terms of their respective licenses, then placed in `datasets/` in the column format
documented in `backend/app/datasets/loaders.py`.

The `backend/scripts/` directory contains example evaluation batteries (`policy_run.py`,
`policy_run_v2.py`, `policy_run_v3.py`) demonstrating programmatic use of the run API across
multiple strategies, ciphers, and objectives with rate-limit-aware batching, suitable as a
starting point for a custom evaluation sweep.

## Data Model and Reproducibility

Every attempt is persisted across five related tables (`runs`, `attempts`, `payloads`,
`responses`, `judgments`), queryable through the DB Explorer page or the read-only SQL
endpoint `POST /api/explorer-sql`. Each payload record stores:

- the plaintext objective and the final ciphertext payload,
- `full_request_sent`: the exact, verbatim message transmitted to the target model,
- `reproduction_steps`: a structured, human-readable procedure for reconstructing and
  resending the identical attack,
- the cipher (if any) used to encode the inter-model payload, recorded as a first-class
  column (`cipher_type`) rather than inferred,
- cryptographic/information-theoretic metrics computed on the payload.

This is sufficient for an independent party to reconstruct any recorded attempt exactly,
without requiring access to this system's source code or runtime state.

Attempts meeting the two-axis bypass criterion may be promoted to a local, operator-only
corpus (`corpus/corpus.db`), which supports re-verification against the live target at a
later date (to detect whether a vendor has since patched the behavior) and generation of a
cryptographically signed, provenance-stamped disclosure report scoped to a single vendor and
target model.

## Deployment

The two halves of this system have different deployment requirements and are deployed
separately.

**Frontend.** The React application in `frontend/` is a static single-page application and
is suitable for deployment on Vercel or an equivalent static host:

```bash
cd frontend
npm run build
```

The build output in `frontend/dist/` can be deployed directly. When deployed separately from
the backend, set the API base URL the frontend should target (see `frontend/src/lib/`) to
the backend's public address, and configure CORS on the backend (`cors_origins` in
`backend/app/config.py`) to permit the deployed frontend's origin.

**Backend.** The FastAPI backend is intentionally not serverless-compatible and should not
be deployed to Vercel or a similar function-per-request platform. It depends on:

- local SQLite files (`telemetry.db`, `corpus/corpus.db`) that must persist across requests
  and survive restarts,
- an in-process, memory-only credential store that is deliberately not externalized to any
  database, cache, or secret store, by design (see Security and Key Handling below),
- a long-lived WebSocket connection for the Live Console,
- long-running background tasks for in-progress evaluation runs.

These properties require a persistent process and a persistent filesystem. Suitable targets
include a self-managed virtual machine, Render, Fly.io, Railway, or an on-premises host. The
provided `backend/Dockerfile` and root `docker-compose.yml` are suitable for any such
platform that accepts a container image.

## Security and Key Handling

- Provider credentials supplied through the API or user interface are held in server
  process memory only. They are never written to either database, to logs, to exported
  reports, or to any corpus record, and do not survive a process restart unless the
  operator explicitly opts into persistence.
- All text fields that may contain operator-supplied content are passed through a
  redaction utility before being included in logs, validation responses, or disclosure
  reports.
- The attack corpus (`corpus/corpus.db`) and any benchmark datasets placed in `datasets/`
  are excluded from version control and from the Docker build context.
- Disclosure reports are signed with an HMAC key generated on first use and stored
  locally with restrictive file permissions; the signature allows a recipient to verify
  that a report has not been altered after generation.
- See `SECURITY.md` for the vulnerability disclosure contact and process for this project
  itself.

Before pointing any credential at a third-party model, the operator is responsible for
confirming that the intended evaluation is permitted under that provider's usage policy or
through its red-teaming program.

## Ethics and Responsible Use

This system is designed for authorized evaluation of guardrail robustness, not for
producing or distributing harmful content. Several design decisions follow from this:

- The only objective dataset shipped with the repository beyond operator-planted canaries
  (`policy_violation`) is constructed so that a confirmed bypass is evidence of a guardrail
  failure (a false identity claim, a false disclaimer of safety constraints) without
  requiring the model to produce genuinely harmful output, such as operational instructions
  for causing real-world harm.
- Benchmark datasets containing sensitive or dual-use behavior descriptions (for example
  HarmBench or JailbreakBench) are never bundled with this repository; obtaining and using
  them is the operator's own responsibility under their respective license terms.
- The attack corpus is local and operator-controlled. It leaves the local machine only
  inside a signed, scoped disclosure report intended for the affected vendor, not as a
  general-purpose dataset release.
- Operators must restrict evaluation to models they own or are explicitly authorized to
  test.

## Related Work

This project builds on and benchmarks against ideas from, among others: CipherChat
(Yuan et al., ICLR 2024); Jailbroken: How Does LLM Safety Training Fail? (Wei et al., 2023);
Exploiting Programmatic Behavior of LLMs (Kang et al., 2023); ACE (Handa et al., 2024);
StrongREJECT (Souly et al., 2024); Many-Shot Jailbreaking (Anthropic, 2024); PAIR
(Chao et al., 2023); and Tree of Attacks with Pruning (Mehrotra et al., 2023). No novelty is
claimed for any individual prompting or encoding technique; the contribution of this
repository is the engineering artifact connecting payload-level cryptographic metrics,
two-axis bypass scoring, and a reproducible disclosure pipeline. OWASP LLM Top 10 and
MITRE ATLAS tags are attached to every recorded run for cross-reference with established
taxonomies.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for the full text.
