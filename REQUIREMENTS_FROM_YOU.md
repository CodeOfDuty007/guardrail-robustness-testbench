# Requirements from you (the operator)

Everything is built and tested against a **mock target** with zero keys. These are the only things
that block the *final execution* phase (real models, real GPU). Check them off as you go.

## A. Needed for the GPU / real-model run (end phase)

| # | What | Why | Where it goes |
|---|------|-----|---------------|
| A1 | **A GPU host** (own box, Lambda/RunPod/Vast, or university cluster) with ≥24 GB VRAM for Llama-3-8B + judge, or ≥48 GB to host attacker + target + judge simultaneously | Real inference | — |
| A2 | **SSH or HTTPS reachability** of that host from the machine running the testbench (VPN, Tailscale, or an SSH tunnel `ssh -L 11434:localhost:11434 gpu`) | Orchestrator → GPU | Compute page → "Add node" |
| A3 | **Inference server running on the GPU host**: Ollama (`ollama serve`) *or* vLLM (`vllm serve <model>`) — both expose an OpenAI-compatible API. `docker-compose.gpu.yml` does this for you | Model serving | `compute/docker-compose.gpu.yml` |
| A4 | **Model names** you want in the arena, pulled on the host, e.g. `llama3:8b`, `qwen2.5:7b` (judge), `llama-guard3:8b`, `tinyllama:1.1b` | Attacker / target / judge roles | Compute page + New Experiment |
| A5 | If using gated HF weights (Llama): your **Hugging Face token** accepted for the license | vLLM download | env on the GPU host only |
| A6 | **Cloud BYOK keys** (OpenAI / Anthropic / Gemini …) *only for models you are authorized to test* — check each provider's usage policy / red-team programme first | Cross-provider arena | Settings page (memory-only by default) |
| A7 | **Budget cap** (USD) per run for cloud targets | Cost guardrail | New Experiment |

## B. Needed before any public release (decisions only you can make)

| # | Decision | Default I used |
|---|----------|----------------|
| B1 | Operator name for provenance headers / `curated_by` | `OPERATOR_NAME` env, falls back to OS user |
| B2 | Vendor contacts + disclosure hold window | 90 days, set per report in the wizard |
| B3 | Real contact address in `SECURITY.md` / `DISCLOSURE.md` | placeholder `security@example.org` |
| B4 | Report-signing secret (HMAC) | auto-generated into `corpus/.signing_key` (0600) on first use |
| B5 | Licence confirmation (Apache-2.0 code, CC-BY results) | Apache-2.0 declared |
| B6 | **Datasets you legally obtain yourself**: JBB-Behaviors, HarmBench-text, StrongReject (never shipped) | drop CSV/JSON in `datasets/` |
| B7 | Python 3.11–3.13 for the pinned PyRIT (3.14 is **not** supported by PyRIT yet) | project venv uses 3.12 via `uv` |

## C. Things I could not verify offline (please confirm on first online run)

| # | Item |
|---|------|
| C1 | Exact PyRIT 1.x version to pin (`engine/pyrit_bridge.py` is import-guarded and the pin in `pyproject.toml` is a placeholder until you run `uv lock` online) |
| C2 | That `litellm` accepts your Ollama / vLLM base URLs (the OpenAI-compatible fallback target works without it) |
| C3 | Llama Guard 3 availability in your Ollama registry |
