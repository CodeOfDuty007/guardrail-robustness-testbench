#!/usr/bin/env bash
# Run on the GPU host after `docker compose -f docker-compose.gpu.yml up -d`
set -euo pipefail
for m in llama3:8b qwen2.5:7b llama-guard3:8b tinyllama:1.1b; do
  docker compose -f "$(dirname "$0")/docker-compose.gpu.yml" exec ollama ollama pull "$m"
done
