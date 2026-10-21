#!/usr/bin/env bash
# CI gate (plan "Cross-cutting requirements"): ruff + mypy (no NEW errors vs baseline) + pytest + vitest + tsc build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
PY=.venv/bin/python
echo "== ruff";   .venv/bin/ruff check app tests
echo "== mypy (no new errors vs backend/mypy-baseline.txt)"
.venv/bin/mypy app 2>/dev/null | grep -E '^app/.*: error:' | sed -E 's/:[0-9]+: error:/: error:/' | sort > /tmp/mypy-now.$$ || true
NEW=$(comm -13 mypy-baseline.txt /tmp/mypy-now.$$ || true); rm -f /tmp/mypy-now.$$
if [ -n "$NEW" ]; then echo "new mypy errors:"; echo "$NEW"; exit 1; fi
echo "== pytest"; $PY -m pytest -q -p no:cacheprovider
cd "$ROOT/frontend"
echo "== vitest"; npm test --silent
echo "== tsc + vite build"; npm run build --silent >/dev/null
echo "GATE GREEN"
