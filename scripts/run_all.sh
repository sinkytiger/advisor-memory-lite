#!/usr/bin/env bash
# The full local run on the test split. Resumable: after an interruption, run the same command again.
# Order matters on a 6 GB GPU: build every LLM-written memory with the writer model first, then read with one
# reader model at a time, so Ollama does not keep swapping models.
set -euo pipefail
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8 MEM0_TELEMETRY=False
"$PY" -m advisor_memory build
"$PY" -m advisor_memory prepare --split test --workers "${WORKERS:-3}"
for reader in llama3.2:3b qwen3:4b; do
  "$PY" -m advisor_memory run --reader "$reader" --split test --out runs/main
done
"$PY" -m advisor_memory report --runs runs/main --split test --out reports/results.md
