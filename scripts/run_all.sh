#!/usr/bin/env bash
# The full local run on the test split. Resumable: after an interruption, run the same command again.
# Order matters on a 6 GB GPU: build every LLM-written memory with the writer model first, then read with one
# reader model at a time, so Ollama does not keep swapping models.
# Each step waits until Ollama answers and is retried if the model server drops out mid-step (an Ollama
# self-update, a restart); finished answers and memories are kept on disk, so a retry continues where it stopped.
set -uo pipefail
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8 MEM0_TELEMETRY=False

wait_ollama() { until ollama ps >/dev/null 2>&1; do sleep 15; done; }
attempt() {
  for i in 1 2 3 4 5 6; do
    wait_ollama
    "$@" && return 0
    echo "step failed (attempt $i): $*; retrying in 60s"
    sleep 60
  done
  echo "giving up on: $*"
  return 1
}

attempt "$PY" -m advisor_memory build || exit 1
attempt "$PY" -m advisor_memory prepare --split test --workers "${WORKERS:-3}" || exit 1
for reader in llama3.2:3b qwen3:4b; do
  attempt "$PY" -m advisor_memory run --reader "$reader" --split test --out runs/main || exit 1
done
"$PY" -m advisor_memory report --runs runs/main --split test --out reports/results.md
