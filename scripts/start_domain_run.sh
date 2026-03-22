#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_INTENT="find RFP websites for data annotation"
INTENT="${INTENT:-$DEFAULT_INTENT}"

STOP_EXISTING="${STOP_EXISTING:-true}"
if [[ "$STOP_EXISTING" == "true" ]]; then
  WORKER_PIDS="$(ps -eo pid,args | awk '/python .*internet_explorer\.cli/ && !/awk/ {print $1}')"
  if [[ -n "$WORKER_PIDS" ]]; then
    echo "stopping existing worker pid(s): $WORKER_PIDS"
    kill -TERM $WORKER_PIDS || true
    sleep 4
  fi
fi

REMAINING_WORKERS="$(ps -eo pid,args | awk '/python .*internet_explorer\.cli/ && !/awk/ {print}')"
if [[ -n "$REMAINING_WORKERS" ]]; then
  echo "worker still running; refusing to start another one:" >&2
  echo "$REMAINING_WORKERS" >&2
  exit 1
fi

mkdir -p run_logs
RUNTIME_LOG="run_logs/runtime-up-$(date +%Y%m%d-%H%M%S).json"
.venv/bin/python scripts/runtime_up.py --skip-mongo > "$RUNTIME_LOG"

RUN_LOG="run_logs/domain-soak-$(date +%Y%m%d-%H%M%S).log"
nohup env \
  CANDIDATE_START_MODE="${CANDIDATE_START_MODE:-domain_homepage}" \
  DISCOVERY_CACHE_MODE="${DISCOVERY_CACHE_MODE:-off}" \
  URL_BATCH_SIZE="${URL_BATCH_SIZE:-40}" \
  MAX_URL_CONCURRENCY="${MAX_URL_CONCURRENCY:-0}" \
  MAX_BROWSER_CONCURRENCY="${MAX_BROWSER_CONCURRENCY:-0}" \
  BROWSER_DELEGATE_TIMEOUT_SECONDS="${BROWSER_DELEGATE_TIMEOUT_SECONDS:-240}" \
  .venv/bin/python -m internet_explorer.cli --intent "$INTENT" > "$RUN_LOG" 2>&1 &

RUN_PID="$!"
echo "pid=$RUN_PID"
echo "run_log=$RUN_LOG"
echo "runtime_log=$RUNTIME_LOG"
