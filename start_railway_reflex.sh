#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-7860}"
export API_CONFIG_PATH="${API_CONFIG_PATH:-configs/api_railway.yaml}"
export EHC_API_BASE="${EHC_API_BASE:-http://localhost:8000}"

echo "[railway-entrypoint] Starting FastAPI inference backend on :8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

cleanup() {
  echo "[railway-entrypoint] Shutting down processes"
  kill "${API_PID}" >/dev/null 2>&1 || true
  kill "${REFLEX_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[railway-entrypoint] Starting Reflex frontend on :${PORT}"
reflex run --env prod --single-port --frontend-port "${PORT}" --backend-port "${PORT}" --backend-host 0.0.0.0 &
REFLEX_PID=$!

wait -n "${API_PID}" "${REFLEX_PID}"
