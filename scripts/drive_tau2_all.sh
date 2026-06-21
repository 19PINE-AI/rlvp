#!/bin/bash
# Driver C+D: tau2 cell-C. Offline x3 + training (outcome/sem-c3/llmcritic).
# Each step gated + retried; the user-sim vLLM is (re)started as needed.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TAU2_DATA_DIR=/tmp/tau2-bench/data
export OPENAI_API_KEY=local OPENAI_API_BASE=http://localhost:8011/v1 OPENAI_BASE_URL=http://localhost:8011/v1
GATE=${GATE:-40}; MAXTRY=${MAXTRY:-8}
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
iters () { wc -l < "$1/train_log.jsonl" 2>/dev/null || echo 0; }
log () { echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
L=results/tau2_cellc; mkdir -p "$L"
VLLM_PID=""
trap 'kill -9 $VLLM_PID 2>/dev/null' EXIT

ensure_server () {  # gate on a window, (re)start user-sim vLLM if not responding
  while [ "$(free_gb)" -lt "$GATE" ]; do sleep 60; done
  if curl -s localhost:8011/v1/models >/dev/null 2>&1; then return 0; fi
  pkill -9 -f "api_server --model Qwen/Qwen3-4B --port 8011" 2>/dev/null; sleep 2
  log "starting user-sim vLLM (free=$(free_gb)GB)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --port 8011 --gpu-memory-utilization 0.18 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    > "$L/usersim_vllm.log" 2>&1 &
  VLLM_PID=$!
  for i in $(seq 1 60); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && return 0; sleep 5; done
  log "user-sim FAILED to come up"; return 1
}

# ---- C) offline cell-C, 3 seeds ----
for i in 1 2 3; do
  try=0
  while [ ! -s "$L/run$i/report.json" ] && [ "$try" -lt "$MAXTRY" ]; do
    try=$((try + 1)); ensure_server || continue
    log "cell-C offline run $i (try $try): rollout"
    .venv-tau2/bin/python scripts/tau2_cellc_rollout.py 15 4 > "$L/rollout$i.log" 2>&1 || { log "rollout$i fail"; continue; }
    python3 scripts/tau2_cellc_critique.py Qwen/Qwen3-4B > "$L/critique$i.log" 2>&1 || { log "critique$i fail"; continue; }
    mkdir -p "$L/run$i"; cp "$L/traj.json" "$L/report.json" "$L/run$i/" 2>/dev/null && log "cell-C run$i done"
  done
done

# ---- D) cell-C training: outcome / semantic-c3 / llmcritic ----
for spec in "outcome run_tau2_cellc_outcome" "c3 run_tau2_cellc_sem" "llmcritic run_tau2_cellc_llmcritic"; do
  set -- $spec; credit=$1; out=$2
  try=0
  while [ "$(iters "results/$out")" -lt 20 ] && [ "$try" -lt "$MAXTRY" ]; do
    try=$((try + 1)); ensure_server || continue
    rm -rf "results/$out"
    log "tau2 TRAIN credit=$credit (try $try) -> $out"
    .venv-tau2/bin/python scripts/tau2_train.py 20 --credit "$credit" --rule-mode semantic \
        --with-policy --out "$out" > "results/${out}.log" 2>&1
    [ "$(iters "results/$out")" -ge 20 ] && log "done tau2-train $credit" || log "incomplete tau2-train $credit (try $try)"
  done
done
log "DRIVER_TAU2_DONE"
