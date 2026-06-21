#!/bin/bash
# tau2 cell-C end-to-end, gated on a free GPU window so it never starves miniF2F.
# Pipeline: user-sim vLLM (system py) -> rollout (.venv-tau2) -> kill server ->
# blind self-critique + score (system py). Needs ~25GB (user-sim 17 + policy 8).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TAU2_DATA_DIR=/tmp/tau2-bench/data
export OPENAI_API_KEY=local OPENAI_API_BASE=http://localhost:8011/v1 OPENAI_BASE_URL=http://localhost:8011/v1
NEED_GB=${NEED_GB:-28}
LOG=results/tau2_cellc
mkdir -p "$LOG"
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
log () { echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

log "waiting for >=${NEED_GB}GB free for tau2 cell-C ..."
while [ "$(free_gb)" -lt "$NEED_GB" ]; do sleep 60; done
log "window open ($(free_gb)GB free). launching user-sim vLLM (Qwen3-4B :8011)"

pkill -9 -f "api_server --model Qwen/Qwen3-4B --port 8011" 2>/dev/null; sleep 2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --port 8011 --gpu-memory-utilization 0.18 \
  --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 > "$LOG/usersim_vllm.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 60); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && break; sleep 5; done
if ! curl -s localhost:8011/v1/models >/dev/null 2>&1; then
  log "user-sim vLLM FAILED to come up"; kill -9 $VLLM_PID 2>/dev/null; exit 1
fi
log "user-sim up. rolling out tau2 episodes (semantic rules)"
.venv-tau2/bin/python scripts/tau2_cellc_rollout.py 15 4 > "$LOG/rollout.log" 2>&1 \
  && log "rollout done" || log "rollout FAILED (see rollout.log)"
kill -9 $VLLM_PID 2>/dev/null; sleep 3   # free the user-sim before the critic loads
log "critiquing (blind self-critique, Qwen3-4B)"
python3 scripts/tau2_cellc_critique.py Qwen/Qwen3-4B > "$LOG/critique.log" 2>&1 \
  && log "critique done" || log "critique FAILED (see critique.log)"
log "CELLC_DONE"
