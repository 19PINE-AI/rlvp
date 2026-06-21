#!/bin/bash
# Patient gated runner for the self-critique vs RLVP study. Polls the GPU every
# 60s; before EACH job it waits until there is enough free memory, so an in-flight
# miniF2F run is never starved or blocked. Captures the GPU the moment a big-enough
# window opens. Executes in order:
#   1. Exp0 critic scale sweep  (4B, 8B) -- structural-vs-capability ceiling
#   2. Exp1 harm-reduction matrix at full horizon (coexistence-capped)
# Writes a heartbeat + stage status to results/sc_autorun.status so a supervisor
# can read progress at a glance. Safe to leave running for hours.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SWEEP_GB=${SWEEP_GB:-30}        # free GB needed for an 8B critic load (~22GB + margin)
MATRIX_GB=${MATRIX_GB:-18}      # free GB needed for the 1.7B LoRA matrix (~15GB cap)
POLL=${POLL:-60}
STATUS=results/sc_autorun.status

log () { echo "[$(date +%H:%M:%S)] $*"; }
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
status () { echo "[$(date +%H:%M:%S)] stage=$1 free=$(free_gb)GB $2" > "$STATUS"; }

wait_free () {  # $1 = GB needed, $2 = stage label
  while [ "$(free_gb)" -lt "$1" ]; do
    status "$2" "waiting (need ${1}GB)"
    sleep "$POLL"
  done
}

# ---- 1. Exp0 scale sweep (same model = policy = critic; on-policy self-critique) ----
for M in Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  if [ -f "results/exp_selfcritic/${M##*/}/report.json" ]; then
    log "skip Exp0 $M (already have report)"; continue
  fi
  log "waiting for >=${SWEEP_GB}GB to run Exp0 sweep $M ..."
  wait_free "$SWEEP_GB" "exp0:$M"
  status "exp0:$M" "RUNNING"
  log "Exp0 scale sweep: $M (free=$(free_gb)GB)"
  RLVP_MEM_FRAC=0 python3 scripts/exp_selfcritic.py "$M" 24 \
      > "results/exp_selfcritic_${M##*/}.log" 2>&1 \
    && log "  done $M" || log "  FAILED $M (see log)"
done

# ---- 2. Exp1 harm-reduction matrix at full horizon (1.7B LoRA, coexistence-capped) ----
log "waiting for >=${MATRIX_GB}GB to run Exp1 matrix ..."
wait_free "$MATRIX_GB" "exp1:matrix"
status "exp1:matrix" "RUNNING"
log "Exp1 matrix (free=$(free_gb)GB)"
RLVP_MEM_FRAC=0.15 ITERS=${ITERS:-40} bash scripts/run_selfcritic_exp1.sh \
    > results/exp1_matrix.log 2>&1 \
  && log "Exp1 matrix done" || log "Exp1 matrix FAILED (see results/exp1_matrix.log)"

python3 scripts/exp1_aggregate.py > results/exp1_summary.txt 2>&1 || true
status "DONE" "all stages complete"
log "ALL DONE"
