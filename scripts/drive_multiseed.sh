#!/bin/bash
# Driver A: multi-seed + frozen-critic on csops (1.7B). c2nodis/llmcritic/
# llmcriticfrozen × seeds 11/22/33. Each run is RETRIED until it lands 40 iters,
# so a mid-run OOM (miniF2F restarting) just restarts next window.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLVP_MEM_FRAC=0.25
GATE=${GATE:-22}; MAXTRY=${MAXTRY:-10}
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
iters () { wc -l < "$1/train_log.jsonl" 2>/dev/null || echo 0; }
log () { echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

SPECS=(
  "c2nodis c2 nodis_sSEED RLVP_BETA=0"
  "llmcritic llmcritic _sSEED -"
  "llmcriticfrozen llmcritic frozen_sSEED RLVP_FROZEN_CRITIC=1"
)
for S in 11 22 33; do
  for spec in "${SPECS[@]}"; do
    set -- $spec; tag=$1; credit=$2; suf=${3/SEED/$S}; extra=$4
    dir="results/exp_sc_train_${tag}_s${S}_csops"
    [ "$extra" = "-" ] && extra=""
    try=0
    while [ "$(iters "$dir")" -lt 40 ] && [ "$try" -lt "$MAXTRY" ]; do
      try=$((try + 1))
      while [ "$(free_gb)" -lt "$GATE" ]; do sleep 60; done
      rm -rf "$dir"
      log "RUN tag=$tag seed=$S try=$try (free=$(free_gb)GB)"
      env RLVP_SEED=$S RLVP_OUT_SUFFIX="$suf" $extra \
          python3 scripts/exp_selfcritic_train.py "$credit" 40 csops \
          > "results/drive_ms_${tag}_s${S}.log" 2>&1
      [ "$(iters "$dir")" -ge 40 ] && log "done $tag s$S" || log "incomplete $tag s$S (try $try, $(iters "$dir") iters)"
    done
  done
done
log "DRIVER_A_DONE"
