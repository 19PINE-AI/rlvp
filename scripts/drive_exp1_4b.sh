#!/bin/bash
# Driver B: Exp1 harm-reduction matrix at 4B. Each run RETRIED until 40 iters.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLVP_MEM_FRAC=0.45
GATE=${GATE:-42}; MAXTRY=${MAXTRY:-10}
free_gb () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
iters () { wc -l < "$1/train_log.jsonl" 2>/dev/null || echo 0; }
log () { echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

SPECS=(
  "outcome fileops _4b -"
  "c3 fileops _4b -"
  "llmcritic fileops _4b -"
  "c3 csops _4b -"
  "llmcritic csops _4b -"
  "c2 csops nodis_4b RLVP_BETA=0"
)
for spec in "${SPECS[@]}"; do
  set -- $spec; credit=$1; domain=$2; suf=$3; extra=$4
  dir="results/exp_sc_train_${credit}${suf}_${domain}"
  [ "$extra" = "-" ] && extra=""
  try=0
  while [ "$(iters "$dir")" -lt 40 ] && [ "$try" -lt "$MAXTRY" ]; do
    try=$((try + 1))
    while [ "$(free_gb)" -lt "$GATE" ]; do sleep 60; done
    rm -rf "$dir"
    log "RUN credit=$credit domain=$domain suffix=$suf try=$try (free=$(free_gb)GB)"
    env RLVP_OUT_SUFFIX="$suf" $extra \
        python3 scripts/exp_selfcritic_train.py "$credit" 40 "$domain" Qwen/Qwen3-4B \
        > "results/drive_4b_${credit}${suf}_${domain}.log" 2>&1
    [ "$(iters "$dir")" -ge 40 ] && log "done $credit$suf $domain" || log "incomplete $credit$suf $domain (try $try)"
  done
done
log "DRIVER_B_DONE"
