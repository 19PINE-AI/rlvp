#!/bin/bash
# Experiment 1 matrix: does self-critique reward reduce TRUE (oracle) violations
# only where Exp 0 found the critic accurate? Runs sequentially (one GPU process
# at a time) under a hard memory cap so a co-resident run is never starved.
set -u
cd "$(dirname "$0")/.."
export RLVP_MEM_FRAC=${RLVP_MEM_FRAC:-0.11}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ITERS=${ITERS:-24}

run () {  # credit domain
  local credit=$1 domain=$2
  local dir="results/exp_sc_train_${credit}_${domain}"
  rm -rf "$dir"                       # clean (drop any smoke leftovers)
  echo "=== $(date +%H:%M:%S) RUN credit=$credit domain=$domain iters=$ITERS ==="
  python3 scripts/exp_selfcritic_train.py "$credit" "$ITERS" "$domain" \
      2>&1 | tail -n 2
}

# fileops: critic was BLIND in Exp0 -> predict llmcritic ~ outcome (no harm cut)
run outcome   fileops
run c3        fileops
run llmcritic fileops
# csops: critic was ACCURATE in Exp0 -> predict llmcritic ~ c3 (cuts violations)
run c3        csops
run llmcritic csops
echo "=== ALL DONE $(date +%H:%M:%S) ==="
