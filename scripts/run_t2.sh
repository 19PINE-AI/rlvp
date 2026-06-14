#!/bin/bash
# T2 (ceiling) + fairness controls + RLVP calls/ep fix validation.
# Runs after the main flagship (separate lock; polls the process).
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec 8>/tmp/rlvp_t2.lock; flock -n 8 || { echo "t2 already running"; exit 0; }
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
tr() { local name=$1 kwargs=$2
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }
ev() { python3 scripts/eval_checkpoint.py "$1" "$2" >> results/evals_paper.log 2>&1 \
  && mark "EVAL $2 OK" || mark "EVAL $2 FAILED"; }

while pgrep -f "run_flagship.sh" >/dev/null; do sleep 120; done

C4="domains=('chain4',), tasks_per_iter=8, gen_batch=32, max_episode_tokens=9000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"
C6="domains=('chain6',), tasks_per_iter=8, gen_batch=24, max_episode_tokens=13000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"
RLVP="credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, anneal_at=40"

# --- Phase 1: fix the calls/ep bloat on chain4 (discharge-credit farming) ---
mark "=== T2-P1: RLVP calls/ep fix validation (chain4) ==="
tr rlvp_sc3  "${RLVP}, step_cost=0.03, ${C4}"   # length penalty
tr rlvp_b10  "credit='c3', lam=0.25, beta=0.10, mix_scripted=True, script_scalar=False, anneal_at=40, ${C4}"  # lower discharge weight
tr ctrl_outmix "credit='outcome', mix_scripted=True, script_scalar=False, ${C4}"  # fairness: demos w/o process channel

# --- Phase 2: ceiling test on the hard chain6 regime ---
# E4 (chain4) showed mixing is redundant and the discharge credit is the driver.
# chain6 is the decisive test of whether mixing is EVER needed: at ~2% base
# success, live rollouts rarely contain a success, so if clean RLVP (no mixing)
# still climbs, mixing can be dropped entirely. If it stalls and only the mixed
# variant climbs, mixing earns its keep in the extreme-sparse regime.
RLVP_CLEAN="credit='c3', lam=0.25, beta=0.25, anneal_at=40"  # penalty+discharge, NO mixing
mark "=== T2-P2: ceiling test (chain6) ==="
tr t2_outcome    "credit='outcome', ${C6}"
tr t2_dapo       "credit='outcome', dynamic_sampling=True, ${C6}"
tr t2_rlvp_clean "${RLVP_CLEAN}, step_cost=0.02, ${C6}"
tr t2_rlvp_mix   "${RLVP}, step_cost=0.02, ${C6}"
tr t2_outmix     "credit='outcome', mix_scripted=True, script_scalar=False, ${C6}"
mark "=== T2 DONE ==="
