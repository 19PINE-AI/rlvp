#!/bin/bash
# CAPSTONE: does an AUTO-DERIVED process reward (tool tags + env errors, no
# hand-written predicates) recover hand-written clean-RLVP's efficiency? The
# maximal-R1-Zero version: no demonstrations AND no hand-written rules.
# Waits for all chain campaigns + tau2 to free the GPU.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec 6>/tmp/rlvp_auto.lock; flock -n 6 || { echo "autorule already running"; exit 0; }
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
tr() { local name=$1 kwargs=$2
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }

while pgrep -f "run_flagship.sh|run_t2.sh|run_tau2_h2h.sh" >/dev/null; do sleep 180; done
mark "=== CAPSTONE: auto-derived rules (chain4) ==="

C4="domains=('chain4',), tasks_per_iter=8, gen_batch=32, max_episode_tokens=9000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6, step_cost=0.02"
# clean RLVP (penalty+discharge, NO mixing): hand rules vs AUTO-derived rules
tr auto_rlvp "credit='c3', lam=0.25, beta=0.25, anneal_at=40, auto_rules=True, ${C4}"
tr hand_rlvp "credit='c3', lam=0.25, beta=0.25, anneal_at=40, ${C4}"   # reference (= clean RLVP)
mark "=== CAPSTONE DONE ==="
