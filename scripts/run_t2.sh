#!/bin/bash
# T2 (ceiling) + fairness controls. Runs after the main flagship (shares lock).
# chain4 showed a 6.7x efficiency win but all methods reach ~1.0 (no ceiling
# gap). chain6 (base succ .02, ~85% all-fail) is the regime where outcome-only
# and DAPO should stall within budget while RLVP climbs -> the ceiling result.
# Fairness control: outcome + demo mixing (isolates process channel vs demos).
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

# wait for the main flagship to finish (lock is advisory; poll the process)
while pgrep -f "run_flagship.sh" >/dev/null; do sleep 120; done

C6="domains=('chain6',), tasks_per_iter=8, gen_batch=24, max_episode_tokens=12000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"

# --- fairness control on chain4: outcome + demo mixing (process channel OFF) ---
C4="domains=('chain4',), tasks_per_iter=8, gen_batch=32, max_episode_tokens=9000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"
mark "=== FAIRNESS: outcome+mixing (chain4) ==="
tr ctrl_outmix "credit='outcome', mix_scripted=True, script_scalar=False, ${C4}"

# --- T2 ceiling: trio on the hard chain6 regime ---
mark "=== T2 ceiling trio (chain6) ==="
tr t2_outcome "credit='outcome', ${C6}"
tr t2_dapo    "credit='outcome', dynamic_sampling=True, ${C6}"
tr t2_rlvp    "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, script_scalar=False, anneal_at=40, ${C6}"
tr t2_outmix  "credit='outcome', mix_scripted=True, script_scalar=False, ${C6}"
mark "=== T2 DONE ==="
