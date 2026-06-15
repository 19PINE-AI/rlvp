#!/bin/bash
# CEILING TEST (T2) on the non-saturating gated task. The chain tasks saturate
# at 1.0 (compositionally easy) so they only show efficiency. The gated task
# has a silent, hard-to-discover precondition gate -> outcome-only should stall
# below 1.0 while RLVP's verifiable gate rule lifts the ceiling. Runs after all
# other GPU work.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec 5>/tmp/rlvp_gated.lock; flock -n 5 || { echo "gated already running"; exit 0; }
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
tr() { local name=$1 kwargs=$2
  [ -d "results/run_${name}/final" ] && { mark "SKIP ${name}"; return; }
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }

while pgrep -f "run_flagship.sh|run_t2.sh|run_tau2_h2h.sh|run_autorule.sh" >/dev/null; do sleep 180; done
mark "=== GATED ceiling test ==="

# calibration: confirm the task is non-saturating (base success low, gate rarely found)
python3 scripts/calib_gated.py Qwen/Qwen3-4B > results/calib_gated.log 2>&1 && mark "GATED CALIB OK" || mark "GATED CALIB FAILED"

G="domains=('gated',), tasks_per_iter=8, group_size=8, gen_batch=48, \
max_episode_tokens=3500, eval_tasks=24, eval_k=2, iters=80, eval_every=8"
# the ceiling comparison (no rules in prompt unless noted)
tr gated_outcome  "credit='outcome', ${G}"
tr gated_dapo     "credit='outcome', dynamic_sampling=True, ${G}"
tr gated_rlvp     "credit='c3', lam=0.25, beta=0.25, anneal_at=60, ${G}"          # clean RLVP
tr gated_outcome_prompt "credit='outcome', include_rules_in_prompt=True, ${G}"     # prompting baseline
mark "=== GATED DONE ==="
