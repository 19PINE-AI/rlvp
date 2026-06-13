#!/bin/bash
# DAPO add-on: completes the GRPO-vs-DAPO-vs-RLVP comparison. Waits for the
# main flagship to finish, then runs the DAPO arm + seeds + paired micro-exp.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }

# read chosen N from the flagship marker
N=$(grep -oE "domain=chain[0-9]+" "$S" | tail -1 | grep -oE "[0-9]+")
N=${N:-4}
FLAG="domains=('chain${N}',), tasks_per_iter=8, gen_batch=32, max_episode_tokens=9000, \
eval_tasks=16, eval_k=2, iters=60, eval_every=6"

tr() { local name=$1 kwargs=$2
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }
ev() { python3 scripts/eval_checkpoint.py "$1" "$2" >> results/evals_paper.log 2>&1 \
  && mark "EVAL $2 OK" || mark "EVAL $2 FAILED"; }

while pgrep -f run_flagship >/dev/null; do sleep 120; done
mark "=== DAPO add-on (chain${N}) ==="

# paired dead-iteration micro-experiment first (cheap, base model)
python3 scripts/paired_dead.py Qwen/Qwen3-4B 20 > results/paired_dead.log 2>&1 \
  && mark "PAIRED-DEAD OK" || mark "PAIRED-DEAD FAILED"

# DAPO main + seeds (dynamic sampling, outcome-only, no process channel)
tr flag_dapo     "credit='outcome', dynamic_sampling=True, ${FLAG}"
ev results/run_flag_dapo/final flag_dapo_eval
for seed in 11 12; do
  tr "flag_dapo_s${seed}" "credit='outcome', dynamic_sampling=True, data_seed=${seed}, ${FLAG}"
done
mark "=== DAPO add-on DONE ==="
