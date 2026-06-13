#!/bin/bash
# Paper campaign, stages A-C (PAPER_PLAN.md). Markers -> results/paper_status.log
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }
RECIPE="credit='c3', lam=0.25, beta=0.25, mix_scripted=True, anneal_at=40"

tr() { local name=$1 kwargs=$2
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', gen_batch=48, out_dir='results/run_${name}', ${kwargs}))
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"; }
ev() { python3 scripts/eval_checkpoint.py "$1" "$2" ${3:-} >> results/evals_paper.log 2>&1 \
  && mark "EVAL $2 OK" || mark "EVAL $2 FAILED"; }

mark "=== STAGE A ==="
tr c3v2_imp "${RECIPE}, imperfect_scripts=True, script_scalar=False"
ev results/run_c3v2_imp/final c3v2_imp_norules
tr c3v2 "${RECIPE}, script_scalar=False"
ev results/run_c3v2/final c3v2_norules
ev results/run_c3v2/final c3v2_rules --rules
mark "=== STAGE A DONE ==="

mark "=== STAGE B (seeds) ==="
for seed in 11 12; do
  tr "c3anneal_s${seed}" "${RECIPE}, data_seed=${seed}"
  ev "results/run_c3anneal_s${seed}/final" "c3anneal_s${seed}_norules"
  tr "outcome_s${seed}" "credit='outcome', data_seed=${seed}"
  ev "results/run_outcome_s${seed}/final" "outcome_s${seed}_norules"
  tr "c3mix_s${seed}" "credit='c3', lam=0.25, beta=0.25, mix_scripted=True, data_seed=${seed}"
  ev "results/run_c3mix_s${seed}/final" "c3mix_s${seed}_norules"
done
# SFT seed variants use task ranges 2000+ and 3000+ (disjoint from RL train
# seeds 0-499 and eval seeds 1000-1029)
python3 scripts/sft_bc.py --seed-offset 2000 > results/sftbc_o2000.log 2>&1 \
  && mark "TRAIN sftbc_o2000 OK" || mark "TRAIN sftbc_o2000 FAILED"
ev results/run_sftbc_o2000/final sftbc_o2000_norules
python3 scripts/sft_bc.py --seed-offset 3000 > results/sftbc_o3000.log 2>&1 \
  && mark "TRAIN sftbc_o3000 OK" || mark "TRAIN sftbc_o3000 FAILED"
ev results/run_sftbc_o3000/final sftbc_o3000_norules
mark "=== STAGE B DONE ==="

mark "=== STAGE C ==="
tr holdout_v2 "${RECIPE}, drop_rules=('untested_submit','no_tz_before_call'), strip_dropped_from_scripts=True"
ev results/run_holdout_v2/final holdout_v2_norules
python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='results/run_c3anneal/final', credit='outcome',
                  iters=60, eval_every=10, gen_batch=48, out_dir='results/run_persist'))
" > results/persist.log 2>&1 && mark "TRAIN persist OK" || mark "TRAIN persist FAILED"
ev results/run_persist/final persist_norules
mark "=== STAGE C DONE ==="
