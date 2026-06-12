#!/bin/bash
# Tier 1-3 master driver. Sequential trainings, light evals co-scheduled.
# Status markers -> results/tier123_status.log
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p results
S=results/tier123_status.log
mark() { echo "$(date '+%H:%M:%S') $1" >> "$S"; }

tr() {  # tr <name> <python-kwargs>
  local name=$1 kwargs=$2
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
cfg = TrainConfig(gen_batch=48, out_dir='results/run_${name}', ${kwargs})
train(cfg)
" > "results/${name}.log" 2>&1 && mark "TRAIN ${name} OK" || mark "TRAIN ${name} FAILED"
}
ev() {  # ev <ckpt> <tag> [extra args]
  python3 scripts/eval_checkpoint.py "$1" "$2" ${3:-} >> "results/evals_tier123.log" 2>&1 \
    && mark "EVAL $2 OK" || mark "EVAL $2 FAILED"
}

mark "=== TIER123 START ==="

# ---------- Tier 1 ----------
# Arm 4 (guardrail, eval-only, ~16GB) co-runs with the first training
( sleep 90
  python3 scripts/eval_guardrail.py results/run_outcome/final guardrail_outcome \
    >> results/evals_tier123.log 2>&1 && mark "EVAL guardrail_outcome OK" || mark "EVAL guardrail_outcome FAILED"
  python3 scripts/eval_guardrail.py Qwen/Qwen3-4B guardrail_base \
    >> results/evals_tier123.log 2>&1 && mark "EVAL guardrail_base OK" || mark "EVAL guardrail_base FAILED"
) &

tr c3mix "model_name='Qwen/Qwen3-4B', credit='c3', lam=0.25, beta=0.25, mix_scripted=True"
ev results/run_c3mix/final c3mix_norules
ev results/run_c3mix/final c3mix_rules --rules
wait  # guardrail evals

python3 scripts/sft_bc.py > results/sftbc.log 2>&1 && mark "TRAIN sftbc OK" || mark "TRAIN sftbc FAILED"
ev results/run_sftbc/final sftbc_norules

tr c3anneal "model_name='Qwen/Qwen3-4B', credit='c3', lam=0.25, beta=0.25, mix_scripted=True, anneal_at=40"
ev results/run_c3anneal/final c3anneal_norules
mark "=== TIER 1 DONE ==="

# ---------- Tier 2 ----------
for seed in 11 12; do
  tr "c1_s${seed}"  "model_name='Qwen/Qwen3-4B', credit='c1', lam=0.5,  beta=0.5,  data_seed=${seed}"
  ev "results/run_c1_s${seed}/final"  "c1_s${seed}_norules"
  tr "c3_s${seed}"  "model_name='Qwen/Qwen3-4B', credit='c3', lam=0.25, beta=0.25, data_seed=${seed}"
  ev "results/run_c3_s${seed}/final"  "c3_s${seed}_norules"
done

tr c3mix_holdout "model_name='Qwen/Qwen3-4B', credit='c3', lam=0.25, beta=0.25, mix_scripted=True, drop_rules=('untested_submit','no_tz_before_call')"
ev results/run_c3mix_holdout/final holdout_norules

# k/temp sweeps (eval-only)
ev results/run_c3mix/final c3mix_k16 "--k 16"
ev results/run_c3mix/final c3mix_t10 "--temp 1.0"
ev Qwen/Qwen3-4B base_k16 "--k 16"
mark "=== TIER 2 DONE ==="

# ---------- Tier 3 ----------
tr c3mix_1p7b "model_name='Qwen/Qwen3-1.7B', credit='c3', lam=0.25, beta=0.25, mix_scripted=True"
ev results/run_c3mix_1p7b/final c3mix_1p7b_norules

tr c3mix_8b_lora "model_name='Qwen/Qwen3-8B', credit='c3', lam=0.25, beta=0.25, mix_scripted=True, lora_r=32, micro_token_budget=3072"
ev results/run_c3mix_8b_lora/final c3mix_8b_norules

bash scripts/tau2_phase0.sh >> results/tau2.log 2>&1 && mark "TAU2 OK" || mark "TAU2 FAILED (see results/tau2.log)"
mark "=== ALL TIERS DONE ==="
