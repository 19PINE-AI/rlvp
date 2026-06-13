#!/bin/bash
# Paper campaign, stages D-F. Waits for the A-C driver to finish.
set -uo pipefail
cd /home/ubuntu/rlvp
export PYTORCH_ALLOC_CONF=expandable_segments:True
S=results/paper_status.log
mark() { echo "$(date '+%m-%d %H:%M') $1" >> "$S"; }

while pgrep -f run_paper_abc >/dev/null; do sleep 120; done

# recipe checkpoint: prefer the Stage-A process-only variant if it trained OK
RC=results/run_c3anneal/final
[ -d results/run_c3v2/final ] && grep -q "TRAIN c3v2 OK" "$S" && RC=results/run_c3v2/final
mark "=== STAGE D (horizon; recipe=$RC) ==="
python3 scripts/eval_horizon.py Qwen/Qwen3-4B horizon_base >> results/evals_paper.log 2>&1 \
  && mark "HORIZON base OK" || mark "HORIZON base FAILED"
python3 scripts/eval_horizon.py Qwen/Qwen3-4B horizon_base_rules --rules >> results/evals_paper.log 2>&1 \
  && mark "HORIZON base_rules OK" || mark "HORIZON base_rules FAILED"
python3 scripts/eval_horizon.py "$RC" horizon_recipe >> results/evals_paper.log 2>&1 \
  && mark "HORIZON recipe OK" || mark "HORIZON recipe FAILED"
mark "=== STAGE D DONE ==="

mark "=== STAGE E (baselines) ==="
python3 scripts/eval_bestofn.py Qwen/Qwen3-4B bestofn_base_rules --rules --n 4 >> results/evals_paper.log 2>&1 \
  && mark "EVAL bestofn OK" || mark "EVAL bestofn FAILED"
for variant in gigpo steptool; do
  python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='Qwen/Qwen3-4B', credit='${variant}', lam=0.25, beta=0.25,
                  gen_batch=48, out_dir='results/run_${variant}'))
" > "results/${variant}.log" 2>&1 && mark "TRAIN ${variant} OK" || mark "TRAIN ${variant} FAILED"
  python3 scripts/eval_checkpoint.py "results/run_${variant}/final" "${variant}_norules" \
    >> results/evals_paper.log 2>&1 && mark "EVAL ${variant} OK" || mark "EVAL ${variant} FAILED"
done
mark "=== STAGE E DONE ==="

mark "=== STAGE F (Mistral) ==="
python3 scripts/eval_checkpoint.py mistralai/Mistral-7B-Instruct-v0.3 mistral_base_norules \
  >> results/evals_paper.log 2>&1 && mark "EVAL mistral_base OK" || mark "EVAL mistral_base FAILED"
python3 -c "
import sys; sys.path.insert(0, '.')
from rlvp.grpo import TrainConfig, train
train(TrainConfig(model_name='mistralai/Mistral-7B-Instruct-v0.3', credit='c3',
                  lam=0.25, beta=0.25, mix_scripted=True, anneal_at=40, lora_r=32,
                  gen_batch=32, micro_token_budget=3072, out_dir='results/run_mistral'))
" > results/mistral.log 2>&1 && mark "TRAIN mistral OK" || mark "TRAIN mistral FAILED"
python3 scripts/eval_checkpoint.py results/run_mistral/final mistral_norules \
  >> results/evals_paper.log 2>&1 && mark "EVAL mistral OK" || mark "EVAL mistral FAILED"
mark "=== STAGE F DONE ==="
