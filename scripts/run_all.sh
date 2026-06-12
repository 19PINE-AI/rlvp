#!/bin/bash
# Sequential Phase 1: train all credit variants, then evaluate everything.
set -uo pipefail
cd /home/ubuntu/rlvp
mkdir -p results
export PYTORCH_ALLOC_CONF=expandable_segments:True

# never overlap with the Phase 0 grid
while pgrep -f phase0_baseline >/dev/null; do sleep 20; done

for credit in c2 c1 outcome c2pos; do
  echo "=== TRAIN $credit $(date) ==="
  python3 scripts/train.py "$credit" 60 || echo "TRAIN $credit FAILED"
done

echo "=== EVALS $(date) ==="
python3 scripts/eval_checkpoint.py Qwen/Qwen3-4B base_norules || true
python3 scripts/eval_checkpoint.py Qwen/Qwen3-4B base_rules --rules || true
for credit in outcome c1 c2 c2pos; do
  if [ -d "results/run_${credit}/final" ]; then
    python3 scripts/eval_checkpoint.py "results/run_${credit}/final" "${credit}_norules" || true
    python3 scripts/eval_checkpoint.py "results/run_${credit}/final" "${credit}_rules" --rules || true
  fi
done
echo "ALL DONE $(date)"
