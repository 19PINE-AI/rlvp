#!/bin/bash
# Controlled 30B miniF2F comparison: aligned-RLVP vs outcome, BOTH Muon lr=1e-3
# (matched optimizer). Aligned = goal-progress discharge only (the structural
# errored-tactic penalty drove a compliance-attractor collapse). Politely waits
# for a free GPU and reaps orphaned vLLM EngineCore between arms.
cd ~/rlvp

wait_gpu() {
  ps -eo pid,user,args | grep '[E]ngineCore' | awk '$2=="ubuntu"{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 8
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 7000 ]; do sleep 60; done
}

echo "=== aligned-RLVP arm (Muon 1e-3, 40 iters) ==="; wait_gpu
python3 scripts/minif2f_train.py 40 --credit c3 --rule-mode aligned --algebra --seed 7 \
    --muon --lr 1e-3 --out minif2f_rlvp_aligned > results/minif2f_rlvp_aligned.log 2>&1
echo "=== aligned-RLVP done: $(wc -l < results/run_minif2f_rlvp_aligned/train_log.jsonl 2>/dev/null) iters ==="

echo "=== outcome arm (Muon 1e-3, 40 iters) ==="; wait_gpu
python3 scripts/minif2f_train.py 40 --credit outcome --algebra --seed 7 \
    --muon --lr 1e-3 --out minif2f_outcome_muon > results/minif2f_outcome_muon.log 2>&1
echo "=== outcome done: $(wc -l < results/run_minif2f_outcome_muon/train_log.jsonl 2>/dev/null) iters ==="

echo "=== COMPARISON (matched Muon optimizer) ==="
python3 scripts/bench_cmp.py minif2f_aln minif2f_rlvp_aligned minif2f_outcome_muon
