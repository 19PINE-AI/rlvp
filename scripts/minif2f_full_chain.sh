#!/bin/bash
# Full miniF2F RLVP-vs-outcome at 30B, both arms with Muon lr=1e-3 (matched
# optimizer = controlled comparison). Politely waits for a free GPU and reaps
# orphaned vLLM EngineCore between arms. Never disturbs other users' procs.
cd ~/rlvp

wait_gpu() {
  ps -eo pid,user,args | grep '[E]ngineCore' | awk '$2=="ubuntu"{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 8
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 7000 ]; do sleep 60; done
}

echo "=== RLVP arm (Muon 1e-3, 40 iters) ==="; wait_gpu
python3 scripts/minif2f_train.py 40 --credit c3 --anneal 24 --algebra --seed 7 \
    --muon --lr 1e-3 --out minif2f_rlvp_muon > results/minif2f_rlvp_muon.log 2>&1
echo "=== RLVP done: $(wc -l < results/run_minif2f_rlvp_muon/train_log.jsonl 2>/dev/null) iters ==="

echo "=== OUTCOME arm (Muon 1e-3, 40 iters) ==="; wait_gpu
python3 scripts/minif2f_train.py 40 --credit outcome --algebra --seed 7 \
    --muon --lr 1e-3 --out minif2f_outcome_muon > results/minif2f_outcome_muon.log 2>&1
echo "=== OUTCOME done: $(wc -l < results/run_minif2f_outcome_muon/train_log.jsonl 2>/dev/null) iters ==="

echo "=== COMPARISON (Muon, matched optimizer) ==="
python3 scripts/bench_cmp.py minif2f_muon minif2f_rlvp_muon minif2f_outcome_muon
