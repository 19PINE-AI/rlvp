#!/bin/bash
# Wait for the RLVP arm to finish, fully free the GPU (kill any orphaned vLLM
# EngineCore that lingers on shutdown), then run the outcome arm + comparison.
cd ~/rlvp

cleanup_gpu() {
  # kill MY orphaned EngineCore procs (no minif2f train running at this point)
  ps -eo pid,user,args | grep '[E]ngineCore' | awk '$2=="ubuntu"{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 8
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 6000 ]; do sleep 10; done
}

echo "waiting for RLVP arm to finish ..."
while pgrep -f "minif2f_train.py 40 --credit c3" >/dev/null; do sleep 30; done
echo "=== RLVP done: $(wc -l < results/run_minif2f_rlvp/train_log.jsonl 2>/dev/null) iters ==="
cleanup_gpu
echo "=== GPU freed ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)); OUTCOME 40 iters ==="
python3 scripts/minif2f_train.py 40 --credit outcome --algebra --seed 7 --out minif2f_outcome > results/minif2f_outcome.log 2>&1
echo "=== OUTCOME done: $(wc -l < results/run_minif2f_outcome/train_log.jsonl 2>/dev/null) iters ==="
cleanup_gpu
python3 scripts/bench_cmp.py minif2f minif2f_rlvp minif2f_outcome
