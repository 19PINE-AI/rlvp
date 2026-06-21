#!/bin/bash
# Run experiments #1 (un-gameability law), #2 (c4 gating), #3 (SWE hard domain)
# in priority order AFTER the current aligned-vs-outcome comparison finishes.
cd ~/rlvp
echo "[$(date +%H:%M)] waiting for current comparison ..."
while [ ! -f results/minif2f_aln_RESULT.json ] && pgrep -f minif2f_full_chain >/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] === #1+#2 un-gameability sweep ==="
bash scripts/ungameability_sweep.sh
echo "[$(date +%H:%M)] === #3 SWE hard domain (structural / c4 / outcome) ==="
wait_gpu() { ps -eo pid,user,args|grep '[E]ngineCore'|awk '$2=="ubuntu"{print $1}'|xargs -r kill -9 2>/dev/null; sleep 8; while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)" -gt 7000 ]; do sleep 60; done; }
for spec in "structural c3 swe_structural" "structural c4 swe_gated" "outcome outcome swe_outcome"; do
  set -- $spec; echo "[$(date +%H:%M)] SWE $3 (rule=$1 credit=$2)"; wait_gpu
  python3 scripts/swe_train_30b.py 16 --rule-mode $1 --credit $2 --muon --lr 1e-3 --out $3 > results/$3.log 2>&1
  echo "  $3 done: $(wc -l < results/run_$3/train_log.jsonl 2>/dev/null) iters"
done
echo "[$(date +%H:%M)] === ALL EXPERIMENTS (#1 #2 #3) COMPLETE ==="
