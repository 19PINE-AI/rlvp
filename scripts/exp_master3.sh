#!/bin/bash
cd ~/rlvp
safe_wait() {  # reap MY orphaned EngineCores only (parent dead); never the user's live judge
  for p in $(ps -eo pid,ppid,comm | awk '$3=="VLLM::EngineCo"||$3=="repl"{print $1":"$2}'); do
    pid=${p%:*}; ppid=${p#*:}; [ "$ppid" = "1" ] && kill -9 $pid 2>/dev/null
  done
  sleep 8
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)" -gt 7000 ]; do sleep 60; done
}
echo "[$(date +%H:%M)] === #1+#2 un-gameability sweep (waits for GPU to fully free) ==="
ITERS=14
for spec in "aligned c3 aligned" "valid c3 valid" "noerror c3 noerror" "structural c3 structural" "validgated c4 valid"; do
  set -- $spec; nm=$1; cr=$2; rm=$3; echo "[$(date +%H:%M)] sweep $nm (credit=$cr rule=$rm)"; safe_wait
  python3 scripts/minif2f_train.py $ITERS --credit $cr --rule-mode $rm --algebra --seed 7 --muon --lr 1e-3 --out swp_$nm > results/swp_$nm.log 2>&1
  echo "  $nm done: $(wc -l < results/run_swp_$nm/train_log.jsonl 2>/dev/null) iters"
done
echo "[$(date +%H:%M)] === #3 SWE hard domain ==="
for spec in "structural c3 swe_structural" "structural c4 swe_gated"; do
  set -- $spec; echo "[$(date +%H:%M)] SWE $3"; safe_wait
  python3 scripts/swe_train_30b.py 16 --rule-mode $1 --credit $2 --muon --lr 1e-3 --out $3 > results/$3.log 2>&1
  echo "  $3 done: $(wc -l < results/run_$3/train_log.jsonl 2>/dev/null) iters"
done
echo "[$(date +%H:%M)] === 30B EXPERIMENTS DONE ===" ; python3 scripts/ungameability_report.py 2>/dev/null
