#!/bin/bash
# EXPERIMENT #1 (un-gameability as a measured law) + #2 (outcome-gating rescue).
# Each Lean process-signal variant has a PRE-REGISTERED cheapest gaming policy;
# train short (matched Muon 1e-3, algebra) and see which collapse vs help.
#   aligned   (c3) : goal-decrease discharge        -> UNGAMEABLE   -> predict HELP
#   valid     (c3) : any-non-error discharge        -> pad no-ops   -> predict FARM/collapse
#   noerror   (c3) : errored penalty only           -> stop trying  -> predict COLLAPSE
#   structural(c3) : penalties + goal-decrease       -> avoid errors -> predict COLLAPSE
#   valid     (c4) : gameable signal, OUTCOME-GATED  -> can't farm   -> predict gating RESCUES
# (aligned@40 already running as the headline; here aligned@14 for matched comparison.)
cd ~/rlvp
ITERS=14
wait_gpu() {
  ps -eo pid,user,args | grep '[E]ngineCore' | awk '$2=="ubuntu"{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 8
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 7000 ]; do sleep 60; done
}
run() {  # name credit rule_mode
  echo "=== sweep $1 (credit=$2 rule=$3, $ITERS iters) ==="; wait_gpu
  python3 scripts/minif2f_train.py $ITERS --credit $2 --rule-mode $3 --algebra --seed 7 \
      --muon --lr 1e-3 --out swp_$1 > results/swp_$1.log 2>&1
  echo "--- $1 done: $(wc -l < results/run_swp_$1/train_log.jsonl 2>/dev/null) iters ---"
}
run aligned    c3 aligned
run valid      c3 valid
run noerror    c3 noerror
run structural c3 structural
run validgated c4 valid
echo "=== UN-GAMEABILITY SWEEP DONE ==="
python3 scripts/ungameability_report.py
