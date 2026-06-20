#!/bin/bash
# Higher-LR, longer success-payoff contrast: coarse(=outcome) vs fine at blind n,
# where progress VARIES within groups (so the potential isn't centered out). Runs
# after the matrix. Predict: fine LEARNS (rising success) where coarse stays dead.
cd ~/rlvp
echo "[$(date +%H:%M)] waiting for matrix to finish ..."
while pgrep -f chain_matrix >/dev/null; do sleep 60; done
sleep 10
for spec in "coarse 4" "fine 4" "coarse 6" "fine 6"; do
  set -- $spec; g=$1; n=$2
  echo "[$(date +%H:%M)] === HI-LR chainpot $g n=$n (40 iters, lr 3e-5) ==="
  python3 scripts/chain_potential_exp.py "$g" "$n" 40 "chainpot_${g}_n${n}_hi" 7 3e-5 \
      > "results/chainpot_${g}_n${n}_hi.log" 2>&1
  echo "  done: $(wc -l < results/run_chainpot_${g}_n${n}_hi/train_log.jsonl 2>/dev/null) iters"
done
echo "[$(date +%H:%M)] === CHAIN HI-LR FOLLOWUP DONE ==="
PY
