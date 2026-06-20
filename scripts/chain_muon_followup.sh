#!/bin/bash
# Recovery: success payoff with MUON (bounded updates) -- AdamW 3e-5 collapsed the
# dense potential (compliance attractor); Muon stabilized the dense signal at 30B.
# Predict: fine-Muon LEARNS stably where fine-AdamW collapsed.
cd ~/rlvp
echo "[$(date +%H:%M)] waiting for AdamW follow-up to finish ..."
while pgrep -f chain_followup >/dev/null || pgrep -f "chain_potential_exp.py fine 6 40" >/dev/null; do sleep 60; done
sleep 10
for spec in "coarse 4" "fine 4" "coarse 6" "fine 6"; do
  set -- $spec; g=$1; n=$2
  echo "[$(date +%H:%M)] === MUON chainpot $g n=$n (40 iters, muon 2e-3) ==="
  python3 scripts/chain_potential_exp.py "$g" "$n" 40 "chainpot_${g}_n${n}_muon" 7 2e-3 muon \
      > "results/chainpot_${g}_n${n}_muon.log" 2>&1
  echo "  done: $(wc -l < results/run_chainpot_${g}_n${n}_muon/train_log.jsonl 2>/dev/null) iters"
done
echo "[$(date +%H:%M)] === CHAIN MUON FOLLOWUP DONE ==="
