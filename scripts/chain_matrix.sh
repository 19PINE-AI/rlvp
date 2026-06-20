#!/bin/bash
# E-A (granularity) x E-B (sparsity) matrix on the 4B synthetic chain. 4B (~16GB)
# coexists with other GPU jobs. credit=c3 throughout; the env's granularity controls
# how fine the verifiable potential Phi is; n_stages controls outcome sparsity.
cd ~/rlvp
ITERS=25
for spec in "coarse 2" "fine 2" "coarse 4" "mid 4" "fine 4" "coarse 6" "fine 6"; do
  set -- $spec; g=$1; n=$2
  echo "[$(date +%H:%M)] === chainpot $g n=$n ($ITERS iters) ==="
  python3 scripts/chain_potential_exp.py "$g" "$n" "$ITERS" "chainpot_${g}_n${n}" 7 \
      > "results/chainpot_${g}_n${n}.log" 2>&1
  echo "  done: $(wc -l < results/run_chainpot_${g}_n${n}/train_log.jsonl 2>/dev/null) iters"
done
echo "[$(date +%H:%M)] === CHAIN POTENTIAL MATRIX DONE ==="
python3 scripts/chain_potential_report.py
