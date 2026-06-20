#!/usr/bin/env python3
"""E-A/E-B report: does RLVP benefit (dead-iter elim + success) scale with potential
granularity, and appear as outcome sparsity (n_stages) grows?"""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
def stats(g, n):
    f = R / f"run_chainpot_{g}_n{n}" / "train_log.jsonl"
    if not f.exists(): return None
    rs = [json.loads(l) for l in open(f)]
    succ = [x.get("succ", 0) for x in rs]
    dead = sum(1 for x in rs if x.get("loss", 0) == 0.0)
    import statistics as st
    return dict(iters=len(rs), succ_last5=round(st.mean(succ[-5:]), 3),
                succ_peak=round(max(succ), 3), dead=dead,
                disch=round(st.mean([x.get("disch_per_ep", 0) for x in rs[-5:]]), 2))
print("granularity x n_stages  (Phi fineness: coarse=outcome < mid < fine)")
print(f"{'cfg':14} {'succ_last5':>10} {'peak':>5} {'dead_iters':>10} {'disch/ep':>8}")
for n in (2, 4, 6):
    for g in ("coarse", "mid", "fine"):
        s = stats(g, n)
        if s:
            print(f"{g+'_n'+str(n):14} {s['succ_last5']:>10} {s['succ_peak']:>5} "
                  f"{s['dead']:>10} {s['disch']:>8}")
print("\nPredict: at fixed n, dead-iters DECREASE and succ INCREASE coarse->mid->fine.")
print("Predict: the fine-vs-coarse gap GROWS with n (outcome blinder -> potential matters more).")
