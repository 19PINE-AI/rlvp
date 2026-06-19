#!/usr/bin/env python3
"""Aggregate RLVP-vs-outcome across seeds for Lean and Terminal.
Reports mean +/- std of the headline metrics over the 3 seeds per arm."""
import json, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"

# seed 7 runs use the original names; seeds 11/12 use the _sNN suffix
BENCH = {
    "LEAN": {
        "RLVP":    ["lean_rlvp6", "lean_rlvp_s11", "lean_rlvp_s12"],
        "OUTCOME": ["lean_outcome6", "lean_outcome_s11", "lean_outcome_s12"],
    },
    "TERM": {
        "RLVP":    ["term_rlvp3", "term_rlvp_s11", "term_rlvp_s12"],
        "OUTCOME": ["term_outcome3", "term_outcome_s11", "term_outcome_s12"],
    },
}


def rows(r):
    f = R / f"run_{r}" / "train_log.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def metrics(r):
    rs = rows(r)
    if not rs:
        return None
    succ = [x.get("succ", 0) for x in rs]
    viol = [x.get("viol_per_ep", 0) for x in rs]
    dead = sum(1 for x in rs if x.get("loss", 0) == 0.0)
    e50 = next((rs[i]["iter"] for i in range(len(rs))
                if sum(succ[max(0, i - 2):i + 1]) / min(3, i + 1) >= 0.5), None)
    return {"dead": dead, "succ_last5": st.mean(succ[-5:]),
            "viol_last5": st.mean(viol[-5:]),
            "iters_to_0.5": e50 if e50 is not None else len(rs)}


def agg(runs):
    ms = [metrics(r) for r in runs]
    ms = [m for m in ms if m]
    out = {"n_seeds": len(ms)}
    for k in ("dead", "succ_last5", "viol_last5", "iters_to_0.5"):
        vals = [m[k] for m in ms]
        out[k] = {"mean": round(st.mean(vals), 3),
                  "std": round(st.pstdev(vals), 3) if len(vals) > 1 else 0.0,
                  "all": [round(v, 3) for v in vals]}
    return out


D = {}
for bench, arms in BENCH.items():
    D[bench] = {arm: agg(runs) for arm, runs in arms.items()}
    print(f"\n=== {bench} (n_seeds per arm shown) ===")
    for arm, a in D[bench].items():
        print(f"  {arm:8s} [{a['n_seeds']} seeds]: "
              f"dead={a['dead']['mean']}±{a['dead']['std']} {a['dead']['all']} | "
              f"succ_last5={a['succ_last5']['mean']}±{a['succ_last5']['std']} | "
              f"viol_last5={a['viol_last5']['mean']}±{a['viol_last5']['std']} | "
              f"iters->0.5={a['iters_to_0.5']['mean']}±{a['iters_to_0.5']['std']}")
(R / "seed_aggregate.json").write_text(json.dumps(D, indent=2))
print("\nwrote", R / "seed_aggregate.json")
