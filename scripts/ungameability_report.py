#!/usr/bin/env python3
"""Tabulate the un-gameability sweep: does each Lean process-signal variant COLLAPSE
or HELP, vs its pre-registered gaming prediction?"""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"

PRED = {  # name: (cheapest gaming policy, prediction)
    "aligned":    ("none (must decrease goals)",      "HELP"),
    "valid":      ("pad trivial valid no-ops",         "FARM/collapse"),
    "noerror":    ("stop attempting",                  "COLLAPSE"),
    "structural": ("avoid errors by inaction",         "COLLAPSE"),
    "validgated": ("gated on success -> can't farm",   "RESCUED (stable)"),
}


def traj(name):
    f = R / f"run_swp_{name}" / "train_log.jsonl"
    if not f.exists():
        return None
    rs = [json.loads(l) for l in open(f)]
    return [r.get("succ", 0) for r in rs]


print(f"{'variant':12} {'pred':16} {'peak':>5} {'final':>6} {'collapsed?':>10}  cheapest-gaming-policy")
for name, (game, pred) in PRED.items():
    s = traj(name)
    if not s:
        print(f"{name:12} {pred:16} {'--':>5} {'--':>6} {'(no data)':>10}")
        continue
    peak, final = max(s), s[-1]
    # collapse = reached >=0.3 then ended <= 0.1 (rose then died)
    collapsed = "YES" if (peak >= 0.3 and final <= 0.1) else "no"
    print(f"{name:12} {pred:16} {peak:5.2f} {final:6.2f} {collapsed:>10}  {game}")
print("\nHypothesis: predicted-misaligned variants collapse; aligned + outcome-gated hold.")
