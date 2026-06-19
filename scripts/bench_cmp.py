#!/usr/bin/env python3
"""Generic RLVP-vs-outcome comparison for a real-benchmark pair.
Usage: bench_cmp.py <tag> <rlvp_run> <outcome_run>"""
import json, sys, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"


def rows(r):
    f = R / f"run_{r}" / "train_log.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def summ(r):
    rs = rows(r)
    if not rs:
        return None
    g = lambda k: [x.get(k, 0) for x in rs]
    rew, succ = g("reward"), g("succ")
    disc, viol = g("disc_per_ep"), g("viol_per_ep")
    # a "dead" iter: no gradient (all-fail group blindness for outcome-only)
    dead = sum(1 for x in rs if x.get("loss", 0) == 0.0)
    return dict(iters=len(rs),
                succ_last5=round(st.mean(succ[-5:]), 3),
                succ_peak=round(max(succ), 3),
                reward_last5=round(st.mean(rew[-5:]), 3),
                disc_ep_last5=round(st.mean(disc[-5:]), 2),
                viol_ep_last5=round(st.mean(viol[-5:]), 2),
                dead_iters=dead)


def main():
    tag, rl, oc = sys.argv[1], sys.argv[2], sys.argv[3]
    out = {"tag": tag, "rlvp": summ(rl), "outcome": summ(oc)}
    print(f"=== {tag.upper()}: RLVP vs OUTCOME ===")
    for name, s in [("RLVP   ", out["rlvp"]), ("OUTCOME", out["outcome"])]:
        print(f"{name}", s)
    (R / f"{tag}_RESULT.json").write_text(json.dumps(out, indent=2))
    print("wrote", R / f"{tag}_RESULT.json")


if __name__ == "__main__":
    main()
