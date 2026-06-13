"""Extract T1/T2 metrics from flagship train logs.

For each run: success-vs-episodes curve (train batch success per iter),
eval-success curve, episodes-to-25%/50% success, final/best eval success,
all-fail-group fraction curve (the GRPO-blindness mechanism metric).

Usage: python3 scripts/curves.py flag_outcome flag_rlvp flag_gigpo flag_steptool
Writes results/curves.json and prints a compact table.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EP_PER_ITER = None  # derived from config


def load(run):
    cfgp = ROOT / f"results/run_{run}/config.json"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    n_live = cfg.get("tasks_per_iter", 16) * (cfg.get("group_size", 8)
                                              - (1 if cfg.get("mix_scripted") else 0))
    rows = []
    for line in open(ROOT / f"results/run_{run}/train_log.jsonl"):
        d = json.loads(line)
        rows.append(d)
    return cfg, n_live, rows


def episodes_to(rows, n_live, thresh, window=3):
    """First episode count where the rolling-mean train success >= thresh."""
    succ = [r["train"]["success"] for r in rows]
    for i in range(len(succ)):
        lo = max(0, i - window + 1)
        if sum(succ[lo:i + 1]) / (i + 1 - lo) >= thresh:
            return (i + 1) * n_live
    return None


def main(runs):
    out = {}
    for run in runs:
        try:
            cfg, n_live, rows = load(run)
        except FileNotFoundError:
            print(f"{run:22s} MISSING")
            continue
        evals = [(r["iter"], r["eval"]) for r in rows if "eval" in r]
        dom = cfg.get("domains", ["?"])[0]
        eval_succ = [(it, ev[dom]["success"]) for it, ev in evals if dom in ev]
        rec = {
            "n_iters": len(rows),
            "episodes_total": len(rows) * n_live,
            "eps_to_25": episodes_to(rows, n_live, 0.25),
            "eps_to_50": episodes_to(rows, n_live, 0.50),
            "final_train_succ_5": sum(r["train"]["success"] for r in rows[-5:]) / max(len(rows[-5:]), 1),
            "best_eval_succ": max((s for _, s in eval_succ), default=None),
            "final_eval_succ": eval_succ[-1][1] if eval_succ else None,
            "eval_curve": eval_succ,
            "train_succ_curve": [round(r["train"]["success"], 3) for r in rows],
            "all_fail_curve": [round(r["train"].get("all_fail_groups", -1), 3) for r in rows],
            "entropy_final": rows[-1].get("entropy") if rows else None,
        }
        out[run] = rec
        print(f"{run:22s} eps25={str(rec['eps_to_25']):>6} eps50={str(rec['eps_to_50']):>6} "
              f"final_eval={rec['final_eval_succ']} best={rec['best_eval_succ']} "
              f"allfail[0:3]={rec['all_fail_curve'][:3]} [-3:]={rec['all_fail_curve'][-3:]}")
    (ROOT / "results/curves.json").write_text(json.dumps(out, indent=2))
    print("wrote results/curves.json")


if __name__ == "__main__":
    main(sys.argv[1:] or ["flag_outcome", "flag_rlvp", "flag_gigpo", "flag_steptool"])
