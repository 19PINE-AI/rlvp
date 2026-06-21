"""Aggregate Exp 1 train logs into the harm-reduction comparison.

For each (credit, domain) run, report the early vs late TRUE (oracle) violation
rate and success, plus -- for llmcritic -- the critic-vs-oracle agreement. The
thesis test: self-critique reward should cut TRUE violations only in the domain
where Exp 0 found the critic accurate (csops), not where it was blind (fileops).
"""
import json
import sys
from pathlib import Path

RUNS = [("outcome", "fileops"), ("c3", "fileops"), ("llmcritic", "fileops"),
        ("c3", "csops"), ("llmcritic", "csops")]


def load(credit, domain):
    p = Path(f"results/exp_sc_train_{credit}_{domain}/train_log.jsonl")
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows or None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, k=4):
    early = rows[:k]
    late = rows[-k:]

    def grab(rs, key):
        return avg([r.get("train", {}).get(key) for r in rs])
    return {
        "iters": len(rows),
        "succ_early": round(grab(early, "success"), 3),
        "succ_late": round(grab(late, "success"), 3),
        "trueviol100_early": round(grab(early, "viol_per_100_calls"), 1),
        "trueviol100_late": round(grab(late, "viol_per_100_calls"), 1),
        "trueviol_ep_early": round(grab(early, "viol_per_episode"), 2),
        "trueviol_ep_late": round(grab(late, "viol_per_episode"), 2),
        "all_fail_late": round(grab(late, "all_fail_groups"), 2),
        "critic_P_late": (round(grab(late, "critic_precision"), 2)
                          if any("critic_precision" in r.get("train", {}) for r in rows) else None),
        "critic_R_late": (round(grab(late, "critic_recall"), 2)
                          if any("critic_recall" in r.get("train", {}) for r in rows) else None),
    }


def main():
    out = {}
    print(f"{'run':22s} {'succ e->l':14s} {'trueViol/100 e->l':20s} {'viol/ep e->l':14s} {'criticP/R':10s}")
    print("-" * 86)
    for credit, domain in RUNS:
        rows = load(credit, domain)
        if not rows:
            print(f"{credit+'/'+domain:22s} (no log yet)")
            continue
        s = summarize(rows)
        out[f"{credit}/{domain}"] = s
        cpr = (f"{s['critic_P_late']}/{s['critic_R_late']}"
               if s["critic_P_late"] is not None else "-")
        print(f"{credit+'/'+domain:22s} "
              f"{str(s['succ_early'])+'->'+str(s['succ_late']):14s} "
              f"{str(s['trueviol100_early'])+'->'+str(s['trueviol100_late']):20s} "
              f"{str(s['trueviol_ep_early'])+'->'+str(s['trueviol_ep_late']):14s} "
              f"{cpr:10s}")
    Path("results/exp_selfcritic").mkdir(parents=True, exist_ok=True)
    Path("results/exp_selfcritic/exp1_summary.json").write_text(json.dumps(out, indent=2))
    print("\n-> results/exp_selfcritic/exp1_summary.json")


if __name__ == "__main__":
    main()
