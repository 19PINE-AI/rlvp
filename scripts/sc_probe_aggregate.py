"""Aggregate the fixed-trajectory cross-model probe: per-rule blind recall on
the IDENTICAL trajectories, across critic sizes. This is the clean scale claim
(does the stateful/masked blind spot persist when capability varies but the
trajectory is held fixed?) and the distillation probe (policy fixed, critic
varied). Reads results/exp_selfcritic/probe/<policy>__<critic>.json
"""
import json
from pathlib import Path

PROBE = Path("results/exp_selfcritic/probe")
RULES = [("no_kb_before_call", "surface"), ("access_without_acl", "surface"),
         ("untested_submit", "stateful"), ("blind_write", "stateful"),
         ("write_without_access", "masked")]
ORDER = ["Qwen3-1_7B", "Qwen3-4B", "Qwen3-8B"]


def main():
    files = sorted(PROBE.glob("*__*.json"))
    if not files:
        print("no probe reports yet"); return
    by_policy = {}
    for f in files:
        rep = json.loads(f.read_text())
        by_policy.setdefault(rep["policy"], {})[rep["critic"]] = rep

    for policy, critics in by_policy.items():
        def slug(c):
            return c.split("/")[-1].replace(".", "_")
        order = sorted(critics, key=lambda c: next(
            (i for i, o in enumerate(ORDER) if o == slug(c)), 99))
        labels = [c.split("/")[-1] for c in order]
        print(f"\n=== fixed trajectories from policy {policy.split('/')[-1]} "
              f"({critics[order[0]]['n_records']} episodes), judged by each critic (BLIND) ===")
        print(f"{'rule':24s} {'class':9s} " + " ".join(f"{l:>12s}" for l in labels))
        for rule, cls in RULES:
            cells = []
            for c in order:
                pr = critics[c]["modes"]["blind"]["per_rule_recall"].get(rule)
                cells.append(f"{pr['recall']}({pr['detected']}/{pr['total']})" if pr else "-")
            print(f"{rule:24s} {cls:9s} " + " ".join(f"{x:>12s}" for x in cells))
        print(f"{'OVERALL blind R':24s} {'':9s} " +
              " ".join(f"{critics[c]['modes']['blind']['recall']:>12}" for c in order))
        print(f"{'OVERALL blind P':24s} {'':9s} " +
              " ".join(f"{critics[c]['modes']['blind']['precision']:>12}" for c in order))


if __name__ == "__main__":
    main()
