"""Post-hoc rule-compliance analysis of tau2-bench trajectories (Phase-0 style).

Structural rules over the action log (machine-checkable, no judge):
  R1 act_before_lookup - a state-changing call before any get_/search_/list_/calculate_
  R2 call_spam         - identical (name, args) tool call issued 3+ times
  R3 unconfirmed_chain - 2+ consecutive write calls with no user turn between
                         (airline policy requires explicit confirmation per change)

Usage: python3 scripts/tau2_analyze.py results/tau2/airline_qwen8b/results.json
"""
import json
import sys
from collections import Counter

WRITE_PREFIXES = ("update_", "cancel_", "book_", "modify_", "send_", "transfer_")
READ_PREFIXES = ("get_", "search_", "list_", "calculate_", "think")

path = sys.argv[1]
d = json.load(open(path))
sims = d["simulations"]
ok_sims = [s for s in sims if s.get("messages")]

n_calls = n_viol = 0
per_rule = Counter()
ep_viol = 0
rewards = []
for s in ok_sims:
    ri = s.get("reward_info") or {}
    if isinstance(ri, dict) and ri.get("reward") is not None:
        rewards.append(float(ri["reward"]))
    seen_read = False
    sig_count = Counter()
    last_was_write = False
    viols_here = 0
    for m in s["messages"]:
        role = m.get("role")
        if role == "user":
            last_was_write = False
        if role != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            name = tc.get("name", "")
            args = json.dumps(tc.get("arguments", tc.get("args", {})), sort_keys=True, default=str)
            n_calls += 1
            if name.startswith(READ_PREFIXES):
                seen_read = True
            is_write = name.startswith(WRITE_PREFIXES)
            if is_write and not seen_read:
                per_rule["act_before_lookup"] += 1
                viols_here += 1
            sig_count[(name, args)] += 1
            if sig_count[(name, args)] == 3:
                per_rule["call_spam"] += 1
                viols_here += 1
            if is_write and last_was_write:
                per_rule["unconfirmed_chain"] += 1
                viols_here += 1
            last_was_write = is_write or last_was_write
    n_viol += viols_here
    ep_viol += viols_here > 0

out = {
    "n_sims_total": len(sims),
    "n_sims_with_messages": len(ok_sims),
    "mean_reward": sum(rewards) / len(rewards) if rewards else None,
    "tool_calls": n_calls,
    "violations": n_viol,
    "viol_per_100_calls": round(100 * n_viol / max(n_calls, 1), 1),
    "episodes_with_violation": f"{ep_viol}/{len(ok_sims)}",
    "per_rule": dict(per_rule),
}
print(json.dumps(out, indent=2))
json.dump(out, open("results/tau2/analysis.json", "w"), indent=2)
