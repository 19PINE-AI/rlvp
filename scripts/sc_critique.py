"""Stage 2 of the fixed-trajectory cross-model probe: judge a saved trajectory
file with a chosen CRITIC model (blind + rule-aware), scored against the rule
oracle. Because every critic judges the IDENTICAL trajectories, differences are
pure critic effect -- this is both the clean scale claim and the distillation
probe (policy fixed, critic varied = how much a bigger judge buys).

Usage: python3 scripts/sc_critique.py <traj_file.json> <critic_model>
Writes results/exp_selfcritic/probe/<policy_slug>__<critic_slug>.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.self_critic import label_records

TRAJ = sys.argv[1]
CRITIC = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
CRITIC_BATCH = 2

frac = float(os.environ.get("RLVP_MEM_FRAC", "0"))
if frac > 0:
    torch.cuda.set_per_process_memory_fraction(frac, 0)

out = Path("results/exp_selfcritic/probe"); out.mkdir(parents=True, exist_ok=True)


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return round(p, 3), round(r, 3), round(2 * p * r / max(p + r, 1e-9), 3)


def score(records, flagged):
    """flagged: list[set] aligned to records. Returns overall + per-rule recall."""
    tp = fp = fn = 0
    per_rule = {}
    for rec, pred in zip(records, flagged):
        gt_turns = {int(t): rs for t, rs in rec["gt_turns"].items()}
        gt = set(gt_turns)
        tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
        for t, rs in gt_turns.items():
            for rname in rs:
                d = per_rule.setdefault(rname, [0, 0])
                d[1] += 1
                if t in pred:
                    d[0] += 1
    p, r, f = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn,
            "per_rule_recall": {k: {"detected": v[0], "total": v[1],
                                    "recall": round(v[0] / max(v[1], 1), 3)}
                                for k, v in sorted(per_rule.items())}}


def main():
    blob = json.loads(Path(TRAJ).read_text())
    records = blob["records"]
    policy = blob.get("policy", "?")
    pslug = policy.split("/")[-1].replace(".", "_")
    cslug = CRITIC.split("/")[-1].replace(".", "_")

    tok = AutoTokenizer.from_pretrained(CRITIC)
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    print(f"loading critic {CRITIC} to judge {len(records)} trajectories from policy {policy} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(CRITIC, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    report = {"policy": policy, "critic": CRITIC, "n_records": len(records), "modes": {}}
    for mode in ("blind", "rule_aware"):
        flagged = label_records(model, tok, records, mode=mode, batch=CRITIC_BATCH, temperature=0.0)
        report["modes"][mode] = score(records, flagged)
        s = report["modes"][mode]
        print(f"  {mode:11s} P={s['precision']} R={s['recall']} F1={s['f1']}", flush=True)

    path = out / f"{pslug}__{cslug}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
