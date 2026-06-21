"""tau2 cell-C, stage 2: judge the rolled-out trajectories with the BLIND
self-critic (same policy model) and test whether self-critique flags the
INTENT-MISS episodes that the verifiable semantic rules structurally cannot.

Cell-C hypothesis: on episodes that FAILED but are semantically CLEAN (rules
silent), the self-critic still flags a mistake -> self-critique supplies
intent-level signal beyond the rules' ceiling (the mirror of Exp0's stateful
blind spot).

Usage: python3 scripts/tau2_cellc_critique.py [critic_model]
Reads results/tau2_cellc/traj.json -> writes results/tau2_cellc/report.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.rollout import set_template
from rlvp.self_critic import label_records

CRITIC = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
OUT = Path("results/tau2_cellc")


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return round(p, 3), round(r, 3), round(2 * p * r / max(p + r, 1e-9), 3)


def main():
    blob = json.loads((OUT / "traj.json").read_text())
    recs = blob["records"]
    set_template(CRITIC)
    tok = AutoTokenizer.from_pretrained(CRITIC)
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    print(f"loading critic {CRITIC} for {len(recs)} tau2 episodes ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(CRITIC, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    flagged = label_records(model, tok, recs, mode="blind", batch=3, temperature=0.0)
    for r, f in zip(recs, flagged):
        r["critic_flag"] = len(f) > 0
        r["critic_turns"] = sorted(f)

    # categorize
    cats = {"succ_clean": [], "succ_dirty": [], "fail_clean": [], "fail_dirty": []}
    for r in recs:
        ok = r["success"]; clean = not r["semantic_viol_turns"]
        key = ("succ" if ok else "fail") + ("_clean" if clean else "_dirty")
        cats[key].append(r)

    def flagrate(rs):
        return round(sum(x["critic_flag"] for x in rs) / max(len(rs), 1), 3)

    # failure-prediction: does flagging a mistake predict outcome failure, and
    # does it beat the semantic rule as a failure predictor?
    def predictor(pred_fn):
        tp = sum(1 for r in recs if pred_fn(r) and not r["success"])
        fp = sum(1 for r in recs if pred_fn(r) and r["success"])
        fn = sum(1 for r in recs if not pred_fn(r) and not r["success"])
        return prf(tp, fp, fn)

    sem_p = predictor(lambda r: bool(r["semantic_viol_turns"]))
    crit_p = predictor(lambda r: r["critic_flag"])

    report = {
        "critic": CRITIC, "n": len(recs),
        "counts": {k: len(v) for k, v in cats.items()},
        "critic_flag_rate": {k: flagrate(v) for k, v in cats.items()},
        "INTENT_MISS_critic_recall": flagrate(cats["fail_clean"]),  # the headline
        "success_false_flag_rate": flagrate(cats["succ_clean"] + cats["succ_dirty"]),
        "failure_prediction": {
            "semantic_rule (P,R,F1)": sem_p,
            "self_critic (P,R,F1)": crit_p,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nHEADLINE: on INTENT-MISS episodes (failed & semantically clean), the "
          f"semantic rules flag 0% by construction; the blind self-critic flags "
          f"{report['INTENT_MISS_critic_recall']*100:.0f}%.")


if __name__ == "__main__":
    main()
