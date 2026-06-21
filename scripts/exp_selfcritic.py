"""Experiment 0 -- offline gating study for on-policy LLM self-critique.

Roll out the policy model on the synthetic envs, then have the SAME model
reflect on its own transcript and flag its mistakes (a) blind -- no rules
given, only domain language; and (b) rule-aware -- told the rules (detection
ceiling). Score both against the deterministic rule oracle.

The go/no-go question for Exp 1: can the on-policy model recover its own
violations at usable precision/recall WITHOUT being told the rules? If blind
precision/recall is near the rule-aware ceiling and both are decent, training
on self-critique is viable. If blind collapses, self-reward will be noisier
than rules -- and that negative result is itself the finding.

Usage:
  python3 scripts/exp_selfcritic.py [model] [n_tasks_per_domain]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.envs import make_env
from rlvp.rollout import run_episodes, set_template, start_episode
from rlvp.self_critic import label_episodes

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
DOMAINS = ("fileops", "csops", "gated")
ROLL_TEMP = 1.0          # match training temp: we WANT mistakes to grade
CRITIC_TEMP = 0.0        # deterministic judgment
GEN_BATCH = 6            # small: share GPU with any in-flight run
CRITIC_BATCH = 2
MAX_EP_TOK = 1800
_slug = MODEL.split("/")[-1].replace(".", "_")
OUT = Path(f"results/exp_selfcritic/{_slug}"); OUT.mkdir(parents=True, exist_ok=True)


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return round(p, 3), round(r, 3), round(f, 3)


def score(episodes, mode_turns):
    """mode_turns: list[set] of critic-flagged turn_idx, aligned to episodes."""
    tp = fp = fn = 0
    ep_detect_tp = ep_detect_fn = 0           # episode-level: any violation found?
    per_rule = {}                              # rule -> [detected, total]
    fp_errored = fp_total = 0                  # are extra flags real failures?
    for e, pred in zip(episodes, mode_turns):
        gt = set(e.turn_violations)
        tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
        if gt:
            if gt & pred:
                ep_detect_tp += 1
            else:
                ep_detect_fn += 1
        for turn, rules in e.turn_violations.items():
            for rname in rules:
                d = per_rule.setdefault(rname, [0, 0])
                d[1] += 1
                if turn in pred:
                    d[0] += 1
        for turn in (pred - gt):
            fp_total += 1
            if turn in e.turn_errors:
                fp_errored += 1
    p, r, f = prf(tp, fp, fn)
    return {
        "turn_precision": p, "turn_recall": r, "turn_f1": f,
        "tp": tp, "fp": fp, "fn": fn,
        "episode_detection_recall": round(ep_detect_tp / max(ep_detect_tp + ep_detect_fn, 1), 3),
        "per_rule_recall": {k: {"detected": v[0], "total": v[1],
                                "recall": round(v[0] / max(v[1], 1), 3)}
                            for k, v in sorted(per_rule.items())},
        "fp_that_were_env_errors": round(fp_errored / max(fp_total, 1), 3),
        "fp_total": fp_total,
    }


def main():
    # Safety: a hard cap on THIS process's share of the GPU, so if we'd exceed
    # our budget we OOM ourselves (recoverable) instead of starving a co-resident
    # in-flight training run. Override with RLVP_MEM_FRAC.
    import os
    frac = float(os.environ.get("RLVP_MEM_FRAC", "0.10"))
    if frac > 0:                       # 0 (or unset-to-0) means NO cap, not "0 bytes"
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"GPU memory cap: {frac:.2f} of device", flush=True)
    else:
        print("GPU memory cap: none", flush=True)
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    print(f"loading {MODEL} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    report = {"model": MODEL, "n_tasks_per_domain": N, "roll_temp": ROLL_TEMP,
              "domains": {}}
    overall = {m: {"tp": 0, "fp": 0, "fn": 0} for m in ("blind", "rule_aware")}

    for domain in DOMAINS:
        print(f"\n=== {domain}: rolling out {N} episodes ===", flush=True)
        eps = [start_episode(tok, make_env(domain, 2000 + s), include_rules=False)
               for s in range(N)]
        run_episodes(model, tok, eps, temperature=ROLL_TEMP, top_p=1.0,
                     gen_batch=GEN_BATCH, max_episode_tokens=MAX_EP_TOK)
        n_viol = sum(len(e.turn_violations) for e in eps)
        n_succ = sum(e.env.success for e in eps)
        print(f"  rolled out: success={n_succ}/{N}, violating_turns={n_viol}", flush=True)

        mode_results, mode_turns_store = {}, {}
        for mode in ("blind", "rule_aware"):
            print(f"  critic[{mode}] ...", flush=True)
            label_episodes(model, tok, eps, mode=mode, batch=CRITIC_BATCH,
                           temperature=CRITIC_TEMP)
            turns = [set(e.critic_turns) for e in eps]   # snapshot before overwrite
            mode_turns_store[mode] = turns
            res = score(eps, turns)
            mode_results[mode] = res
            for k in ("tp", "fp", "fn"):
                overall[mode][k] += res[k]
            print(f"    P={res['turn_precision']} R={res['turn_recall']} "
                  f"F1={res['turn_f1']} (tp={res['tp']} fp={res['fp']} fn={res['fn']})",
                  flush=True)

        # one qualitative example: an episode with a violation, blind-mode reply
        ex = None
        for e, turns in zip(eps, mode_turns_store["blind"]):
            if e.turn_violations:
                ex = {"gt_violating_turns": {str(t): r for t, r in e.turn_violations.items()},
                      "blind_flagged_turns": sorted(turns),
                      "success": e.env.success}
                break
        report["domains"][domain] = {
            "rollout": {"success": n_succ, "n": N, "violating_turns": n_viol},
            "blind": mode_results["blind"],
            "rule_aware": mode_results["rule_aware"],
            "example": ex,
        }
        # free per-domain episode tensors
        del eps
        torch.cuda.empty_cache()

    report["overall"] = {}
    for mode in ("blind", "rule_aware"):
        o = overall[mode]
        p, r, f = prf(o["tp"], o["fp"], o["fn"])
        report["overall"][mode] = {"turn_precision": p, "turn_recall": r,
                                   "turn_f1": f, **o}

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print("OVERALL (micro-averaged over turns, all domains):")
    for mode in ("blind", "rule_aware"):
        o = report["overall"][mode]
        print(f"  {mode:11s}  P={o['turn_precision']}  R={o['turn_recall']}  "
              f"F1={o['turn_f1']}")
    print(f"\nfull report -> {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
