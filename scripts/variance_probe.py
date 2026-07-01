"""E1: measure the variance vacuum. Sample groups on the fileops+csops task pool and, per
group, compute the within-group variance of the OUTCOME (success in {0,1}) and of the
PENALTY channel (# rule violations per episode). Bin groups by their success rate to show
that Var_G(outcome) -> 0 at both ends (all-fail, all-success) while Var_G(penalty) stays > 0.
This turns the conceptual Fig 2 into a measured panel.

Usage: variance_probe.py [model_or_ckpt] [--G 8] [--tasks 60] [--out variance_probe]
"""
import json, sys, statistics as st
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlvp.rollout import run_episodes, set_template, start_episode   # noqa: E402
from rlvp.envs import make_env                                        # noqa: E402


def arg(k, d, cast=str):
    for i, a in enumerate(sys.argv):
        if a == "--" + k:
            return cast(sys.argv[i + 1])
    return d


MODEL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "Qwen/Qwen3-4B"
G = arg("G", 8, int)
TASKS = arg("tasks", 60, int)
OUT = arg("out", "variance_probe")
DOMAINS = ("fileops", "csops")
OUTD = ROOT / "results" / f"run_{OUT}"; OUTD.mkdir(parents=True, exist_ok=True)


def main():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL if "/" in MODEL and not Path(MODEL).exists() else MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import random
    rng = random.Random(7)
    rows = []
    for t in range(TASKS):
        dom = DOMAINS[t % len(DOMAINS)]
        s = rng.randint(0, 500)
        grp = [start_episode(tok, make_env(dom, s)) for _ in range(G)]
        run_episodes(model, tok, grp, temperature=1.0, top_p=1.0, gen_batch=48,
                     max_new_tokens=200, max_episode_tokens=3000)
        succ = [1.0 if e.env.success else 0.0 for e in grp]
        pen = [float(len(e.env.violations)) for e in grp]     # penalty channel magnitude
        rec = {"domain": dom, "success_rate": round(sum(succ) / len(succ), 3),
               "var_outcome": round(st.pvariance(succ), 4),
               "var_penalty": round(st.pvariance(pen), 4),
               "mean_viol": round(sum(pen) / len(pen), 2)}
        rows.append(rec)
        with open(OUTD / "per_group.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    # bin by success rate
    bins = [(0.0, 0.05), (0.05, 0.35), (0.35, 0.65), (0.65, 0.95), (0.95, 1.01)]
    summary = []
    for lo, hi in bins:
        b = [r for r in rows if lo <= r["success_rate"] < hi]
        if not b:
            continue
        summary.append({"bin": f"[{lo},{hi})", "n": len(b),
                        "mean_var_outcome": round(sum(r["var_outcome"] for r in b) / len(b), 4),
                        "mean_var_penalty": round(sum(r["var_penalty"] for r in b) / len(b), 4)})
    (OUTD / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY:", json.dumps(summary), flush=True)
    print("VARIANCE PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
