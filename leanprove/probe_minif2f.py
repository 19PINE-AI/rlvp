"""PROSPECTIVE-PREDICTION probe: reachability of the goal-progress potential on the
HELD-OUT *hard* miniF2F slice (easy_only=False minus the easy set -- IMO/AIME-level
theorems NEVER trained on; all prior 30B Lean runs used easy_only=True).

This is the cheap, no-training step of the prospective prediction (PROSPECTIVE_PREDICTION.md):
run the BASE 30B policy for G rollouts per theorem and measure the within-group variance
Var_G(Phi), Phi = #goal_progress discharges (a strict goal-count decrease -- exactly the
event the aligned `c3` credit rewards). The criterion's rule, pre-registered BEFORE training:

    aligned `c3` HELPS  iff  the base policy reaches intermediate goal-progress on the
    held-out theorems, i.e. Var_G(Phi) is meaningfully > 0 (some rollouts decrease goals
    while others do not). If Var_G(Phi) ~ 0 (no partial progress, like the SWE/E-C null),
    the potential is UNREACHABLE and aligned gives no gradient over outcome-only.

Usage: python3 leanprove/probe_minif2f.py [--G 8] [--tasks 24] [--maxsteps 8] [--seed 7]
"""
import json, os, sys, threading, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ELAN_HOME = ROOT / "leanprove" / ".elan"
os.environ["ELAN_HOME"] = str(ELAN_HOME)
os.environ["PATH"] = f"{ELAN_HOME / 'bin'}:{os.environ.get('PATH', '')}"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer
from rlvp.rollout import set_template
from rlvp.lean_adapter import run_lean_episode
from rlvp.vllm_gen import VLLMGenServer

sys.path.insert(0, str(ROOT / "leanprove"))
from load_minif2f import load_minif2f          # noqa: E402
from mathlib_repl_wrapper import MathlibREPL    # noqa: E402


def arg(k, d, cast=str):
    for i, a in enumerate(sys.argv):
        if a == "--" + k:
            return cast(sys.argv[i + 1])
    return d


G        = arg("G", 8, int)
TASKS    = arg("tasks", 24, int)
MAX_STEPS= arg("maxsteps", 8, int)
SEED     = arg("seed", 7, int)
MODEL_GEN= "Qwen/Qwen3-30B-A3B-FP8"
MODEL_HF = "Qwen/Qwen3-30B-A3B"
RULE_MODE= "aligned"          # goal_progress discharge ONLY -- the Phi the c3 credit rewards
OUT      = ROOT / "results" / "probe_minif2f_hard"; OUT.mkdir(parents=True, exist_ok=True)

_TLS, _ALL_REPLS, _RECYCLE = threading.local(), [], 16


def _repl():
    r = getattr(_TLS, "repl", None); n = getattr(_TLS, "n", 0)
    if r is None or r.proc.poll() is not None or n >= _RECYCLE:
        if r is not None:
            try: r.close()
            except Exception: pass
        r = MathlibREPL(); _TLS.repl = r; _ALL_REPLS.append(r); n = 0
    _TLS.n = n + 1
    return r


def _phi(ep):
    """Potential = #goal_progress discharges (strict goal-count decreases). success => the
    proof closed, the maximal reachable Phi for that theorem."""
    if ep is None or ep.env is None:
        return None
    return sum(1 for _, r in ep.env.discharges if r == "goal_progress")


def _rollout(thm, gen, tok):
    return run_lean_episode(thm, gen, tok, rule_mode=RULE_MODE,
                            max_steps=MAX_STEPS, repl=_repl())


def main():
    set_template(MODEL_GEN)
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # held-out HARD slice = full Valid minus the easy set (never trained)
    easy = {t["name"] for t in load_minif2f(split="Valid", easy_only=True)}
    allt = load_minif2f(split="Valid", easy_only=False)
    hard = [t for t in allt if t["name"] not in easy]
    import random
    rng = random.Random(SEED)
    rng.shuffle(hard)
    hard = hard[:TASKS]
    print(f"HELD-OUT hard miniF2F: {len(hard)} theorems, G={G}, max_steps={MAX_STEPS}, "
          f"mode={RULE_MODE}", flush=True)

    print("starting vLLM fp8 generator (base, no LoRA) ...", flush=True)
    gen_srv = VLLMGenServer(MODEL_GEN, tok, max_new_tokens=200, temperature=1.0,
                            gpu_mem=0.48, max_model_len=4096, max_batch=32,
                            enable_lora=False, enforce_eager=True)

    ex = ThreadPoolExecutor(max_workers=G)
    rows, group_vars, succ_all, phi_all, reach_groups = [], [], [], [], 0
    for ti, thm in enumerate(hard):
        t0 = time.time()
        fs = [ex.submit(_rollout, thm, gen_srv.generate, tok) for _ in range(G)]
        eps = []
        for f in fs:
            try: eps.append(f.result())
            except Exception: eps.append(None)
        phis = [p for p in (_phi(e) for e in eps) if p is not None]
        succ = [1.0 if (e is not None and e.env is not None and e.env.success) else 0.0
                for e in eps]
        if len(phis) < 2:
            continue
        v = st.pvariance(phis)
        group_vars.append(v); succ_all += succ; phi_all += phis
        reached = any(p > 0 for p in phis)
        reach_groups += int(reached)
        row = {"thm": thm["name"], "var_phi": round(v, 4),
               "phi_max": max(phis), "phi_mean": round(sum(phis) / len(phis), 3),
               "reached": reached, "succ": round(sum(succ) / len(succ), 3),
               "n": len(phis), "wall_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(OUT / "per_theorem.jsonl", "a") as fp:
            fp.write(json.dumps(row) + "\n")

    summary = {
        "held_out": "miniF2F hard (easy_only=False minus easy)",
        "n_theorems": len(rows), "G": G, "max_steps": MAX_STEPS, "seed": SEED,
        "mean_var_phi": round(sum(group_vars) / len(group_vars), 4) if group_vars else 0.0,
        "frac_groups_reachable": round(reach_groups / len(rows), 3) if rows else 0.0,
        "mean_phi": round(sum(phi_all) / len(phi_all), 3) if phi_all else 0.0,
        "base_succ": round(sum(succ_all) / len(succ_all), 4) if succ_all else 0.0,
        "max_phi_seen": max(phi_all) if phi_all else 0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("PROBE SUMMARY:", json.dumps(summary), flush=True)
    for r in _ALL_REPLS:
        try: r.close()
        except Exception: pass
    gen_srv.stop()
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
