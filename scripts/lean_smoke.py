"""Smoke test (GATE, no training) for the Lean RLVP adapter.

Loads Qwen3-1.7B, rolls out 3 episodes on the self-contained synthetic theorems
via run_lean_episode, and prints per episode: n_tactics, the tactics emitted,
the goals trajectory, reward, and firing violations/discharges.

Run:  cd /home/ubuntu/rlvp && python3 scripts/lean_smoke.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The REPL needs the self-contained elan toolchain on PATH (same as lean_repl.py).
ELAN_HOME = ROOT / "leanprove" / ".elan"
os.environ["ELAN_HOME"] = str(ELAN_HOME)
os.environ["PATH"] = f"{ELAN_HOME / 'bin'}:{os.environ.get('PATH', '')}"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.rollout import set_template
from rlvp.lean_adapter import run_lean_episode
from rlvp.tau2_adapter import GenServer

sys.path.insert(0, str(ROOT / "leanprove"))
from theorems_synth import load_synth_theorems  # noqa: E402
from lean_repl import LeanREPL  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-1.7B")
N_EPISODES = int(os.environ.get("SMOKE_N", "3"))
RULE_MODE = os.environ.get("SMOKE_RULE_MODE", "structural")
MAX_STEPS = int(os.environ.get("SMOKE_MAX_STEPS", "8"))


def main():
    print(f"=== Lean RLVP adapter smoke test ===", flush=True)
    print(f"model={MODEL} rule_mode={RULE_MODE} max_steps={MAX_STEPS}", flush=True)

    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"model loaded in {time.time() - t_load:.1f}s", flush=True)

    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=120, max_batch=8)

    theorems = load_synth_theorems()[:N_EPISODES]
    # one persistent REPL reused across episodes (fresh proof state per theorem)
    repl = LeanREPL()
    rolled = 0
    rule_fired_any = False
    try:
        for i, thm in enumerate(theorems):
            t0 = time.time()
            ep = run_lean_episode(thm, gen_srv.generate, tok, rule_mode=RULE_MODE,
                                  max_steps=MAX_STEPS, repl=repl)
            dt = time.time() - t0
            env = ep.env
            v = env.violations
            d = env.discharges
            rule_fired_any = rule_fired_any or bool(v) or bool(d)
            rolled += 1
            print(f"\n--- episode {i+1}/{len(theorems)}: {thm['name']} ---", flush=True)
            print(f"  statement : {thm['statement']}")
            print(f"  n_tactics : {ep.n_turns}")
            print(f"  tactics   : {env.calls}")
            print(f"  reward    : {env.outcome_reward()}  (success={env.success})")
            print(f"  violations: {v}")
            print(f"  discharges: {d}")
            print(f"  wall_time : {dt:.2f}s", flush=True)
    finally:
        repl.close()
        gen_srv.stop()

    print(f"\n=== SMOKE SUMMARY ===", flush=True)
    print(f"episodes rolled out : {rolled}/{len(theorems)}")
    print(f"rules fired (any ep): {rule_fired_any}")
    gate = rolled == len(theorems)
    print(f"GATE: {'PASS' if gate else 'FAIL'} "
          f"({rolled} episodes rolled out cleanly via the REPL oracle)", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
