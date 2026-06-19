#!/usr/bin/env python3
"""Does 30B-A3B prove any REAL miniF2F theorems zero-shot? (the go/no-go for the
miniF2F RL harness). Serves the model via vLLM, drives the proven Mathlib REPL
through the existing Lean episode loop. Sequential -> one warm MathlibREPL.

Usage: probe_minif2f.py [model] [n] [gpu_mem]
"""
import sys
from transformers import AutoTokenizer

sys.path.insert(0, ".")
sys.path.insert(0, "leanprove")
from rlvp.rollout import set_template            # noqa: E402
from rlvp.vllm_gen import VLLMGenServer          # noqa: E402
from rlvp.lean_adapter import run_lean_episode   # noqa: E402
from load_minif2f import load_minif2f            # noqa: E402
from mathlib_repl_wrapper import MathlibREPL     # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-FP8"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
GPU_MEM = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55


def main():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    thms = load_minif2f(split="Valid", easy_only=True, n=N)
    print(f"miniF2F easy theorems: {len(thms)} | loading {MODEL} ...", flush=True)
    gen = VLLMGenServer(MODEL, tok, max_new_tokens=512, temperature=0.7,
                        gpu_mem=GPU_MEM, max_model_len=8192, max_batch=8)
    g = gen.generate
    print("warming Mathlib REPL (import Mathlib ~12s) ...", flush=True)
    repl = MathlibREPL()
    print(f"  base env = {repl.base_env}", flush=True)

    solved = progressed = ep_ok = 0
    for thm in thms:
        try:
            ep = run_lean_episode(thm, g, tok, rule_mode="structural",
                                  max_steps=12, repl=repl)
        except Exception as e:
            print(f"  {thm['name']}: ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        ep_ok += 1
        r = ep.env.outcome_reward()
        disc = len(getattr(ep.env, "discharges", []))
        solved += int(r > 0)
        progressed += int(disc > 0)
        print(f"  {thm['name']:42s} solved={int(r>0)} goal-progress={disc}", flush=True)
    print(f"\nminiF2F n={len(thms)}: episodes={ep_ok} SOLVED={solved} "
          f"made-progress={progressed}", flush=True)
    repl.close()
    gen.stop()
    print("MINIF2F PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
