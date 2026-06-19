#!/usr/bin/env python3
"""Capability de-risk for the big model BEFORE building the training harness.
Serves Qwen3-30B-A3B-FP8 via vLLM and runs zero-shot rollouts through the PROVEN
oracles (SWE-dask worktree+pytest; Lean REPL) to answer one question: does a 30B-
class model get success OFF THE 0% FLOOR where 4B failed? If yes -> build the RL
harness; if no -> stronger model won't rescue these benchmarks either.

Usage: probe_30b.py [model_path] [n_swe] [n_lean]
"""
import sys
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from rlvp.rollout import set_template          # noqa: E402
from rlvp.vllm_gen import VLLMGenServer        # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-FP8"
N_SWE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N_LEAN = int(sys.argv[3]) if len(sys.argv) > 3 else 12
# cap GPU so we can share with a running seed sweep; fp8 30B ~30GB + KV
GPU_MEM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.42


def main():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"loading {MODEL} in vLLM (gpu_mem={GPU_MEM}) ...", flush=True)
    gen = VLLMGenServer(MODEL, tok, max_new_tokens=512, temperature=0.7,
                        gpu_mem=GPU_MEM, max_model_len=8192, max_batch=32)
    g = gen.generate

    # ---- SWE-dask: the benchmark where 4B scored 0% ----
    print("\n=== SWE-dask probe ===", flush=True)
    from rlvp.swe_adapter import load_clean_instances, run_swe_episode
    insts = load_clean_instances()[:N_SWE]
    n_ep = n_succ = n_tests = 0
    for inst in insts:
        try:
            ep = run_swe_episode(inst, g, tok, rule_mode="structural", max_steps=14)
        except Exception as e:
            print(f"  {inst['instance_id']}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if ep is None:
            print(f"  {inst['instance_id']}: setup-failed/no-episode", flush=True)
            continue
        n_ep += 1
        r = ep.env.outcome_reward()
        ran = len(getattr(ep.env, "discharges", []))
        n_succ += int(r > 0)
        n_tests += int(ran > 0)
        print(f"  {inst['instance_id']}: reward={r} discharges={ran}", flush=True)
    print(f"SWE: episodes={n_ep}/{len(insts)} solved={n_succ} ran-tests={n_tests}", flush=True)

    # ---- Lean (synthetic, fast core REPL) as a theorem-proving sanity signal ----
    print("\n=== Lean probe (synthetic, capability sanity) ===", flush=True)
    from rlvp.lean_adapter import run_lean_episode
    sys.path.insert(0, "leanprove")
    from theorems_synth import load_synth_theorems
    thms = load_synth_theorems()[:N_LEAN]
    ls = lsucc = 0
    for thm in thms:
        try:
            ep = run_lean_episode(thm, g, tok, rule_mode="structural", max_steps=10)
        except Exception as e:
            print(f"  {thm.get('name','?')}: ERROR {e}", flush=True)
            continue
        ls += 1
        lsucc += int(ep.env.outcome_reward() > 0)
    print(f"Lean: episodes={ls}/{len(thms)} solved={lsucc}", flush=True)

    gen.stop()
    print("\nPROBE DONE", flush=True)


if __name__ == "__main__":
    main()
