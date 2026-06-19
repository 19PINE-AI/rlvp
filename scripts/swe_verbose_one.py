#!/usr/bin/env python3
"""Run ONE oracle small-patch SWE episode and dump the full trajectory, to tell
capability-failure (model writes a wrong fix) from mechanical-failure (fix never
lands / patch doesn't apply / tests not actually run)."""
import sys
from transformers import AutoTokenizer
sys.path.insert(0, ".")
from rlvp.rollout import set_template, TEMPLATE
from rlvp.vllm_gen import VLLMGenServer
from rlvp.swe_adapter import load_small_patch_instances, run_swe_episode, oracle_hint

MODEL = "Qwen/Qwen3-30B-A3B-FP8"
set_template(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
inst = load_small_patch_instances(max_changed=6, max_files=1, max_hunks=1, n=1)[0]
print("INSTANCE:", inst["instance_id"])
print("ORACLE HINT:", oracle_hint(inst))
print("GOLD PATCH:\n", inst["patch"][:600])
gen = VLLMGenServer(MODEL, tok, max_new_tokens=700, temperature=0.5,
                    gpu_mem=0.40, max_model_len=12288, max_batch=4)
ep = run_swe_episode(inst, gen.generate, tok, rule_mode="structural",
                     max_steps=16, oracle=True, verbose=True)
print("\n=== TRAJECTORY (decoded turns) ===")
full = tok.decode(ep.ids)
# print just the assistant/tool turns, truncated
print(full[-6000:])
print("\nREWARD:", ep.env.outcome_reward(), "discharges:", len(ep.env.discharges),
      "violations:", len(ep.env.violations))
gen.stop()
print("VERBOSE DONE")
