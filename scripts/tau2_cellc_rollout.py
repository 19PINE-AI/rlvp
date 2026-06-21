"""tau2 cell-C, stage 1 (run under .venv-tau2/bin/python):
roll out airline episodes with the SEMANTIC rule tracker and serialize each as a
critic-ready record (transcript + outcome reward + semantic-rule violations).

Cell-C question: in a domain where verifiable rules cover policy *validity* but
not task *intent* (tau2, FINDINGS §5), does blind self-critique flag the
INTENT-MISS episodes (outcome failed, semantic rules clean) that rules
structurally cannot? This stage produces the trajectories; tau2_cellc_critique.py
judges them.

Needs a user-sim LLM at localhost:8011 (system-python vLLM, Qwen3-4B).
Usage: .venv-tau2/bin/python scripts/tau2_cellc_rollout.py [n_tasks] [trials] [--no-policy]
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TAU2_DATA_DIR", "/tmp/tau2-bench/data")
os.environ.setdefault("OPENAI_API_KEY", "local")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8011/v1")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8011/v1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvp.rollout import set_template
from rlvp.tau2_adapter import GenServer, run_one_sim
from tau2.domains.airline.environment import get_tasks

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
INCLUDE_POLICY = "--no-policy" not in sys.argv
POLICY_MODEL = "Qwen/Qwen3-4B"
USER_LLM = "openai/Qwen/Qwen3-4B"
MAX_STEPS = 10
OUT = ROOT / "results" / "tau2_cellc"; OUT.mkdir(parents=True, exist_ok=True)

BLOCK = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)
THINK = re.compile(r"^<think>.*?</think>\s*", re.S)


def parse_record(ep, tok, task):
    """Decode the episode token stream into a critic-ready record."""
    full = tok.decode(ep.ids, skip_special_tokens=False)
    blocks = BLOCK.findall(full)
    system = next((c for r, c in blocks if r == "system"), "")
    # ordered non-system turns: user(obs/customer) / assistant(action), alternating
    seq = [(r, c) for r, c in blocks if r in ("user", "assistant")]
    goal = next((c.strip() for r, c in seq if r == "user"), "")
    # pair each assistant action with the user block that follows it
    steps, turn_ids, k = [], [], 0
    for i, (r, c) in enumerate(seq):
        if r != "assistant":
            continue
        action = THINK.sub("", c).strip()
        m = list(re.finditer(r"Action:\s*\S+", action))
        if m:
            action = action[m[-1].start():].strip()
        result = ""
        if i + 1 < len(seq) and seq[i + 1][0] == "user":
            result = seq[i + 1][1].strip()
        if len(result) > 400:
            result = result[:400] + " ...[truncated]"
        steps.append(f"Step {k + 1}:\n  action: {action}\n  result: {result}")
        turn_ids.append(k)
        k += 1
    return {
        "task_id": str(getattr(task, "id", "?")),
        "reward": float(ep.env.outcome_reward()),
        "success": bool(ep.env.success),
        "semantic_viol_turns": sorted({t for t, _ in ep.env.violations}),
        "semantic_viols": [list(v) for v in ep.env.violations],
        "n_actions": k,
        "domain_sys": system.strip(),
        "rules_block": "",                 # blind only (tau2 rules aren't prompt text)
        "goal": goal,
        "transcript": "\n".join(steps),
        "turn_ids": turn_ids,
    }


def main():
    set_template(POLICY_MODEL)
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL)
    print(f"loading policy {POLICY_MODEL} (include_policy={INCLUDE_POLICY}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    gen = GenServer(model, tok, temperature=1.0, max_batch=12)

    tasks = get_tasks()[:N_TASKS]
    print(f"{len(tasks)} tasks x {TRIALS} trials = {len(tasks)*TRIALS} sims", flush=True)
    records, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [(t, ex.submit(run_one_sim, t, gen.generate, tok, INCLUDE_POLICY,
                              USER_LLM, MAX_STEPS, "semantic"))
                for t in tasks for _ in range(TRIALS)]
        for t, f in futs:
            try:
                ep = f.result()
            except Exception as exc:
                print("sim error:", str(exc)[:100], flush=True); ep = None
            done += 1
            if ep is None or ep.n_turns == 0:
                continue
            records.append(parse_record(ep, tok, t))
            if done % 10 == 0:
                print(f"  {done}/{len(futs)} sims, {len(records)} usable", flush=True)
    gen.stop()
    nsucc = sum(r["success"] for r in records)
    nclean = sum(1 for r in records if not r["semantic_viol_turns"])
    nintent = sum(1 for r in records if not r["success"] and not r["semantic_viol_turns"])
    print(f"\n{len(records)} episodes: success={nsucc}, semantically-clean={nclean}, "
          f"INTENT-MISS (fail & clean)={nintent}", flush=True)
    path = OUT / "traj.json"
    path.write_text(json.dumps({"policy": POLICY_MODEL, "include_policy": INCLUDE_POLICY,
                                "records": records}))
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
