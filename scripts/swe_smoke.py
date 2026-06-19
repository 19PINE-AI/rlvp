"""SWE adapter smoke test (GATE, no training).

Rolls out a few episodes on 1-2 clean dask instances via run_swe_episode and
prints, per episode: n_turns, tools called, whether it ran tests / edited files,
the terminal oracle reward, and violations/discharges.

The gate is that the ROLLOUT MECHANICS work end to end on a REAL repo:
  * real worktree+venv setup, real file edits, real pytest execution,
  * the real FAIL_TO_PASS/PASS_TO_PASS oracle computing a 0/1 reward,
  * the process rules firing with token-exact turn bookkeeping.
A 1.7B/4B model almost never actually fixes the bug (reward usually 0) -- that
is EXPECTED; we are testing the harness, not the policy.

Modes:
  --fake   scripted policy (no GPU): validates env/tools/reward/rules. Runs a
           GOLD-fix script that SHOULD earn reward 1.0, proving the oracle.
  (default) real Qwen3-1.7B policy on the GPU.

Usage:
  cd /home/ubuntu/rlvp && python3 scripts/swe_smoke.py [--fake] [--model NAME]
                                                       [--episodes N] [--insts K]
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rlvp.swe_adapter import (load_clean_instances, run_swe_episode,
                              SweWorktree, SweRuleTracker, build_system_prompt,
                              build_task_msg)

FAKE = "--fake" in sys.argv
MODEL = "Qwen/Qwen3-1.7B"
N_EPISODES = 3
N_INSTS = 2
for i, a in enumerate(sys.argv):
    if a == "--model":
        MODEL = sys.argv[i + 1]
    if a == "--episodes":
        N_EPISODES = int(sys.argv[i + 1])
    if a == "--insts":
        N_INSTS = int(sys.argv[i + 1])


def report(ep, iid, wall):
    if ep is None:
        print(f"  [{iid}] EPISODE=None (setup failed)\n", flush=True)
        return
    tools = getattr(ep, "_swe_tools", [t for _, t in ep.env.calls])
    viol = ep.env.violations
    disch = ep.env.discharges
    print(f"  instance        : {iid}")
    print(f"  n_turns         : {ep.n_turns}")
    print(f"  tools called    : {tools}")
    print(f"  ran_tests       : {getattr(ep, '_swe_ran_tests', None)}")
    print(f"  edited_files    : {getattr(ep, '_swe_edited', None)}")
    print(f"  truncated       : {ep.truncated}")
    print(f"  REWARD (oracle) : {ep.env.outcome_reward()}   success={ep.env.success}")
    print(f"  violations      : {viol}")
    print(f"  discharges      : {disch}")
    print(f"  wall_s          : {round(wall, 1)}")
    print(f"  total tokens    : {len(ep.ids)}\n", flush=True)


# ----------------------------------------------------------------------------
# FAKE policy: a scripted generator. We hand-write a GOLD-fix trajectory per
# instance (read the buggy file, apply the gold patch via write_file, run tests,
# submit). This validates the FULL non-GPU stack including a reward=1.0 path.
# ----------------------------------------------------------------------------
def make_fake_gen(tok, scripts):
    """scripts: list of assistant texts to emit, in order. Returns a `gen` with
    the GenServer signature (ids -> generated ids ending in EOT)."""
    from rlvp.rollout import TEMPLATE
    eot = tok.convert_tokens_to_ids(TEMPLATE.eot)
    state = {"i": 0}

    def gen(ids):
        i = state["i"]
        text = scripts[i] if i < len(scripts) else 'done.\nAction: submit {}'
        state["i"] = i + 1
        return tok(text, add_special_tokens=False).input_ids + [eot]
    return gen


def gold_script(instance):
    """Assistant turns that apply the GOLD source patch through write_file, then
    run tests and submit -- exercises reproduce-before-patch, edit, verify."""
    import json as _json
    import re
    # parse gold patch into {path: new_full_file_content} by applying it to base.
    # Simpler + robust: reconstruct each touched file from a git worktree+apply.
    # We compute the gold file contents by setting up a throwaway worktree,
    # applying the gold patch, and reading the resulting files.
    return None  # replaced at runtime by gold_texts() which needs the worktree


def gold_texts(instance, wt):
    """Build the scripted assistant turns using the REAL gold file contents
    (obtained by applying the gold patch in a sibling worktree)."""
    import json as _json
    import re
    import swe_env_setup as H
    files = re.findall(r"^\+\+\+ b/(.+)$", instance["patch"], re.M)
    files = [f for f in files if f != "/dev/null"]
    # sibling worktree with the gold patch applied -> read final file contents
    sib = os.path.join("/tmp/swe_gold", instance["instance_id"])
    os.makedirs(sib, exist_ok=True)
    gdir = os.path.join(sib, "dask")
    H.make_worktree(instance["base_commit"], gdir)
    gold = {}
    try:
        H._apply_patch(gdir, instance["test_patch"], "tp")
        H._apply_patch(gdir, instance["patch"], "gp")
        for f in files:
            with open(os.path.join(gdir, f)) as fh:
                gold[f] = fh.read()
    finally:
        H._run(f"git -C {H.BARE} worktree remove --force {gdir}", check=False)
        import shutil
        shutil.rmtree(sib, ignore_errors=True)

    texts = []
    # 1. reproduce-before-patch: run the failing tests first (discharge)
    texts.append("First reproduce the failure.\nAction: run_tests {}")
    # 2. read each file we will change
    for f in files:
        texts.append("Read the buggy file.\nAction: read_file "
                     + _json.dumps({"path": f}))
    # 3. write the gold contents (the patch)
    for f in files:
        texts.append("Apply the fix.\nAction: write_file "
                     + _json.dumps({"path": f, "content": gold[f]}))
    # 4. verify (ran_tests discharge) then submit
    texts.append("Verify the fix.\nAction: run_tests {}")
    texts.append("Done.\nAction: submit {}")
    return texts


def run_fake():
    from transformers import AutoTokenizer
    from rlvp.rollout import set_template
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)

    insts = load_clean_instances(n=N_INSTS)
    print(f"FAKE (scripted gold) smoke: {len(insts)} instance(s), model tok={MODEL}\n", flush=True)
    ok = 0
    for k in range(N_EPISODES):
        inst = insts[k % len(insts)]
        iid = inst["instance_id"]
        print(f"=== EPISODE {k+1}/{N_EPISODES}  ({iid}) ===", flush=True)
        # build gold script (needs a worktree to read final file contents)
        t0 = time.time()
        wt = SweWorktree(inst, os.path.join("/tmp/swe_probe", iid)).setup()
        if not wt.setup_ok:
            print("  setup failed:", wt.setup_error, "\n", flush=True)
            wt.close()
            continue
        wt.close()
        scripts = gold_texts(inst, wt)
        gen = make_fake_gen(tok, scripts)
        ep = run_swe_episode(inst, gen, tok, rule_mode="structural",
                             max_steps=len(scripts) + 2, verbose=True)
        report(ep, iid, time.time() - t0)
        if ep is not None and ep.env.success:
            ok += 1
    print(f"FAKE GATE: {ok}/{N_EPISODES} gold scripts earned reward=1.0 "
          f"(oracle proven)" if ok else
          f"FAKE GATE: 0/{N_EPISODES} earned reward -- oracle/edit path broken",
          flush=True)


def run_real():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rlvp.rollout import set_template
    from rlvp.swe_adapter import GenServer
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"loading {MODEL} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    gen_srv = GenServer(model, tok, temperature=1.0, max_new_tokens=400, max_batch=4)

    insts = load_clean_instances(n=N_INSTS)
    print(f"REAL smoke: {len(insts)} instance(s), model={MODEL}\n", flush=True)
    try:
        for k in range(N_EPISODES):
            inst = insts[k % len(insts)]
            iid = inst["instance_id"]
            print(f"=== EPISODE {k+1}/{N_EPISODES}  ({iid}) ===", flush=True)
            t0 = time.time()
            ep = run_swe_episode(inst, gen_srv.generate, tok,
                                 rule_mode="structural", max_steps=12, verbose=True)
            report(ep, iid, time.time() - t0)
    finally:
        gen_srv.stop()
    print("REAL smoke done.", flush=True)


if __name__ == "__main__":
    if FAKE:
        run_fake()
    else:
        run_real()
