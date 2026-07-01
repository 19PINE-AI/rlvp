"""One RLVP compliance/harm run on the DIVERSE FileOps+CSOps task pool (outcome-neutral
rules). Arms validate the penalty-design criteria. Usage: recipe_run.py <arm> <seed>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvp.grpo import TrainConfig, train

arm = sys.argv[1]
seed = int(sys.argv[2])
common = dict(model_name="Qwen/Qwen3-4B", iters=60, gen_batch=48, data_seed=seed)

if arm == "outcome":                    # baseline: solves task, but violates (clean low)
    cfg = TrainConfig(credit="outcome", out_dir=f"results/run_rvp_outcome_s{seed}", **common)
elif arm == "penalty_only":             # rule 2 test: pure penalty, NO discharge, NO mix
    cfg = TrainConfig(credit="c3", lam=0.5, beta=0.0, mix_scripted=False,
                      script_scalar=False, anneal_at=40,
                      out_dir=f"results/run_rvp_penonly_s{seed}", **common)
elif arm == "nomix":                    # rule 4 test: penalty+discharge, NO reachability seed
    cfg = TrainConfig(credit="c3", lam=0.5, beta=0.5, mix_scripted=False,
                      script_scalar=False, anneal_at=40,
                      out_dir=f"results/run_rvp_nomix_s{seed}", **common)
elif arm == "recipe":                   # full: penalty + discharge + mixing + anneal
    cfg = TrainConfig(credit="c3", lam=0.5, beta=0.5, mix_scripted=True,
                      script_scalar=False, anneal_at=40,
                      out_dir=f"results/run_rvp_recipe_s{seed}", **common)
else:
    raise SystemExit(f"unknown arm {arm}")
print(f"ARM {arm} seed {seed}: {cfg.out_dir}", flush=True)
train(cfg)
print("DONE", flush=True)
