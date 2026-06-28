"""Phase diagram of WHEN dense process rewards help (see PHASE_DIAGRAM_DESIGN.md).

Two failure modes bound the "helps" region: low outcome-sparsity (process reward unneeded)
and low reachability Var_G(Phi)=0 (process reward vacuous). The variance proposition
(paper/theorem.tex) predicts benefit tracks the within-group potential variance.

Grid: sparsity = n_stages x reachability = model size. Per cell:
  probe : G base rollouts -> within-group Var_G(Phi)  (Phi = #stages completed), NO training.
  grid  : train dense(c3) vs outcome short arms -> benefit = final_succ(c3)-final_succ(outcome).

Usage:
  phase_diagram.py probe  --model Qwen/Qwen3-1.7B --nstages 4 [--G 8 --tasks 8]
  phase_diagram.py grid   --model Qwen/Qwen3-1.7B --nstages 4 [--iters 12 --seed 7]
"""
import json, sys, time, statistics as st
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlvp.grpo import TrainConfig, build_advantages, update_policy        # noqa: E402
from rlvp.rollout import run_episodes, set_template, start_episode         # noqa: E402
from rlvp.envs.fileops import make_chain_potential_env                     # noqa: E402

A = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[2:]}
def arg(k, d, cast=str):
    for i, a in enumerate(sys.argv):
        if a == "--" + k: return cast(sys.argv[i + 1])
    return d

MODE   = sys.argv[1] if len(sys.argv) > 1 else "probe"
MODEL  = arg("model", "Qwen/Qwen3-1.7B")
NSTAGES= arg("nstages", 4, int)
G      = arg("G", 8, int)
TASKS  = arg("tasks", 8, int)
ITERS  = arg("iters", 12, int)
SEED   = arg("seed", 7, int)
OUTD   = ROOT / "results" / "phase_diagram"; OUTD.mkdir(parents=True, exist_ok=True)
TAG    = f"{MODEL.split('/')[-1]}_n{NSTAGES}"


def _phi(ep):
    """Potential = fraction of stages completed (verifiable). Uses the env's
    stage-progress discharges; success => all stages."""
    if ep.env is None:
        return 0.0
    if getattr(ep.env, "success", False):
        return 1.0
    n = len({t for t, r in ep.env.discharges if r == "stage_progress"})
    return min(1.0, n / NSTAGES)


def _model_tok():
    set_template(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    return m, tok


def mode_probe():
    """Measure within-group Var_G(Phi) on the BASE policy (no training) -> the
    proposition's predictor for whether a dense reward can help this cell."""
    import random
    m, tok = _model_tok(); m.eval()
    rng = random.Random(SEED)
    group_vars, base_succ = [], []
    for _ in range(TASKS):
        s = rng.randint(0, 10**6)
        grp = [start_episode(tok, make_chain_potential_env(s, NSTAGES, granularity="fine"))
               for _ in range(G)]
        run_episodes(m, tok, grp, temperature=1.0, top_p=1.0, gen_batch=32,
                     max_new_tokens=160, max_episode_tokens=3500)
        phis = [_phi(e) for e in grp]
        group_vars.append(st.pvariance(phis))
        base_succ += [1.0 if getattr(e.env, "success", False) else 0.0 for e in grp]
    rec = {"model": MODEL, "n_stages": NSTAGES, "tasks": TASKS, "G": G,
           "mean_var_phi": round(sum(group_vars) / len(group_vars), 4),
           "base_succ": round(sum(base_succ) / len(base_succ), 4),
           "sparsity": round(1 - sum(base_succ) / len(base_succ), 4)}
    (OUTD / f"probe_{TAG}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec), flush=True)


def _train(credit):
    """Short training arm; returns final-3 success."""
    import random
    m, tok = _model_tok()
    m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.eval()
    from rlvp.muon import Muon
    opt = Muon([p for p in m.parameters() if p.requires_grad], lr=5e-4, momentum=0.95)
    cfg = TrainConfig(credit=credit, beta=0.5, lam=0.0, max_episode_tokens=3500)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup)
    rng = random.Random(SEED); succ_hist = []
    for it in range(1, ITERS + 1):
        groups, eps = [], []
        for _ in range(4):
            s = rng.randint(0, 10**6)
            grp = [start_episode(tok, make_chain_potential_env(s, NSTAGES, granularity="fine"))
                   for _ in range(G)]
            groups.append(grp); eps += grp
        m.config.use_cache = True
        run_episodes(m, tok, eps, temperature=1.0, top_p=1.0, gen_batch=32,
                     max_new_tokens=160, max_episode_tokens=cfg.max_episode_tokens)
        succ_hist.append(sum(e.env.success for e in eps) / len(eps))
        m.config.use_cache = False
        update_policy(m, tok, build_advantages(groups, cfg), cfg, opt, sched)
    del m; torch.cuda.empty_cache()
    return sum(succ_hist[-3:]) / min(3, len(succ_hist))


def mode_grid():
    dense = _train("c3"); outcome = _train("outcome")
    rec = {"model": MODEL, "n_stages": NSTAGES, "iters": ITERS, "seed": SEED,
           "succ_dense": round(dense, 4), "succ_outcome": round(outcome, 4),
           "benefit": round(dense - outcome, 4)}
    with open(OUTD / "grid.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    {"probe": mode_probe, "grid": mode_grid}[MODE]()
