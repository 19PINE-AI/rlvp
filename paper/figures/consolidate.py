"""Consolidate all RLVP run/eval logs into one clean paper_data.json for figures.
Reads results/run_*/train_log.jsonl and results/*.json. No hand-typed numbers."""
import json
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parent / "paper_data.json"


def rows(run):
    f = R / f"run_{run}" / "train_log.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def cfg(run):
    f = R / f"run_{run}" / "config.json"
    return json.loads(f.read_text()) if f.exists() else {}


def n_live(run):
    c = cfg(run)
    return c.get("tasks_per_iter", 8) * (c.get("group_size", 8) - (1 if c.get("mix_scripted") else 0))


def cum_gen(run):
    rs, c, acc = rows(run), [], 0
    for r in rs:
        acc += r["train"].get("episodes_generated", n_live(run))
        c.append(acc)
    return c


def eval_curve(run, dom):
    return [(r["iter"], r["eval"][dom]["success"]) for r in rows(run)
            if "eval" in r and dom in r["eval"]]


def train_succ(run):
    return [r["train"]["success"] for r in rows(run)]


def dead(run):
    return sum(1 for r in rows(run) if r.get("upd_s", 1) == 0.0)


def eps_to(run, thr=0.5, w=3):
    s, g = train_succ(run), cum_gen(run)
    for i in range(len(s)):
        lo = max(0, i - w + 1)
        if sum(s[lo:i + 1]) / (i + 1 - lo) >= thr:
            return g[i]
    return None


def final_eval(run, dom):
    ec = eval_curve(run, dom)
    return ec[-1][1] if ec else None


def best_eval(run, dom):
    ec = eval_curve(run, dom)
    return max((s for _, s in ec), default=None)


def oversample(run):
    g = cum_gen(run)[-1] if rows(run) else 0
    u = len(rows(run)) * n_live(run)
    return round(g / max(u, 1), 2)


D = {}

# --- chain4 5-way efficiency, with per-seed for the headline trio ---
D["chain4"] = {}
for run in ["flag_outcome", "flag_dapo", "flag_rlvp", "flag_gigpo", "flag_steptool"]:
    D["chain4"][run] = {
        "eval_curve": eval_curve(run, "chain4"),
        "gen_curve": cum_gen(run),
        "train_succ": train_succ(run),
        "eps50": eps_to(run), "final": final_eval(run, "chain4"),
        "best": best_eval(run, "chain4"), "dead": dead(run),
        "oversample": oversample(run), "n_iters": len(rows(run)),
    }

# headline trio seeds -> mean/std of eps50, final, dead
D["seeds"] = {}
trio = {"GRPO": ["flag_outcome", "flag_outcome_s11", "flag_outcome_s12"],
        "DAPO": ["flag_dapo", "flag_dapo_s11", "flag_dapo_s12"],
        "RLVP": ["flag_rlvp", "flag_rlvp_s11", "flag_rlvp_s12"]}
for name, runs in trio.items():
    e = [eps_to(r) for r in runs if eps_to(r)]
    fi = [final_eval(r, "chain4") for r in runs if final_eval(r, "chain4") is not None]
    dd = [dead(r) for r in runs]
    ov = [oversample(r) for r in runs]
    D["seeds"][name] = {
        "eps50_mean": st.mean(e), "eps50_std": st.pstdev(e) if len(e) > 1 else 0,
        "eps50_all": e,
        "final_mean": st.mean(fi), "final_std": st.pstdev(fi) if len(fi) > 1 else 0,
        "dead_mean": st.mean(dd), "oversample_mean": st.mean(ov),
    }

# --- paired dead-iteration ---
D["paired_dead"] = json.loads((R / "paired_dead.json").read_text())

# --- component ablation (chain4) ---
D["ablation"] = {}
abl = {"clean (penalty+fulfillment+anneal)": "abl_nomix",
       "+ mixing (full)": "flag_rlvp",
       "- fulfillment": "abl_nodisch",
       "- token channel (scalar-fold)": "abl_scalaronly",
       "- anneal": "abl_noanneal",
       "+ step-cost": "rlvp_sc3"}
for label, run in abl.items():
    if rows(run):
        D["ablation"][label] = {"eps50": eps_to(run), "final": final_eval(run, "chain4"),
                                "dead": dead(run)}

# --- fairness control: process vs demos vs outcome ---
D["fairness"] = {
    "outcome-only": eps_to("flag_outcome"),
    "outcome + demos": eps_to("ctrl_outmix"),
    "process channel (clean RLVP)": eps_to("abl_nomix"),
}

# --- capstone: auto vs hand rules ---
D["capstone"] = {
    "auto-derived rules": {"eps50": eps_to("auto_rlvp"), "final": final_eval("auto_rlvp", "chain4")},
    "hand-written rules": {"eps50": eps_to("hand_rlvp"), "final": final_eval("hand_rlvp", "chain4")},
    "outcome-only": {"eps50": eps_to("flag_outcome"), "final": final_eval("flag_outcome", "chain4")},
}

# --- gated ceiling (non-saturating) ---
D["gated"] = {
    "base": json.loads((R / "calib_gated.json").read_text()),
    "outcome": {"final": final_eval("gated_outcome", "gated"), "curve": eval_curve("gated_outcome", "gated"), "dead": dead("gated_outcome")},
    "dapo": {"final": final_eval("gated_dapo", "gated"), "dead": dead("gated_dapo"), "oversample": oversample("gated_dapo")},
    "clean_rlvp": {"final": final_eval("gated_rlvp", "gated"), "curve": eval_curve("gated_rlvp", "gated"), "dead": dead("gated_rlvp")},
    "prompted": {"final": final_eval("gated_outcome_prompt", "gated")},
    "rlvp_mix": {"curve": eval_curve("gated_rlvp_mix", "gated"),
                 "train": train_succ("gated_rlvp_mix"),
                 "final": final_eval("gated_rlvp_mix", "gated")},
}

# --- chain6 (saturates; DAPO cost) ---
D["chain6"] = {
    "outcome": {"final": final_eval("t2_outcome", "chain6"), "gen": cum_gen("t2_outcome")[-1] if rows("t2_outcome") else None},
    "dapo": {"final": final_eval("t2_dapo", "chain6"), "gen": cum_gen("t2_dapo")[-1] if rows("t2_dapo") else None, "oversample": oversample("t2_dapo")},
    "rlvp_clean": {"curve": eval_curve("t2_rlvp_clean", "chain6")},
}

# --- tau2 real benchmark ---
def tev(tag):
    f = R / f"tau2_eval_tau2_{tag}_nopolicy.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"reward": d["mean_reward"], "viol": d["viol_per_ep"]}
D["tau2"] = {
    "base": tev("base"), "outcome": tev("outcome"),
    "rlvp": tev("rlvp"), "rlvp_anneal": tev("rlvp_anneal"),
    "aligned": tev("aligned"),
    "outcome_train": [r["reward"] for r in rows("tau2_outcome")],
    "outcome_viol": [r.get("viol_per_ep", 0) for r in rows("tau2_outcome")],
    "rlvp_train": [r["reward"] for r in rows("tau2_rlvp")],
    "rlvp_viol": [r.get("viol_per_ep", 0) for r in rows("tau2_rlvp")],
    "aligned_train": [r["reward"] for r in rows("tau2_aligned")],
    "aligned_viol": [r.get("viol_per_ep", 0) for r in rows("tau2_aligned")],
    "semantic_train": [r["reward"] for r in rows("tau2_semantic")],
    # four-tier coverage gradient, measured on training reward (last-5 mean;
    # eval is too small-k to separate the middle tiers reliably)
    "tiers_train": {
        "outcome-only\nGRPO": st.mean([r["reward"] for r in rows("tau2_outcome")][-5:]),
        "RLVP: generic\n(orthogonal)": st.mean([r["reward"] for r in rows("tau2_rlvp")][-5:]),
        "RLVP: aligned\nprocedural": st.mean([r["reward"] for r in rows("tau2_aligned")][-5:]),
        "RLVP: aligned\nsemantic": st.mean([r["reward"] for r in rows("tau2_semantic")][-5:]),
    },
    "tiers_peak": {
        "outcome-only\nGRPO": max(r["reward"] for r in rows("tau2_outcome")),
        "RLVP: generic\n(orthogonal)": max(r["reward"] for r in rows("tau2_rlvp")),
        "RLVP: aligned\nprocedural": max(r["reward"] for r in rows("tau2_aligned")),
        "RLVP: aligned\nsemantic": max(r["reward"] for r in rows("tau2_semantic")),
    },
}

# --- phase0 all-fail mechanism (chain difficulty calibration) ---
D["calib"] = json.loads((R / "horizon_calib.json").read_text())

# --- model scale ---
D["scale"] = {}
for tag, run in [("1.7B", "c3mix_1p7b"), ("8B-LoRA", "c3mix_8b_lora")]:
    f = R / f"eval_{run}_norules.json"
    if f.exists():
        d = json.loads(f.read_text())
        D["scale"][tag] = {dm: d[dm]["perfect^k"] for dm in ("fileops", "csops")}

# --- self-critique vs rules ablation (SELFCRITIC.md) ---
def srows(name):
    """Read results/<name>/train_log.jsonl (arbitrary dir, not just run_*)."""
    f = R / name / "train_log.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def _late(vals, k=4):
    vals = [v for v in vals if v is not None]
    return st.mean(vals[-k:]) if vals else None


def _ms(tag, key, k=4):  # late-mean per seed, across seeds 11/22/33 (csops, nested 'train')
    out = []
    for s in (11, 22, 33):
        v = _late([r["train"].get(key) for r in srows(f"exp_sc_train_{tag}_s{s}_csops")], k)
        if v is not None:
            out.append(v)
    return out


def _msstat(tag, key):
    xs = _ms(tag, key)
    return {"mean": st.mean(xs) if xs else None,
            "std": (st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else None,
            "seeds": [round(x, 3) for x in xs], "n": len(xs)}


def _screport(slug):
    f = R / "exp_selfcritic" / slug / "report.json"
    return json.loads(f.read_text()) if f.exists() else None


def _rule_recall(rep, rule):
    for dd in (rep or {}).get("domains", {}).values():
        pr = dd.get("blind", {}).get("per_rule_recall", {})
        if rule in pr:
            return pr[rule]["recall"]
    return None


sc = {"multiseed_csops": {}, "scale": {}, "tau2_offline": {}, "tau2_train": {}}
# Driver A: penalty-only rule vs live vs frozen self-critic (all-fail regime, 3 seeds)
for label, tag in [("rule", "c2nodis"), ("live", "llmcritic"), ("frozen", "llmcriticfrozen")]:
    e = {"viol": _msstat(tag, "viol_per_episode"), "succ": _msstat(tag, "success")}
    if label in ("live", "frozen"):
        cp, cr = _ms(tag, "critic_precision"), _ms(tag, "critic_recall")
        e["critic_P"] = round(st.mean(cp), 3) if cp else None
        e["critic_R"] = round(st.mean(cr), 3) if cr else None
    sc["multiseed_csops"][label] = e
# cross-model scale sweep (blind detection): overall recall + stateful per-rule recall
for size, slug in [("1.7B", "Qwen3-1_7B"), ("4B", "Qwen3-4B"), ("8B", "Qwen3-8B")]:
    rep = _screport(slug)
    if rep:
        sc["scale"][size] = {
            "overall_blind_recall": rep["overall"]["blind"]["turn_recall"],
            "blind_write": _rule_recall(rep, "blind_write"),
            "untested_submit": _rule_recall(rep, "untested_submit"),
        }
# tau2 cell-C offline: intent-miss recall + failure-prediction F1 (self-critic vs rules)
_cc = [json.loads(p.read_text()) for p in sorted((R / "tau2_cellc").glob("run*/report.json"))]
if _cc:
    imr = [r["INTENT_MISS_critic_recall"] for r in _cc]
    sf1 = [r["failure_prediction"]["self_critic (P,R,F1)"][2] for r in _cc]
    rf1 = [r["failure_prediction"]["semantic_rule (P,R,F1)"][2] for r in _cc]
    msd = lambda x: {"mean": st.mean(x), "std": st.pstdev(x) if len(x) > 1 else 0.0, "n": len(x)}
    sc["tau2_offline"] = {"intent_recall": msd(imr), "selfcritic_F1": msd(sf1),
                          "semantic_F1": msd(rf1)}
# tau2 cell-C training: early->late reward for outcome / semantic-c3 / llmcritic
for label, run in [("outcome", "tau2_cellc_outcome"), ("semantic", "tau2_cellc_sem"),
                   ("llmcritic", "tau2_cellc_llmcritic")]:
    rw = [r.get("reward") for r in rows(run)]
    if rw:
        sc["tau2_train"][label] = {"early": _late(rw[:4]) if len(rw) >= 1 else None,
                                   "late": _late(rw), "curve": rw}
D["selfcritique"] = sc

OUT.write_text(json.dumps(D, indent=2, default=str))
print("wrote", OUT)
print("selfcritique multiseed viol:",
      {k: round(v["viol"]["mean"], 2) for k, v in D["selfcritique"]["multiseed_csops"].items() if v["viol"]["mean"] is not None})
print("selfcritique tau2 train late:",
      {k: round(v["late"], 2) for k, v in D["selfcritique"]["tau2_train"].items()})
print("selfcritique tau2 offline F1 self/sem:",
      round(D["selfcritique"]["tau2_offline"].get("selfcritic_F1", {}).get("mean", 0), 2),
      round(D["selfcritique"]["tau2_offline"].get("semantic_F1", {}).get("mean", 0), 2))
# sanity print
print("chain4 RLVP eps50:", D["chain4"]["flag_rlvp"]["eps50"])
print("seeds:", {k: (round(v["eps50_mean"]), round(v["eps50_std"])) for k, v in D["seeds"].items()})
print("gated rlvp_mix final:", D["gated"]["rlvp_mix"]["final"])
print("tau2:", D["tau2"]["outcome"], D["tau2"]["rlvp"])
