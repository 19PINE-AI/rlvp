"""Page-1 dual headline: the two deployment needs the paper delivers.
 (a) Penalize the path -> reach the violation-free (deployable) axis GRPO cannot.
 (b) Reward verifiable progress where reachable -> reach competence in fewer rollouts.
Reads results/run_rvp_{recipe,outcome}_s* (eval.clean) and
results/run_lean_p0_4b_{aligned,outcome}_s* (succ). Writes fig_headline.pdf."""
import glob
import json
import os
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06, "font.family": "sans-serif", "axes.linewidth": 0.9,
})
GREEN, GRAY, BLUE, RED, DARK = "#50B86C", "#9CA3AF", "#4A90D9", "#D94A4A", "#4B5563"
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "..", "results")


def rvp_clean(arm, gen_batch=48):
    """(episodes, clean-rate mean, std) over seeds, from held-out eval."""
    per = {}
    for d in sorted(glob.glob(os.path.join(RES, f"run_rvp_{arm}_s*"))):
        f = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(f):
            continue
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if not (isinstance(r, dict) and r.get("eval") and r.get("iter")):
                continue
            doms = [k for k in ("fileops", "csops") if k in r["eval"]]
            if doms:
                per.setdefault(r["iter"], []).append(
                    st.mean(r["eval"][k]["clean"] for k in doms))
    its = sorted(per)
    return ([i * gen_batch for i in its], [st.mean(per[i]) for i in its],
            [st.pstdev(per[i]) for i in its])


def lean_succ(arm):
    """Per-iter success mean over stable 4B seeds."""
    curves = []
    for d in sorted(glob.glob(os.path.join(RES, f"run_lean_p0_4b_{arm}_s*"))):
        if "Anneal" in d:
            continue
        rows = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"))]
        c = [r["succ"] for r in rows if "succ" in r]
        if len(c) >= 30 and not any(r.get("DIVERGED") for r in rows):
            curves.append(c)
    n = min(len(c) for c in curves)
    mat = np.array([c[:n] for c in curves])
    return np.arange(1, n + 1), mat.mean(0), mat.std(0)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.0))

# ---- (a) Penalize the path: violation-free episodes ----
eo, co, ceo = rvp_clean("outcome")
er, cr, cer = rvp_clean("recipe")
axL.plot(eo, co, "-o", color=GRAY, lw=2.2, ms=5, label="outcome-only (GRPO)")
axL.fill_between(eo, np.array(co) - np.array(ceo), np.array(co) + np.array(ceo), color=GRAY, alpha=0.15)
axL.plot(er, cr, "-o", color=GREEN, lw=2.4, ms=5, label="reward outcome, penalize path")
axL.fill_between(er, np.array(cr) - np.array(cer), np.array(cr) + np.array(cer), color=GREEN, alpha=0.18)
axL.axhline(1.0, ls=":", lw=0.8, color=GRAY)
axL.set_ylim(-0.04, 1.08); axL.set_xlim(left=0)
axL.set_xlabel("episodes generated"); axL.set_ylabel("violation-free rate")
axL.set_title("(a) Penalize the path", fontweight="bold")
axL.annotate("the deployable axis\noutcome-only never reaches", xy=(eo[-1], co[-1]),
             xytext=(eo[len(eo)//2], 0.30), fontsize=10.5, color=GRAY, ha="center",
             arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))
axL.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=11)
axL.spines[["top", "right"]].set_visible(False)

# ---- (b) Verifiable progress: fewer rollouts to competence ----
it, ma, sa = lean_succ("aligned")
_, mo, so = lean_succ("outcome")
axR.plot(it, mo, "-", color=GRAY, lw=2.2, label="outcome-only")
axR.fill_between(it, mo - so, mo + so, color=GRAY, alpha=0.15)
axR.plot(it, ma, "-", color=BLUE, lw=2.4, label="+ verifiable progress potential")
axR.fill_between(it, ma - sa, ma + sa, color=BLUE, alpha=0.18)
axR.axhline(0.9, ls=":", lw=0.8, color=GRAY)
axR.set_ylim(-0.04, 1.08); axR.set_xlim(left=1)
axR.set_xlabel("training iterations (each = costly rollouts)")
axR.set_ylabel("task success")
axR.set_title("(b) Reward progress where reachable", fontweight="bold")
axR.annotate("competence in\nfewer rollouts", xy=(4.4, 0.9), xytext=(9, 0.42),
             fontsize=10.5, color=DARK, arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))
axR.legend(frameon=False, loc="lower right", fontsize=11)
axR.spines[["top", "right"]].set_visible(False)

fig.suptitle("Two verifiable path signals for agents that learn from costly, real-world rollouts",
             fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(HERE, "fig_headline.pdf"))
print("wrote fig_headline.pdf (dual)")
