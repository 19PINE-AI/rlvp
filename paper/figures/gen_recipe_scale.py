"""fig_recipe_scale: the P0 aligned-potential recipe matrix on real theorems (miniF2F algebra).
Panel (a) 4B matched-optimizer (Muon): aligned vs outcome, 5 seeds each.
Panel (b) 30B: aligned (Muon) vs outcome (Muon, 3/5 diverge) vs outcome (AdamW, stable baseline).
Reads results/run_*/train_log.jsonl directly; no hand-typed numbers.
NOTE: regenerate when 30B AdamW-outcome seeds 12/13/14 finish (currently 2/5 seeds).
Matches the house style of gen_penalty_figures.py."""
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14.5, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "legend.fontsize": 11, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "font.family": "sans-serif", "axes.linewidth": 0.9,
})
BLUE, ORANGE, RED = "#4A90D9", "#F5A623", "#D94A4A"
HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "..", "results")


def runs(pattern):
    out = []
    for d in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        rows = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"))]
        succ = [r["succ"] for r in rows if "succ" in r]
        diverged = any(r.get("DIVERGED") for r in rows)
        if len(succ) < 30 and not diverged:
            continue  # still training (e.g. 30B AdamW seeds 12-14)
        out.append((succ, diverged))
    return out


def draw_arm(ax, arm, color, label, ls="-"):
    n_div = sum(1 for _, d in arm if d)
    for succ, diverged in arm:
        ax.plot(range(1, len(succ) + 1), succ, color=color, alpha=0.35, lw=1.0, ls=ls)
        if diverged:
            ax.plot(len(succ), succ[-1], marker="x", color=color, ms=8, mew=2.2, zorder=5)
    stable = [succ for succ, d in arm if not d]
    full = max(len(s) for s in stable)
    mat = np.full((len(stable), full), np.nan)
    for i, succ in enumerate(stable):
        mat[i, :len(succ)] = succ
    mean = np.nanmean(mat, axis=0)
    suffix = f", {n_div}/{len(arm)} diverge" if n_div else f", {len(arm)}/{len(arm)} stable"
    ax.plot(range(1, full + 1), mean, color=color, lw=2.6, ls=ls, label=label + suffix)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)

ax = axes[0]
draw_arm(ax, runs("run_lean_p0_4b_aligned_s*"), BLUE, "aligned potential (Muon)")
draw_arm(ax, runs("run_lean_p0_4b_outcome_s*"), RED, "outcome-only (Muon)")
ax.set_title("(a) Qwen3-4B, matched optimizer (5 seeds each)")

ax = axes[1]
draw_arm(ax, runs("run_lean_potential_s*"), BLUE, "aligned potential (Muon)")
draw_arm(ax, runs("run_lean_outcome_s*"), RED, "outcome-only (Muon)")
draw_arm(ax, runs("run_lean_p0_30b_outcomeAdamw_s*"), ORANGE,
         "outcome-only (AdamW)", ls="--")
ax.set_title("(b) Qwen3-30B-A3B (5 seeds per arm)")

for ax in axes:
    ax.axhline(0.9, color="#9CA3AF", lw=0.9, ls=":")
    ax.set_xlabel("training iteration")
    ax.set_ylim(-0.03, 1.05)
    # legend below the axes so it never overlaps the curves; single column
    # keeps each legend well inside its own panel's width
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=1, frameon=False, handlelength=1.8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("training success")
axes[0].text(29.5, 0.92, "0.9 threshold", ha="right", va="bottom",
             fontsize=10.5, color="#6B7280")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_recipe_scale.pdf"))
print("wrote fig_recipe_scale.pdf")
