"""New flagship figures for the verifiable-potential reframe:
  fig_criterion_map      - the decision criterion as a flow, with the 5 settings as leaves
  fig_ungameability_sweep- 30B Lean sweep (real seed-7 data): penalty kills, gating rescues
  fig_ec_reachability    - E-C: finer potential exists structurally but is unreachable (phi=0)
Reads real data from ../../results.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE, GREEN, ORANGE, RED = "#4A90D9", "#50B86C", "#F5A623", "#D94A4A"
PURPLE, GRAY, DARK = "#8B5CF6", "#9CA3AF", "#4B5563"
LBLUE, LGREEN, LORANGE, LRED, LPUR = "#DBEAFE", "#D1FAE5", "#FEF3C7", "#FEE2E2", "#EDE9FE"
RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def _load(arm):
    p = os.path.join(RES, f"run_swp_{arm}", "train_log.jsonl")
    return [json.loads(l)["succ"] for l in open(p)]


# ---------------------------------------------------------------------------
# Figure: the criterion as a decision flow, settings as leaves
# ---------------------------------------------------------------------------
def fig_criterion_map():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, fc, ec, text, fs=9, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.06", facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=DARK)

    def arrow(x1, y1, x2, y2, label="", color=DARK, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=13, color=color, linewidth=1.3, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                    va="center", fontsize=8.5, color=color, fontweight="bold")

    # three gating questions, left to right
    box(0.2, 2.2, 2.05, 1.4, "white", DARK,
        "Domain", fs=11, bold=True)
    q1 = (2.9, 2.0, 2.25, 1.8)
    box(*q1, LBLUE, BLUE,
        "Is there a\nVERIFIABLE potential\nfiner than the\noutcome?", fs=8.5)
    q2 = (6.0, 2.0, 2.25, 1.8)
    box(*q2, LBLUE, BLUE,
        "Is it\nUN-GAMEABLE?\n(cheapest maximizer\nmust solve task)", fs=8.5)
    q3 = (9.1, 2.0, 2.25, 1.8)
    box(*q3, LBLUE, BLUE,
        "Are intermediate\nvalues\nREACHABLE?\n(within-group var > 0)", fs=8.5)

    arrow(2.25, 2.9, 2.9, 2.9)
    arrow(q1[0] + q1[2], 2.9, q2[0], 2.9, "yes", GREEN, ly=0.22)
    arrow(q2[0] + q2[2], 2.9, q3[0], 2.9, "yes", GREEN, ly=0.22)

    # NO branches drop down to failure leaves
    box(2.9, 0.15, 2.25, 1.2, LORANGE, ORANGE,
        "No usable signal:\noutcome-only ceiling\n(intent)", fs=8)
    arrow(q1[0] + q1[2] / 2, 2.0, q1[0] + q1[2] / 2, 1.35, "no", ORANGE, lx=0.26)
    box(6.0, 0.15, 2.25, 1.2, LRED, RED,
        "COMPLIANCE\nCOLLAPSE\n(misaligned penalty)", fs=8)
    arrow(q2[0] + q2[2] / 2, 2.0, q2[0] + q2[2] / 2, 1.35, "no", RED, lx=0.26)
    box(9.1, 0.15, 2.25, 1.2, LRED, RED,
        "VACUOUS signal:\nzero gradient\n(unreachable)", fs=8)
    arrow(q3[0] + q3[2] / 2, 2.0, q3[0] + q3[2] / 2, 1.35, "no", RED, lx=0.26)

    # YES terminal
    box(9.1, 4.05, 2.7, 1.0, LGREEN, GREEN,
        "DENSE GRADIENT\nfrom failed episodes\n+ harm bounded", fs=8.5, bold=True)
    arrow(q3[0] + q3[2] / 2, q3[1] + q3[3], 10.45, 4.05, "yes", GREEN, lx=0.3)

    # setting labels under leaves
    def tag(x, y, t, c):
        ax.text(x, y, t, ha="center", va="center", fontsize=7.4, style="italic",
                color=c)
    tag(4.02, -0.15, r"$\tau^2$ customer service", ORANGE)
    tag(7.12, -0.15, "Lean penalty arms", RED)
    tag(10.22, -0.15, "SWE software repair", RED)
    tag(10.45, 3.78, "Lean progress; sysadmin harm", GREEN)
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_criterion_map.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: the un-gameability sweep (real seed-7 data)
# ---------------------------------------------------------------------------
def fig_ungameability_sweep():
    arms = [
        ("validgated", "outcome-gated discharge", BLUE, "-"),
        ("aligned",    "aligned discharge",       GREEN, "-"),
        ("valid",      "gameable discharge",      ORANGE, "-"),
        ("structural", "discharge + penalty",     RED, "--"),
        ("noerror",    "penalty only",            PURPLE, "--"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    # left: trajectories
    for arm, lab, c, ls in arms:
        s = _load(arm)
        axL.plot(range(1, len(s) + 1), s, ls, color=c, linewidth=1.8,
                 marker="o", markersize=3, label=lab, alpha=0.9)
    axL.set_xlabel("training iteration")
    axL.set_ylabel("task success")
    axL.set_title("(a) Success trajectory per signal (Qwen3-30B, miniF2F)", fontsize=10.5)
    axL.set_ylim(-0.03, 0.72); axL.grid(alpha=0.25)
    axL.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=1)
    axL.axhspan(-0.03, 0.06, color=RED, alpha=0.07)
    axL.text(7, 0.015, "dead zone", color=RED, fontsize=7.5, ha="center", style="italic")

    # right: terminal (last3) bars, colored by admissibility
    names, vals, cols, dead = [], [], [], []
    for arm, lab, c, ls in arms:
        s = _load(arm)
        names.append(lab.replace(" ", "\n", 1)); vals.append(np.mean(s[-3:])); cols.append(c)
        dead.append(np.mean(s[-3:]) <= 0.05)
    y = np.arange(len(names))[::-1]
    axR.barh(y, vals, color=cols, alpha=0.9, edgecolor="white")
    for yi, v, d in zip(y, vals, dead):
        axR.text(v + 0.012, yi, ("DEAD" if d else f"{v:.2f}"),
                 va="center", fontsize=8.5,
                 fontweight="bold", color=(RED if d else DARK))
    axR.set_yticks(y); axR.set_yticklabels(names, fontsize=8)
    axR.set_xlabel("terminal success (last 3 iters)")
    axR.set_xlim(0, 0.40)
    axR.set_title("(b) Terminal state: penalty kills,\ngating rescues", fontsize=10.5)
    axR.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_ungameability_sweep.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: E-C reachability
# ---------------------------------------------------------------------------
def fig_ec_reachability():
    struct = json.load(open(os.path.join(RES, "ec_f2p", "structural.json")))
    roll = [json.loads(l) for l in open(os.path.join(RES, "ec_f2p", "rollout.jsonl"))]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    # left: structural split -- where a finer potential exists at all
    axL.bar([0], [struct["single"]], color=GRAY, alpha=0.85, width=0.6,
            label="single-F2P: $\\Phi\\equiv$ outcome")
    axL.bar([1], [struct["multi"]], color=PURPLE, alpha=0.85, width=0.6,
            label="multi-F2P: finer $\\Phi$ exists")
    for x, v in [(0, struct["single"]), (1, struct["multi"])]:
        axL.text(x, v + 0.6, str(v), ha="center", fontsize=11, fontweight="bold", color=DARK)
    axL.set_xticks([0, 1]); axL.set_xticklabels(["single-F2P\n(32/48)", "multi-F2P\n(16/48)"])
    axL.set_ylabel("# instances")
    axL.set_title("(a) A finer potential is structurally\nrare: 2/3 are all-or-nothing", fontsize=10.5)
    axL.set_ylim(0, 38); axL.legend(fontsize=8, loc="upper right")

    # right: phi reachability -- every rollout sits at 0, the reachable band is empty
    multi = [r for r in roll if r["kind"] == "multi"]
    single = [r for r in roll if r["kind"] == "single"]
    rng = np.random.default_rng(0)
    # shaded "reachable but unrealised" band for multi-F2P (0,1)
    axR.axhspan(0.02, 1.0, color=PURPLE, alpha=0.07)
    axR.text(0.5, 0.55, "intermediate $\\Phi$ exists for multi-F2P\nbut NO rollout reaches it",
             ha="center", va="center", fontsize=8.5, color=PURPLE, style="italic")
    for i, r in enumerate(multi):
        xs = 0.15 + rng.uniform(-0.05, 0.05, r["G"])
        axR.scatter(xs, [v for v in r["phi_vals"]], s=14, color=PURPLE, alpha=0.5,
                    edgecolors="none")
    for i, r in enumerate(single):
        xs = 0.6 + rng.uniform(-0.05, 0.05, r["G"])
        axR.scatter(xs, [v for v in r["phi_vals"]], s=14, color=GRAY, alpha=0.5,
                    edgecolors="none")
    nmulti = sum(r["G"] for r in multi); nsingle = sum(r["G"] for r in single)
    axR.set_xticks([0.15, 0.6])
    axR.set_xticklabels([f"multi-F2P\n({nmulti} rollouts)", f"single-F2P\n({nsingle} rollouts)"])
    axR.set_ylabel("verifiable potential $\\Phi$ = frac. F2P tests passing")
    axR.set_ylim(-0.05, 1.05)
    axR.set_title("(b) Every rollout sits at $\\Phi{=}0$:\nthe potential is unreachable at 30B", fontsize=10.5)
    axR.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_ec_reachability.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_criterion_map()
    fig_ungameability_sweep()
    fig_ec_reachability()
    print("wrote fig_criterion_map, fig_ungameability_sweep, fig_ec_reachability")
