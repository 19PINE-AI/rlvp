"""Figures for the RLVP (penalties) rewrite. No-data / conceptual + data-in-hand figures:
  fig_two_channel        (Fig 1) reward the outcome, penalize the path
  fig_variance_vacuum    (Fig 2) outcome-only is blind at both ends; penalty fills the vacuum
  fig_penalty_design     (Fig 5) the four-rule design procedure
  fig_potential_fragility(Fig 7) reachability wall + sign-flip
Data-dependent figs (fig_recipe_positive, fig_penalty_ablation) are generated once the n=5
recipe batch lands. Matches the house style of gen_new_figures.py."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "font.family": "sans-serif", "axes.linewidth": 0.9,
})
BLUE, GREEN, ORANGE, RED = "#4A90D9", "#50B86C", "#F5A623", "#D94A4A"
PURPLE, GRAY, DARK = "#8B5CF6", "#9CA3AF", "#4B5563"
LBLUE, LGREEN, LORANGE, LRED, LPUR = "#DBEAFE", "#D1FAE5", "#FEF3C7", "#FEE2E2", "#EDE9FE"
HERE = os.path.dirname(__file__)


def box(ax, x, y, w, h, text, fc, ec, fs=10, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=ms, color=color, linewidth=lw, shrinkA=3, shrinkB=3, zorder=1))


# ---------------------------------------------------------------- Fig 1
def fig_two_channel():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.text(6, 5.0, "Reward the outcome.  Penalize the path.",
            ha="center", fontsize=15, fontweight="bold", color=DARK)

    # trajectory of actions
    xs = [1.4, 3.2, 5.0, 6.8, 8.6]
    acts = ["read\nfiles", "edit\ncode", "rm -rf\n(destructive)", "run\ntests", "submit"]
    fcs = [LBLUE, LBLUE, LRED, LGREEN, LBLUE]
    ecs = [BLUE, BLUE, RED, GREEN, BLUE]
    for x, t, fc, ec in zip(xs, acts, fcs, ecs):
        box(ax, x, 3.2, 1.5, 0.95, t, fc, ec, fs=9.5)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 0.75, 3.2, xs[i + 1] - 0.75, 3.2, color=GRAY, lw=1.4)
    ax.text(0.35, 3.2, "start", ha="center", va="center", fontsize=9, color=GRAY, style="italic")
    arrow(ax, 0.7, 3.2, xs[0] - 0.75, 3.2, color=GRAY, lw=1.4)

    # outcome channel (end)
    box(ax, 10.7, 3.2, 1.6, 0.95, "task\nsolved  ✓", LGREEN, GREEN, fs=9.5, bold=True)
    arrow(ax, xs[-1] + 0.75, 3.2, 10.7 - 0.8, 3.2, color=GRAY, lw=1.4)
    box(ax, 10.7, 4.55, 1.9, 0.7, "Outcome reward\n(drives the task)", "white", GREEN, fs=9.5, bold=True, tc=GREEN)
    arrow(ax, 10.7, 4.2, 10.7, 3.7, color=GREEN, lw=1.6)

    # path channel (per-action penalties / discharges)
    box(ax, 5.0, 1.55, 1.9, 0.72, r"$-\lambda$  destructive", LRED, RED, fs=9, bold=True, tc=RED)
    arrow(ax, 5.0, 2.72, 5.0, 1.95, color=RED, lw=1.6)
    box(ax, 6.8, 1.55, 1.9, 0.72, r"$+\beta$  tested first", LGREEN, GREEN, fs=9, bold=True, tc=GREEN)
    arrow(ax, 6.8, 2.72, 6.8, 1.95, color=GREEN, lw=1.6)
    box(ax, 2.9, 0.5, 3.5, 0.7, "Path channel: verifiable penalties  (teaches the outcome-neutral\nconstraints the outcome cannot see)", "white", DARK, fs=9, bold=True, tc=DARK)

    ax.text(4.7, 4.35, "verifier asymmetry:  a bad move is a one-line check;  progress is the whole problem",
            ha="center", fontsize=8.6, color=GRAY, style="italic")
    fig.savefig(os.path.join(HERE, "fig_two_channel.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_variance_vacuum():
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    p = np.linspace(0, 1, 400)
    out = 4 * p * (1 - p)                       # within-group variance of the binary outcome (norm to 1)
    pen = 0.72 + 0.06 * np.sin(6 * p)           # penalty variance: high & roughly flat everywhere
    pot = np.clip(3.4 * (p - 0.12) * (0.92 - p), 0, 1)  # potential: low at low success, ~0 at both ends

    ax.axvspan(0, 0.14, color=RED, alpha=0.10)
    ax.axvspan(0.86, 1.0, color=RED, alpha=0.10)
    ax.plot(p, out, color=DARK, lw=2.6, label="outcome (binary)")
    ax.plot(p, pen, color=BLUE, lw=2.6, label="verifiable penalty")
    ax.plot(p, pot, color=PURPLE, lw=2.4, ls="--", label="dense progress potential")

    ax.text(0.07, 0.9, "all-fail\ngroups\n(early)", ha="center", va="top", fontsize=8.5,
            color=RED, fontweight="bold")
    ax.text(0.93, 0.9, "all-success\ngroups\n(late)", ha="center", va="top", fontsize=8.5,
            color=RED, fontweight="bold")
    ax.annotate("outcome blind\n(zero variance)", xy=(0.02, 0.04), xytext=(0.22, 0.30),
                fontsize=8.7, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.text(0.5, 0.80, "penalty supplies variance\nin both vacuums", ha="center",
            fontsize=9, color=BLUE, fontweight="bold")
    ax.text(0.5, 0.16, "potential needs reachable\npartial success", ha="center",
            fontsize=9, color=PURPLE, style="italic")

    ax.set_xlabel("group success rate"); ax.set_ylabel("within-group variance  (usable gradient)")
    ax.set_title("Outcome-only RL is blind at both ends; the penalty is not", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=8.8, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(os.path.join(HERE, "fig_variance_vacuum.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig_penalty_design():
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(3.4, 5.15, "Designing a verifiable penalty", ha="center", fontsize=13.5,
            fontweight="bold", color=DARK)
    rules = [
        ("1.  Penalize a verifiable ACTION,\n     not lack of progress",
         "penalizing omission rewards doing nothing", 4.25),
        ("2.  Keep the outcome reward as the driver;\n     never optimize a penalty alone",
         "a lone penalty stalls: the inaction trap", 3.15),
        ("3.  Pair the penalty with a DISCHARGE\n     (reward the compliant action)",
         "pull toward doing it right, not just away", 2.05),
        ("4.  Make the compliant path reachable\n     and its target un-gameable",
         "seed a demo; avoid a learned proxy", 0.95),
    ]
    for txt, warn, y in rules:
        box(ax, 3.4, y, 5.3, 0.86, txt, LBLUE, BLUE, fs=10.2, bold=True, tc=DARK)
        ax.text(6.35, y, warn, ha="left", va="center", fontsize=8.4, color=RED, style="italic")
    for y0, y1 in [(4.25, 3.15), (3.15, 2.05), (2.05, 0.95)]:
        arrow(ax, 3.4, y0 - 0.43, 3.4, y1 + 0.43, color=BLUE, lw=1.6)
    box(ax, 3.4, 0.02, 6.6, 0.56, "a penalty that bounds the path without stalling the task",
        LGREEN, GREEN, fs=9.2, bold=True, tc=GREEN)
    fig.savefig(os.path.join(HERE, "fig_penalty_design.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7
def fig_potential_fragility():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # left: reachability wall on SWE (256 rollouts, all Phi=0)
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1, 256)
    axL.scatter(xs, np.zeros(256), s=16, color=PURPLE, alpha=0.45, edgecolors="none")
    axL.axhspan(-0.02, 0.02, color=PURPLE, alpha=0.08)
    axL.set_ylim(-0.05, 1.0); axL.set_xlim(0, 1)
    axL.set_yticks([0, 0.5, 1.0]); axL.set_xticks([])
    axL.set_ylabel(r"partial progress  $\Phi$  (tests passed)")
    axL.set_title("Reachability wall (software repair)", fontsize=11)
    axL.text(0.5, 0.55, "256 rollouts, every $\\Phi=0$\n"
             r"$\Rightarrow \mathrm{Var}_G(\Phi)=0$, no gradient",
             ha="center", fontsize=9.2, color=PURPLE, fontweight="bold")
    axL.text(0.5, 0.03, "all rollouts pinned at zero", ha="center", va="bottom",
             fontsize=8.2, color=GRAY, style="italic")

    # right: sign-flip of the "efficiency win" across identical-config runs
    labels = ["run A", "run B\n(re-seeded)"]
    aligned = [4, 25]      # iters-to-mastery (25 = did not master within budget)
    outcome = [12, 4]
    x = np.arange(2); w = 0.36
    axR.bar(x - w / 2, aligned, w, color=BLUE, label="dense potential", alpha=0.9)
    axR.bar(x + w / 2, outcome, w, color=DARK, label="outcome-only", alpha=0.9)
    for xi, a, o in zip(x, aligned, outcome):
        axR.text(xi - w / 2, a + 0.5, "master@4" if a == 4 else "no master",
                 ha="center", fontsize=8, color=BLUE, fontweight="bold")
        axR.text(xi + w / 2, o + 0.5, f"master@{o}", ha="center", fontsize=8,
                 color=DARK, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels(labels)
    axR.set_ylabel("iterations to mastery  (lower = faster)")
    axR.set_title("The speed-up sign-flips under re-seeding", fontsize=11)
    axR.set_ylim(0, 30)
    axR.annotate("winner flips", xy=(1, 6), xytext=(0.35, 22), fontsize=9,
                 color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    axR.legend(frameon=False, fontsize=8.6, loc="upper left")
    fig.suptitle("Dense potentials are reachability-gated and fragile", fontsize=12.5,
                 fontweight="bold", y=1.03)
    fig.savefig(os.path.join(HERE, "fig_potential_fragility.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_two_channel(); print("fig_two_channel OK")
    fig_variance_vacuum(); print("fig_variance_vacuum OK")
    fig_penalty_design(); print("fig_penalty_design OK")
    fig_potential_fragility(); print("fig_potential_fragility OK")
    print("ALL FIGURES DONE")
