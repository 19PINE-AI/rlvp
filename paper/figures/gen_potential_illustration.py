"""fig_potential_illustration: how the aligned potential (verifiable progress) works,
on a REAL miniF2F theorem used in the paper (mathd_algebra_109). As the proof advances,
the Lean kernel verifies each tactic and the number of remaining goals falls; each
verified decrease pays a dense +beta fulfillment, while the outcome pays a single +1
only when the proof closes. An errored tactic (an alternate rollout) pays -lambda.
This is the potential mirror of the penalty architecture figure. Large fonts."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 12, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06, "font.family": "sans-serif",
})
GREEN, RED, GRAY, DARK, BLUE = "#3EA55B", "#D94A4A", "#9CA3AF", "#374151", "#4A90D9"
LGREEN, LBLUE, LGRAY = "#D6F1DF", "#DBEAFE", "#EDEFF2"
HERE = os.path.dirname(__file__)


def box(ax, x, y, w, h, text, fc, ec, fs=11.5, bold=False, tc="black", mono=False):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3,
            family=("monospace" if mono else "sans-serif"))


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.8, style="-|>", ms=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                 color=color, linewidth=lw, shrinkA=2, shrinkB=2, zorder=1))


fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")

# real miniF2F theorem (Valid/mathd_algebra_109)
ax.text(6.5, 6.35, "miniF2F  mathd_algebra_109:   "
        r"$(3a+2b=12),\ (a=4)\ \vdash\ b=0$",
        ha="center", fontsize=13.5, fontweight="bold", color=DARK)

# proof states (obligations remaining) + tactics between them
XS = [1.55, 5.0, 8.45, 11.5]
STATES = ["3a+2b=12\na=4\n⊢ b=0", "12+2b=12\n⊢ b=0", "2b=0\n⊢ b=0", "∎  proved"]
GOALS = [3, 2, 1, 0]                         # remaining obligations (the potential Phi)
TACS = ["subst h₁", "norm_num", "linarith"]
YS = 3.85
for i, (x, s, g) in enumerate(zip(XS, STATES, GOALS)):
    last = (i == len(XS) - 1)
    box(ax, x, YS, 2.35 if not last else 1.9, 1.25,
        s, LGREEN if last else LBLUE, GREEN if last else BLUE,
        fs=11.5, bold=last, tc=(GREEN if last else DARK), mono=not last)
    ax.text(x, YS - 1.02, f"obligations remaining: {g}", ha="center", fontsize=10.0,
            color=(GREEN if g == 0 else GRAY), fontweight="bold" if g == 0 else "normal")

# tactic arrows + dense +beta fulfillment for each verified goal-count drop
BETA_X = [3.275, 6.725, 9.7]                  # nudge the last +beta box left, clear of outcome
for i in range(3):
    xm = (XS[i] + XS[i + 1]) / 2
    arrow(ax, XS[i] + 1.18, YS, XS[i + 1] - (1.18 if i < 2 else 0.98), YS,
          color=GREEN, lw=2.2)
    ax.text(xm, YS + 0.42, f"`{TACS[i]}`", ha="center", fontsize=10.5, color=DARK,
            family="monospace")
    bx = BETA_X[i]
    box(ax, bx, YS + 1.25, 1.8, 0.62, r"$+\beta$  progress", LGREEN, GREEN,
        fs=11, bold=True, tc=GREEN)
    arrow(ax, bx, YS + 0.94, xm, YS + 0.62, color=GREEN, lw=1.4)

# outcome channel: a single +1, only at the end
box(ax, 12.0, YS + 2.0, 1.85, 0.66, r"outcome $+1$", "white", GREEN, fs=11, bold=True, tc=GREEN)
arrow(ax, 12.0, YS + 1.67, 11.6, YS + 0.68, color=GREEN, lw=1.6)

# bottom takeaway (plain text; matplotlib does not render LaTeX macros)
ax.text(6.5, 1.55, "Outcome is sparse — it pays once, at ∎.  The aligned potential is dense: "
        "every kernel-verified drop in obligations pays $+\\beta$.", ha="center", fontsize=11.5,
        color=DARK)
ax.text(6.5, 1.05, "So failing early rollouts that get further score higher — within-group "
        "variance, hence gradient, where the outcome gives none.", ha="center",
        fontsize=11.5, color=DARK, style="italic")

# an errored alternate tactic -> -lambda: the channel is signed and kernel-verifiable
box(ax, 6.5, 0.4, 5.2, 0.6, "an errored tactic in another rollout pays $-\\lambda$",
    LGRAY, RED, fs=10.5, bold=True, tc=RED)

fig.savefig(os.path.join(HERE, "fig_potential_illustration.pdf"))
print("wrote fig_potential_illustration.pdf")
