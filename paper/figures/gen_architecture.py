"""fig_architecture: how the two-channel system works, in the real-world (phone-agent)
setting. Left = RLVR (reward the outcome only); Right = RLVP (reward the outcome AND
penalize the path). Same episode; the difference is the per-action path channel."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 10, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "font.family": "sans-serif",
})
BLUE, GREEN, ORANGE, RED = "#4A90D9", "#50B86C", "#F5A623", "#D94A4A"
GRAY, DARK = "#9CA3AF", "#4B5563"
LBLUE, LGREEN, LRED, LGRAY = "#DBEAFE", "#D1FAE5", "#FEE2E2", "#EEF0F2"
HERE = os.path.dirname(__file__)


def box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.5, style="-|>", ms=12, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                 color=color, linewidth=lw, linestyle=ls, shrinkA=2, shrinkB=2, zorder=1))


# shared phone-agent episode: two actions carry verifiable path signal
XS = [1.35, 3.5, 5.65, 7.8]
ACTS = ["call user", "call again\n(no answer)", "authenticate\n(DOB, acct #)", "update\naccount"]
VIOL = 1   # "call again" is a path violation (over-calling)
DISC = 2   # "authenticate" discharges a required precondition


def draw_panel(ax, rlvp):
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
    title = ("RLVP: reward the outcome, penalize the path" if rlvp
             else "RLVR: reward the outcome only")
    ax.text(5.5, 6.35, title, ha="center", fontsize=11.5, fontweight="bold",
            color=(DARK if rlvp else GRAY))

    # trajectory of agent actions
    for i, (x, t) in enumerate(zip(XS, ACTS)):
        if rlvp and i == VIOL:
            fc, ec = LRED, RED
        elif rlvp and i == DISC:
            fc, ec = LGREEN, GREEN
        else:
            fc, ec = LBLUE, BLUE
        box(ax, x, 4.3, 1.7, 0.95, t, fc, ec, fs=8.6)
    ax.text(0.35, 4.3, "start", ha="center", va="center", fontsize=8.4, color=GRAY, style="italic")
    arrow(ax, 0.7, 4.3, XS[0]-0.85, 4.3, color=GRAY, lw=1.3)
    for i in range(len(XS)-1):
        arrow(ax, XS[i]+0.85, 4.3, XS[i+1]-0.85, 4.3, color=GRAY, lw=1.3)
    box(ax, 9.75, 4.3, 1.5, 0.95, "dispute\nresolved ✓", LGREEN, GREEN, fs=8.6, bold=True)
    arrow(ax, XS[-1]+0.88, 4.3, 9.75-0.78, 4.3, color=GRAY, lw=1.3)

    # outcome channel (both): terminal reward scales the whole trajectory
    box(ax, 9.75, 5.75, 1.9, 0.7, "outcome reward\n(drives the task)", "white", GREEN, fs=8.4, bold=True, tc=GREEN)
    arrow(ax, 9.75, 5.4, 9.75, 4.8, color=GREEN, lw=1.6)

    if not rlvp:
        # RLVR: the path is unseen -- no per-action signal
        ax.text(4.6, 2.6, "the path is invisible to the reward:\nover-calling and acting without auth"
                "\nare never penalized", ha="center", fontsize=8.6, color=GRAY, style="italic")
        for i in (VIOL, DISC):
            arrow(ax, XS[i], 3.82, XS[i], 3.25, color=GRAY, lw=1.0, style="-|>", ms=9, ls=(0, (2, 2)))
            ax.text(XS[i], 3.05, "?", ha="center", fontsize=11, color=GRAY, fontweight="bold")
    else:
        # RLVP: a per-action rule engine emits penalties / discharges
        box(ax, 3.5, 2.5, 1.85, 0.72, r"$-\lambda$ over-call", LRED, RED, fs=8.2, bold=True, tc=RED)
        arrow(ax, 3.5, 3.82, 3.5, 2.9, color=RED, lw=1.6)
        box(ax, 5.65, 2.5, 2.05, 0.72, r"$+\beta$ precond. met", LGREEN, GREEN, fs=8.2, bold=True, tc=GREEN)
        arrow(ax, 5.65, 3.82, 5.65, 2.9, color=GREEN, lw=1.6)
        box(ax, 4.6, 1.15, 4.9, 0.7, "verifiable rule engine  (per-action, deterministic)",
            LGRAY, DARK, fs=8.4, bold=True, tc=DARK)
        arrow(ax, 3.5, 2.12, 3.9, 1.5, color=DARK, lw=1.0)
        arrow(ax, 5.65, 2.12, 5.3, 1.5, color=DARK, lw=1.0)


fig, axes = plt.subplots(1, 2, figsize=(13, 3.7))
draw_panel(axes[0], rlvp=False)
draw_panel(axes[1], rlvp=True)
# divider
fig.subplots_adjust(wspace=0.06)
fig.savefig(os.path.join(HERE, "fig_architecture.pdf"))
print("wrote fig_architecture.pdf")
