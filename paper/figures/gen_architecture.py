"""fig_architecture: how the two-channel system works, in the real-world (phone-agent)
setting. Left = RLVR (reward the outcome only); Right = RLVP (reward the outcome AND
penalize the path). Same episode; the difference is the per-action path channel.
Fewer boxes + large fonts for legibility at \\textwidth."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 12, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06, "font.family": "sans-serif",
})
BLUE, GREEN, RED, GRAY, DARK = "#4A90D9", "#3EA55B", "#D94A4A", "#9CA3AF", "#374151"
LBLUE, LGREEN, LRED, LGRAY = "#DBEAFE", "#D6F1DF", "#FDE1E1", "#EDEFF2"
HERE = os.path.dirname(__file__)


def box(ax, x, y, w, h, text, fc, ec, fs=12, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.8, style="-|>", ms=15, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                 color=color, linewidth=lw, linestyle=ls, shrinkA=2, shrinkB=2, zorder=1))


# shared 3-action phone-agent episode
XS = [1.7, 4.5, 7.3]
ACTS = ["call user", "call again\n(no answer)", "authenticate\n(give DOB)"]
VIOL, DISC = 1, 2


def draw_panel(ax, rlvp):
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.05); ax.axis("off")
    title = ("RLVP: reward the outcome, penalize the path" if rlvp
             else "RLVR: reward the outcome only")
    ax.text(5.5, 6.85, title, ha="center", fontsize=14, fontweight="bold",
            color=(DARK if rlvp else GRAY))

    for i, (x, t) in enumerate(zip(XS, ACTS)):
        if rlvp and i == VIOL:
            fc, ec = LRED, RED
        elif rlvp and i == DISC:
            fc, ec = LGREEN, GREEN
        else:
            fc, ec = LBLUE, BLUE
        box(ax, x, 4.55, 2.05, 1.15, t, fc, ec, fs=12)
    ax.text(0.4, 4.55, "start", ha="center", va="center", fontsize=11, color=GRAY, style="italic")
    arrow(ax, 0.85, 4.55, XS[0]-1.05, 4.55, color=GRAY, lw=1.5)
    for i in range(len(XS)-1):
        arrow(ax, XS[i]+1.05, 4.55, XS[i+1]-1.05, 4.55, color=GRAY, lw=1.5)
    box(ax, 9.9, 4.55, 1.75, 1.15, "dispute\nresolved ✓", LGREEN, GREEN, fs=12, bold=True)
    arrow(ax, XS[-1]+1.05, 4.55, 9.9-0.9, 4.55, color=GRAY, lw=1.5)

    # outcome channel (both)
    box(ax, 9.9, 5.9, 1.9, 0.92, "outcome\nreward", "white", GREEN, fs=11, bold=True, tc=GREEN)
    arrow(ax, 9.9, 5.42, 9.9, 5.15, color=GREEN, lw=2.0)

    if not rlvp:
        for i in (VIOL, DISC):
            arrow(ax, XS[i], 3.95, XS[i], 3.45, color=GRAY, lw=1.2, ls=(0, (2, 2)))
            ax.text(XS[i], 3.15, "?", ha="center", fontsize=15, color=GRAY, fontweight="bold")
        ax.text(4.5, 1.9, "the path is invisible to the reward:\nover-calling and acting without"
                " authentication\nare never penalized", ha="center", fontsize=12, color=GRAY, style="italic")
    else:
        box(ax, XS[VIOL], 2.45, 1.75, 1.05, "$-\\lambda$\nover-call", LRED, RED, fs=12.5, bold=True, tc=RED)
        arrow(ax, XS[VIOL], 3.95, XS[VIOL], 3.0, color=RED, lw=2.0)
        box(ax, XS[DISC], 2.45, 1.9, 1.05, "$+\\beta$\nprecond. met", LGREEN, GREEN, fs=12.5, bold=True, tc=GREEN)
        arrow(ax, XS[DISC], 3.95, XS[DISC], 3.0, color=GREEN, lw=2.0)
        box(ax, 5.5, 1.0, 4.7, 1.1, "verifiable rule engine\n(per-action, deterministic)",
            LGRAY, DARK, fs=11.5, bold=True, tc=DARK)
        arrow(ax, XS[VIOL], 1.92, 4.2, 1.5, color=DARK, lw=1.3)
        arrow(ax, XS[DISC], 1.92, 6.8, 1.5, color=DARK, lw=1.3)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
draw_panel(axes[0], rlvp=False)
draw_panel(axes[1], rlvp=True)
fig.subplots_adjust(wspace=0.05)
fig.savefig(os.path.join(HERE, "fig_architecture.pdf"))
print("wrote fig_architecture.pdf")
