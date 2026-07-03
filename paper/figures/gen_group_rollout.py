"""fig_group_rollout: why the path channel creates gradient where the outcome cannot.
A group of G=4 rollouts of the SAME coding task, all of which FAIL the hidden test
(outcome = 0 for every one). Left: outcome-only -> every advantage is 0 (dead update).
Right: the verifiable path channel scores the rollouts differently (a test run, a
destructive command, real progress, an edited test file), so advantages are non-zero
-- learning happens even at 0% task success. Large text on purpose."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.size": 12, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08, "font.family": "sans-serif",
})
BLUE, GREEN, RED, GRAY, DARK = "#4A90D9", "#3EA55B", "#D94A4A", "#9CA3AF", "#374151"
LRED, LGREEN = "#FDE8E8", "#E3F5E9"

# four rollouts of one bug-fix task; ALL fail the hidden test (outcome 0)
ROLLOUTS = ["read files, run tests", "edit, then rm -rf build/",
            "localize + edit the fix", "overwrite the failing test"]
PATH = [+2.0, -2.0, +1.0, -1.5]          # verifiable path score (discharges - penalties)
PATH_TAG = [r"$+\beta$ ran tests", r"$-\lambda$ destructive",
            r"$+\beta$ progress", r"$-\lambda$ edited test"]
YS = [3.35, 2.45, 1.55, 0.65]
Z = 6.9                                    # x of the advantage zero-line
SC = 0.62                                  # bar scale
HERE = os.path.dirname(__file__)


def panel(ax, right):
    ax.set_xlim(0, 10); ax.set_ylim(0.0, 4.55); ax.axis("off")
    title = ("Reward outcome $+$ verifiable path" if right else "Outcome-only (GRPO)")
    ax.text(5, 4.4, title, ha="center", fontsize=15, fontweight="bold",
            color=(DARK if right else GRAY))
    vals = (np.array(PATH) - np.mean(PATH)) if right else np.zeros(4)
    ax.plot([Z, Z], [0.35, 3.75], color=DARK, lw=1.2, zorder=1)
    ax.text(Z, 0.12, "advantage $=$ reward $-$ group mean", ha="center", fontsize=9.5, color=DARK)
    for y, adv, roll, tag in zip(YS, vals, ROLLOUTS, PATH_TAG):
        ax.text(0.15, y + 0.13, roll, ha="left", va="center", fontsize=11, color=DARK)
        ax.text(0.15, y - 0.16, "outcome: test fails (0)" + (f"   {tag}" if right else ""),
                ha="left", va="center", fontsize=9.5, color=RED, style="italic")
        c = GREEN if adv > 0.01 else (RED if adv < -0.01 else GRAY)
        w = adv * SC
        ax.add_patch(FancyBboxPatch((Z + min(0, w), y - 0.12), abs(w) if adv else 0.02, 0.24,
                     boxstyle="square,pad=0", facecolor=c, edgecolor="none", zorder=3))
    note = ("path scores differ  →  variance  →  gradient at 0% success"
            if right else "all-fail  →  every advantage is 0  →  dead update")
    ax.text(5, 4.02, note, ha="center", fontsize=11, fontweight="bold",
            color=(GREEN if right else RED),
            bbox=dict(boxstyle="round,pad=0.3", fc=(LGREEN if right else LRED), ec="none"))


fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
panel(axes[0], right=False)
panel(axes[1], right=True)
fig.suptitle("Same group, all rollouts fail the test: the verifiable path channel still learns",
             fontsize=14.5, fontweight="bold", y=1.02)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(HERE, "fig_group_rollout.pdf"))
print("wrote fig_group_rollout.pdf")
