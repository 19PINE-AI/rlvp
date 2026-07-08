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


def _last3_seeds(arm):
    """last3 success for seeds 7 (no suffix), 8, 9."""
    out = []
    for suf in ["", "_s8", "_s9"]:
        p = os.path.join(RES, f"run_swp_{arm}{suf}", "train_log.jsonl")
        r = [json.loads(l)["succ"] for l in open(p)]
        out.append(sum(r[-3:]) / len(r[-3:]))
    return out


# ---------------------------------------------------------------------------
# Figure: the criterion as a decision flow, settings as leaves
# ---------------------------------------------------------------------------
def fig_criterion_map():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.75, 6.15); ax.axis("off")

    def box(x, y, w, h, fc, ec, text, fs=10, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.06", facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="black")

    def arrow(x1, y1, x2, y2, label="", color=DARK, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=16, color=color, linewidth=1.6, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                    va="center", fontsize=11, color="black", fontweight="bold")

    # three gating questions, left to right (larger boxes + larger text)
    box(0.15, 2.45, 2.0, 1.5, "white", DARK, "Domain", fs=13, bold=True)
    q1 = (2.7, 2.15, 2.55, 2.05)
    box(*q1, LBLUE, BLUE, "Is there a\nVERIFIABLE potential\nfiner than the\noutcome?", fs=10)
    q2 = (6.05, 2.15, 2.55, 2.05)
    box(*q2, LBLUE, BLUE, "Is it\nUN-GAMEABLE?\n(cheapest maximizer\nmust solve task)", fs=10)
    q3 = (9.4, 2.15, 2.55, 2.05)
    box(*q3, LBLUE, BLUE, "Are intermediate\nvalues REACHABLE?\n(within-group\nvariance > 0)", fs=10)

    midy = q1[1] + q1[3] / 2
    arrow(2.15, midy, q1[0], midy)
    arrow(q1[0] + q1[2], midy, q2[0], midy, "yes", GREEN, ly=0.28)
    arrow(q2[0] + q2[2], midy, q3[0], midy, "yes", GREEN, ly=0.28)

    # NO branches drop to failure leaves
    for (q, fc, ec, txt) in [
        (q1, LORANGE, ORANGE, "No usable signal:\noutcome-only\nceiling (intent)"),
        (q2, LRED, RED, "COMPLIANCE\nCOLLAPSE\n(misaligned penalty)"),
        (q3, LRED, RED, "VACUOUS signal:\nzero gradient\n(unreachable)")]:
        cx = q[0] + q[2] / 2
        box(q[0], 0.05, q[2], 1.45, fc, ec, txt, fs=9.5)
        arrow(cx, q[1], cx, 1.5, "no", ec, lx=0.32)

    # YES terminal box ABOVE gate 3 -> clean vertical arrow, no overlap
    gx = q3[0] + q3[2] / 2
    box(gx - 1.45, 4.55, 2.9, 1.1, LGREEN, GREEN,
        "DENSE GRADIENT\nfrom failed episodes\n+ harm bounded", fs=10, bold=True)
    arrow(gx, q3[1] + q3[3], gx, 4.55, "yes", GREEN, lx=0.34)

    # setting labels (italic): failure leaves below their box, success leaf above its box
    def tag(x, y, t, c):
        ax.text(x, y, t, ha="center", va="center", fontsize=9, style="italic", color="black")
    tag(q1[0] + q1[2] / 2, -0.4, r"$\tau^2$ customer service", ORANGE)
    tag(q2[0] + q2[2] / 2, -0.4, "Lean penalty arms", RED)
    tag(q3[0] + q3[2] / 2, -0.4, "SWE software repair", RED)
    tag(gx, 5.92, "Lean progress; sysadmin harm", GREEN)
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_criterion_map.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: the un-gameability sweep (real seed-7 data)
# ---------------------------------------------------------------------------
def fig_ungameability_sweep():
    # arms ordered: penalty-free (survive) then penalty-bearing; label, colour, has-penalty
    arms = [
        ("validgated", "outcome-gated\nfulfillment", BLUE, False),
        ("aligned",    "aligned\nfulfillment",       GREEN, False),
        ("valid",      "gameable\nfulfillment",      "#14B8A6", False),
        ("structural", "fulfillment\n+ penalty",     RED, True),
        ("noerror",    "penalty\nonly",            PURPLE, True),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.1),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) per-arm terminal success across 3 seeds: dots + mean + std error bar
    axL.axhspan(-0.04, 0.05, color=RED, alpha=0.07)
    axL.text(len(arms) - 1, 0.0, "dead", color=RED, fontsize=7.5, ha="center",
             va="center", style="italic")
    rng = np.random.default_rng(1)
    for i, (arm, lab, c, pen) in enumerate(arms):
        v = _last3_seeds(arm)
        m, sd = np.mean(v), np.std(v)
        xs = i + rng.uniform(-0.07, 0.07, len(v))
        axL.scatter(xs, v, s=34, color=c, alpha=0.55, edgecolors="none", zorder=3)
        axL.errorbar(i, m, yerr=sd, fmt="o", color=c, markersize=8, capsize=4,
                     linewidth=1.8, markeredgecolor="white", zorder=4)
    axL.set_xticks(range(len(arms)))
    axL.set_xticklabels([a[1] for a in arms], fontsize=8)
    axL.set_ylabel("terminal success (last 3 iters)")
    axL.set_ylim(-0.06, 1.06)
    axL.set_title("(a) Terminal success over 3 seeds (Qwen3-30B, miniF2F)\n"
                  "mean $\\pm$ std; dots are seeds 7/8/9", fontsize=10)
    axL.axvline(2.5, color=GRAY, ls=":", lw=1)
    axL.text(1.0, 1.0, "penalty-free: survive", color=GREEN, fontsize=8.5,
             ha="center", fontweight="bold")
    axL.text(3.5, 1.0, "penalty-bearing", color=RED, fontsize=8.5,
             ha="center", fontweight="bold")
    axL.grid(axis="y", alpha=0.25)

    # (b) the two penalty arms across seeds: structural bimodal vs noerror reliably dead
    for arm, c, lab in [("structural", RED, "fulfillment+penalty"),
                        ("noerror", PURPLE, "penalty only")]:
        for k, suf in enumerate(["", "_s8", "_s9"]):
            p = os.path.join(RES, f"run_swp_{arm}{suf}", "train_log.jsonl")
            s = [json.loads(l)["succ"] for l in open(p)]
            axR.plot(range(1, len(s) + 1), s, "-" if arm == "structural" else "--",
                     color=c, alpha=0.75, linewidth=1.5,
                     label=(lab if k == 0 else None))
    axR.axhspan(-0.04, 0.05, color=RED, alpha=0.07)
    axR.set_xlabel("training iteration"); axR.set_ylabel("task success")
    axR.set_ylim(-0.06, 1.06)
    axR.set_title("(b) Why the big error bar: structural\n"
                  "is bimodal (1 seed to 1.0, 2 collapse)", fontsize=10)
    axR.legend(loc="center right", fontsize=8, framealpha=0.9)
    axR.grid(alpha=0.25)
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


def fig_verifier_bound():
    """Systems observation: verifiable agentic RL is verifier(CPU)-bound -- GPU memory
    full but compute idle, because the CPU verifier dominates each rollout step."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.3),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) memory full vs compute idle
    bars = axL.bar([0, 1], [88, 8], color=[BLUE, RED], alpha=0.9, width=0.6,
                   edgecolor="white")
    for x, v in [(0, 88), (1, 8)]:
        axL.text(x, v + 2.5, f"{v}%", ha="center", fontweight="bold", fontsize=11,
                 color=DARK)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["GPU memory\n(86 / 98 GB)", "GPU compute\n(SM %, avg)"])
    axL.set_ylabel("utilization (%)"); axL.set_ylim(0, 100)
    axL.set_title("(a) 30B theorem-proving RL:\nfull memory, idle compute", fontsize=10)
    axL.grid(axis="y", alpha=0.25)

    # (b) one rollout step: brief GPU generate, long CPU verify
    # schematic timeline (illustrative proportions: generate ~ms, verify ~seconds)
    axR.barh([1], [0.4], left=[0], color=BLUE, alpha=0.9, height=0.5,
             label="GPU: generate tactic (~ms)")
    axR.barh([1], [3.2], left=[0.4], color=ORANGE, alpha=0.9, height=0.5,
             label="CPU verifier: Lean kernel (~s)")
    axR.barh([0], [0.4], left=[0], color=BLUE, alpha=0.9, height=0.5)
    axR.barh([0], [3.2], left=[0.4], color=ORANGE, alpha=0.9, height=0.5)
    axR.text(0.4 + 3.2 / 2, 1.0, "GPU idle here", ha="center", va="center",
             fontsize=8.5, color=DARK, style="italic")
    axR.set_yticks([0, 1]); axR.set_yticklabels(["step $i{+}1$", "step $i$"])
    axR.set_xlabel("wall-clock per rollout step (schematic)")
    axR.set_xlim(0, 3.8); axR.set_ylim(-0.6, 1.7)
    axR.set_title("(b) Each step: a millisecond of generation,\nthen seconds on the CPU verifier", fontsize=10)
    axR.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_verifier_bound.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_verifier_bound()


def fig_efficiency_at_scale():
    """#1: positive-at-scale. Aligned verifiable-progress potential vs outcome-only on
    hard miniF2F at 30B -- aligned reaches mastery far faster and skips the all-fail stall."""
    def load(arm):
        p = os.path.join(RES, f"run_hard_{arm}", "train_log.jsonl")
        return [json.loads(l)["succ"] for l in open(p)]
    al, oc = load("aligned"), load("outcome")
    def first1(s):
        for i, v in enumerate(s, 1):
            if v >= 0.999:
                return i
        return None
    fa, fo = first1(al), first1(oc)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhspan(-0.03, 0.06, color=RED, alpha=0.06)
    ax.text(4, 0.012, "outcome-only stalled (all-fail dead-zone)", color=RED,
            fontsize=8, ha="center", style="italic")
    ax.plot(range(1, len(al) + 1), al, "-o", color=GREEN, lw=2.2, ms=4.5,
            label="aligned (verifiable progress potential)")
    ax.plot(range(1, len(oc) + 1), oc, "-s", color=GRAY, lw=2.2, ms=4.5,
            label="outcome-only")
    for f, c, dx, dy in [(fa, GREEN, 0.25, -0.16), (fo, DARK, -2.4, -0.16)]:
        ax.axvline(f, color=c, ls=":", lw=1.2, alpha=0.6)
        ax.annotate(f"mastery\niter {f}", (f, 1.0), xytext=(f + dx, 1.0 + dy),
                    color=c, fontsize=9.5, fontweight="bold", ha="center")
    ax.set_xlabel("training iteration"); ax.set_ylabel("task success")
    ax.set_ylim(-0.05, 1.08); ax.set_xlim(0.5, max(len(al), len(oc)) + 0.5)
    ax.set_title("Efficiency at scale (Qwen3-30B, hard miniF2F):\n"
                 f"the progress potential reaches mastery in {fa} iters vs. outcome-only's {fo}",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_efficiency_at_scale.pdf"))
    plt.close(fig)


def fig_phase_diagram():
    """Reachability is cheap to measure (probe) and gates the dense-reward benefit (anchors)."""
    import glob
    pr = {}
    for f in glob.glob(os.path.join(RES, "phase_diagram", "probe_*.json")):
        d = json.load(open(f)); pr[(d["model"].split("/")[-1], d["n_stages"])] = d["mean_var_phi"]
    models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]; ns = [2, 4, 6, 8]
    M = np.array([[pr.get((m, n), np.nan) for n in ns] for m in models])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.15]})
    # (a) probe Var_G(Phi) heatmap
    im = axA.imshow(M, cmap="viridis", aspect="auto", origin="lower")
    axA.set_xticks(range(len(ns))); axA.set_xticklabels([f"n={n}" for n in ns])
    axA.set_yticks(range(len(models))); axA.set_yticklabels(["0.6B", "1.7B", "4B"])
    axA.set_xlabel("outcome sparsity (chain length)"); axA.set_ylabel("model capability")
    for i in range(len(models)):
        for j in range(len(ns)):
            axA.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.025 else "black", fontsize=9)
    axA.add_patch(mpatches.Rectangle((1.5, 1.5), 2, 1, fill=False, edgecolor=RED, lw=2.2))
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label=r"base-rollout $\mathrm{Var}_G(\Phi)$")
    axA.set_title("(a) Reachability map (red = sparse+reachable sweet-spot)", fontsize=10)

    # (b) benefit vs reachability: floor (~0) + 4B reachable (E-A/E-B) + real anchors
    axB.axhspan(-0.03, 0.05, color=RED, alpha=0.06)
    # controlled chain floor: 0.6B/1.7B grid cells (benefit ~0)
    try:
        gr = [json.loads(l) for l in open(os.path.join(RES, "phase_diagram", "grid.jsonl"))]
        fx = [pr.get((d["model"].split("/")[-1], d["n_stages"]), 0) for d in gr]
        fy = [d["benefit"] for d in gr]
        axB.scatter(fx, fy, s=40, color=GRAY, alpha=0.7, label="chain 0.6B/1.7B (too weak)", zorder=3)
    except Exception:
        pass
    # 4B reachable sweet-spot: documented E-A/E-B (fine 0.34 vs outcome 0.01 at n6)
    axB.scatter([0.042], [0.33], s=120, marker="*", color=GREEN, edgecolors="black",
                linewidths=0.6, label="chain 4B, n6 (reachable)", zorder=5)
    axB.annotate("dense 0.34\nvs outcome 0.01", (0.042, 0.33), xytext=(0.030, 0.55),
                 fontsize=8.5, color=DARK, ha="center")
    # real anchors
    axB.scatter([0.0], [0.0], s=120, marker="X", color=RED, edgecolors="black",
                linewidths=0.6, label="SWE-bench 30B ($\\Phi{=}0$)", zorder=5)
    axB.annotate("unreachable\n$\\to$ no help", (0.0, 0.0), xytext=(0.006, 0.20),
                 fontsize=8.5, color=RED)
    axB.axvline(0.025, color=GRAY, ls=":", lw=1.2)
    axB.text(0.057, 0.05, "miniF2F 30B (\\#1): reachable\n$\\to$ mastery in 5 iters vs 12",
             fontsize=8.5, color=GREEN, ha="center", style="italic")
    axB.set_xlabel(r"reachability  $\mathrm{Var}_G(\Phi)$ (base rollouts)")
    axB.set_ylabel("dense-reward benefit\n(final success: dense $-$ outcome)")
    axB.set_xlim(-0.004, 0.072); axB.set_ylim(-0.05, 0.7)
    axB.set_title("(b) Benefit appears only past a reachability threshold", fontsize=10)
    axB.legend(loc="upper left", fontsize=7.6, framealpha=0.92); axB.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "fig_phase_diagram.pdf"))
    plt.close(fig)
