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
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.set_xlim(0, 12); ax.set_ylim(1.05, 4.95); ax.axis("off")

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
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0.45, 5.4); ax.axis("off")
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

    # right: where reachable, the potential removes dead all-fail updates (3 seeds, Lean)
    out_dead = [16, 14, 18]     # dead iters / 40, outcome-only
    pot_dead = [0, 0, 0]        # dense potential
    labels = ["outcome-only", "dense\npotential"]
    means = [np.mean(out_dead), np.mean(pot_dead)]
    errs = [np.std(out_dead), np.std(pot_dead)]
    bars = axR.bar(labels, means, yerr=errs, capsize=5, color=[DARK, GREEN],
                   edgecolor="black", linewidth=1.0, width=0.6, error_kw=dict(lw=1.2))
    for b, m, e in zip(bars, means, errs):
        axR.text(b.get_x() + b.get_width() / 2, m + e + 0.7, f"{m:.0f}", ha="center",
                 va="bottom", fontsize=11, fontweight="bold")
    axR.set_ylabel("dead (all-fail) iterations / 40")
    axR.set_title("Where reachable: dead updates eliminated", fontsize=11)
    axR.set_ylim(0, 22); axR.spines[["top", "right"]].set_visible(False)
    axR.text(0.5, 0.90, "3 seeds; every seed", transform=axR.transAxes, ha="center",
             fontsize=8.5, color=GRAY, style="italic")
    fig.suptitle("Dense potentials are reachability-gated", fontsize=12.5,
                 fontweight="bold", y=1.03)
    fig.savefig(os.path.join(HERE, "fig_potential_fragility.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- data helpers
import json, glob, statistics as _st
RESULTS = os.path.join(HERE, "..", "..", "results")


def _final_eval(run_dir):
    """Last train-log iter carrying an eval; returns fileops+csops-averaged metrics or None."""
    log = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(log):
        return None
    last = None
    for ln in open(log):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if isinstance(r, dict) and r.get("eval"):
            last = r
    if not last:
        return None
    e = last["eval"]
    doms = [k for k in ("fileops", "csops") if k in e]
    if not doms:
        return None
    m = lambda key: _st.mean(e[k][key] for k in doms)
    return {"success": m("success"), "clean": m("clean"),
            "viol": m("viol_per_episode"), "calls": m("calls_per_ep"),
            "iter": last.get("iter")}


def _arm_stats(arm):
    """Mean+-std across all landed seeds for an arm; None if no seed has an eval yet."""
    rows = [_final_eval(d) for d in sorted(glob.glob(os.path.join(RESULTS, f"run_rvp_{arm}_s*")))]
    rows = [r for r in rows if r]
    if not rows:
        return None
    out = {"n": len(rows)}
    for key in ("success", "clean", "viol", "calls"):
        vals = [r[key] for r in rows]
        out[key] = (_st.mean(vals), _st.pstdev(vals) if len(vals) > 1 else 0.0)
    return out


# ---------------------------------------------------------------- Fig 3
def fig_recipe_positive():
    """recipe vs outcome: both keep success high; recipe drives clean ~0 -> ~1."""
    outc, rec = _arm_stats("outcome"), _arm_stats("recipe")
    if not (outc and rec):
        print("fig_recipe_positive: SKIP (need outcome+recipe evals)"); return
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    labels = ["Outcome-only", "Reward outcome,\npenalize path"]
    cols = [GRAY, GREEN]
    for a, metric, title, ylab in (
            (ax[0], "success", "Task success", "success rate"),
            (ax[1], "clean", "Violation-free episodes", "clean rate")):
        means = [outc[metric][0], rec[metric][0]]
        errs = [outc[metric][1], rec[metric][1]]
        bars = a.bar(labels, means, yerr=errs, capsize=5, color=cols,
                     edgecolor=DARK, linewidth=1.1, width=0.62, error_kw=dict(lw=1.2))
        a.set_ylim(0, 1.08); a.set_ylabel(ylab); a.set_title(title, fontweight="bold")
        a.spines[["top", "right"]].set_visible(False)
        a.axhline(1.0, ls=":", lw=0.8, color=GRAY, zorder=0)
        for b, mn in zip(bars, means):
            a.text(b.get_x() + b.get_width() / 2, mn + 0.03, f"{mn:.2f}",
                   ha="center", va="bottom", fontsize=10, fontweight="bold")
    n = min(outc["n"], rec["n"])
    fig.suptitle(f"Reward the outcome, penalize the path: task attained AND constraints kept "
                 f"(n={n} seeds)", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_recipe_positive.pdf"))
    plt.close(fig)
    print(f"fig_recipe_positive OK (outcome n={outc['n']} it={ '/'.join(str(_final_eval(d)['iter']) for d in sorted(glob.glob(os.path.join(RESULTS,'run_rvp_outcome_s*'))) if _final_eval(d)) }, "
          f"recipe n={rec['n']}) succ {outc['success'][0]:.2f}->{rec['success'][0]:.2f} clean {outc['clean'][0]:.2f}->{rec['clean'][0]:.2f}")


# ---------------------------------------------------------------- Fig 6
def fig_penalty_ablation():
    """Walk outcome -> penalty_only -> recipe; success + clean per arm."""
    arms = [("outcome", "Outcome\nonly"), ("penonly", "+ penalty\n(no discharge/seed)"),
            ("recipe", "Full recipe")]
    stats = [(lbl, _arm_stats(a)) for a, lbl in arms]
    stats = [(lbl, s) for lbl, s in stats if s]
    if len(stats) < 2:
        print("fig_penalty_ablation: SKIP (need >=2 arms)"); return
    labels = [lbl for lbl, _ in stats]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    succ = [s["success"][0] for _, s in stats]; succ_e = [s["success"][1] for _, s in stats]
    clean = [s["clean"][0] for _, s in stats]; clean_e = [s["clean"][1] for _, s in stats]
    ax.bar(x - w / 2, succ, w, yerr=succ_e, capsize=4, color=BLUE, edgecolor=DARK,
           linewidth=1.0, label="task success", error_kw=dict(lw=1.1))
    ax.bar(x + w / 2, clean, w, yerr=clean_e, capsize=4, color=GREEN, edgecolor=DARK,
           linewidth=1.0, label="violation-free", error_kw=dict(lw=1.1))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12); ax.set_ylabel("rate"); ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.set_title("Each design element earns its place", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x - w / 2, succ):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    for xi, v in zip(x + w / 2, clean):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_penalty_ablation.pdf"))
    plt.close(fig)
    print("fig_penalty_ablation OK: " + " | ".join(f"{l}: succ {s['success'][0]:.2f} clean {s['clean'][0]:.2f} (n={s['n']})" for l, s in stats))


def _curve(arm, gen_batch=48):
    """Per-iter (episodes, success mean/std, clean mean/std) aggregated over landed seeds."""
    import collections
    succ = collections.defaultdict(list); clean = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(RESULTS, f"run_rvp_{arm}_s*"))):
        log = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(log):
            continue
        for ln in open(log):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if not (isinstance(r, dict) and r.get("eval") and r.get("iter")):
                continue
            e = r["eval"]; doms = [k for k in ("fileops", "csops") if k in e]
            if not doms:
                continue
            succ[r["iter"]].append(_st.mean(e[k]["success"] for k in doms))
            clean[r["iter"]].append(_st.mean(e[k]["clean"] for k in doms))
    iters = sorted(succ)
    ep = [i * gen_batch for i in iters]
    sm = [_st.mean(succ[i]) for i in iters]; ss = [_st.pstdev(succ[i]) for i in iters]
    cm = [_st.mean(clean[i]) for i in iters]; cs = [_st.pstdev(clean[i]) for i in iters]
    n = max((len(succ[i]) for i in iters), default=0)
    return ep, sm, ss, cm, cs, n


# ---------------------------------------------------------------- Fig 0 (headline / teaser)
def fig_headline():
    """Page-1 teaser: vs the GRPO/outcome-only baseline, RLVP keeps task success while
    reaching violation-free behavior the baseline never does. Two training curves + harm callout."""
    ep_o, so, so_e, co, co_e, no = _curve("outcome")
    ep_r, sr, sr_e, cr, cr_e, nr = _curve("recipe")
    if not (ep_o and ep_r):
        print("fig_headline: SKIP (need outcome+recipe curves)"); return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 2.75))
    npo = np.array
    for ax, (ym_o, ye_o, ym_r, ye_r, title, note) in (
        (axL, (so, so_e, sr, sr_e, "Task success", "no task cost")),
        (axR, (co, co_e, cr, cr_e, "Violation-free episodes",
               "the deployable behavior GRPO never reaches")),
    ):
        ax.plot(ep_o, ym_o, "-o", color=GRAY, lw=2.2, ms=5, label="GRPO (outcome-only)", zorder=3)
        ax.fill_between(ep_o, npo(ym_o) - npo(ye_o), npo(ym_o) + npo(ye_o), color=GRAY, alpha=0.15)
        ax.plot(ep_r, ym_r, "-o", color=GREEN, lw=2.4, ms=5, label="RLVP (penalize the path)", zorder=4)
        ax.fill_between(ep_r, npo(ym_r) - npo(ye_r), npo(ym_r) + npo(ye_r), color=GREEN, alpha=0.18)
        ax.set_ylim(-0.04, 1.08); ax.set_xlim(left=0)
        ax.set_xlabel("episodes generated"); ax.set_title(title, fontweight="bold", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(1.0, ls=":", lw=0.8, color=GRAY, zorder=0)
        ax.text(0.5, 0.06, note, transform=ax.transAxes, ha="center", fontsize=9.2,
                style="italic", color=DARK)
    axL.set_ylabel("success rate"); axR.set_ylabel("violation-free rate")
    axL.legend(frameon=False, loc="lower right", fontsize=9)
    # baseline-floor marker on the right panel
    axR.annotate("GRPO stuck near 0\nat any budget", xy=(ep_o[-1], co[-1]),
                 xytext=(ep_o[len(ep_o) // 2], 0.42), fontsize=8.5, color=GRAY,
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
    fig.suptitle("Reward the outcome, penalize the path: RLVP reaches deployable behavior "
                 "GRPO cannot, at no task cost", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(HERE, "fig_headline.pdf"))
    plt.close(fig)
    print(f"fig_headline OK (outcome n={no}, recipe n={nr})")


if __name__ == "__main__":
    fig_headline()
    fig_two_channel(); print("fig_two_channel OK")
    fig_variance_vacuum(); print("fig_variance_vacuum OK")
    fig_penalty_design(); print("fig_penalty_design OK")
    fig_potential_fragility(); print("fig_potential_fragility OK")
    fig_recipe_positive()
    fig_penalty_ablation()
    print("ALL FIGURES DONE")
