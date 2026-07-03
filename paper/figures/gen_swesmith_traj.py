"""fig_swesmith_traj: in the all-fail regime (SWE-smith @ 8B, ~0% solve), the
process reward drives rising programming discipline where outcome-only is flat.
Reads results/run_swe_{outcome,c3}_s{7,11}/train_log.jsonl (phi-bearing lines).
Left: productive-actions/ep over training (the learning trend). Right: final-3
trajectory-quality bars, outcome vs c3."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "font.family": "sans-serif", "axes.linewidth": 0.9,
})
BLUE, RED, GRAY = "#4A90D9", "#D94A4A", "#9CA3AF"
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "..", "results")


def series(run, k):
    f = os.path.join(RES, run, "train_log.jsonl")
    return [x.get(k, 0) for x in (json.loads(l) for l in open(f) if '"phi"' in l)]


def m3(run, k):
    s = series(run, k)
    return sum(s[-3:]) / min(3, len(s)) if s else 0


fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 3.6))

# ---- Left: productive-actions/ep trend, both seeds ----
for s, ls in [(7, "-"), (11, "--")]:
    o = series(f"run_swe_outcome_s{s}", "d_made_progress")
    c = series(f"run_swe_c3_s{s}", "d_made_progress")
    axL.plot(range(1, len(o) + 1), o, color=RED, ls=ls, lw=1.8,
             label=f"outcome-only (s{s})")
    axL.plot(range(1, len(c) + 1), c, color=BLUE, ls=ls, lw=1.8,
             label=f"process reward c3 (s{s})")
axL.set_xlabel("training iteration")
axL.set_ylabel("productive actions / episode")
axL.set_title("(a) Trajectory quality over training (0% task success)")
axL.legend(loc="upper left", framealpha=0.9)
axL.spines[["top", "right"]].set_visible(False)

# ---- Right: final-3 trajectory metrics, outcome vs c3 (mean over seeds) ----
metrics = [("d_made_progress", "productive\nedits/ep"),
           ("d_ran_tests", "test-runs/ep"),
           ("v_repeat_error", "repeat-error/ep\n(penalized)"),
           ("v_untested_edit", "untested-edit/ep\n(penalized)")]
seeds = [7, 11]
ovals = [np.mean([m3(f"run_swe_outcome_s{s}", k) for s in seeds]) for k, _ in metrics]
cvals = [np.mean([m3(f"run_swe_c3_s{s}", k) for s in seeds]) for k, _ in metrics]
x = np.arange(len(metrics))
w = 0.38
axR.bar(x - w / 2, ovals, w, color=RED, label="outcome-only")
axR.bar(x + w / 2, cvals, w, color=BLUE, label="process reward c3")
axR.set_xticks(x)
axR.set_xticklabels([lab for _, lab in metrics], fontsize=8.5)
axR.set_ylabel("actions / episode (final 3 iters, mean of 2 seeds)")
axR.set_title("(b) Productive up, penalized behaviors down")
axR.legend(framealpha=0.9)
axR.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_swesmith_traj.pdf"))
print("wrote fig_swesmith_traj.pdf")
