# Phase diagram: *when* do dense process rewards help?

**Goal.** Turn the criterion from a yes/no checklist into a **quantitative phase boundary**,
predicted by the variance proposition (`paper/theorem.tex`): the benefit of a dense process
reward should track the within-group variance of the potential, `Var_G(Phi)`, and the
"helps" region should be bounded by *two* failure modes, not one.

## The two axes (and why benefit is a ridge, not a step)
Difficulty moves two things in opposite directions:
- **Outcome sparsity** (all-fail rate) *rises* with difficulty -> more room for a process
  reward to help (outcome-only goes blind: Var_G(O)->0).
- **Reachability** of Phi (`Var_G(Phi)` on base rollouts) *falls* with difficulty -> the
  potential becomes vacuous (the SWE null).

So benefit is **non-monotone**: near-zero when easy (outcome-only already learns), **peaks**
when hard-but-reachable, and **collapses** when so hard that Phi is unreachable. The figure
is a 2-D region — *process rewards help in the upper band: sparse outcome AND reachable
potential* — bounded below by "unneeded" and above by "vacuous". This is more thought-
provoking than a single threshold and it is exactly what the proposition predicts.

```
 reachability (Var_G Phi)  ^
   high (potential moves)  |  [unneeded]      ####### HELPS #######
                           |  easy/algebra    hard-miniF2F (#1: aligned 1.0@it5)
                           |                  ####################
   low  (Phi vacuous)      |  [unneeded]      [vacuous]  SWE/E-C (Phi=0)
                           +----------------------------------------------->
                              low sparsity         high sparsity (all-fail)
```

## Controlled grid (synthetic chain, cheap, seed-robust, fits small GPU)
Reuse `rlvp/envs/fileops.py::ChainPotentialEnv` + `scripts/chain_potential_exp.py`.
- **Sparsity axis** = `n_stages in {2, 4, 6, 8}` (success is product of stages -> base succ
  falls; this is the existing E-B knob).
- **Reachability axis** = **model capability** = `{Qwen3-0.6B, 1.7B, 4B}` (weaker model ->
  lower base Var_G(Phi); needs NO env change). [Alt knob if wanted: per-stage difficulty.]
- Grid = 4 sparsities x 3 models = 12 cells.
- Per cell, run TWO short arms (12 iters, 3 seeds if budget allows, else 1): `c3` (dense
  potential) vs `outcome`. **Benefit** = `final_succ(c3) - final_succ(outcome)` (and the
  iters-to-threshold gap). Cheap: small models, short runs.

## The mechanistic predictor (the cheap probe, no training)
Per cell, BEFORE training, run `G` base-policy rollouts on `N` tasks and measure
`Var_G(Phi)` (Phi = #stages completed). The proposition says **benefit should be a function
of probe `Var_G(Phi)` x sparsity**, with `Var_G(Phi)~0 => benefit~0` regardless of sparsity.
Plotting `benefit` vs `probe Var_G(Phi)` should collapse the grid onto one rising curve that
saturates — the law, not the grid. This is the figure that proves the theorem empirically.

## Real-domain anchors (already in hand — no new 30B runs needed)
Place three measured 30B points on the same axes to show the controlled law holds in the wild:
- **hard-miniF2F (#1):** high sparsity, reachable (base Var_G(Phi)>0) -> **large benefit**
  (aligned 1.0 by iter5 vs outcome-only stalled). UPPER-RIGHT "helps".
- **SWE / E-C:** high sparsity, **Var_G(Phi)=0** (156 rollouts all Phi=0) -> **zero benefit**
  (both arms 0%). "vacuous" boundary.
- **easy algebra (existing 30B sweep):** low sparsity -> small benefit (outcome also learns).
  "unneeded" boundary.

## Deliverables
1. `scripts/phase_diagram.py` (skeleton below): `probe` mode (Var_G(Phi) per cell) +
   `grid` mode (dense-vs-outcome benefit per cell).
2. Figure `paper/figures/fig_phase_diagram.pdf`: (a) 2-D heatmap benefit over
   sparsity x reachability with the three real 30B anchors overlaid; (b) benefit vs
   probe-`Var_G(Phi)` collapse curve (the theorem confirmed).
3. Paragraph in paper Sec "predictive law" tying the proposition -> phase boundary -> the
   anchors, and a **prospective prediction**: pick one held-out cell, predict its verdict
   from the probe alone, then train and confirm.

## Cost / scheduling
Controlled grid is small-model + short -> fits even alongside other GPU users (0.6B/1.7B are
~2-4GB). Queue behind #1/#2. The real anchors need no new runs (reuse #1, E-C, sweep).
The probe is cheap (no training) and can run first to de-risk.
