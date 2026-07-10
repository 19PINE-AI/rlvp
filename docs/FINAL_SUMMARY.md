# RLVP — Final Summary of the Bulletproofing Program (2026-06-29)

Written autonomously overnight. This consolidates everything the replication/instrumentation
program established, separates what is **bulletproof** from what had to be **scoped or refuted**,
and proposes the paper's final framing. Full detail in FINDINGS.md secs 14-22; related work in
paper/body.tex.

---

## The one-sentence thesis (what survived)

**A group-relative (GRPO) advantage is a within-group variance; it vanishes on both all-fail and
all-success groups. A dense, verifiable potential helps an agentic-RL run *exactly insofar as it
supplies reachable within-group variance the outcome lacks* — i.e. only where the policy already
reaches intermediate values that differ across rollouts (var_phi > 0). That condition usually
fails, which is why the headline benefits of process rewards are narrow and often do not
reproduce; the one robust, reachability-free win is an un-gameable harm-bounding penalty.**

This is a *mechanism + honest-negative* paper, not a "here is a method that helps" paper.

---

## BULLETPROOF (robust, keep as core claims)

1. **The variance lens (mechanism).** The GRPO advantage is a within-group variance:
   Var_G(R) = Var_G(O) + beta^2 Var_G(Phi) + 2 beta Cov_G(O,Phi). It is identically zero on
   all-fail AND all-success groups — GRPO is "blind at both ends." Definitional + externally
   corroborated: DAPO's dynamic sampling explicitly *drops* groups with accuracy 1 or 0 (its
   Eq. 11), and the problem is named "advantage collapse" in the literature. (FINDINGS 20-22.)

2. **The reachability gate (the unifying empirical law).** A potential supplies usable variance
   only where the policy *reaches* differing intermediate values (var_phi > 0). Three measured
   faces of one quantity:
   - SWE-bench: var_phi = 0 (256+ rollouts, all Phi=0) -> a structurally-finer potential gives
     zero benefit (the reachability wall).
   - Dead-iteration rescue (E4, 4B, n=5): the fine potential rescues all-fail iters only where
     var_phi>0; it yields **2.15x fewer dead iters** (fine 6.8 vs coarse 14.6, fine<coarse every
     seed) but **never zero** — truly-dead iters (var_phi=0) remain. Partial, reachability-gated.
   - All-success: a completion-potential *saturates* at success (var_phi -> 0), so it cannot hold
     a mastered task; outcome-only then drifts off solved states (all-success blindness).

3. **Real-task efficiency does NOT replicate.** The "process rewards make agentic RL faster at
   scale" result (#1: 30B aligned mastered @it5 vs outcome @it12) **flips sign on re-running the
   identical config** (outcome beats aligned); at 8B-dense over n=4 seeds the efficiency gap is
   noise (peak ~equal, both arms break through on one seed). vLLM rollouts are unseeded, 30B-MoE
   training is bimodal. Literature-corroborated: A Sober Look (15% seed variance, gains often
   insignificant), Spurious Rewards (random rewards "work" via bias), and the Practitioner's
   Guide (dense accelerates but final perf depends on the estimator). Our instance is sharper:
   even the *sign* of the benefit is not reproducible. (FINDINGS 20.)

4. **Harm-bounding is the one robust positive (reachability-free) -- RE-VERIFIED at n=5.** An
   un-gameable penalty cut destructive actions ~5.7x (violations 0.66+-0.63 vs 3.71+-0.52,
   non-overlapping ranges) at statistically-equal task success (0.097 vs 0.122, within 1 sigma,
   both at floor), while taking ~3x MORE productive actions (13.4 vs 4.2). TerminalBench, 4B,
   5 seeds {7,11,12,13,14}. It needs no reachability because its target ("do not do the bad
   thing") is always reachable and the cheapest maximizer *is* the desired behavior. This is the
   paper's bulletproof, deployable positive (FINDINGS sec 23).

5. **Reproducibility as a contribution.** The sign-flip under re-seeding (plus unseeded vLLM
   rollouts) is a clean, literature-aligned cautionary result for the agentic-process-reward
   subfield: "re-seed before you believe a process-reward speed-up."

---

## SCOPED or REFUTED (do not claim as originally stated)

- **Efficiency-as-speed at scale (#1):** REFUTED on real tasks (does not replicate). Keep only a
  controlled-synthetic, reachability-gated dead-iter statement.
- **Dead-iteration elimination "0 vs 16/40":** SCOPED to "~2x fewer, reachability-gated, never
  zero" (E4 n=5). The clean "0" only appears in high-reachability/low-lr regimes.
- **All-success "a potential fixes the drift":** NOT demonstrable in the controlled harness
  (completion-Phi saturates; no clean mastery; var_O~0 homogeneous groups). Keep all-success
  blindness as a *diagnostic* (DAPO-corroborated), not a fix.
- **Reachability-ALONE criterion:** REFUTED (prospective test): reachability is necessary not
  sufficient; a learnable foothold lets outcome-only bootstrap, so cross-task generalization
  rescues all-fail groups WITHOUT a dense reward.
- **"A verifiable-potential criterion that called every success and failure across 5 settings":**
  RETIRE this framing. The five-settings table over-claimed predictiveness; the honest object is
  the variance/reachability mechanism plus a much narrower set of robust effects.

---

## Proposed final framing + title

**Title (candidate):** *The Variance Vacuum: When Dense Process Rewards Help Agentic RL — and
Why the Speed-ups Don't Replicate.*

**Arc:** (1) GRPO advantage = within-group variance, blind at both ends [mechanism, DAPO-
corroborated]. (2) A verifiable potential helps only by supplying *reachable* within-group
variance — one quantity governing dead-iter rescue, all-success holding, and efficiency. (3)
That condition usually fails: SWE wall (unreachable), all-success saturation, learnable-foothold
redundancy; and on real tasks the apparent efficiency gain does not survive re-seeding. (4) The
robust, deployable use is an un-gameable harm-bounding penalty, which is reachability-free. (5)
Methodological takeaway: re-seed; single-run at-scale process-reward gains are fragile.

**Why this is stronger than the old paper:** it is honest, hard to attack, and unifies everything
around a single measurable quantity (reachable within-group potential variance). It trades an
over-claimed "criterion that works" for a defensible mechanism + a clear-eyed account of how
narrow the wins really are, with external literature support at every step.

---

## Figures (plan)

- CUT: fig_efficiency_at_scale (#1, invalidated).
- REDESIGN: fig_criterion_map -> variance-vacuum schematic (Var_G(O)->0 at both ends of success).
- ADD: dead-iter rescue (E4 n=5, fine 6.8 vs coarse 14.6 with var_phi rescue overlay);
  reproducibility panel (sign-flip + 8B n=4 spread).
- KEEP: ungameability_sweep, ec_reachability (SWE wall), verifier_bound, phase_diagram (with the
  necessary-not-sufficient caveat).

## Open items for the user to decide
1. ~~Re-verify harm-bounding at n>=5~~ DONE (2026-06-30, n=5): ROBUST, ~5.7x fewer violations at
   equal success (FINDINGS sec 23; table_harm.tex updated to 5 seeds).
2. Decide whether to keep the τ²/learned-critic/discovery-wall "no" sections or trim to the
   variance-vacuum spine.
3. Optional: a clean within-success-variance synthetic task to *positively* demonstrate the
   all-success hold (needs env eng; deemed not worth it for now — left as a prediction).
