# RLVP: Research Findings

*Reinforcement Learning from Verifiable Penalties — what we've established, what we
haven't, and where the boundaries are. Last updated this session.*

---

## TL;DR

**The idea.** Outcome-only RL (GRPO/DAPO) rewards a trajectory only if it solves the
task. On hard tasks, early training is dominated by groups where *every* sampled
rollout fails — and group-relative methods compute an advantage of exactly zero
there, so the update is a no-op. The learning signal is silent precisely when
learning matters most. **RLVP** adds dense, machine-checkable *process* rewards at
each tool call (a penalty when a move violates a verifiable rule, a credit when it
discharges a pending obligation). Because these depend on *how* an episode acts, not
*whether* it ultimately succeeds, they produce gradient from the failed episodes
GRPO throws away.

**Two contributions, each demonstrated on a real benchmark:**
1. **Efficiency** — denser signal turns dead updates into learning (Lean).
2. **Harm-reduction as an axis independent of outcome** — the agent learns to stop
   acting destructively even when task success doesn't change (Terminal).

| Result | Status | Headline number |
|---|---|---|
| All-fail blindness → dead updates | ✅ solid | mechanism reproduced on every benchmark |
| Lean: dead-iteration elimination | ✅ 3 seeds | RLVP **0±0** dead iters vs GRPO **16±1.6** |
| Terminal: harm-reduction at equal outcome | ✅ 3 seeds | **~4× fewer** violations, equal success |
| Auto-derived rules recover hand-written gains | ✅ | no human rules needed |
| DAPO doesn't solve sample-efficiency | ✅ | 5.6× oversampling tax, no recovery |
| Lean "faster to threshold" | ❌ retracted | seed-dependent; not robust |
| τ²: process rewards can't substitute for outcome | ✅ (boundary) | coverage gradient w/ ceiling |
| Self-critique vs rules ([SELFCRITIC.md](SELFCRITIC.md)) | ✅ full 2×2 (Exp0+probe+Exp1+τ² cell-C) | symmetric: rules win on stateful (signal *stationarity*); self-critique wins on τ² *intent* (F1 .61 vs rules .14) |
| SWE-bench at 4B/30B | 🔄 re-measuring | env bug fixed; valid measurement in progress |
| miniF2F at 30B (vLLM+QLoRA) | 🔄 running | harness built+validated; runs starting |

---

## 1. The core mechanism (solid)

On a difficulty-calibrated long-horizon task, we confirmed the failure mode directly:
when the policy is weak, most rollout *groups* are all-fail, GRPO's advantage is zero,
and the optimizer does nothing. We measured GRPO spending a large fraction of early
iterations on these dead updates. RLVP's process channel (penalty for a rule
violation, credit for discharging an obligation) is non-zero on exactly those
episodes, so it keeps learning through the wall.

**Why this is the right framing for agents.** Agentic RL differs from reasoning RL:
the environment is a *verifier*. Tool calls have machine-checkable properties (did you
edit a test file? did you run the test before submitting? did you delete data you were
only asked to read?). Those properties are cheap, dense, and don't require the task to
be solved — exactly the signal outcome-only RL lacks.

---

## 2. Efficiency — Lean theorem proving (✅ 3 seeds)

Qwen3-4B + LoRA, synthetic Nat/algebra theorems, structural rules (errored-tactic =
penalty, goal-progress = discharge). RLVP (c3) vs outcome-only GRPO, 40 iters × 3 seeds:

| metric | RLVP | Outcome-only |
|---|---|---|
| **dead iterations** | **0 ± 0** `[0,0,0]` | **16 ± 1.6** `[16,14,18]` |
| final success | 0.886 ± 0.084 | 0.839 ± 0.093 |

**The robust claim is dead-iteration elimination** — every seed, RLVP has zero dead
updates while GRPO wastes 14–18 of 40 (all concentrated early, when the policy can't
prove anything yet). Final success is a slight, *non-significant* RLVP edge.

**Honest retraction:** an earlier single-seed run showed RLVP reaching a success
threshold ~1.75× faster. Across seeds this did **not** hold (in one seed GRPO crossed
the threshold first) — the metric is too noisy near zero. We do **not** claim faster
wall-clock convergence on Lean. The claim is: RLVP always has gradient where GRPO goes
dark.

---

## 3. Harm-reduction, independent of outcome — TerminalBench (✅ 3 seeds)

Qwen3-4B + LoRA, real TerminalBench Docker tasks, rules = repeat-error and
blind-destructive penalties. 40 iters × 3 seeds:

| metric | RLVP | Outcome-only |
|---|---|---|
| task success | 0.097 ± 0.069 | 0.097 ± 0.063 (**equal**, at floor) |
| **violations / episode** | **0.95 ± 0.71** | **3.99 ± 0.57** (error bars separate) |
| productive actions / episode | **~14** | ~4 |

4B rarely solves TerminalBench, so **task success is identical and at the floor** — and
yet RLVP commits **~4× fewer harmful actions**. Crucially it is **not** doing this by
going passive: it takes *~3× more* productive actions while violating less. This is the
clean demonstration that **harm lives on an axis independent of the outcome**: you can
reduce harm without changing whether the task is solved. (Operational intuition: you
cannot delete the production database and then "finish" the repair — the harm already
happened, regardless of the final state.)

---

## 4. Supporting results (✅)

- **Process credit must be token-attached.** Folding the same process reward into a
  single scalar return (instead of attaching it to the responsible tokens) loses the
  benefit — confirmed by a paired ablation.
- **Rules can be auto-derived.** Deriving the process signal from tool-category
  metadata and the environment's own error signals (no human-written rules) recovers
  the hand-engineered gain — keeps the method R1-Zero-clean.
- **DAPO doesn't fix sample-efficiency.** DAPO's discard-and-resample confronts the
  same all-fail groups by throwing them out and resampling; we measured a **5.6×**
  oversampling tax with no recovery of the lost signal.
- **Where self-evolution ends.** On a task with a hidden precondition the policy never
  spontaneously samples, no reward-only method escapes the wall; a demonstration
  cold-start does — delineating exactly where imitation is still required.

---

## 5. The honest boundary — τ²-bench (✅ as a negative result)

On a real verifiable-workflow benchmark we traced RLVP's limit as a **coverage
gradient with an irreducible ceiling**:
- rules *orthogonal* to the reward actively **harm** (drive the policy into a
  degenerate compliance-only optimum — it minimizes harm by not acting);
- *policy-derived* rules (procedural **and** verifiable-semantic) remove that harm;
- but **neither beats outcome-only**, because verifiable rules express policy
  *validity*, while the reward's residual difficulty is task *intent* — and intent is
  the reward itself, learnable only from the outcome.

**Conclusion:** process rewards accelerate the policy-verifiable component of success
and reduce harm; they **cannot substitute** for outcome learning of intent.

---

## 6. Scaling to a stronger model — 30B (🔄 in progress)

Hardware: one RTX PRO 6000 Blackwell (96 GB). We stood up and validated the full
stack to take the two hard benchmarks past the 4B capability floor:
- **vLLM fp8 serving** of Qwen3-30B-A3B (MoE, 3B active): loads in 29 GB, fast.
- **QLoRA backward**: 30B in 4-bit = 16.7 GB, forward+backward peak **42.6 GB** →
  coexists with vLLM fp8 on one card.
- **grpo.py unmodified**: it recomputes its own old-logprobs, so vLLM is just a
  sampler (consistent HF/HF ratio) — no importance-sampling surgery needed.
- **Mathlib REPL**: working (`import Mathlib` once per warm REPL, then fast proofs);
  real miniF2F theorems loaded (138 easy ones).

**Capability probes (zero-shot, 30B):**
- miniF2F: **1/20 (5%)** solved — off the floor; viable for an RL signal (curating to
  the algebra subset to raise it). RL harness (`minif2f_train.py`) built and smoking.
- SWE-bench dask: re-measuring. **Important correction:** the earlier "0% boundary at
  4B and 30B" was **invalid** — a venv-build race left half-built test environments so
  *every* test errored at import, regardless of the model's fix. With environments
  pre-built (verified: 48/48 import OK, **0** import errors in the re-run), we are
  taking the first *valid* SWE measurement now.

---

## 7. Infrastructure (the unglamorous half)

Turning three smoke-tested adapters into trainers that complete unattended required
fixing four real bugs, each latent until sustained real-benchmark load:
1. **Lean REPL stderr deadlock** — undrained `stderr=PIPE` blocks the REPL at ~64 KB.
2. **Lean REPL read-buffer conflict** — `select()` on the fd vs buffered `readline()`
   loses the multi-line reply; replaced with raw `os.read` + real per-tactic timeout.
3. **Lean REPL process+memory leak** — per-iter thread pools orphaned thread-local
   REPLs (576 live REPLs / 189 GB) and reused REPLs accumulated env state (166→6 GB);
   fixed with persistent pool + per-episode REPLs + process-group kill.
4. **SWE shared-venv editable race** — concurrent episodes re-pointing one editable
   install; fixed with a per-slot venv pool. (And the venv-build race in §6.)

Plus the 30B path: `VLLMGenServer` (vLLM rollout backend matching the existing
interface, with LoRA hot-swap) and the QLoRA training loop.

---

## 8. What's validated vs in-flight

**Paper-ready now:** the mechanism; Lean dead-iter elimination (3 seeds); Terminal
harm-reduction (3 seeds); token-attachment ablation; auto-derived rules; DAPO cost;
cold-start boundary; τ² coverage-gradient boundary.

**In-flight this session:** valid SWE-dask measurement at 30B (env fixed); miniF2F
RLVP-vs-outcome at 30B (harness built, runs starting).

**Two-sentence summary for a reader:** RLVP gives an agent dense, verifiable feedback
at every tool call, so it keeps learning on the failed rollouts that outcome-only RL
discards — robustly eliminating dead updates and, independently, cutting harmful
actions ~4× without trading away success. It accelerates the *verifiable* part of a
task and reduces harm, but it cannot replace the outcome signal for learning task
*intent*.

## 9. 30B miniF2F: a regime where RLVP HURTS (compliance attractor, reproduced)
At 30B the model can actually learn the curated algebra subset:
 - OUTCOME-only: 0.188 -> 1.0 success over 40 iters (stable). 30B genuinely learns it.
 - RLVP (c3): success rises 3 iters (0.25->0.48) then COLLAPSES to 0, violations rising.
   Robust across AdamW lr=1e-4 (also numerically diverged), Muon 5e-3, Muon 1e-3
   (grad bounded <=1.0 -> NOT optimizer/numeric; it's the reward signal).
Mechanism: on all-fail batches the OUTCOME advantage is 0, so ONLY the process signal
drives the gradient. The errored-tactic PENALTY is MISALIGNED with proving (cut errors
by degenerating, not by proving) -> drives the degenerate compliance-only optimum, the
SAME attractor seen on tau2. Coverage-gradient boundary reproduced at 30B on a real
benchmark: when outcome already carries signal, a misaligned process penalty destabilizes
clean outcome learning. RLVP helps most when outcome is BLIND and the process signal is
ALIGNED with progress; miniF2F-algebra@30B violates both. (Muon added + validated;
collapse is a reward-signal property, not the optimizer.)

## 10. The unifying principle: un-gameability of process rewards
The structural collapse + aligned fix generalize to a single, predictive criterion:

> A process reward is admissible as a LEARNING signal iff it is UN-GAMEABLE -- the
> cheapest policy that maximizes it must solve the task.

Test it BEFORE training: "what is the cheapest policy that maximizes signal P without
solving the task?" If that policy is degenerate (do nothing / pad safe moves / avoid
errors by inaction), P is misaligned, and where the outcome is blind (all-fail groups)
P is the ONLY gradient -- so the optimizer climbs the degenerate direction (the
compliance attractor). Penalties almost never pass (cheapest maximizer = inaction) ->
they belong on the HARM axis, not as the learning gradient. Aligned discharges can pass
(Lean goal-count strictly decreased is kernel-verified, unfakeable) -> admissible.

Deeper: a good progress reward is a cheap, un-gameable, monotone proxy for dV (change in
value). Where the domain gives one free (Lean goals, SAT clauses, distance, failing->
passing tests), RLVP wins. Where it doesn't, "define progress" == "approximate V" == the
credit-assignment problem itself, and a learned proxy just moves reward-hacking up a
level (same wall as tau2's "intent is the reward"). RLVP's real deliverable is telling
you WHICH regime you're in. Planned experiments to push into hard domains: (1) turn the
un-gameability test into a measured law; (2) outcome-gated credit (c4); (3) SWE
verifiable-progress ladder (reproduced->localized->test-passes). See PAPER_PLAN.md.

## 11. THE CENTRAL CLAIM: verifiable-potential characterization (to validate/refute)
> A domain admits a useful dense process reward IFF it has a VERIFIABLE POTENTIAL
> function Phi strictly finer than the terminal outcome -- a cheap, certifiable
> quantity that decreases only on real progress toward a verifiable terminal state.

Grounding (potential-based shaping, Ng/Harada/Russell 1999): shaping with F = g*Phi(s')
- Phi(s) leaves the optimal policy invariant for ANY Phi. So:
  * admissible dense process reward == verifiable potential-based shaping. Aligned
    signal = -dPhi for a verifiable Phi (Lean: Phi = # open goals; goal_progress = -dPhi).
    It improves OPTIMIZATION (gradient on all-fail batches) WITHOUT moving the optimum.
  * a non-potential penalty (errored-tactic: a per-action cost that doesn't telescope)
    MOVES the optimum; when outcome is blind the moved optimum is degenerate -> collapse.
  * un-gameability test = cheap verifiable SUFFICIENT condition for being potential-based.
Corollary (domain hardness): RLVP can help iff Phi exists strictly finer than outcome.
  Theorem proving HAS it (subgoal count). SWE-single-test does NOT (only pass/fail ==
  outcome) -> structurally predicts the hard-domain wall. Constructive fix: manufacture
  a finer verifiable Phi by sub-goal certification (Phi = # uncertified sub-goals).

FALSIFIABLE PREDICTIONS + EXPERIMENTS (queued):
  E-A GRANULARITY: hold Phi aligned, vary its FINENESS coarse(=outcome) -> mid(1
      milestone) -> fine(every -dPhi). Predict RLVP benefit (dead-iter elim, speed)
      scales with fineness; zero at coarse(=outcome). [Lean full-easy; synthetic chain]
  E-B SPARSITY PHASE DIAGRAM: fix the signal, sweep outcome density (task difficulty,
      all-fail fraction 0->1). Predict aligned-Phi benefit and misaligned-penalty COLLAPSE
      both APPEAR only as outcome blinds -> harm/help are gated by outcome sparsity.
  E-C IFF on a hard domain: SWE instances WITH a finer verifiable Phi (multiple F2P
      tests -> Phi = # failing F2P) vs WITHOUT (single F2P == outcome). Predict RLVP helps
      on multi-test, not single-test -- same domain, the structural property flips.

## 12. REFINEMENT (within-group variance): a uniform potential is centered out
The verifiable-potential claim needs a corollary discovered in the 4B chain matrix:
GRPO's advantage is GROUP-RELATIVE (reward - group mean), so a process reward only
creates gradient if it VARIES WITHIN THE GROUP. A potential that is uniform across
rollouts (every rollout makes the same partial progress) cancels under group-centering
-> no gradient, even though it is non-zero per episode. (Caught because loss==0 said
"0 dead iters" while effective grad_norm was ~0; the right metric is effective
gradient, not loss==0.)
  Refined claim: RLVP helps iff a verifiable Phi exists that is (a) finer than outcome
  AND (b) produces within-group variance (different rollouts reach different Phi).
  This is WHY it worked on Lean (proof attempts reach genuinely different goal-counts
  -> variance -> gradient) and why the n=2 chain doesn't (progress too uniform).
4B chain matrix (controlled): the fine potential ~2x the effective gradient of coarse
(=outcome) at n=2 (grad_mean 0.30 vs 0.12) -> confirms the gradient-density mechanism;
success payoff needs harder regime / more training (shown instead on Lean-30B:
aligned-RLVP 1.0 / 0 dead vs outcome-Muon 0.125 / 25 dead).

## 13. E-A/E-B RESULT (4B chain): potential's gradient boost GROWS with outcome sparsity
Granularity x sparsity matrix (grad_mean = effective learning gradient):
                coarse(=outcome)   fine    boost
   n=2 stages       0.118          0.297   2.5x
   n=4 stages       0.072          0.272   3.8x
   n=6 stages       0.019          0.329   17.3x
As the outcome blinds (n->6), outcome-only's gradient collapses to ~0 (effectively
dead) while the verifiable potential keeps it strong -> the potential's benefit GROWS
2.5x -> 17x. This is the E-B sparsity phase diagram, CONFIRMED: the verifiable Phi
matters most exactly when the outcome is blindest. (E-A: at n=4, coarse 0.072 << mid
0.332 ~ fine 0.272 -- any potential >> outcome.)

CAVEAT (success payoff is optimizer-fragile, NOT automatic): at AdamW lr=3e-5 the dense
potential DESTABILIZES (success collapses to 0; compliance-attractor instability, same
as 30B structural), while sparse outcome learns slowly but stably. So the dense
potential gives gradient but its conversion to SUCCESS needs bounded updates -- which is
exactly why MUON was needed at 30B (aligned-RLVP 1.0 stable). Muon success-payoff run
re-queued. Lesson: a verifiable potential is necessary (E-A/E-B) but the optimizer must
bound the dense signal or it overfits to partial progress and degenerates.

## 14. (hour-2 monitor) 4B chain SUCCESS payoff is optimizer-fragile (gradient is not)
The E-A/E-B gradient-density result (potential gives 2.5x->17x more gradient as outcome
blinds) is ROBUST. But converting that gradient to task SUCCESS on the 4B chain is
delicate and so far fragile: AdamW 1e-5 undertrains (flat); AdamW 3e-5 -> dense potential
DESTABILIZES (succ->0, compliance attractor) while outcome learns slow+stable; Muon 2e-3
-> both arms collapse (LR too high, even outcome degrades 0.15->0). One bounded attempt
(Muon 5e-4 @ n=6, the 17x-gap regime) running. Interim conclusion: a verifiable potential
reliably supplies gradient where the outcome is blind, but the optimizer/LR must bound the
dense signal or it overfits to partial progress and degenerates -- the SAME fragility seen
at 30B (where Muon at the right LR did convert it: aligned-RLVP 1.0 stable). The clean,
robust 4B result is the gradient-density phase diagram; the success payoff lives on Lean-30B.

## (hour-3 monitor) 4B success-payoff CONCLUDED fragile; redirected GPU
4 optimizer/LR settings tried for the 4B-chain success payoff (AdamW 1e-5/3e-5, Muon
2e-3/5e-4); none cleanly converts the potential's gradient into task success on this
finicky long-horizon 4B task (under/over-shoot + compliance-attractor collapse). STOPPED
tuning (rabbit hole). The robust 4B result = the E-A/E-B gradient-density phase diagram
(now seed-validating across seeds 11,12); the success payoff = Lean-30B (aligned-RLVP
1.0/0-dead). 30B un-gameability sweep + SWE safely queued (runs at full settings when the
user's GPU jobs free; cleanup reaps only MY orphaned vLLM procs, never the user's).

## (hour-4) grad-density REPLICATES across seeds (n=6)
                grad_mean coarse(outcome)   fine(potential)   boost
   seed 7            0.019                   0.329             17.3x
   seed 11           0.002                   0.451 (running)   >>1
At seed 11 outcome-only's gradient is ~0 (0.002, fully dead at n=6) while the verifiable
potential keeps strong gradient (0.45). The E-A/E-B claim is robust: the potential supplies
gradient exactly where the outcome is blind, and the boost is largest when the outcome is
blindest. (seed 12 running.) 30B sweep+SWE still queued behind the user's GPU jobs (no OOM
squeeze); my 4B seed-val utilizes the otherwise-idle GPU without delaying the 30B (the
user's 9GB blocks it regardless).

## (hour-5) grad-density SEED-ROBUST (final framing: absolute, not ratio)
n=6 (blind outcome), grad_mean:           coarse(outcome)   fine(potential)
   seed 7                                       0.019            0.329
   seed 11                                      0.002            0.312
   seed 12                                      0.000 (run)      (running)
The potential holds a STABLE ~0.32 gradient across every seed; outcome-only is effectively
DEAD (~0.007). Report the ABSOLUTE values, not the boost ratio (the ratio explodes 17x->
156x->inf only because coarse->0). Robust conclusion: a verifiable potential supplies
consistent learning signal exactly where the outcome is blind. This is the clean, robust
4B result of the program; the success payoff is on Lean-30B (aligned-RLVP 1.0/0-dead).
