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
| Self-critique vs rules ([SELFCRITIC.md](SELFCRITIC.md)) | ✅ complete (Exp0+probe+Exp1 3-seed+frozen+4B+τ² offline+train) | rules>self-critique even at equal structure+perfect recall (critic's 6% FP, not non-stationarity; all-fail-regime-specific); self-critique only useful as an *offline intent detector* (F1 .63 vs .23) — as a training reward it **collapses** (0.0 vs outcome 0.5) |
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


## 13. E-A/E-B VALIDATION (4B synthetic chain): the verifiable potential supplies gradient where outcome is blind
Controlled test of the central claim. Knobs: granularity of the verifiable potential
Phi=#satisfied-stages (coarse=outcome / mid=1 milestone / fine=every -dPhi) x outcome
sparsity (n_stages 2/4/6; more stages -> blinder outcome). All arms credit=c3 (potential-
based shaping, no penalties).

E-A/E-B grad-density (effective learning gradient, grad_mean):
                coarse(=outcome)   fine(potential)   boost
   n=2 stages       0.118            0.297            2.5x
   n=4 stages       0.072            0.272            3.8x   (mid 0.332 ~ fine: any potential >> outcome)
   n=6 stages       0.019            0.329            17x
The potential's gradient advantage GROWS as the outcome blinds (2.5x -> 17x): outcome-only
collapses toward zero gradient (effectively dead) while the potential stays strong. This is
the sparsity phase-diagram prediction, confirmed.

SEED-ROBUST (n=6, 3 seeds 7/11/12):
   coarse(outcome) : [0.019, 0.002, 0.008] -> 0.010 mean   (DEAD)
   fine(potential) : [0.329, 0.312, 0.376] -> 0.339 +/- 0.027  (ALIVE, tight)
~34x gradient difference, tight across seeds.

TWO REFINEMENTS the chain exposed (both important, both real):
 (a) WITHIN-GROUP VARIANCE: GRPO's advantage is group-relative, so a potential only gives
     gradient if it VARIES across rollouts in a group; a uniform potential is centered out.
     (This is WHY Lean works -- proof attempts reach different goal-counts -- and a too-easy
     task with uniform progress would not.) The metric must be EFFECTIVE gradient, not loss==0.
 (b) SUCCESS PAYOFF IS OPTIMIZER-FRAGILE: the potential reliably supplies gradient, but
     converting it to task SUCCESS on the 4B chain was fragile across 4 optimizer/LR settings
     (AdamW under/over-shoots; the dense signal can hit the compliance-attractor collapse).
     The success payoff is demonstrated cleanly on Lean-30B (aligned-RLVP 1.0 / 0 dead,
     stable under Muon) -- the optimizer must BOUND the dense signal or it overfits to
     partial progress and degenerates.

NET: the verifiable potential provides learning signal exactly where the outcome is blind
(robust, seed-validated); converting that signal to success needs an un-gameable potential
(sec 11) AND a bounded optimizer. Refines, does not refute, the central claim.

## STATUS (end of overnight monitoring block)
DONE + seed-robust: core mechanism; Lean efficiency (3 seeds); Terminal harm (3 seeds);
tau2 boundary; 30B miniF2F compliance-attractor + aligned-RLVP fix; un-gameability /
verifiable-potential principle; E-A/E-B grad-density (3 seeds). QUEUED (GPU-blocked by the
user's jobs, runs at gpu_mem 0.48 when the GPU frees): 30B un-gameability sweep
(aligned/valid/noerror/structural/c4-gated) + SWE structural/gated; E-C (SWE multi-vs-
single-F2P) not yet built.

## 14. SYSTEMS OBSERVATION: the bottleneck is TASK-DEPENDENT -- our Lean run was verifier(CPU)-bound
SCOPE: this is a measured observation about ONE workload (30B miniF2F theorem-proving), NOT a
general law about verifiable agentic RL. Whether a run is GPU-bound or CPU/verifier-bound
depends on the specific task -- do not assume either.

What we measured (Lean theorem proving, 30B): GPU MEMORY ~86GB (full -- 30B weights + KV +
QLoRA) but GPU COMPUTE only ~4-15% SM (avg ~8%, bursty 0->28%). In THIS task the binding
resource is the CPU-side verifier, not the model: each rollout step generates ONE short tactic
on the GPU (~ms) then waits on the Mathlib REPL where the Lean KERNEL checks it on CPU
(~seconds); across ~8 sequential proof steps x 48 episodes, wall-clock is dominated by
environment latency + Python orchestration, while the GPU-heavy parts (vLLM gen, QLoRA
backward) are brief bursts. So for THIS run, scaling the GPU does little and CPU contention
directly throttles it (observed: a co-tenant 27-core job pushed load to 38/32 and starved our
rollouts; when it freed, load dropped to ~4.5 and iters sped up).

Where this does and does NOT transfer (the bottleneck is per-task, must be profiled):
 * CPU/verifier-bound when the checker is a cheap CPU process AND generation is short:
   Lean kernel (measured), and plausibly pytest (SWE), Docker exec (Terminal) -- but verify
   per task, not assume.
 * GPU-bound when generation dominates or the verifier itself runs on the GPU: long-CoT /
   long-horizon rollouts with large outputs, big batches, a LEARNED reward model or LLM-judge
   verifier, or simulators/renderers that run on the GPU. There the GPU stays busy and the
   CPU verifier is cheap.
 * Mixed/shifting: the same pipeline can flip between regimes as you change batch size,
   sequence length, model size, or verifier cost.
Takeaway: PROFILE each task (SM% vs verifier latency) rather than assuming "GPU is the
bottleneck." For the verifier-bound case specifically, the levers are environment-verification
parallelism (more REPL/test workers, async rollouts overlapping CPU-verify with GPU-generate)
and co-locating many verifier-bound runs per GPU; for the GPU-bound case those do nothing and
the usual GPU-scaling intuitions apply. The practical point for RLVP: a cheap machine-checkable
verifier can move the bottleneck OFF the GPU and onto the CPU -- when it does, the systems
design must follow -- but this is a property of the task's verifier, not of RLVP in general.

---

## 15. 30B UN-GAMEABILITY SWEEP (complete) — penalty-is-lethal, gating rescues, hard-domain limit
Pre-registered sweep on Qwen3-30B-A3B (vLLM-fp8 rollouts + 4-bit QLoRA + Muon, lr=1e-3,
14 iters Lean / 16 iters SWE, seed 7). Each arm = one "cheapest gaming policy"; we ask
whether the process signal survives an agent that games it.

### Lean (miniF2F algebra) — 5 arms
| arm | rule / credit | signal | first3 | last3 | terminal |
|-----|---------------|--------|--------|-------|----------|
| validgated | valid / **c4 (outcome-gated)** | gameable discharge, gated on success | 0.361 | **0.305** | ALIVE — best |
| aligned    | aligned / c3 | goal_progress discharge only (no penalty) | 0.368 | 0.243 | ALIVE |
| valid      | valid / c3   | any-non-error discharge (gameable), ungated | 0.354 | 0.180 | ALIVE, declines |
| structural | structural / c3 | goal_progress discharge **+ errored/no_progress PENALTIES** | 0.382 | **0.00** | DEAD (stays 0) |
| noerror    | noerror / c3 | errored **PENALTY only** (no discharge) | 0.34 | **0.00** | DEAD (stays 0) |

### SWE-bench dask (hard / blind domain) — 2 arms
| arm | credit | succ (all 16 iters) | disch/ep | reading |
|-----|--------|---------------------|----------|---------|
| swe_structural | c3 | **0/16 (0.0 every iter)** | ~1.0 flat (peak 2.19) | FARMS the discharge, never solves |
| swe_gated      | c4 | **0/16 (0.0 every iter)** | bounces 0.4–2.5 | gating WITHHOLDS credit, ~0 usable signal |

### Three findings
1. **The PENALTY is the lethal ingredient.** Both penalty-bearing Lean arms (structural,
   noerror) collapse to the compliance/inaction attractor and STAY dead (last 4–5 iters = 0);
   both penalty-free arms (aligned, valid) survive. A misaligned penalty — satisfiable by not
   acting — is what kills the policy, not the presence of a dense signal per se. (Confirms the
   original compliance-attractor result at 30B.)
2. **Outcome-gating RESCUES a gameable discharge.** The same gameable "any-valid-tactic"
   discharge is harmful ungated (valid declines to 0.180 as it farms the signal — discharge/ep
   climbs) but BEST when gated on the real outcome (validgated recovers to 0.305). Gating the
   process credit on terminal success converts a farmable signal into a useful one — the core
   RLVP design lever.
3. **Hard domain → no free un-gameable progress.** On blind SWE (0% solve), neither SWE arm
   moves success off zero: structural farms its discharge (~1/ep) without ever fixing a bug,
   and gating withholds the credit so there is ~0 learning signal. When no cheap verifiable
   progress metric exists that the agent can't game, there is no safe dense signal to add —
   the dense-reward win is domain-gated, exactly as the verifiable-potential frame predicts.

### Honest caveats
- 30B at lr=1e-3, single seed, small batch is HIGH-VARIANCE: every arm swings iter-to-iter
  (aligned hits 0.0 twice mid-run and recovers to 0.65). The robust discriminator is the
  **terminal state over last3 + whether it STAYS dead**, NOT a single final iter. (The helper
  `ungameability_report.py` uses a single-final-iter heuristic and therefore over-flags aligned
  as "collapsed" because aligned's very last iter was 0.04, though its last3 is 0.243 and it is
  clearly alive.) Multi-seed runs would firm up the exact last3 numbers; the qualitative
  ordering (penalty-free survive, penalty-bearing die, gating rescues) is stable.
- SWE arms are a 0%-success regime: they demonstrate the ABSENCE of a usable dense signal in a
  blind hard domain, not a positive training result.

---

## 16. E-C — SWE verifiable-potential: a finer Phi is STRUCTURALLY RARE (CPU part done)
Tests the verifiable-potential claim (sec 11) in SWE: Phi = (#FAIL_TO_PASS tests passing)/
total. A dense process reward can only help if Phi is strictly FINER than the all-pass
outcome AND intermediate Phi is reachable.

**Structural split (48 small-patch dask instances, metadata only):**
- **SINGLE-F2P (|F2P|==1): 32 / 48 (2/3).** Phi in {0,1} == outcome — *no finer potential
  exists at all*; the claim predicts dense process reward cannot help on these by construction.
- **MULTI-F2P (|F2P|>=2): 16 / 48 (1/3).** A strictly finer Phi exists — but coarse:
  F2P-count distribution {1:32, 2:10, 3:3, 4:1, 5:1, 9:1}, i.e. most multi-F2P instances
  have only 2–3 sub-tests.
- **Finding: in SWE a finer verifiable potential is structurally RARE and shallow** — most
  bug-fix instances are all-or-nothing. This is a domain-level reason the sec-15 SWE arms
  found no usable dense signal: for 2/3 of instances there is no finer Phi to exploit even
  in principle.

**Instrument validated (CPU, no GPU):** for 4 instances (3 multi + 1 single), Phi reads
(0, total) on the un-fixed base and (total, total) after the gold patch — instrument_ok on
all. Phi measures exactly #F2P-pass; the 0 and full extremes are reachable.

**PENDING (GPU, queued):** rollout-reachability — run G 30B rollouts/instance with
measure_phi and compare WITHIN-GROUP Phi variance on multi-F2P vs single-F2P. The sharp test:
do multi-F2P rollouts ever land at INTERMEDIATE Phi (0<Phi<1)? If yes -> multi-F2P has a
usable finer potential single-F2P lacks (predicts dense reward helps there). If multi-F2P
rollouts are also all-or-nothing -> the finer potential is VACUOUS (unreachable), explaining
the sec-15 null directly. Run: `python3 scripts/ec_f2p.py rollout --n 16 --iters 6` when GPU free.

**E-C rollout-reachability (30B, COMPLETE on the multi side):** ran G=6 rollouts on each of
the 16 multi-F2P instances (96 rollouts) + 10/16 single-F2P (60 rollouts), measuring Phi per
rollout. **Every one of the 156 rollouts scored Phi=0** (n_partial=0, n_solved=0, phi_var=0
in every group) -- the 30B never made even a SINGLE FAIL_TO_PASS test pass, let alone an
intermediate fraction. So although a finer Phi STRUCTURALLY exists for the 16 multi-F2P
instances, it is **empirically VACUOUS at 30B: unreachable, zero within-group variance**.

**Why this matters (mechanism):** GRPO's advantage is group-relative, so a process potential
only yields gradient if it VARIES across the rollouts in a group. Phi here is a constant 0 in
every group -> it would be centered out -> **zero gradient**. This is the direct mechanistic
cause of the sec-15 SWE null: the dense SWE signal failed not because it was gameable but
because **the policy gets no partial traction at all** -- there is no reachable intermediate
state for any potential, gameable or not, to reward. "No verifiable Phi finer than outcome
that the policy can move" is the operative condition, and it is empirically true here.

**Honest caveat:** this is measured at 30B on SWE-bench dask, a 0%-solve regime for this model.
The finding is conditional on REACHABILITY -- a stronger model (or easier instances) that
achieves partial fixes would make the multi-F2P potential non-vacuous and potentially useful.
The claim is therefore: a finer verifiable potential helps iff it both EXISTS (structural:
multi-F2P) AND is REACHABLE (empirical: partial Phi occurs with within-group variance). In SWE
at 30B, the first holds for 1/3 of instances but the second fails everywhere we measured.

## 15b. MULTI-SEED FIRMING (n=3, seeds 7/8/9) — sharpens the un-gameability sweep
The single-seed (seed-7) sweep in sec 15 was high-variance; we re-ran all 5 Lean arms at
seeds 8 and 9 (Qwen3-30B, 14 iters, Muon lr=1e-3). Per-arm terminal success (last3),
mean +/- std over 3 seeds:

| arm | rule / credit | seed7 | seed8 | seed9 | mean +/- std | verdict |
|-----|---------------|-------|-------|-------|--------------|---------|
| aligned    | aligned / c3       | 0.243 | 0.993 | 0.993 | 0.74 +/- 0.35 | ALIVE (variable) |
| validgated | valid / c4 (gated) | 0.305 | 0.333 | 0.743 | 0.46 +/- 0.20 | ALIVE |
| valid      | valid / c3         | 0.180 | 0.236 | 0.215 | 0.21 +/- 0.02 | ALIVE (very stable) |
| structural | structural / c3    | 0.000 | 1.000 | 0.007 | 0.34 +/- 0.47 | EXTREME variance |
| noerror    | noerror / c3       | 0.000 | 0.007 | 0.000 | 0.002 +/- 0.003 | RELIABLY DEAD |

REVISED (more honest) FINDINGS — the n=3 data refines the single-seed "penalty kills":
1. **Penalty-free signals reliably SURVIVE.** All three (aligned, valid, validgated) stay
   well above the dead zone on every seed. valid is the most stable (0.21+/-0.02);
   outcome-gated validgated is the best stable survivor (0.46+/-0.20); aligned is highest on
   average but seed-variable (one low seed).
2. **PURE penalty (noerror, no discharge) RELIABLY DIES** — 0.002+/-0.003, the only arm at
   ~0 on every seed. This is the robust compliance-attractor result.
3. **Penalty+discharge (structural) is EXTREME-variance, not reliably dead.** 0.00/1.00/0.007
   across seeds: it collapses on 2 of 3 seeds but the goal-progress discharge can occasionally
   rescue it all the way to 1.0. So the single-seed claim "structural dies" was a seed artifact;
   the honest claim is "structural usually collapses but is high-variance — the discharge
   sometimes saves it." The large error bar IS the result.
NET: the robust law is penalty-free-survives vs PURE-penalty-reliably-dies; adding a discharge
to a penalty does NOT reliably save it. Outcome-gating yields the most consistent survivor.
This is what the paper's Figure (fig_ungameability_sweep, now n=3 with error bars) shows.

## 17. POSITIVE-AT-SCALE (#1): efficiency win at 30B on real hard theorems
Closes the "all positive results are 4B/synthetic" gap. Aligned verifiable-progress potential
(c3) vs outcome-only baseline, Qwen3-30B, HARD miniF2F (easy_only, NOT algebra-only -> harder,
more all-fail), 16 iters, seed 7, identical optimizer (Muon lr 1e-3, only the credit differs).
- aligned succ: [0.19,0.29,0.35,0.94,1.0,...x12] -> first reaches 1.0 at ITER 5, holds.
- outcome succ: [0.17,0.23,0.02,0.02,0.04,0.48,0.27,0.65,0.75,0.88,0.79,1.0,0.98,1.0,1.0,1.0]
  -> STALLS near 0 (iters 3-5, the all-fail dead-zone) then climbs, first 1.0 at ITER 12.
FINDING: aligned reaches mastery >2x faster (5 vs 12 iters) AND skips the early all-fail stall.
HONEST framing: both eventually saturate (the hard set IS learnable by 30B given budget), so
the win is SPEED + dead-zone-avoidance, not final success. The criterion's YES at scale:
hard-enough that all-fail dominates AND reachable (goal-count decreases occur) -> dense reward
has the most uniquely-available gradient. NB recovered from the vLLM LoRA-hot-swap leak (fixed:
free old adapter + max_loras=1). Figure fig_efficiency_at_scale.pdf; written into paper Sec
"Sample-Efficiency in Theorem Proving". Doubles as the phase-diagram "sparse + reachable ->
big benefit" anchor.
