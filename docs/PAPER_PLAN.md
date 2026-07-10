# RLVP v2 — Plan for review (2026-06-13)

## CORE FRAMING (paper spine — do not lose this when writing) [added 2026-06-17]

**Agentic RL is not reasoning RL with more steps.** Reasoning models (math,
competitive programming) were the origin of RLVR (RL from Verifiable Rewards):
those tasks have a verifiable OUTCOME but no checkable PROCEDURE — you cannot
verify a chain-of-thought step mid-stream without redoing it — so outcome-only
RL is not a choice, it is the only signal available. The agentic era inherited
that recipe as if it were a law. But an agentic trajectory is a sequence of
ENVIRONMENT INTERACTIONS, and the environment is itself a VERIFIER: error codes,
tool results, state transitions, precondition/postcondition checks are all
verifiable intermediate signal that reasoning never had. Outcome-only RL throws
it away. So the case for moving beyond outcome-only is STRONGER in agentic
settings than in reasoning — the opposite of how the field treats it.
RLVR (verifiable outcomes) and RLVP (verifiable procedures) are COMPLEMENTARY
channels, not rivals: RLVP densifies + anchors early, RLVR carries terminal
truth. RLVR is the ceiling when no procedure is verifiable; RLVP is what
agentic environments additionally make available and inherited-RLVR leaves on
the table.

**TWO contributions (not four).**
1. **EFFICIENCY** (denser rewards == faster learning; these are ONE thing —
   denser verifiable signal -> more efficient sampling -> faster convergence).
   The process channel produces gradient from FAILED episodes, exactly where
   outcome-only GRPO is blind (all-fail groups -> zero advantage -> dead
   updates). Measured: ~4-7x fewer episodes to target success, near-zero seed
   variance. Operates when rules are OUTCOME-INSTRUMENTAL.
2. **HARM REDUCTION as an INDEPENDENT contribution** (merge "reduced harm" and
   "harm on an independent axis" — same concept). The outcome reward is defined
   on the TERMINAL STATE; harm lives on the PATH. An agent can reach a correct
   final state via an irreversible harmful path (delete the production DB, then
   rebuild it -> outcome=success, but catastrophic irreversible harm in the
   window). Outcome-only RL reinforces "delete and rebuild" because it only sees
   where you ended, not what you destroyed. Because harm is often IRREVERSIBLE,
   a successful terminal state does NOT redeem it. In deployment, NOT violating
   operating principles can matter MORE than achieving the goal (a goal
   not-yet-done is recoverable; a deleted DB is not). Outcome-only RL is
   CONSTITUTIONALLY BLIND to orthogonal harm (by definition: if it moved the
   reward it would not be a separate axis). RLVP is the only lever that reaches
   it. This is QUALITATIVELY beyond RLVR, not quantitatively better -> co-headline.
   A reward on terminal states is the WRONG OBJECT to express a constraint on
   paths; process rewards are NECESSARY, not merely efficient.

**The two axes ARE the two regimes of the coverage gradient.** Instrumental
rules -> efficiency (axis 1). Orthogonal-but-principled rules -> harm reduction
(axis 2). Same two-channel machinery; rule-to-reward alignment selects the axis.
CAVEAT (the tau2 collapse): harm reduction done naively triggers the
compliance-only attractor (minimize harm by not acting -> safe + useless). The
contribution is reducing harm WITHOUT the collapse — that is what two SEPARATE
channels + annealing + outcome-gating are FOR (protect the outcome axis from the
harm axis). Independence is what makes harm-reduction uniquely RLVP's to give
AND what makes it dangerous to give carelessly — same coin.

**Benchmark targets = high verifiable-FRACTION tasks (RLVP should GAIN, not just
neutralize harm):** SWE-bench Verified via SWE-Gym, TerminalBench, formal
theorem proving (Lean/miniF2F, per-tactic verification). Contrast: math /
competitive programming have a verifiable OUTCOME but no verifiable PROCEDURE
(single hard computation) -> RLVR ceiling, RLVP cannot help. tau2 is a HYBRID:
verifiable policy/validity layer + non-verifiable task-INTENT layer -> RLVP
neutralizes harm but cannot gain, because intent IS the reward and not a rule
(coverage gradient saturates at policy-validity; task-intent is irreducible).

## Positioning: RLVP as R1-Zero extended to long-horizon agents

R1-Zero self-evolves from a verifiable OUTCOME reward alone — elegant, but it
assumes the base policy samples *some* successes (true on math/code). On
long-horizon agents this breaks: ~85% all-fail groups at chain6, GRPO dead
39/60 iters -> pure outcome reward stalls (measured). RLVP is the MINIMAL
departure that restores self-evolution, and the departure is SPECIFICATION,
not DEMONSTRATION:
- rules (penalty/discharge) = a richer verifier (write-once, no solved examples,
  no per-example labels) — the R1-Zero-analog input.
- mixing (scripted demos) = a cold-start / SFT-analog (requires being able to
  produce a correct solution) — the un-R1-Zero input.
Headline method = CLEAN RLVP (rules only, NO mixing): demonstration-free,
self-evolving. Mixing kept only as the characterized R1-style cold-start for
the extreme-sparse regime; chain6/tau2 draw the boundary (R1-Zero vs R1 parallel).
Supervision spectrum of the rules (frontier = push all to automatic):
  automatic (repeat_failure <- env error signals) | general principle
  (read-before-mutate, verify-before-submit <- tool types) | domain policy
  (call-limits/auth <- provided policy doc, free with the benchmark).
CAPSTONE (future/stretch): auto-derive the process signal from environment
structure (tool pre/postconditions, error codes, state transitions) so even
the rules aren't hand-written -> maximal R1-Zero purity, no demos AND no
hand-written rules.

## Thesis (revised)

**Verifiable process rewards are a better RL training method than outcome-only
GRPO — measured by the outcome itself.** Two claims:

- **T1 (efficiency):** with the same episode budget, RLVP reaches any given
  outcome level in fewer samples than outcome-only GRPO/DAPO. [STRONGLY
  SUPPORTED: chain4 eps-to-50% RLVP 336+-0 vs GRPO 1280+-837 vs DAPO 2557.]
- **T1b (consistency):** RLVP has near-zero seed variance; outcome-only is a
  high-variance lottery (the all-fail cold-start). [SUPPORTED, emerged from
  seeds — promote to a headline claim alongside efficiency.]
- **T2 (ceiling):** RLVP's converged score is higher. [NOT SUPPORTED on chain
  tasks — they are compositionally easy, so outcome-only also reaches 1.0 given
  budget. Ceiling gap needs an irreducibly-hard subtask; best shot = tau2
  (compliance != success). Either demonstrate there or SCOPE THE CLAIM DOWN to
  efficiency+consistency.]

Compliance/internalization is demoted to a *secondary benefits* section
(real, already proven, but distillable via prompts on easy behaviors — not
the headline). The headline is benchmark score, judged by the outcome reward
alone, with the process machinery as the *means*.

## Why this should be true (mechanism, already measured)

1. **GRPO is blind in all-fail groups** (group-centered advantage ≡ 0). On
   hard tasks, early training is mostly all-fail groups → outcome-only GRPO
   has near-zero gradient exactly when learning matters most. The process
   channel (rule penalties + obligation-discharge credits) produces
   verifiable gradient FROM FAILED EPISODES. Measured already on chain4:
   outcome-only logs DEAD iterations (upd_s=0, all-fail-group fraction 1.0);
   RLVP logs zero dead iterations.
2. **Unsampled good behavior cannot be reinforced** (gradient ∝ 1-p; 0/48
   base sampling of key actions). Process-channel-only demo mixing injects
   the behavior even from demos that FAIL the task — measured: BC on failing
   demos → 0.00 success; RLVP with the same demos → 0.97.
3. Together: RLVP converts "verifiable wrongness/obligation" signals — cheap
   to specify, available at every step — into outcome-relevant gradient that
   outcome-only GRPO simply does not have.

## Positioning vs DAPO (the key baseline for the efficiency claim)

DAPO's **dynamic sampling** confronts the SAME all-fail-group problem but with
the opposite move: it DISCARDS groups with accuracy 0 or 1 and resamples until
the batch is full of informative (0<acc<1) groups. This guarantees every
gradient step is informative (no dead updates) — but it does NOT solve sampling
efficiency: the rollouts spent on discarded all-fail groups are sunk cost, and
on hard/low-success tasks the oversampling factor blows up (chain4 ~2x,
chain6 ~6x; can stall entirely if no informative group is ever found).

DAPO converts a zero-GRADIENT problem into a sampling-COST problem. RLVP instead
makes the all-fail group itself informative: on a binary outcome all failures
are identical (all reward 0, hence DAPO can only discard them), but verifiable
process rewards distinguish them (one reproduced-then-timed-out, one never read
the file) → gradient from the failures, no discard, no resample.

THEREFORE every efficiency comparison is measured in **episodes GENERATED**
(counting DAPO's discarded rollouts), not episodes used — else DAPO's resample
cost is hidden. DAPO's orthogonal tricks (clip-higher, token-level loss,
overlong shaping) are compatible with RLVP and noted as stackable, not
competing.

## Experiments

### E1. Difficulty calibration (no training, ~1h)
Chained multi-stage tasks (all-or-nothing outcome). Eval base Qwen3-4B on
chain-2/4/6/8; pick N* = smallest N with base success <= 0.3. This is the
hard sparse-outcome testbed. (Risk check: also confirms outcome-only has a
cold-start problem here: % all-fail groups at iteration 1.)

### E2. FLAGSHIP — 5-way on chain-N*, same budget (~25h + seeds)
Train: (a) outcome-only GRPO, (b) DAPO (outcome + dynamic sampling), (c) RLVP
= hybrid credit + process-channel-only demo mixing + anneal@40, (d) GiGPO-style
step advantages, (e) StepTool-style per-call rewards. 60 iters x 8 tasks x G=8;
eval every 6 iters. Headline trio (a) GRPO vs (b) DAPO vs (c) RLVP gets 3 seeds.

Metrics (all OUTCOME-side), X-AXIS = EPISODES GENERATED (counts DAPO resamples):
- success-vs-episodes-generated curve; episodes-generated-to-25%/50% (T1)
- final success + pass^8 at convergence; extend +40 iters if not plateaued (T2)
- all-fail-group fraction + DEAD-iteration count over time (GRPO mechanism)
- DAPO oversampling factor over time (episodes_generated / episodes_used) —
  the cost dynamic sampling pays and RLVP does not
- wall-clock per point of success (practical efficiency)

Kill criterion: if (c) RLVP does not beat BOTH (a) GRPO and (b) DAPO on
episodes-generated-to-50% AND final success at n=3 seeds, T1/T2 are refuted
on this testbed and we report it.

### E2b. Paired dead-iteration micro-experiment (controlled, ~1h)
Feed the IDENTICAL rollout batches to outcome-only vs RLVP credit and count
dead (zero-gradient) updates per method — makes the dead-iteration claim paired
rather than across-run. (Addresses the "different rollouts" caveat.)

### E3. tau2-bench head-to-head (~6-8h)
Real benchmark, base reward 0.44 = real headroom. Train outcome-only vs RLVP
(LoRA policy, tau2 user simulator on local vLLM, ENV-evaluator reward, NO
policy document in the training prompt). Report tau2 reward curves + final.
Secondary: the no-policy-prompt condition doubles as the deployment win
(drop ~10k tokens/call) if compliance internalizes along the way.

### E4. Component attribution on chain-N* (~8h, after E2 readout)
Which component buys the outcome gain? RLVP minus mixing; minus discharge
credits (penalties only); minus anneal; minus token channel (scalar only);
naive demo mixing (demos in the scalar baseline, LUFFY-style) vs our
process-channel-only mixing. One seed each, ranked by episodes-to-50%.

### E5. Integrity checks (cheap, alongside)
Outcome is env ground truth so the metric can't be gamed, but verify: no
overcaution regression (calls/ep, timeout rate), no reward-hacking of
discharge credits (idempotence audit), entropy/diversity retained.

### Secondary section (already in hand, no new compute)
Compliance internalization results (clean^k=1.00 no-prompt, guardrail and BC
comparisons, imperfect-demo result), to be reframed as supporting evidence
that the process channel teaches real behavior, not prompt-following.

## What changes vs the v1 plan

- CANCELLED: easy-env seed farm (B), clean-holdout + persistence (C),
  Mistral replication (F) — compliance-thesis robustness, now optional.
- REPURPOSED: horizon chains (D) -> the flagship testbed; GiGPO/StepTool
  (E) -> flagship comparators on outcome; tau2 (G) -> head-to-head with an
  outcome-only baseline arm (was RLVP-only).
- KEPT from in-flight work: Stage A (process-channel-only mixing validation)
  — it is the core RLVP component; its run finishes and the old driver stops.

## Risks / honest caveats

- Chain-N* may be too hard for BOTH methods (success ~0 throughout). Fallback:
  curriculum (start N=2, switch to N*) or increase mixing to 2 demos/group.
- GiGPO/StepTool are reimplementation-style adaptations (sparse-outcome
  setting); labeled as such, not as faithful reproductions.
- tau2 sims are expensive (~1-2 min each); 30 iters x 24 sims is the budget
  ceiling locally. If too noisy, reduce to telecom-small or raise G.
- Toy-env saturation was the v1 blind spot; E1 exists so we never again run
  a comparison without verified headroom.

## Execution order and est. GPU time

E1 (1h) -> E2 main (20h) -> E3 (8h) -> E2 seeds (10h) -> E4 (8h) -> writeup.
~2.5 days GPU total. Driver: scripts/run_flagship.sh (written, NOT launched —
awaiting review).

================================================================================
## NEW CORE PRINCIPLE (do NOT forget — promote to a main paper section)
### "Admissibility of process rewards: the un-gameability criterion"
================================================================================

UNIFYING CLAIM (ties tau2 + 30B miniF2F into ONE principle, stronger than 3 obs):
  A process reward is admissible as a LEARNING signal iff it is UN-GAMEABLE: the
  cheapest policy that maximizes it must solve the task. Operational test, applied
  BEFORE training: "what is the cheapest policy that maximizes signal P WITHOUT
  solving the task?" If that policy is degenerate (do nothing / pad safe moves /
  avoid errors by inaction), P is misaligned and -- in the regime where outcome is
  BLIND (all-fail groups) and P is the ONLY gradient -- the optimizer WILL climb the
  degenerate direction. This is the compliance attractor.

  * A PENALTY almost never passes (cheapest maximizer = inaction). Penalties belong
    on the HARM axis (bound the path), NEVER as the learning gradient when outcome
    is blind.
  * An ALIGNED DISCHARGE can pass: Lean goal-count STRICTLY DECREASED is kernel-
    verified and cannot be earned by degenerating -> admissible, dense, safe.

EMPIRICAL SWEEP — COMPLETE (30B Qwen3-30B-A3B, miniF2F algebra, vLLM-fp8 + QLoRA +
Muon, lr=1e-3, 14 iters, seed 7; full table in FINDINGS.md sec 15). Five process-signal
variants, each a pre-registered cheapest-gaming-policy, scored by terminal success (last3
of 14) + whether it STAYS dead:
  arm         signal                                    last3   terminal
  validgated  gameable discharge, OUTCOME-GATED (c4)     0.305   ALIVE (best)
  aligned     goal-progress discharge only (no penalty)  0.243   ALIVE
  valid       any-valid discharge, ungated (gameable)    0.180   ALIVE, declines
  structural  discharge + errored/no_progress PENALTIES  0.00    DEAD (stays 0)
  noerror     errored PENALTY only (no discharge)        0.00    DEAD (stays 0)

  THREE RESULTS:
  (1) PENALTY IS THE LETHAL INGREDIENT. Both penalty-bearing arms (structural, noerror)
      collapse to the compliance/inaction attractor and STAY dead; both penalty-free arms
      survive. It is the misaligned penalty -- satisfiable by not acting -- that kills the
      policy, not the presence of a dense signal. (Refines the earlier "structural
      collapses" obs: the discharge is not what saves structural; removing the penalty is.)
  (2) OUTCOME-GATING RESCUES a gameable discharge. The SAME farmable any-valid discharge is
      harmful ungated (valid declines to 0.180 as discharge/ep climbs) but BEST gated on
      success (validgated 0.305). c4-gating converts a farmable signal into a useful one ->
      this is experiment #2 below, CONFIRMED.
  (3) Robust across the single-seed noise (see caveat); the qualitative ordering
      penalty-free-survive > penalty-bearing-die, and gated > ungated, is stable.

  HONEST CAVEAT (in the paper): 30B/lr=1e-3/single-seed is HIGH-VARIANCE -- every arm swings
  iter-to-iter (aligned touches 0.0 twice mid-run, recovers to 0.65). The robust
  discriminator is last3 + STAYS-dead, NOT a single final iter (a naive last-iter heuristic
  over-flags aligned as collapsed). Multi-seed (seeds 8,9; running) firms the exact numbers;
  ordering already stable.

DEEP POINT (for discussion): a good progress reward is a cheap, un-gameable, monotone
proxy for dV (change in value function). When the domain provides one for free
(Lean goals, SAT clauses, distance, failing->passing tests) RLVP shines. When it
does NOT, "define progress" == "approximate V" == the credit-assignment problem
itself; a learned proxy (LLM-critic) just relocates reward-hacking one level up.
This is the same wall as tau2 "intent is the reward, learnable only from outcome".

COMPLETED EMPIRICAL ABLATION (the learned-proxy arm; DONE, see SELFCRITIC.md + paper
\S"Verifiable Rules vs. a Learned Self-Critic", sec:selfcritique):
  Tests the line-264 claim directly. Same-model on-policy self-critique (no distillation)
  as a drop-in for the rule channel (credit="llmcritic" in grpo.py), across a 2x2 of
  (rule specifiable?) x (critic detects?). Results:
  * DETECTOR: blind self-critique is structurally blind to stateful-bookkeeping rules
    (recall ~0, even told the rules), persists 1.7B->8B on fixed trajectories, overall
    recall flat 0.46/0.38/0.41 -> does NOT scale (one masked rule is capability-gated =
    distillation-only).
  * REWARD (all-fail regime, csops, 1.7B, 3 seeds): a penalty-only RULE with identical
    credit structure + PERFECT recall is decisive every seed (viol->~0.05 stuck, or
    escape to 0.99 success); live AND frozen self-critics are inert. frozen~=live =>
    cause is the critic's ~6% false-POSITIVE imprecision, NOT non-stationarity. Gap is
    all-fail-regime-specific (washes out at 4B once outcome dominates).
  * INTENT frontier (tau2, 3 seeds): self-critique DETECTS intent failures offline
    (failure-pred F1 0.63 vs rules 0.23; flags 0.42 of rule-clean failures) but COLLAPSES
    as a training reward (0.01->0.00 vs outcome 0.50, sem-rules 0.35).
  * No self-rewarding bootstrap (critic-oracle agreement flat across training).
  PAYOFF: empirically substantiates "a learned proxy relocates reward-hacking one level
  up" and closes the "why not let the model judge itself?" reviewer attack; upgrades the
  tau2 boundary (E3/sec:boundary) into a TWO-SIDED result. Tables tab:sc-train,
  tab:sc-tau2. Positioning added vs self-rewarding LMs / RLAIF / Constitutional AI / LLM-judge.

PAPER INTEGRATION:
  - New section after the mechanism: "Admissibility of process rewards" w/ the
    un-gameability test + the triptych figure (3 curves: outcome / structural-collapse
    / aligned-stable).
  - Learned-critic ablation (above) goes right AFTER the alignment-boundary section
    (sec:boundary -> sec:selfcritique): it is the boundary's empirical companion (rules
    can't reach intent; a learned critic detects intent but can't train it).
  - Reframe harm result (Terminal) as the LEGITIMATE penalty use: bound harm on the
    independent axis, NOT a learning signal.
  - Fold tau2 coverage-gradient + miniF2F collapse as the SAME phenomenon, 2 scales.

EXPERIMENTS — STATUS (FINDINGS.md sec 15):
  #1 Un-gameability as a MEASURED LAW [DONE]: 5 Lean process-signal variants spanning
     aligned->misaligned (aligned / valid / noerror / structural / validgated), each
     pre-registered with its cheapest gaming policy. RESULT: penalty-bearing variants die,
     penalty-free survive (see sweep table above). Criterion -> measured test, confirmed.
  #2 Outcome-gated process credit (c4) [DONE]: discharge paid ONLY if the episode
     eventually succeeds. RESULT: the SAME gameable discharge declines ungated (valid 0.180)
     but recovers to best gated (validgated 0.305) -> outcome-gating restores admissibility.
     Confirmed.
  #3 HARD-DOMAIN (SWE-bench dask, blind 0% domain) [DONE, as a NEGATIVE/boundary result]:
     two arms, structural(c3) vs gated(c4), 16 iters each. RESULT: BOTH stay at 0% success
     -- structural farms its discharge (~1/ep) without ever fixing a bug; gating withholds
     the credit so there is ~0 usable signal. Demonstrates the ABSENCE of a free un-gameable
     progress metric in a blind hard domain: when no cheap verifiable progress exists that
     the agent can't game, there is no safe dense signal to add. Matches the
     verifiable-potential frame (no Phi strictly finer than outcome that any rollout
     actually moves). [E-C below probes WHY: is a finer Phi even reachable on these
     instances?]

## VERIFIABLE-POTENTIAL CLAIM (central, see FINDINGS.md sec 11): RLVP helps iff domain has a verifiable Phi strictly finer than outcome. Formalize via potential-based shaping (Ng 1999). Experiments E-A granularity / E-B sparsity-phase-diagram / E-C SWE multi-vs-single-F2P-test. Validate or refute before paper.

## 2026-06-29 FRAMING PIVOT (supersedes the "criterion that always works" framing; see FINDINGS sec 20)

The prospective-prediction refutation + multi-seed firming + 8B-Lean replacement force a pivot
from a PRESCRIPTIVE criterion to a MECHANISM + honest-negative paper.

NEW THESIS (one line): a group-relative advantage IS a within-group variance, which collapses
to zero at BOTH ends of the success rate -- on all-fail groups (early) AND all-success groups
(late). A dense verifiable potential helps iff it supplies variance in one of these vacuums and
its intermediate values are reachable. Everywhere else it is redundant -- and on real multi-task
distributions, "everywhere else" is most of training, which is why headline efficiency gains
do not replicate.

NEW TITLE (candidate): "The Variance Vacuum: What Dense Process Rewards Actually Do for Agentic
RL (and Why the Speed-ups Don't Replicate)."

STANDS (robust, keep): variance identity / mechanism (now BOTH ends); 4B 3-seed dead-iteration
elimination SCOPED to controlled synthetic (all-fail rescue, re-confirmed: var_phi>0 while
var_O=0); 4B 3-seed harm reduction ~4x at equal success; SWE reachability wall (Var_G(Phi)=0);
the un-gameability sweep (compliance attractor); intent-ceiling / learned-critic / discovery wall.
SCOPED/REFUTED (correct, do not overclaim): real-task efficiency-as-SPEED (#1) does NOT replicate
-- DROP it; reachability-ALONE is necessary not sufficient (a learnable foothold lets outcome
bootstrap); 30B/8B-Lean efficiency is noise-dominated (sign-flips under re-seeding).

FOUR DEFENSIBLE CONTRIBUTIONS (the related-work survey 2026-06-29 shows the naive "process
rewards help" and "progress=advantage" are NOT novel -- position carefully):
  1. the VERIFIABLE / un-gameable instantiation (vs Setlur progress=advantage, which uses a
     learned hackable prover; vs DeepSeek-R1 abandoning PRMs for hacking).
  2. the BOTH-ENDS unification: all-success blindness, not just all-fail (DAPO's dynamic
     sampling drops all-correct groups too -> external corroboration; we SUPPLY vs DROP).
  3. a rigorous HONEST-NEGATIVE + REPRODUCIBILITY result for AGENTIC process rewards (sign-flip
     under re-seeding) -- the agentic instance of the Sober-Look / Spurious-Rewards crisis;
     ours is sharper (sign, not magnitude). cf. Practitioner's-Guide: dense accelerates but
     final perf depends on estimator.
  4. per-tool-call VERIFIABLE potentials as penalty-based shaping = genuine whitespace.

FIGURES: CUT fig_efficiency_at_scale (#1, invalidated). REDESIGN fig_criterion_map -> a
variance-vacuum schematic (Var_G(O)->0 at both ends of success rate). ADD: all-success drift
figure (aligned holds vs outcome drifts) + reproducibility panel (sign-flip / 8B n=4 spread).
KEEP: ungameability_sweep, ec_reachability, verifier_bound, phase_diagram (w/ necessary-not-
sufficient caveat). Related work REWRITTEN (body.tex, 5 paras) + ~23 verified bib entries added.

EXPERIMENT PROGRAM (E0 done; E1 is the GATE for whether all-success-drift becomes a headline):
  E0 re-analysis: drift real+common in outcome but confounded by noisy vLLM Lean (DONE, inconcl).
  E1 KEYSTONE (running): controlled all-success drift, 4B chain, instrumented (var_O, var_phi),
     n-seeds, past mastery. CRITICAL SUBTLETY: completion-Phi SATURATES at success (var_phi->0
     at top) so it drifts like outcome; a HOLDING potential needs WITHIN-SUCCESS variance (as
     Lean proof-length naturally has). If E1 shows outcome drifts when var_O->0 AND a within-
     success-varying potential holds -> headline. If not -> all-success stays a suggestive
     extension, lead with the negative+mechanism+harm-bounding (deflationary framing).
  E2 mechanism isolation (coarse/saturating vs fine), E3 2nd domain, E4 re-firm kept positives
     n=5, E5 reproducibility distribution, E6 more 8B-Lean seeds. Gated on E1.

RESULT (2026-06-29, FINDINGS 21-22, FINAL_SUMMARY.md): E1 gate FAILED for the all-success "fix"
(completion-Phi saturates; no clean mastery; var_O~0) -> deflationary framing locked. E4 DONE
n=5: dead-iter rescue is PARTIAL/reachability-gated -- fine 6.8 vs coarse 14.6 truly-dead iters
(2.15x fewer, fine<coarse every seed, NEVER 0) -> SCOPE the "0 vs 16/40" claim. E2/E3/E6 NOT run
(gate failed; mechanism is supported by E4 var_phi measurement + DAPO corroboration + SWE wall).
Consolidated ledger + final title ("The Variance Vacuum...") in FINAL_SUMMARY.md. The unifying
spine: a potential helps only insofar as it supplies REACHABLE within-group variance (var_phi>0),
which governs dead-iter rescue, all-success holding, and efficiency -- and usually fails. Robust
deployable win = un-gameable harm-bounding (TODO: re-verify n>=5).
