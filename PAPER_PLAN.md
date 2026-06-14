# RLVP v2 — Plan for review (2026-06-13)

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
  outcome level in fewer samples / less wall-clock than outcome-only GRPO.
- **T2 (ceiling):** at convergence, RLVP's final outcome score is higher —
  outcome-only GRPO can permanently stall on hard tasks; RLVP does not.

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
