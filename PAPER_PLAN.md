# RLVP v2 — Plan for review (2026-06-13)

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
   verifiable gradient FROM FAILED EPISODES. Measured already: grad-norm
   0.001 vs 0.6 (500x) depending on whether a non-outcome channel exists.
2. **Unsampled good behavior cannot be reinforced** (gradient ∝ 1-p; 0/48
   base sampling of key actions). Process-channel-only demo mixing injects
   the behavior even from demos that FAIL the task — measured: BC on failing
   demos → 0.00 success; RLVP with the same demos → 0.97.
3. Together: RLVP converts "verifiable wrongness/obligation" signals — cheap
   to specify, available at every step — into outcome-relevant gradient that
   outcome-only GRPO simply does not have.

## Experiments

### E1. Difficulty calibration (no training, ~1h)
Chained multi-stage tasks (all-or-nothing outcome). Eval base Qwen3-4B on
chain-2/4/6/8; pick N* = smallest N with base success <= 0.3. This is the
hard sparse-outcome testbed. (Risk check: also confirms outcome-only has a
cold-start problem here: % all-fail groups at iteration 1.)

### E2. FLAGSHIP — 4-way on chain-N*, same budget (~20h + ~10h seeds)
Train: (a) outcome-only GRPO, (b) RLVP = hybrid credit + process-channel-only
demo mixing + anneal@40, (c) GiGPO-style step advantages, (d) StepTool-style
per-call rewards. 60 iters x 8 tasks x G=8; eval every 6 iters.
Headline pair (a) vs (b) gets 3 seeds.

Metrics (all OUTCOME-side):
- success-vs-episodes curve; episodes-to-reach 25%/50% success (T1)
- final success + pass^8 at convergence; extend +40 iters if not plateaued (T2)
- fraction of all-fail groups over time (the mechanism plot: outcome-only
  should sit at ~1.0 early; RLVP should exit faster)
- wall-clock per point of success (practical efficiency)

Kill criterion: if (b) does not beat (a) on BOTH episodes-to-50% and final
success at n=3 seeds, T1/T2 are refuted on this testbed and we say so.

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
