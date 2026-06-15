# RLVP — Research Progress (current)

**Last updated:** 2026-06-15. Canonical status doc. Detailed chronological log:
`RESULTS_LOG.md`. Plan: `PAPER_PLAN.md`. (`REPORT.md` is the older compliance-
thesis writeup, superseded by the thesis below.)

---

## Thesis

**Verifiable process rewards are a better RL *training method* than outcome-only
GRPO — measured by the outcome itself.** Mechanism: GRPO's group-relative
advantage is ~zero when all sampled rollouts fail (all-fail groups), which
dominates early training on hard tasks. A dense, verifiable process signal
produces gradient *from failed episodes*, exactly where GRPO is blind.

Three sub-claims:
- **T1 (efficiency):** fewer episodes to a given outcome level. **VALIDATED.**
- **T1b (consistency):** near-zero seed variance vs GRPO's lottery. **VALIDATED.**
- **T2 (ceiling):** higher converged score. **OPEN** — chains saturate, so the
  ceiling tests run on non-saturating tasks (gated, tau2), in progress.

Setup: Qwen3-4B, custom GRPO, chain-N (N-stage tool tasks), tau2-bench airline.
Efficiency measured in **episodes GENERATED** (counts DAPO's resampling cost).

---

## Scorecard

### VALIDATED
| claim | evidence (chain4, n=3 unless noted) |
|---|---|
| RLVP >> outcome-only on efficiency | eps-to-50%: **RLVP 336±0** vs GRPO 1280±837 vs DAPO 2557±743 |
| RLVP is consistent; GRPO is a lottery | RLVP 0 variance; GRPO final 0.69–1.0 across seeds |
| **DAPO does NOT fix it** | 5.6× oversampling, no benefit; chain6: 20,696 vs 3,840 eps for same 1.0 |
| Mechanism is causal (paired) | identical batches: GRPO **25% dead** updates, RLVP **0%** |
| Token-attached credit = efficiency lever | scalar-fold is 2× slower (eps50 672 vs 336) |
| Annealing = ceiling/stability lever | no-anneal final 0.66 vs 1.0; controls episode bloat |
| Process channel > demonstrations alone | clean RLVP 320 vs outcome+demos 616 vs outcome 2240 |
| Mixing (demos) is redundant | clean (no mixing) is the best variant (eps50 320, final 1.0) |
| **Auto-derived rules = hand rules** (capstone) | tags+errors only: eps50 **320 = 320**, 7× vs outcome |

### REFUTED / SCOPED DOWN
| claim | status |
|---|---|
| T2 ceiling on chain tasks | **REFUTED** — chains are compositionally easy, outcome-only also reaches 1.0. Ceiling test moved to non-saturating tasks. |
| "discharge credit is the hero" (earlier) | corrected — token-channel is the efficiency lever; discharge is secondary (dead-iter reduction) |
| step_cost as a universal bloat fix | harmful on short horizons (final 0.84<1.0); length control is HORIZON-DEPENDENT |
| pure penalties / positive-only rewards | inert / useless (earlier toy-env results, still hold) |

### OPEN / RUNNING
- **Gated ceiling test** — non-saturating task (silent precondition gate);
  calibration pending, then outcome-vs-RLVP ceiling comparison.
- **tau2 head-to-head** — real benchmark (base ~0.5), trained without the policy
  doc in prompt; reduced footprint for the shared GPU; runs last.
- auto_rlvp seeds (capstone is n=1 so far).

---

## The recipe (each component earned by an ablation)

**Clean RLVP = sparse outcome + verifiable per-tool-call process terms
(penalty for violations, credit for discharged obligations), delivered via
TOKEN-ATTACHED credit, with annealing after compliance saturates. NO mixing.
NO step_cost on short horizons (add small for very long).** The process signal
can be **auto-derived** from tool category tags + env error signals (capstone)
— no hand-written rules, no demonstrations.

---

## Why it matters (positioning)

RLVP = **R1-Zero extended to long-horizon agents.** R1-Zero self-evolves from a
verifiable outcome reward but stalls when the base policy rarely samples success
(measured: 85% all-fail groups, 42/60 dead GRPO iters). RLVP's minimal addition
restores self-evolution and is **specification, not demonstration** (and, per the
capstone, can be auto-derived). DAPO confronts the same all-fail problem by
discard-and-resample — fixing dead gradient but not sample cost (5.6× tax).

---

## Experiment index
E1 chain calibration · E2 5-way efficiency (outcome/DAPO/RLVP/GiGPO/StepTool) +
seeds · E2b paired dead-iteration · E4 component ablations · fairness control ·
chain6 (saturates) · **capstone (auto rules)** · [running] gated ceiling ·
[queued] tau2 head-to-head.
