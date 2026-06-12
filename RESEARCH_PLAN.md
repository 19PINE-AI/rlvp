# RLVP: Reinforcement Learning from Verifiable Penalties
### Penalty-only process rewards with step-aware credit for reliable long-horizon agents

Working title: **"Don't Do Anything Wrong: Penalty-Only Process Rewards for Reliable Long-Horizon Agents"**

## 1. Framing

**Problem.** Outcome rewards are too sparse for long-horizon agents (one bit after 100+
turns); token-level process rewards are unimplementable because intermediate reasoning is
unverifiable. The practical middle layer is the tool-call boundary — but existing
step-level work (StepTool, ToolRL, AgentPRM, ARTIST) tries to estimate *positive*
contribution of each step, which is exactly the noisy, hard-to-verify part.

**Key observation — verifier asymmetry.** For intermediate actions, "is this move right?"
is hard (did the refund call fail because of the tactic or the rep?), but "is this move
wrong?" is often trivially checkable: 6th consecutive call to the same number, destructive
shell command without prior inspection, commit without running tests, call initiated
without a KB lookup. Wrongness is verifiable; rightness is not.

**Proposal.** Attach *penalty-only*, deterministic, rule-based rewards at each tool call
(RLVR-style: pure functions of state + action, no LLM judge), keep the sparse outcome
reward as the only positive signal, and train R1-style (GRPO family) with *step-aware
credit assignment* so each penalty lands on the tokens of the violating call.

**Three claims that distinguish this from prior step-reward work:**

1. **Reliability, not capability.** The base model (8B class) already exhibits compliant
   behavior *sometimes*; RL's job is moving probability mass, not discovery. The headline
   metric is therefore **pass^k** (all k trials succeed/comply), not pass@1. With a 90%
   per-decision compliance rate and 30 rule-relevant decisions, clean-run probability is
   0.9^30 ≈ 4% — sampling-dependent compliance cannot survive long horizons.
2. **Penalty-only preserves entropy where it matters.** Positive step rewards prescribe a
   blessed path and collapse the policy onto it; penalties only carve out forbidden
   regions. The unverifiable core of the task (negotiation tactic, patch strategy) keeps
   its diversity — which is exactly where diversity has value (retries, varied reps,
   pass@k). This is the principled answer to "constrain compliance, keep strategy
   stochastic."
3. **Internalization beats runtime guardrails.** A runtime mask catches a violation after
   the model proposes it (wasted turns, latency, and ordering rules like "check KB first"
   can't be enforced by masking without blocking legitimate actions). An internalized
   policy plans around constraints. Test: does compliance persist when rules are removed
   from the prompt, and on held-out tasks/domains?

**Why penalties are most of the gradient, not a nicety:** in week-scale tasks the outcome
reward arrives after thousands of decisions; at almost every step, rule penalties are the
*only* signal that exists.

## 2. Method

### 2.1 Rule engine
Rules are deterministic predicates `r_i(s_t, a_t) → {0, -c_i}` over the agent state
(visible history, tool-call log, environment metadata) and the proposed tool call.
Categories, in order of research interest:

| Category | Examples (per benchmark) |
|---|---|
| Ordering / precondition | KB lookup before call; reproduce failing test before patching; read file before editing it |
| Rate / budget | ≤2 calls to same number per session; ≤N retries of an identical failing command |
| Irreversibility / safety | no `rm -rf` outside workspace; no force-push; no destructive op without a prior read of the target |
| Protocol / format | valid tool args, required fields (sanity tier; ToolRL territory, keep but don't claim) |

All rules are machine-checkable from the trajectory — no LLM judge in the training loop.
Each rule ships with a *loophole test suite* (adversarial trajectories that should/should
not trigger it) written before training, per standard verifier-design hygiene.

### 2.2 Reward and credit assignment (the technical contribution)
Two reward channels, normalized separately:

- **Outcome channel:** terminal task reward, group-normalized as in GRPO/RLOO →
  trajectory-level advantage `A_out`, applied to all tokens.
- **Penalty channel:** `p_t = Σ_i r_i(s_t, a_t) ≤ 0`, attached **only to the tokens of the
  violating tool call** (plus optionally the preceding thought segment), normalized
  within the group **at the step level**, weighted by λ.

Per-token advantage: `A = A_out + λ · A_pen(t)`.

Why two channels: (a) summing penalties into the scalar return dilutes them across the
whole trajectory and is provably what vanilla GRPO would do; (b) single-channel group
normalization cancels the signal when all rollouts in a group violate similar rules
(constant shift); (c) separate normalization stops outcome variance from swamping the
penalty signal. Compare three credit schemes:

- **C1 (naive):** penalties summed into terminal return, vanilla GRPO. Expected to
  underperform — this is the strawman that motivates the design.
- **C2 (token-attached, two-channel):** as above. Primary method.
- **C3 (GiGPO-style):** anchor-state grouping for step advantages, penalties included in
  step returns. Strong alternative; whichever wins becomes the recipe.

**λ scheduling:** λ small enough that "comply but fail the task" never beats "solve the
task" (calibrate so max total penalty < min outcome advantage gap), large enough to
shape. After compliance saturates, **anneal λ → 0 and keep training on outcome only**;
persistent compliance after annealing is direct evidence of internalization rather than
ongoing coercion.

### 2.3 Training setup
- Base: Qwen3-8B (τ²/terminal), Qwen2.5-Coder-7B or Qwen3-8B (SWE).
- Trainer: verl (multi-turn agentic RL, token-level advantage hooks) or SkyRL; GRPO/RLOO
  with token-level loss. Needs GPU cluster — not the no-GPU devbox.
- Rollout envs: τ²-bench simulator (LLM user simulator — budget for simulator tokens),
  terminal-bench-style sandboxed containers, SWE-Gym / SWE-smith task instances for
  SWE training (eval on SWE-bench Verified — never train on it).

## 3. Benchmarks and rule sets

| Benchmark | Train | Eval | ~Rules | Example rules |
|---|---|---|---|---|
| **τ²-bench** (+ verified) | retail + airline tasks | held-in test split, **telecom fully held out** (rule & domain transfer) | 10–15/domain, compiled from the official policy doc (machine-checkable subset) | confirm user identity before account actions; no cancellation without explicit confirmation; check order status before promising refund |
| **terminal-bench** | synthetic task variants / train split | official tasks | ~10 | inspect (ls/cat) before destructive ops; no `rm -rf` outside workspace; no editing a file never read; bounded identical retries |
| **SWE-bench Verified** | SWE-Gym or SWE-smith | SWE-bench Verified | ~8 | run repro test before patching; run tests after edit and before submit; never modify test files to pass; localize (read) before editing |

τ²-bench is the centerpiece: it's the only benchmark with an explicit policy document and
pass^k already as its native reliability metric — we provide the *training recipe* its
evaluation was waiting for.

## 4. Baselines

1. Outcome-only GRPO (R1-style) — the field's default.
2. Rules in system prompt, no RL (prompting baseline).
3. Outcome-only GRPO + rules in prompt (does RL alone fix prompted compliance?).
4. **Runtime guardrail:** outcome-only policy + inference-time action masking/rejection
   of rule-violating calls (the production status quo; for claim 3).
5. Rejection sampling + SFT on rule-compliant trajectories (the "why RL?" baseline).
6. **Positive process reward variant:** +c for satisfying each rule instead of −c for
   violating (for the entropy-preservation claim — same information, opposite sign).
7. C1 naive credit (for the step-aware credit claim).

## 5. Metrics

- **pass^k** (k=8) and pass@1 on each benchmark; pass@k for diversity-value evidence.
- Violation rate per 100 tool calls, by rule category.
- **Internalization:** compliance with rules *removed from prompt*; after λ annealing;
  on held-out domain (telecom); on held-out rules (train on subset, measure siblings).
- **Entropy preservation:** policy entropy during training; tactic diversity across k
  rollouts (embedding/self-BLEU diversity of tool-call sequences) — penalty-only (ours)
  vs positive-reward (baseline 6).
- **Hacking audit:** compliance *quality* (mutation-score of written tests, not just
  "tests were run"); inaction/abandonment rate (does the model avoid penalties by doing
  less?); action count distributions.
- Sample-efficiency curves (success & compliance vs training steps).

## 6. Risks

| Risk | Mitigation / measurement |
|---|---|
| Penalty avoidance by inaction | outcome reward dominant; monitor action counts & abandonment explicitly |
| Superficial compliance (throwaway tests/specs) | quality metrics in audit; rules check artifacts not just events where cheap |
| Group normalization cancels penalty signal | two-channel design; C1 ablation quantifies the failure |
| λ mis-calibration | sweep λ ∈ {0.1, 0.3, 1.0} in Phase 1 sandbox before scaling |
| A rule that conflicts with task success | report success-cost of compliance per rule; drop or re-spec offenders |
| τ² user-simulator noise | use τ²-bench-verified fixes; fixed simulator model + seeds for eval |

## 7. Phasing

- **Phase 0 (≈1–2 wks, no training):** build rule engine + trajectory logger; run base
  model rollouts on all three benchmarks; report base violation rates and the
  violation↔failure correlation. This is the motivation section, costs almost nothing,
  and validates the rules. (Optionally: same measurement on Pine production traces — the
  strongest possible motivation figure.)
- **Phase 1 (≈2–3 wks):** terminal-bench sandbox (cheapest env). Small-scale GRPO; settle
  C1 vs C2 vs C3 and the λ sweep. Kill criterion: if C2/C3 don't beat C1 on
  compliance-per-sample here, the credit-assignment story dies cheaply.
- **Phase 2 (≈3–4 wks):** τ²-bench main results + all baselines + internalization and
  entropy studies. This is the paper's core table.
- **Phase 3 (≈3–4 wks):** SWE-bench Verified (expensive rollouts); transfer + annealing
  studies; hacking audit.
- **Phase 4 (≈2 wks):** writeup.

## 8. Local pilot results (2026-06-12) — revisions for the cluster-scale version

A full local pilot (Qwen3-4B, two synthetic domains, 4 credit variants x 7,680
episodes, single RTX PRO 6000) is in `results/REPORT.md`. It confirmed the
thesis (outcome-only RL never fixes outcome-neutral rules; penalty-trained
compliance internalizes and beats prompting at pass^k) and forces four
revisions to the method section:

1. **Per-channel advantage normalization is mandatory** (Sec 2.2 was right but
   underspecified): separate centering is not enough — separate SCALING is the
   load-bearing part, otherwise penalty gradients vanish when outcome
   saturates (measured 500x grad-norm collapse).
2. **Pure penalty-only is insufficient against saturated habits**: gradient
   scales with (1-p) of the violating token. Every penalty rule needs a paired
   verifiable *obligation-discharge credit* (+beta when a pending obligation is
   discharged; idempotent, bounded). This keeps rewards rule-derived and
   verifiable while giving suppressed probability mass a destination.
3. **The exploration wall is the real bottleneck** (0/48 base sampling of the
   compliant alternative at temp 1.1): plan an off-policy guidance arm —
   LUFFY-style mixing of ONE rule-engine-synthesized compliant episode per
   group — rather than relying on temperature.
4. **Scalar-fold (c1) vs token-attached (c2) is not a strict ordering**: c1 =
   fast whole-trajectory cloning, entropy collapse to 0, never cracks
   saturation; c2 = cracks saturation via phase transition, preserves entropy,
   but needs lam calibration against overcaution and is slow on rare-event
   rules. **The hybrid (c3: both channels, each normalized separately) is the
   winning recipe — fileops perfect^8 = 1.00 with no rules in the prompt, no
   overcaution, entropy preserved.** Two stability notes: (a) GRPO is
   degenerate in all-fail groups, so an unopposed process channel creates a
   compliance-only attractor (c2 at any lam) — anneal lam,beta after
   compliance saturates; (b) outcome-gating process credits (c4) prevents the
   attractor but throws away the practice episodes and stalls compliance —
   prefer annealing.

## 9. Related work to position against
StepTool (step-grained tool RL, pre-R1, PPO-era), ToolRL (fine-grained format/correctness
rewards, single-turn-ish), AgentPRM / iStar (learned PRMs — we're rule-based, no judge),
GiGPO / HCAPO / turn-level reward design (credit machinery — we adopt, not compete),
Rubrics-as-Rewards & Agentic Rubrics (outcome-time checklists — ours are dense and
online), τ/τ²-bench (evaluation of policy compliance — we contribute the training side),
constrained RL & shielding (classical framing of penalty-only constraints — bridge to
LLM agents).
