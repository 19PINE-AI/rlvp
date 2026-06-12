# RLVP: Local Phase 0/1 Results

Hardware: 1x RTX PRO 6000 Blackwell (96GB, shared). Models: Qwen3-1.7B/4B/8B.
Environments: FileOps (terminal-style) and CSOps (Pine-flavored synthetic customer
service), 4 penalty-only rules each, deterministic — see `rlvp/envs/`.
Eval: 30 held-out tasks/domain (seeds 1000+), temp 0.7. Phase 0: k=2.

## Phase 0 — base models are capable but non-compliant (no training)

| Model | Domain | Rules in prompt | Success | Clean eps | Viol/100 calls | Dominant violations |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | FileOps | off | 33% | **0%** | 40.0 | untested_submit 60/60, blind_write |
| Qwen3-1.7B | FileOps | on  | 65% | **0%** | 26.9 | untested_submit 60/60 |
| Qwen3-1.7B | CSOps   | off | 0%  | 0%     | 20.8 | no_kb_before_call 60/60 |
| Qwen3-1.7B | CSOps   | on  | 100% | 100%  | 0.0  | — |
| Qwen3-4B   | FileOps | off | 60% | **0%** | 47.3 | untested_submit 60/60, blind_write, blind_delete |
| Qwen3-4B   | FileOps | on  | 97% | 88%    | 2.8  | untested_submit 7/60 |
| Qwen3-4B   | CSOps   | off | 100% | **0%** | 27.2 | no_tz_before_call 63, unverified_call 3 |
| Qwen3-4B   | CSOps   | on  | 100% | 53%   | 10.3 | no_tz_before_call 28 |
| Qwen3-8B   | FileOps | off | 67% | **0%** | 47.1 | untested_submit 60/60, blind_write |
| Qwen3-8B   | FileOps | on  | 95% | 100%  | 0.0  | — |
| Qwen3-8B   | CSOps   | off | 47% | **0%** | 137.7 | call_spam 100(!), no_tz 195, no_kb 166 |
| Qwen3-8B   | CSOps   | on  | 100% | 100%  | 0.0  | — |

Findings:

1. **The premise holds at every size.** With no rules in the prompt, 0% of episodes
   are clean at 1.7B, 4B, AND 8B, even when task success is 47–100%. The
   test-before-submit discipline (`untested_submit`) is violated in 60/60 episodes
   by all three models.
2. **Outcome pressure only fixes outcome-instrumental rules.** Qwen3-4B on CSOps
   discovers KB-first by itself (it's needed to find the phone number: 0 violations,
   100% success) but skips the outcome-neutral timezone check in 100% of episodes.
   This is the cleanest demonstration of why outcome rewards alone cannot deliver
   workflow compliance: the optimal-for-outcome policy simply doesn't include
   outcome-neutral steps.
3. **Bigger models retry harder.** Qwen3-8B with no rules generates 100 call-spam
   violations in 60 CSOps episodes — redialing the same number 3+ times. Capability
   without constraints amplifies the violation rate (137.7/100 calls, the worst cell
   in the grid).
4. **Prompting helps with scale but is not reliable at deployment sizes.** Rules in
   the prompt give 8B perfect compliance, but 4B still violates timezone-check in
   47% of CSOps episodes and 1.7B ignores test-before-submit entirely (60/60).
   For the 4–8B models that production agent stacks actually deploy, prompted
   compliance is exactly the "remembers sometimes" failure mode RLVP targets.

Training subject: **Qwen3-4B** — high success (live outcome channel), heavy
violations (live penalty channel), prompted compliance imperfect (headroom vs
the prompting baseline).

## Interim methodological findings (from failed/slow Phase 1 attempts)

**Finding A — token-attached penalties vanish under global normalization.**
With a single loss normalizer (divide by total action tokens in the batch), the
penalty gradient is proportional to the fraction of tokens carrying penalties
(~600 of ~190k). Measured: once train success saturated at 1.0 (outcome
advantages all zero), grad-norm collapsed from ~0.6 to 0.001-0.003 — a 500x
drop, exactly when penalties became the only learning signal. Fix: per-channel
normalization (outcome channel / its token count, penalty channel / its token
count) — this is the operational content of the plan's "two-channel advantage";
separate *centering* alone is not enough. After the fix, grad norms stay 0.3-2.3
throughout.

**Finding B — pure penalties are nearly powerless against a saturated prior.**
The policy emits the violating action (e.g. `submit` right after a write) with
p~0.995; the policy-gradient of log p for the sampled token scales with (1-p),
so penalty-only pressure moves it ~200x slower than an unsaturated token.
Empirically: moderately-saturated rules (blind_write, unverified_call) were
eliminated within ~6 iterations, while fully-saturated habits (untested_submit,
no_tz_before_call: 100% violation at base) stayed at 100% for 16+ iterations.
Penalizing `submit` also doesn't say what to do instead — the suppressed mass
spreads over the whole vocabulary. Fix while keeping rewards verifiable and
rule-derived: **obligation-discharge credits** (+beta on the call that
discharges a currently-pending rule obligation, e.g. `run_tests` while
untested mutations exist; each obligation pays at most once, idempotent by
construction). This gives the probability mass a verifiable destination without
prescribing a full blessed path. Both C1 and C2 receive the same reward terms;
they still differ only in WHERE the credit lands (scalar vs violating/
discharging turn's tokens), so the credit-assignment comparison stays fair.

Practical takeaways for the cluster-scale version: (1) per-channel advantage
normalization is mandatory, (2) pair every penalty rule with a discharge
definition, (3) saturated habits need exploration heat (rollout temp >1) or
they are never sampled differently.

## Phase 1 — GRPO credit-assignment ablation (COMPLETE)

Setup: Qwen3-4B full fine-tune, 60 iters x 16 tasks x G=8 = 7,680 episodes/run,
lr 6e-6, 3 inner PPO-clipped epochs, rollout temp 1.1, lam=beta=0.5,
per-channel normalization, **no rules in the training prompt** — compliance can
only come from the reward. Eval: 30 held-out tasks/domain, k=8, temp 0.7.

### Exploration-wall measurement (the decisive number)

At rollout temp 1.1 the BASE model discharges the saturated obligations in
**0/48 episodes** in both domains (run_tests-after-write; check_timezone).
The compliant alternative is essentially never sampled, which bounds what any
on-policy method can learn and explains everything below.

### Main results (no rules in prompt at eval; internalization condition)

| Variant | FileOps pass@1 | FileOps clean@1 / clean^8 | CSOps pass@1 | CSOps clean@1 / clean^8 | Final entropy |
|---|---|---|---|---|---|
| base        | 0.61 | 0.00 / 0.00 | 1.00 | 0.03 / 0.00 | 0.022 |
| outcome     | 1.00 | 0.00 / 0.00 | 1.00 | 0.00 / 0.00 | **0.000** (collapsed) |
| c1          | 1.00 | 0.00 / 0.00 | 1.00 | **1.00 / 1.00** | **0.000** (collapsed) |
| c2          | 0.67 | **1.00 / 1.00** | 1.00 | 0.00 / 0.00 | 0.010-0.013 (preserved) |
| c2pos       | 1.00 | 0.00 / 0.00 | 1.00 | 0.00 / 0.00 | 0.002 |

Reference cells with rules in the prompt: base_rules fileops clean 0.88 / csops
0.55; c1_rules = perfect on both domains (clean^8 1.00, perfect^8 0.97/1.00);
c2_rules csops perfect 1.00 but fileops pass@1 0.37 (overcaution, see below).

### Findings

1. **Outcome-only RL never fixes outcome-neutral rules** (the motivating
   claim). 60 iters of outcome GRPO: success 1.00 everywhere, untested_submit
   still 240/240, and unverified_call violations INCREASED vs base (23 vs 9).
   Optimizing outcome actively reinforces the non-compliant shortest path.
2. **Internalization is real and beats prompting.** c2 (FileOps) and c1 (CSOps)
   reach clean^8 = 1.00 with NO rules in the prompt — better than the BASE
   model WITH rules in the prompt (0.88 FileOps / 0.55 CSOps). The discipline
   lives in the weights: reliable at k=8, not "remembers sometimes".
3. **C1 and C2 have complementary failure modes — the kill-criterion verdict
   is split, not a c2 sweep.**
   - c1 (scalar fold) cracked CSOps in <10 iters by wholesale amplification of
     lucky-compliant trajectories (BC-like), but NEVER cracked the saturated
     FileOps habit in 60 iters (untested_submit 240/240 to the end), and its
     token entropy collapsed to 0.0000.
   - c2 (token-attached) cracked the saturated FileOps habit via a phase
     transition (clean 0.00 -> 1.00 between iters 30-40) that c1 never
     achieved, and kept entropy at ~0.012. But its localized rare-event credit
     didn't crack CSOps timezone in 60 iters, and at lam=0.5 the process
     channel overpowered the outcome channel (overcaution: 8.3 calls/ep,
     pass@1 0.67; with rules in prompt it loops run_tests into the turn cap,
     pass@1 0.37).
4. **Entropy preservation, measured.** Trajectory-level credit (outcome, c1)
   collapses token entropy to 0.000; token-attached credit (c2) retains
   ~0.010-0.013 (base: 0.022). This is the predicted mechanism: scalar
   advantages push the whole trajectory distribution onto one mode; localized
   penalties only carve out the violating regions.
5. **Positive-only process rewards do nothing for compliance** (c2pos):
   violations unchanged (240/240 on both saturated rules), and it DEGRADED
   prompted compliance (csops clean 0.30 vs base_rules 0.55). Rewarding clean
   turns reinforces the existing (violating) trajectory shape.
6. **The exploration wall is the binding constraint** for both credit schemes:
   each variant only fixed the rule whose compliant alternative it happened to
   sample and amplify. Penalty pressure alone cannot create behavior that is
   never sampled (gradient scales with 1-p). At scale this argues for
   off-policy guidance into the group (LUFFY-style scripted-compliant
   episodes; the rule engine can synthesize them) rather than temperature.

### Appendix A — c2 at lam=beta=0.25 (calibration alone does NOT fix overcaution)

c2cal_norules: fileops clean^8 1.00 but pass@1 0.00 — every episode hits the
12-turn cap (12.0 calls/ep). Curve: compliance phase transition at it~50, then
success collapse with no recovery. Mechanism, now precisely identified:
**once a group is all-failure, group-centered outcome advantages are
identically zero — GRPO is degenerate at 0% (and 100%) group success — while
the per-channel-normalized discharge credit keeps boosting the safe actions.
The policy falls into a compliance-only attractor with no restoring force.**
Halving lam only delayed the crack (it50 vs it40) and removed the outcome
channel's ability to catch the overshoot. (With rules in the prompt the same
checkpoint is perfect on csops — perfect^8 1.00 — and clean^8 1.00 on fileops
at pass@1 0.56: the discipline transferred; the stop-condition broke.)

Recipe implications (for cluster-scale):
1. **Anneal lam, beta -> 0 once compliance saturates** (already in the plan as
   the internalization probe; it is ALSO the stability mechanism — the
   process channel must not outlive its job).
2. Alternative composite worth testing: **gate process credits on episode
   outcome** (discharge credit pays only in episodes that end successfully) —
   makes the compliance-only attractor unreachable by construction.
3. Give timeouts a distinct outcome value (success 1 / clean-fail 0 /
   timeout -0.2) so all-fail groups retain internal contrast.

### Appendix B — c3 hybrid (scalar + token channels, lam=beta=0.25): BEST VARIANT

| Eval (no rules in prompt) | pass@1 | pass^8 | clean^8 | perfect^8 | calls/ep |
|---|---|---|---|---|---|
| FileOps | **1.00** | **1.00** | **1.00** | **1.00** | 4.9 |
| CSOps   | 1.00 | 1.00 | 0.00 | 0.00 | 4.0 |

The hybrid is the first variant to achieve a PERFECT no-rules domain: FileOps
perfect^8 = 1.00 — full task success AND full compliance at k=8, with no
overcaution (4.9 calls/ep vs c2's 8.3-12.0) and entropy preserved (~0.010).
The curve shows the designed mechanism working: compliance cracks at it30
(success dips to 0.67, same overshoot as c2), then the scalar channel pulls
success back to 1.00 by it40 — the restoring force that pure token credit
lacked. With rules in the prompt, c3 is perfect on every metric in BOTH
domains (the belt-and-suspenders deployment configuration).

CSOps timezone stayed uncracked this run (vs c1@lam=0.5 which cracked it):
at halved channel weights the lucky-trajectory amplification is weaker —
consistent with the exploration-wall reading: WHICH rare habit cracks is
stochastic in who samples it; the cure is off-policy guidance, not weights.

### Appendix C — c4 (outcome-gated discharge credits, lam=beta=0.5)

Gating discharge credits on episode success makes the compliance-only
attractor unreachable — confirmed: no overcaution (5.7 calls/ep), no entropy
collapse (0.015-0.018, the healthiest of all variants), no success collapse.
But it also weakens the compliance drive: fileops froze in a mixed mode
(clean 0.33, success 0.67) from it30 to it60. The gate throws away exactly
the episodes where the new behavior is being practiced but the task hasn't
been re-mastered yet. Verdict: too conservative; annealing (c3 + decay) is
the better stability mechanism.

### Final variant ranking (FileOps, no rules in prompt, perfect^8)

c3 hybrid (1.00) > c2 (0.67, overcautious) > c4 (0.00, clean 0.33)
> c1 = outcome = c2pos (0.00, clean 0.00)

### Qualitative: internalized workflow (c3, held-out task, NO rules in prompt)

    Action: run_tests {}                      <- reproduces the failure first
    Tests: FAIL (1/3) - a .tmp file still exists...
    Action: list_dir {"path": "/data"}        <- inspects before deleting
    Action: delete {"path": "/data/sess38.tmp"}
    Action: run_tests {}                      <- verifies after the mutation
    Tests: PASS (3/3)
    Action: submit {}

The reproduce-first behavior was never rewarded directly — it emerged from
the test-after-mutation discharge credit. SUCCESS, zero violations.

## Recommended recipe (for the cluster-scale paper version)

1. Reward = sparse outcome + per-tool-call verifiable rule terms:
   -lam per violation, +beta per pending-obligation discharge (idempotent).
2. Credit = HYBRID: group-centered scalar (outcome + process terms folded)
   on all tokens, PLUS token-attached process terms on the violating/
   discharging turn, each channel normalized by its own token count.
3. Anneal lam, beta -> 0 after compliance saturates (stability + the
   internalization probe in one mechanism).
4. Break the exploration wall with rule-engine-synthesized compliant episodes
   mixed into GRPO groups (LUFFY-style), not with temperature.
5. Report pass^k and perfect^k (= all k succeed AND comply), not pass@1.
