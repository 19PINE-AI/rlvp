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

## Tier 1 follow-up arms (same setup; eval k=8, no rules in prompt)

### Arm 1 — scripted-compliant group mixing: the exploration wall falls

c3 + 1 rule-engine-synthesized compliant episode per group (7 live + 1
scripted), lam=beta=0.25:

| Domain | pass@1 | pass^8 | clean^8 | perfect^8 | tactic div |
|---|---|---|---|---|---|
| FileOps | 0.86 | 0.43 | 1.00 | 0.43 | 0.57 |
| CSOps   | 1.00 | 1.00 | **1.00** | **1.00** | 0.27 |

The CSOps timezone rule — uncracked by c2, c2cal, c3, c4 and cracked only by
luck in c1 — is now learned deterministically. Off-policy guidance through
the group, not temperature, is what breaks the exploration wall. Side-effect:
final entropy 0.43 (vs 0.01 elsewhere) — the off-policy tokens keep the
policy hot, which costs FileOps reliability (pass^8 0.43).

### Arm 3 — annealing (mix + anneal lam,beta->0 at it40): the recipe

| Domain | pass@1 | pass^8 | clean^8 | perfect^8 | tactic div |
|---|---|---|---|---|---|
| FileOps | 0.97 | 0.87 | **1.00** | **0.87** | 0.32 |
| CSOps   | 1.00 | 1.00 | **1.00** | **1.00** | 0.23 |

Annealing the process channel after compliance saturates recovers the
reliability the hot mixing phase costs (perfect^8 0.43 -> 0.87) while keeping
clean^8 = 1.00 on BOTH domains and 2x the tactic diversity of every
non-mixed variant. **mix + anneal is the final local recipe.**

### Arm 2 — SFT-BC control: strong in toy envs, with a visible ceiling

BC on 800 perfect scripted episodes: perfect^8 0.93 / 1.00 — statistically
indistinguishable from the RL recipe here, BUT with tactic diversity
collapsed to 0.14/0.12 (it IS the scripted path). Honest reading: in a
deterministic toy env a perfect demonstration policy exists, so BC suffices;
the discriminator is Tier 1.5 (imperfect scripts: compliant-but-failing
demos — BC must clone the failure, RL group advantage filters it).

### Arm 4 — runtime guardrail baseline: enforcement is not internalization

Action masking (reject violating calls, agent retries) on FileOps:

| Policy | pass@1 | timeout rate | blocked/ep | episodes hitting guardrail |
|---|---|---|---|---|
| base + guardrail    | 0.38 | 0.59 | 6.3 | 100% |
| outcome + guardrail | 0.67 | 0.33 | 3.4 | 100% |
| c3 internalized     | 1.00* | 0.00 | 0 | 0% |

(*c3anneal pass@1.) Blocking `submit` does not produce `run_tests` — the
masked model retries into the turn cap, exactly the predicted failure of
inference-time enforcement on ordering rules. Internalization is not a
luxury; masking costs 33-62 points of pass@1 on this domain.

## Tier 2 — robustness

**Seeds (3x c1, 3x c3-no-mixing).** c1 is perfectly reproducible (3/3 seeds:
CSOps clean^8 1.00, FileOps untested_submit 240/240 — the scalar-credit
asymmetry is deterministic). Bare c3 is NOT: the saturated-habit crack
happened fully (s0), partially (s11, +a new retry-loop pathology), and not at
all (s12). Token-attached credit's phase transition is real but stochastic in
whether the alternative gets sampled — **scripted mixing is therefore a
necessary component, not an optimization**: c3mix cracks both domains
deterministically.

**Reliability stress.** c3mix at k=16: clean^16 0.97/1.00 (base 0.00/0.00).
At eval temp 1.0: clean^8 1.00/1.00, perfect^8 0.77/1.00. Internalized
compliance survives both more trials and deployment-level stochasticity.

**Held-out rules (confounded — flagged, not claimed).** Training with
untested_submit + no_tz_before_call dropped still reduced their eval
violation rates 57-62% vs base (clean@1 0.57/0.62). CONFOUND: the mixing
scripts still demonstrate the held-out behaviors, so this is partly script
imitation rather than cross-rule generalization. Clean version requires
held-out steps stripped from the scripts; deferred.

## Tier 1.5 — imperfect demonstrations: the "why RL" verdict

Demos made compliant-but-task-failing (wrong file content / missing policy
citation; workflow scaffolding intact). This is the realistic regime: in real
domains you can synthesize compliant scaffolding but not correct solutions.

| Training on imperfect demos | pass@1 (fo/cs) | clean@1 (fo/cs) |
|---|---|---|
| SFT-BC  | **0.00 / 0.00** | 1.00 / 1.00 |
| RL c3mix | **0.97 / 0.97** | 0.00 / 0.63 |

BC clones the failure wholesale — perfect workflow, zero task success, no
mechanism to filter demonstration quality. RL's group-relative advantage
discards the failing demo's solution while training proceeds normally.
So: perfect demos -> BC suffices (toy-env regime); imperfect demos -> BC is
catastrophic and RL is required. Compliance absorption from failing demos
was partial (cs 0.63 / fo 0.00) because the demo's negative scalar advantage
suppresses its compliant scaffolding along with its mistake — cluster-scale
fix: **scripted episodes should contribute through the process channel only**
(zero their scalar advantage, keep their discharge attachments).

## Tier 3 — model scale (recipe = c3 + mixing, lam=beta=0.25)

| Model | FileOps perfect^8 | CSOps perfect^8 | clean^8 (fo/cs) |
|---|---|---|---|
| Qwen3-1.7B (full FT) | 0.80 | 1.00 | 0.80 / 1.00 |
| Qwen3-4B (full FT, +anneal) | 0.87 | 1.00 | 1.00 / 1.00 |
| Qwen3-8B (LoRA r=32) | 0.77 | 0.97 | 1.00 / 0.97 |

The recipe transfers down to 1.7B (full FT) and up to 8B through LoRA —
notable because LoRA-RL is the cheap path for production-size models, and
because the 8B base was the WORST offender in Phase 0 (137 violations/100
calls, 100 redials). tau2-bench Phase-0 measurement pending (py3.12 venv
built; run queued behind Tier 1.5).

## Tier 3b — tau2-bench Phase-0 (fully local: vLLM Qwen3-8B agent + user sim)

20 airline tasks, 16 completed simulations, 36 tool calls, mean reward 0.44.
Structural rule analysis (lookup-before-write, call spam, unconfirmed write
chains; tool-name prefixes verified against the actual airline API):
**0 violations / 36 calls.**

Reading this honestly: tau2's protocol always carries the FULL policy
document in the system prompt — so this is Phase 0's rules-in-prompt
condition, where Qwen3-8B was already perfectly compliant on our domains
(clean 1.00/1.00). The result is consistent, not contradictory. The failure
mass on tau2 (reward 0.44) is SEMANTIC policy compliance (fare rules,
modification constraints), which requires compiling the policy doc into
domain-semantic checkers — the cluster-scale Phase 0 task.

The RLVP opportunity on tau2 is therefore the internalization condition:
train with rule rewards and EVALUATE WITHOUT the ~10k-token policy document
in the prompt. If compliance holds (as it did locally: trained-no-rules beat
base-with-rules), the policy doc can be dropped from every production call —
a direct latency/cost win on top of the reliability win.

Infra note: the full local stack works end-to-end (py3.12 venv, vLLM with
hermes tool-call parser + qwen3 reasoning parser, 32k context); five setup
landmines documented in scripts/tau2_phase0.sh.

## Recommended recipe (final — every component now individually evidenced)

1. Reward = sparse outcome + per-tool-call verifiable rule terms:
   -lam per violation, +beta per pending-obligation discharge (idempotent).
   [Pure penalties refuted by Finding B; c2pos refuted positive-only.]
2. Credit = HYBRID: group-centered scalar on all tokens PLUS token-attached
   process terms on the violating/discharging turn, each channel normalized
   by its OWN token count. [c1/c2 complementary failures; c3 wins; global
   normalizer refuted by Finding A.]
3. Scripted-compliant episode mixed into each group — NECESSARY, not an
   optimization: bare-c3's saturation crack is luck (3 seeds: full/partial/
   none); with mixing it is deterministic (clean^8 = 1.00 both domains).
   Scripted episodes should contribute via the PROCESS CHANNEL ONLY (zero
   scalar advantage) so imperfect demos donate scaffolding without their
   mistakes — Tier 1.5 showed the negative scalar otherwise suppresses both.
4. Anneal lam, beta -> 0 after compliance saturates: recovers success
   reliability the hot mixing phase costs (perfect^8 0.43 -> 0.87) and
   doubles as the internalization probe. [Outcome-gating (c4) refuted as the
   stability mechanism — it discards the practice episodes.]
5. Train-time internalization, not inference-time masking: guardrails cost
   33-62 pts pass@1 with 100% of episodes hitting blocks (ordering rules
   cannot be masked into existence).
6. RL, not BC, whenever demonstrations can be imperfect: BC on compliant-
   but-failing demos = 0.00 pass@1; RL with the same demos = 0.97.
7. Report pass^k and perfect^k, not pass@1. Compliance held at clean^16
   0.97-1.00 and at eval temp 1.0.

## v2 flagship (chain4, outcome thesis) — preliminary

### E2b paired dead-iteration (controlled: identical batches, both credits)
On 20 identical rollout batches (Qwen3-4B base, chain4):
- outcome-only (GRPO) credit: 25% of batches yield a DEAD (zero-gradient) update
- RLVP credit: 0% dead
Same rollouts, different credit -> the dead-iteration gap is causal, not across-run.
GRPO wastes 1 in 4 updates on all-fail batches; RLVP's process channel extracts
gradient from those same failures.

### E1 calibration (base Qwen3-4B, chain-N, k=2)
chain2 succ .29 (~6% all-fail groups) | chain4 succ .083 (~50%) |
chain6 succ .021 (~85%) | chain8 succ .0. Picked chain4: real outcome headroom
AND a high all-fail-group fraction (the regime where GRPO is blind).

### E2 flagship 5-way (chain4, held-out eval) — HONEST
Matched budget (episodes GENERATED), held-out success:
  336 eps:  RLVP 0.78 | GRPO 0.25 | DAPO 0.59(@1720 gen!) | StepTool 0.62 | GiGPO 0.25
  final:    RLVP 0.91 | GRPO 1.00 | DAPO 1.00 | StepTool 1.00 | GiGPO 0.69
eps-to-50% (generated): RLVP 336 | StepTool 512 | GiGPO 832 | GRPO 2240 | DAPO 2312
dead iters/60: GRPO 42 | DAPO 31(+5.59x oversample) | GiGPO 8 | RLVP 2 | StepTool 0

T1 (efficiency): STRONG. At 336 eps RLVP 0.78 vs GRPO 0.25 (held-out). GRPO sits
at base ~0.25 for ~50 iters (42/60 dead) then converges only at the very end.
DAPO confirms the thesis: 5.59x oversampling AND 31 stalls -> no efficiency gain
over vanilla GRPO; on episodes-generated it is the WORST.
CAVEAT (real): RLVP final 0.91 < StepTool/GRPO 1.0, and RLVP calls/ep bloats
11->26 by iter24 (BEFORE anneal). Cause: discharge credit farmed by reading
unneeded files on multi-file chains (read-before-write pays per first read).
Fix: per-step cost (length penalty). chain4 not hard enough for a CEILING gap
(all reach ~1.0) -> chain6 T2 test pending.

### E2 seeds (chain4, episodes-generated axis, mean±sd)
                eps-to-50%     final-succ    dead  oversample
  RLVP (n=2)    336 ± 0        0.92 ± 0.02   5.5   1.0x
  GRPO (n=3)    1280 ± 837     0.88 ± 0.17   39    1.0x
  DAPO (n=3)    2557 ± 743     1.00 ± 0.00   33    5.6x
T1 efficiency CONFIRMED with error bars: RLVP 336 eps to 50% (zero variance) vs
GRPO 3.8x / DAPO 7.6x more. GRPO is a lottery (final 0.69-1.0); RLVP/DAPO stable.
DAPO always reaches 1.0 final but pays 5.6x oversampling -> slowest on the fair
axis. chain4 too easy for a CEILING gap (DAPO/GRPO eventually reach 1.0) -> chain6.
RLVP final 0.92 < DAPO 1.0: the calls/ep bloat caps the ceiling (step_cost fix pending).

### E4 ablation (chain4) — abl_nomix is the key surprise
  full RLVP (c3+mix+anneal):  eps50=336 final=0.91 best=0.97  (oscillates after anneal)
  RLVP - mixing (c3+anneal):  eps50=320 final=1.00 best=1.00  (clean monotone to 1.0)
=> On chain4 the PROCESS CHANNEL (penalties+discharge) is the whole efficiency win;
   demo-mixing is redundant here (8% base succ -> ~44% of groups already have a live
   success) and slightly HURTS ceiling/stability + adds calls bloat. Reframe: process
   channel = hero; mixing = regime-specific tool for near-zero-success tasks (-> chain6
   prediction: mixing earns its keep only when live rollouts rarely contain a success).

### E4 attribution (chain4) + CORRECTION
  RLVP-mixing (penalty+discharge+anneal):  eps50=320 final=1.00  <- BEST, = "clean RLVP"
  full RLVP (+mixing):                     eps50=336 final=0.91  (mixing redundant/harmful here)
  RLVP-discharge (penalty+mixing):         eps50=336 final=0.66  all-fail stays 0.25-0.5 <- discharge ESSENTIAL
CONCLUSION: discharge credit is the load-bearing component (positive process signal
that escapes all-fail groups); penalties alone inert; mixing redundant on chain4.
=> "clean RLVP" = penalties + discharge, NO mixing.
CORRECTION to earlier note: calls/ep bloat is NOT discharge-farming. abl_nodisch (no
discharge) still bloats (24.7 calls/ep), so the cause is PENALTY-AVOIDANCE (defensive
reads before writes to dodge blind_write penalty), present with or without discharge.
step_cost still the right fix (per-action cost discourages defensive over-reading).
OPEN: is mixing EVER needed? chain6 (~2% succ) + tau2 (compliance != success) decide.

### E4 FULL ablation (chain4, final) — corrected attribution
  clean (pen+disch+anneal, NO mix):  eps50=320 final=1.00  <- BEST = the recipe
  full (+mixing):                    eps50=336 final=0.91  (mixing redundant/harmful)
  mixing in scalar baseline:         eps50=392 final=1.00
  NO discharge:                      eps50=336 final=0.94 dead=11  (discharge: fewer dead iters)
  scalar-fold (NO token channel):    eps50=672 final=0.94  <- token channel = 2x SPEED
  NO anneal:                         eps50=336 final=0.66 (31 calls/ep, 5-6x slower)
  clean + step_cost=0.03:            eps50=336 final=0.84  <- step_cost HURTS
CORRECTED ATTRIBUTION (supersedes earlier "discharge is the hero"):
  - TOKEN-ATTACHED credit channel = the efficiency lever (336 vs 672 scalar-fold, 2x).
  - ANNEALING = the ceiling lever (0.66 -> 1.0; also controls the episode bloat).
  - DISCHARGE credit = secondary (reduces dead iters 11->5; chain4 reachable without it).
  - MIXING = redundant/harmful here (clean is best).
  - step_cost = HARMFUL (0.84<1.0); the calls/ep bloat does NOT block success (clean
    reaches 1.0 with bloat) -> DROPPED from the recipe.
RECIPE = penalty + discharge + token-attached credit + anneal, NO mixing, NO step_cost.
chain6 arms corrected to this (step_cost=0 removed before they ran).

### chain4 fairness control + chain6 ceiling (honest)
chain4 eps-to-50% (what speeds up outcome learning):
  pure outcome (GRPO):              2240
  outcome + demos (no proc chan):    616   <- demos help
  process channel, no demos (clean): 320   <- process channel helps MORE
  => process channel ALONE beats demonstrations alone, and needs no demo synthesis.
     Adding demos to the process channel does not help (full RLVP 336 >= clean 320).
chain6 ceiling: outcome-only ALSO reaches final 1.0 (eps50 448, single seed). Chain
tasks are COMPOSITIONALLY EASY (master per-stage pattern -> all-stages success
multiplicatively), so any method that gets a few successes converges. => T2 (higher
CEILING) is NOT demonstrated on chains; the benefit is EFFICIENCY + CONSISTENCY (T1),
not ceiling. A true ceiling gap needs an irreducibly-hard subtask (best shot: tau2,
where compliance != success). REFRAME: lead with T1 (efficiency/consistency); T2
unsupported on chains, defer ceiling claim to tau2 or drop it.
OPEN: chain6 clean-vs-mixed still informative — does mixing EVER help at ~2% base?

### chain6 (partial, then deprioritized) + throughput finding
  t2_outcome: final 1.0 (3,840 eps)  | t2_dapo: final 1.0 (20,696 eps, 5.4x oversample)
  t2_rlvp_clean: eval 0.62->0.91 by iter30 (did NOT stall at 2% base success)
=> chain6 SATURATES (outcome reaches 1.0) -> no ceiling gap (confirms reframe: chains
   show efficiency, not ceiling). DAPO 5.4x compute for ZERO benefit (strongest
   anti-DAPO point). Clean RLVP reaches 0.91 at 2% base -> does NOT stall in the
   sparse regime -> mixing NOT needed even there (answers clean-vs-mixed).
THROUGHPUT FINDING: on chain6 (50-turn cap) clean RLVP bloats to ~46 calls/ep
WITHOUT step_cost, collapsing throughput to ~28 min/iter. => length control is
HORIZON-DEPENDENT: harmful on short chain4 (step_cost 0.03 hurt) but NEEDED on very
long horizons. Recipe note: scale step_cost with horizon (~0/short, small/long).
DECISION: dropped the slow confirmatory chain6 mix/outmix arms (saturating task,
expensive) to prioritize the NON-SATURATING ceiling tests (gated, tau2) the user
correctly flagged as what actually validates T2.

### CAPSTONE: auto-derived process reward (chain4) — R1-Zero-pure WIN
  AUTO rules (tool tags + env errors, NO hand predicates): eps50=320 final=1.0 dead=0
  HAND rules (clean RLVP):                                  eps50=320 final=1.0 dead=5
  outcome-only (no process):                                eps50=2240 final=1.0 dead=42
=> Auto-derived process reward EXACTLY recovers hand-written-rule efficiency (320 vs
   320, 7x faster than outcome-only). The maximal-R1-Zero RLVP: NO demonstrations
   (mixing dropped) AND NO hand-written rules. Only human input = tagging tools by
   category (observe/mutate/verify/terminal) — task-agnostic metadata real tool
   schemas (OpenAPI/function-calling) often already expose. Plus the env's own error
   signals (fully automatic). [n=1; auto seeds worth adding for the paper.]

### GATED ceiling test — calibration (VALID non-saturating testbed)
base Qwen3-4B on gated: success 0.00 | read_acl 0.00 | tried_request_access 1.00 | viol/ep 2.0
=> Model discovers request_access but NEVER the hidden precondition (read /acl first,
   else request_access silently no-ops). base success 0.00 -> fully non-saturating,
   unlike chains. Valid T2 ceiling testbed: outcome-only must discover read-/acl from a
   0/1 signal (no hint); RLVP's rule rewards reading /acl. Ceiling comparison running.

### GATED ceiling test — results: EXPLORATION WALL (all clean methods fail)
  gated_outcome:        final 0.00, dead 80/80 (every iter all-fail)
  gated_dapo:           final 0.00, stalled, 6x oversample
  gated_rlvp (clean):   final 0.00, dead 61/80
  gated_outcome_prompt: final 0.00 (PROMPTED still never reads /acl!)
ROOT CAUSE: base model NEVER samples reading /acl (read_acl 0.00), even when PROMPTED.
So clean RLVP's discharge credit for reading /acl CAN'T FIRE (can't reward an unsampled
behavior); its penalty only suppresses request_access-without-acl, doesn't INDUCE
reading /acl. => clean RLVP (process channel alone) has a DISCOVERY CEILING: it cannot
induce a never-sampled precondition. This is the R1-Zero vs R1 boundary: self-evolution
from reward stalls on un-discoverable preconditions; DEMONSTRATIONS (mixing) are the fix.
DECISIVE TEST RUNNING: gated_rlvp_mix (clean RLVP + scripted demo that reads /acl). If it
lifts success >0 while all others stay 0 -> (a) T2 ceiling gap demonstrated, (b) mixing
proven NECESSARY in the hidden-precondition regime (vs redundant on discoverable chains).
If mixing also fails -> task mis-calibrated (too hard); add a discoverable-hint variant.

### GATED ceiling test — RESOLVED: T2 ceiling gap + mixing-necessity boundary
gated success (non-saturating, base 0.00):
  outcome-only / DAPO / clean-RLVP / prompted:  0.00  (all stuck — discovery ceiling)
  RLVP + MIXING:                                1.00  (eval 0.0->1.0 phase transition @iter38, generalizes)
=> T2 CEILING GAP DEMONSTRATED on a non-saturating task: outcome 0.0 vs RLVP+mixing 1.0
   (reachable vs UNreachable, not just faster). And the MIXING BOUNDARY is now precise:
   - discoverable-workflow tasks (chains): clean RLVP suffices, mixing REDUNDANT.
   - hidden-precondition tasks (gated): clean RLVP hits a DISCOVERY CEILING (can't induce
     a never-sampled precondition -> discharge never fires); mixing is NECESSARY.
   Maps to R1-Zero<->R1: reward-only self-evolution works on discoverable workflows but
   needs a demonstration cold-start for un-discoverable preconditions. Demo injection
   succeeds where PROMPTING failed (changes weights, not just the ask).

### E3 tau2 head-to-head (real benchmark) — HONEST NEGATIVE / scoping result
tau2 airline, no policy doc in prompt, k=2 eval (~20 sims, HIGH variance):
  base:          reward 0.20  viol/ep 0.55
  outcome-only:  reward 0.60  viol/ep 0.90
  RLVP:          reward 0.45  viol/ep 0.55
=> On tau2, outcome-only GRPO BEAT RLVP on reward (0.60 vs 0.45); RLVP lowered
   violations (0.55 vs 0.90) -> traded outcome for compliance. The process channel
   worked but optimized HYGIENE that is ORTHOGONAL to the tau2 ENV reward.
KEY SCOPING (refines, not breaks, the thesis): RLVP improves the OUTCOME only when the
process rules are OUTCOME-INSTRUMENTAL (aligned with what success needs). chains/gated:
rules ARE the bottleneck (read-before-write, the /acl gate) -> outcome lifted. tau2: I
used GENERIC structural rules (write-before-lookup, call-spam) orthogonal to the airline
reward (semantic policy: auth/refund eligibility) -> compliance up, reward not (slightly
down). CAVEATS: (1) small n, high variance; (2) tau2-SPECIFIC policy-derived rules NOT
tested -> a fair "aligned-RLVP on tau2" test = compile rules from the policy doc (future).

### E3 tau2 — CORRECTION (full train trajectory)
tau2_rlvp reward FLAT at 0.0 (whole run); viol/ep 0.4 -> 0.0 (sustained). This is the
COMPLIANCE-ONLY ATTRACTOR (drive violations to 0 by not attempting risky task actions ->
reward collapses), the failure mode found early in the toy envs. The eval 0.45 was k=2
variance over a collapsed policy (true reward ~0). outcome-only climbed reward 0.1->0.5
(viol ~1.3, ignores compliance). CRUCIAL: the tau2 RLVP arm OMITTED annealing (anneal_at
unset) — the KNOWN mechanism to escape this attractor. So this is NOT a fair RLVP test.
Re-running tau2 RLVP WITH anneal for a fair comparison. Expected: anneal prevents the
collapse; with ORTHOGONAL generic rules RLVP should land ~= outcome (no benefit, no harm).
Real benefit on tau2 needs OUTCOME-INSTRUMENTAL rules from the policy doc (future work).

### E3 tau2 — FINAL (fair re-run with anneal)
  outcome-only:      reward 0.60  viol 0.90  (learns the task)
  RLVP (no anneal):  reward ~0.45 viol 0.55  (train reward collapsed to 0)
  RLVP (anneal@12):  reward 0.375 viol 0.28  (collapsed by iter 5-7, BEFORE anneal)
Anneal@12 did NOT rescue it: the compliance-only collapse happens in the first ~7 iters;
post-anneal pure-outcome can't recover from reward-0 (no success -> no gradient).
FINAL CONCLUSION (real benchmark): with GENERIC structural rules orthogonal to the tau2
reward, RLVP does NOT help outcome and modestly hurts it (drives compliance-collapse).
This is the BOUNDARY of the thesis, demonstrated on a real benchmark: RLVP's outcome
gains (chains, gated) REQUIRE outcome-instrumental rules; misaligned rules -> no gain.
Fixes for a real tau2 win (future work): (1) compile OUTCOME-INSTRUMENTAL rules from the
policy doc, (2) much earlier anneal or outcome-gating (c4) to prevent the early collapse.

### E3b tau2 POLICY-DERIVED (outcome-instrumental) rules — three-tier boundary
tau2 airline, no policy in prompt:
  outcome-only:           eval 0.60  viol 0.9   train ~0.50
  RLVP generic rules:     eval 0.45* viol 0.55  train 0.00 (COLLAPSED; *eval=variance over dead policy)
  RLVP aligned + anneal8: eval 0.35  viol 0.00  train 0.37 (HELD, no collapse)
Aligned rules compiled from policy.md (modify_without_user/reservation, write_without_confirm,
discharges for the prerequisite lookups) + early anneal:
=> TIER 1 generic/orthogonal rules -> HARM (compliance-only collapse, train->0).
   TIER 2 aligned procedural rules -> NO HARM (collapse FIXED: train 0.37, viol 0; alignment +
          early anneal closes the attractor).
   TIER 3 still NO GAIN over outcome (0.37<0.50): procedural rules fix the WORKFLOW but not the
          CONTENT; tau2's reward needs SEMANTIC correctness (refund amount, flight, cabin) that
          procedural rules don't capture.
BOUNDARY is a COVERAGE GRADIENT, not a line: misaligned->harm, aligned-procedural->neutral,
semantic-coverage->(predicted) gain. The remaining gap names the next rule set, not a failure.

### E3c tau2 SEMANTIC-coverage rules — the coverage gradient SATURATES
tau2 airline, no policy in prompt, train reward (last5 / peak):
  outcome-only:            0.48 / 0.71
  RLVP generic rules:      0.0  / 0.12   (collapse / HARM)
  RLVP aligned-procedural: 0.37 / 0.50   (no harm, no gain)
  RLVP aligned-semantic:   0.38 / 0.67   (no harm; peak approaches outcome; avg NO gain)
Semantic rules = verifiable content-validity vs DB (modify_basic_economy,
change_passenger_count, payment_not_in_profile). They FIRED (viol 0.4->2.4 early,
agent attempted invalid modifications, then learned to avoid -> viol 0 by iter 10).
=> Semantic coverage did NOT close the gap to outcome-only. The gradient SATURATES:
   verifiable rules can cover GENERAL POLICY (procedure + validity) but NOT task-specific
   INTENT (which specific valid change THIS user wants). Task intent IS the reward; a rule
   encoding it would just be the task spec. So RLVP accelerates the policy-verifiable part
   of success and CANNOT substitute for outcome learning of task intent. This is the
   IRREDUCIBLE limit of verifiable process rewards, measured.
