# RLVP: What we learned

**Thesis:** for long-horizon agents, workflow compliance is a *sampling-and-
credit* problem, not a reward-specification problem. Verifiable rules tell
you instantly when a move is wrong; that is not the hard part. The hard parts
are (a) the compliant alternative is almost never sampled, (b) credit must
land on the responsible tokens without losing the task, and (c) the training
scaffolding must be removed once the discipline is in the weights.

Setup in one line: two deterministic tool-call domains (terminal-style +
customer-service), 4 machine-checkable rules each, Qwen3-1.7B/4B/8B +
Mistral-7B, on-policy GRPO, ~20 training runs, all metrics at pass^k
(all k trials must succeed/comply), held-out tasks, k=8 unless noted.
Full data: `RESULTS_LOG.md`. Method recipe: end of this file.

## Seven findings

**1. Outcome RL cannot buy compliance, at any budget.**
60 iterations to 100% task success; outcome-neutral rules still violated in
240/240 episodes, and one violation type INCREASED (23 vs 9). Mechanism: the
outcome-optimal policy excludes outcome-neutral steps by construction, so
more outcome optimization means less compliance, not more.

**2. The binding constraint is sampling, not reward design.**
The compliant alternative (run tests before submitting; check the timezone)
was sampled in 0/48 base episodes at temp 1.1. The policy gradient for a
sampled token scales with (1-p); against a p~0.995 habit, penalties are
inert (~200x slower) and the suppressed mass has no destination. Corollary
measured directly: whether token-level credit cracks a saturated habit is a
COIN FLIP across seeds (full / partial / never in 3 seeds). One rule-engine-
synthesized compliant episode per GRPO group makes it deterministic:
clean^8 = 1.00 on both domains, every seed.

**3. Credit shape determines WHAT is learned, not just how fast.**
Scalar (fold rules into the return): clones whole lucky trajectories —
fast on easily-sampled rules, token entropy collapses to 0.000, NEVER cracks
saturated habits (240/240 to the end). Token-attached: targeted surgery —
cracks the saturated habit via a phase transition, preserves entropy
(~0.012), but has no restoring force when compliance pressure overshoots
(success 1.00 -> 0.00, episodes loop forever). Only the HYBRID of both
channels reaches perfect^8 = 1.00 (success AND compliance, all 8 trials).

**4. GRPO is blind exactly where the process channel misbehaves.**
Group-centered advantages are identically zero in all-fail groups, so once
process pressure tips the policy into a compliance-only mode there is no
gradient back — at ANY penalty weight (halving lam just delayed the collapse).
Fixes that work / don't: annealing lam,beta -> 0 after compliance saturates
(works — and compliance persists, which is also the internalization proof);
gating process credits on success (fails — it discards exactly the episodes
where the new behavior is being practiced). Two silent implementation
killers found the same way: global advantage normalization (penalty
gradient collapses 500x at the moment outcome saturates and penalties become
the only signal — normalize each channel by its own token count), and
letting demo episodes into the scalar baseline (an imperfect demo's negative
advantage suppresses its own compliant scaffolding — demos must contribute
through the process channel only).

**5. Internalization is real, and beats prompting where it matters.**
Trained with NO rules in the prompt: clean^8 = 1.00, vs 0.88/0.55 for the
base model WITH rules in its prompt. Survives k=16 (0.97-1.00) and
deployment temperature. Emergent generalization: the trained model
reproduces failures before patching — a discipline never directly rewarded.
At 8B with rules prompted, compliance is perfect ANYWAY (Phase 0) — so the
RLVP payoff at production scale is dropping the policy text from every call
(tau2's policy document is ~10k tokens) while keeping reliability.

**6. Enforcement is not internalization.**
Runtime action-masking cannot create the missing action: blocking `submit`
does not produce `run_tests`; the model retries into the turn cap. Measured:
-33 to -62 points pass@1, 59% timeouts, 100% of episodes hitting the
guardrail. The internalized policy needs zero interventions at full success.

**7. RL beats behavior cloning exactly when demonstrations are imperfect —
which is the only realistic regime.**
With PERFECT demos (only possible in toy envs), BC matches RL (0.93 vs 0.87
perfect^8) — at the price of cloned, zero-diversity behavior. With realistic
demos (compliant workflow, failing solution — what a rule engine can
actually synthesize for real tasks): BC pass@1 = 0.00 on both domains (it
clones the failure); RL with the SAME demos = 0.97 (group advantage filters
the failure, the process channel keeps the scaffolding).

## The recipe (each component exists because a simpler thing failed)

1. Sparse outcome + per-tool-call verifiable terms: -lam per violation,
   +beta per discharged pending obligation  [pure penalties: inert, F2]
2. Hybrid credit: scalar + token channels, separately normalized
   [either alone fails complementarily, F3; global norm collapses, F4]
3. One synthesized compliant episode per group, process-channel-only
   [exploration wall, F2; baseline poisoning, F4]
4. Anneal lam,beta -> 0 after compliance saturates
   [compliance-only attractor, F4; doubles as internalization proof, F5]
5. Report pass^k / perfect^k  [pass@1 hides exactly the reliability
   failures this method exists to fix]

## Status / what's still running

Paper campaign (PAPER_PLAN.md): seeds for all main rows, clean holdout,
persistence, horizon-scaling figure, best-of-n + GiGPO/StepTool baselines,
Mistral-7B replication, tau2-bench training with policy-prompt-elimination
eval. Results land in RESULTS_LOG.md as stages complete.
