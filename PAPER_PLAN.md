# RLVP → top-tier paper: experiment plan (2026-06-13)

Priority order (single local GPU; code for later stages is built while earlier
stages train). RECIPE = c3, lam=beta=0.25, mix_scripted, anneal_at=40.

## Stage A — P4: process-channel-only scripted episodes (the untested fix)
- `script_scalar=False`: scripted episodes get ZERO scalar advantage (excluded
  from the group baseline) and contribute only token-attached process terms.
- Runs: A1 `c3v2_imp` (imperfect scripts + fix + anneal) — does compliance
  absorption rise from 0.63/0.00 toward 1.0 while pass@1 stays ~0.97?
  A2 `c3v2` (perfect scripts + fix + anneal) — candidate final recipe.

## Stage B — P2: seeds + CIs for every main table row
- +2 seeds each: RECIPE (c3anneal), outcome, c3mix(hot), SFT-BC (disjoint
  task ranges). Existing: c1 n=3, bare-c3 n=3. -> all main rows n>=3.
- `scripts/aggregate.py`: mean±std tables from all eval jsons.

## Stage C — P5a/P5c: clean internalization suite
- C1 `holdout_v2`: drop_rules + scripts STRIPPED of held-out steps
  (unconfounded transfer number).
- C2 `persist`: resume RECIPE checkpoint, 60 outcome-only iters,
  eval compliance drift every 10 (decay = not internalized).

## Stage D — P6: horizon-scaling figure (no training)
- Chained FileOps tasks (1/2/4 stages = ~3/6/12 rule-relevant decisions),
  zero-shot eval of base / base+rules-prompt / RECIPE checkpoint.
- Deliverable: clean-episode probability vs horizon (exponential decay vs
  flat line). Bonus: recipe generalizes to longer-horizon unseen structure.

## Stage E — P3: baselines beyond self-ablations
- E1 best-of-n inference baseline: sample n=4 episodes, deploy first
  compliant one; measure success/compliance/latency-cost.
- E2 `gigpo` credit variant (reimpl: per-(task,turn) return-to-go groups).
- E3 `steptool` credit variant (reimpl: +per-successful-call step reward).
- E4 DPO on compliant-vs-violating pairs (own implementation) — if time.
- Learned-PRM (AgentPRM) comparison: cluster-scale, documented as deferred.

## Stage F — P7: second model family
- Mistral-7B-Instruct-v0.3 (cached) via LoRA + RECIPE; requires chat-template
  abstraction in rollout.py (Qwen3 + Mistral formats).

## Stage G — P1: tau2-bench training (the long pole; timeboxed)
- Adapter: tau2 airline/retail domain tools+DB+tasks wrapped as ToolEnv;
  user simulator = vLLM Qwen3-8B (low mem fraction) called over HTTP;
  reward from tau2 evaluator on final DB state.
- ~12 semantic rules + discharges compiled from the policy doc.
- Train 4B RECIPE; eval WITH and WITHOUT the policy document in the prompt
  (policy-prompt elimination headline). Fallback if adapter exceeds timebox:
  telecom-small subset, reduced rule set.

## Stage H — P9 + assembly
- (1-p) gradient-saturation probe figure; GRPO all-fail degeneracy lemma;
  aggregate tables; REPORT.md -> paper-results structure; final commits.

Out of scope locally (documented for cluster): AgentPRM/learned PRM, SWE-Gym
+ mutation-score hacking audit, full tau2 multi-domain training, Llama-70B.
