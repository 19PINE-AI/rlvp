# RLVP vs. On-Policy Self-Critique — a constructive ablation

*Why verifiable rules, and not the model's own reflection, are the dense channel —
and where the reverse is true. Companion to [FINDINGS.md](FINDINGS.md). Last updated
this session.*

---

## The question

RLVP's dense signal comes from **verifiable rules** (hand-written or auto-derived).
A natural challenge: do we need rules at all? Could the policy **reflect on its own
trajectory** and produce the per-turn penalties itself — *self-rewarding* /
on-policy RLAIF, no rule predicates? Constraint, to keep it honest: the critic is the
**same model** as the policy (no larger judge), so any signal is genuine on-policy
self-reflection, **not distillation**.

This is not a tweak to RLVP — it is the opposing corner of the design space
(verifiability traded for generality). We treat it as a constructive ablation with a
**symmetric** thesis: *each channel is blind exactly where the other sees.*

## The frame (a 2×2, not "rules vs no-rules")

A naive "rules beat no-rules" comparison invites the obvious rebuttal — *of course more
information wins.* The real contribution is the **map** of when each channel works and
**why**, over two independent axes:

|                          | **Critic detects from trajectory** | **Critic blind** |
|--------------------------|-----------------------------------|------------------|
| **Rule cheaply specifiable** | **A.** both work → self-critique is a cheap *substitute* (cf. auto-rules) | **B.** **RLVP wins**; self-critique has a structural ceiling |
| **No cheap rule**            | **C.** **self-critique is all you have — and it works** | **D.** open; neither channel |

- **Result 2** = cell **B** — where rules exist *and* the violation is not
  self-evident, RLVP strictly beats self-critique.
- **Result 1** = cell **C** — where no cheap rule exists but the norm is reflection-
  detectable (soft/semantic intent, cf. τ² in FINDINGS §5), self-critique is the only
  option and it has signal.
- **A** is the control (tie); **D** is the open frontier.

---

## Summary of findings

*Status: all 6 experiments complete (Exp0, cross-model probe, Exp1 matrix, multi-seed +
frozen ablation, τ² cell-C offline + training).*

1. **Self-critique has a structural blind spot** (Exp0 + probe). Blind self-critique
   recovers surface-evident *ordering* norms but is blind to *stateful-bookkeeping* ones
   (`blind_write`/`untested_submit` ≈ 0) — even when told the rules, and **persisting
   1.7B→8B**. On-policy critic recall is **scale-flat (0.46/0.38/0.41)**: it does not
   scale. (The one "masked" rule is capability-gated, 0→0.75→0.83 — but only a bigger
   judge closes it = distillation, which the on-policy constraint forbids.)
2. **Where a rule exists, the rule wins — robustly (3 seeds) and for a known reason.**
   In the all-fail regime the penalty-only rule is decisive every seed; both live and
   **frozen** self-critics are inert every seed. Frozen ≈ live ⇒ the cause is the
   critic's **6% false-positive imprecision** (P=0.94 vs rule 1.00), **not**
   non-stationarity. Detection accuracy alone is not enough.
3. **But that gap is regime-specific** (4B). When a stronger policy escapes the all-fail
   wall, the outcome channel dominates and the rule-vs-critic distinction washes out —
   on-thesis (process rewards matter most exactly on all-fail groups).
4. **Self-critique is a good intent *detector* but a failed intent *reward*** (τ² cell-C,
   3 seeds offline + training). Offline, where rules can't reach *intent*, blind
   self-critique is the only channel with signal: it flags **0.42±0.01** of intent-miss
   failures (rules: 0%) and predicts failure at **F1 0.63±0.02 vs rules 0.23±0.06**. But
   used as a dense training reward it **collapses** the policy (reward 0.01→0.00 vs
   `outcome` 0.09→0.50, rules 0.10→0.35) — the noisy per-turn penalties corrupt learning.
5. **No self-rewarding bootstrap** — critic↔oracle agreement stays flat across training.

**Spine:** *a deterministic verifiable rule is the better dense reward wherever one
exists — it beats on-policy self-critique even at equal structure and perfect recall
(it's the critic's imprecision), at least while the policy is stuck on all-fail groups.
Where no rule reaches the reward (intent), self-critique can **diagnose** failures
offline that rules can't — but it does **not** convert into a training reward (it
collapses). Use rules for what's verifiable; use self-critique for offline diagnosis,
not as a reward; and don't expect it to scale or bootstrap.*

---

## Experiment 0 — can the model recover its own violations *without* the rules? (✅ at 1.7B)

**Method.** Roll out Qwen3-1.7B (policy = critic, on-policy) on 24 tasks each in
fileops / csops / gated. Then the *same model* reflects on its own numbered transcript
and flags mistakes, in two modes: **blind** (given only the domain tools + goal, **no
rules**) and **rule-aware** (told the rules — the *detection ceiling*). Score flagged
turns against the deterministic rule oracle.
Code: `rlvp/self_critic.py`, `scripts/exp_selfcritic.py`.

**Headline (turn-level, micro-averaged over all domains):**

| mode | precision | recall | F1 |
|---|---|---|---|
| blind | 0.88 | 0.47 | 0.61 |
| rule-aware (ceiling) | 0.78 | 0.54 | 0.64 |

The average hides the result. **Per-rule recall is the finding:**

| domain | rule | blind R | rule-aware R | class |
|---|---|---|---|---|
| csops | `no_kb_before_call` | **1.00** | 1.00 | surface-evident ordering |
| gated | `access_without_acl` | **1.00** | 1.00 | surface-evident ordering |
| fileops | `untested_submit` | 0.04 | 0.33 | stateful bookkeeping |
| fileops | `blind_write` | **0.00** | **0.00** | stateful bookkeeping |
| gated | `write_without_access` | **0.00** | **0.00** | consequence-masked |

**Interpretation.** Blind self-critique reliably recovers violations that are
**self-evident from the trajectory surface** — common-sense ordering you can read off
the text ("called before doing any research"; "requested access before reading the
policy file"). It **fails** on two distinct classes, and crucially the rule-aware
ceiling **barely helps** on them:

1. **Stateful bookkeeping** (`blind_write`, `untested_submit`): requires reconstructing
   history — *did I read **this** path? did I test after **my last** mutation?* The
   model can't, **even when handed the rule** (`blind_write` = 0/8 with the rule given).
2. **Consequence-masked** (`write_without_access`): the env hides the failure (the
   illegal write reports "Wrote N bytes" but silently no-ops), so there is nothing for
   reflection to latch onto — blind by construction.

That the rule-aware ceiling doesn't rescue these means the gap is an **information
limit, not a knowledge gap**: telling the model the norm doesn't help if it cannot
detect the norm's antecedents in the trajectory. These two classes are precisely where
verifiable rules — which carry the bookkeeping and see the true env state — earn their
keep. This populates cells **A** (csops/gated ordering) and **B** (fileops stateful,
gated masked) of the 2×2 with a mechanism.

**Honest caveats.**
- The csops P=R=1.0 is on a **degenerate distribution**: the 1.7B policy fails so early
  it commits only *one* blatant violation type, so precision is trivial. Read it as
  "this rule is easy," not "csops is solved." A stronger test needs a more competent
  policy whose trajectories are mixed (→ the scale sweep).
- Single model (1.7B) so far. The "upper bound of self-critique" claim requires scale.

## Upper-bound studies

### Naive scale sweep (✅) — and why it was inconclusive

Re-ran Exp0 at 4B and 8B (`scripts/exp_selfcritic.py <model>`). Overall blind recall
did **not** improve (0.47 → 0.34 → 0.45), but per-rule numbers were **noisy and
non-monotonic** (`write_without_access` 0 → 0.92 → 0). Cause: *each model produces
different trajectories* (different success → different violation mix), so the per-rule
cells aren't apples-to-apples. This motivated the fixed-trajectory probe.

### Fixed-trajectory cross-model probe (✅ — the clean scale + distillation result)

Roll out **one** policy (1.7B, 72 episodes), then have **1.7B / 4B / 8B critics judge
the *identical* trajectories** (`scripts/sc_rollout.py` + `sc_critique.py`). Differences
are now pure critic effect — and policy-fixed/critic-varied **is** the distillation
probe. Blind per-rule recall:

| rule | class | 1.7B | 4B | 8B |
|---|---|---|---|---|
| `no_kb_before_call` | surface | 1.00 | 0.95 | 1.00 |
| `access_without_acl` | surface | **1.00** | **0.00** | **0.00** |
| `untested_submit` | stateful | 0.04 | 0.00 | 0.00 |
| `blind_write` | stateful | **0.00** | **0.00** | **0.00** |
| `write_without_access` | masked | **0.00** | **0.75** | **0.83** |
| **overall blind recall** | | **0.46** | **0.38** | **0.41** |

**What this corrects and establishes:**
1. **The stateful-bookkeeping blind spot is the robust structural ceiling.**
   `blind_write` = 0 at every critic size; `untested_submit` ≈ 0 even *rule-aware* at
   4B/8B. Scaling the critic 5× does **not** recover violations that need reconstructing
   trajectory history. This is the core "why verifiable rules."
2. **"Consequence-masked" was partly a capability gap, not structural** (a correction to
   the 1.7B-only read): `write_without_access` rises **0 → 0.75 → 0.83** with critic
   size on the *same* trajectories — the /acl policy text *is* in the transcript, and a
   capable-enough critic infers the violation. But the only way to buy that detection is
   a **bigger judge = distillation**, which the on-policy constraint forbids.
3. **On-policy self-critique does not scale.** Overall blind recall is flat (0.46 →
   0.38 → 0.41): gains on the masked rule are offset by an idiosyncratic collapse on
   `access_without_acl` (small critic 1.0, big critics 0.0 — critic behavior is not
   monotone). The *upper bound* of on-policy self-reflection is low and ~scale-flat.

**Self-rewarding bootstrap** — tracked via the training-time monitor (critic P/R across
iters); see Exp1.

## Experiment 1 — training with self-critique as the dense reward (✅ 1.7B, 40 iters)

New GRPO credit mode `llmcritic` (`rlvp/grpo.py`, `rlvp/self_critic.py`): the per-turn
penalty comes from the policy's **own blind self-critique** instead of rule predicates —
same weights = policy = critic, on-policy, no distillation. The rule oracle runs purely
as a **reward-hacking monitor** (logged `critic_precision`/`recall` vs. the unused
oracle, and the *true* violation rate). Matrix: `scripts/run_selfcritic_exp1.sh`;
aggregate: `scripts/exp1_aggregate.py`. Harm metric = **violations/episode** (viol/100-
calls is confounded by call-count changes).

| run | credit structure | viol/ep early→late | success early→late | critic R |
|---|---|---|---|---|
| outcome/fileops | outcome only | 1.40 → 1.34 | 0.22 → 0.38 | — |
| `c2`/fileops | rule: token + discharge | 1.41 → 1.34 | 0.22 → 0.29 | — |
| `c3`/fileops | rule: scalar + token + discharge | 1.40 → **1.25** | 0.22 → 0.25 | — |
| `llmcritic`/fileops | critic: token only | 1.41 → 1.34 | 0.22 → **0.54** | 0.19 |
| `c2`/csops | rule: token + discharge | 0.96 → **0.04** | 0 → 0 | — |
| `c3`/csops | rule: scalar + token + discharge | 1.00 → **0.01** | 0 → 0 | — |
| `llmcritic`/csops | critic: token only | 1.02 → **0.92** | 0 → 0 | **1.00** |

**Findings:**
1. **fileops (critic blind, R=0.19): self-critique does not reduce true violations.**
   `llmcritic` viol/ep ≈ `outcome` (1.34), while rules (`c3`) cut most (1.25). And
   `llmcritic` reached the *highest success* (0.54) while *not* cutting true violations
   — its low-precision penalties didn't target real harm (a reward-hacking signature:
   the policy improves the task without the intended harm reduction).
2. **csops (critic accurate, R=1.0): even perfect detection isn't enough.**
   `llmcritic` barely moves violations (1.02 → 0.92, −10%) while *both* rule variants
   eliminate them (`c2` 0.96 → 0.04; `c3` 1.00 → 0.01). So **rules win even without
   `c3`'s scalar-folding** (`c2` ≈ `c3`) — and even when the critic detects the same
   violations the rule does.
3. **Why rules win at csops — imperfect *precision*, confirmed by 3 seeds + a frozen-
   critic ablation.** Isolation cell `c2nodis` = penalty-only rule, β=0 (no discharge),
   **identical credit structure to `llmcritic`** (and to `llmcriticfrozen`, which judges
   with a never-updated copy of the base model). Multi-seed result (csops, seeds
   11/22/33; mean±std of late viol/ep and late success):

   | csops variant | detection (P/R) | late viol/ep | late success | per-seed behaviour |
   |---|---|---|---|---|
   | `c2nodis` (rule) | 1.00/1.00 | 0.38 ± 0.44 | 0.33 ± 0.47 | **decisive every seed**: viol→~0.06 (s22,s33, stuck) **or** success→0.99 (s11, escaped) |
   | `llmcritic` (live critic) | 0.94/1.00 | 0.94 ± 0.05 | 0.00 | does ~nothing, every seed |
   | `llmcriticfrozen` (frozen) | 0.94/1.00 | 0.93 ± 0.02 | 0.00 | does ~nothing, every seed |

   Two robust conclusions:
   - **Rules >> self-critique holds across all seeds** — but on a *seed-dependent axis*.
     The clean rule signal is always decisive: in the stuck-all-fail regime it drives
     violations to ~0.05 (matching the original seed-7 0.04), and when it instead lets
     the policy **escape to 99% success** (seed-11) the remaining violation is a single
     outcome-neutral one. The self-critic is inert on *both* axes in *every* seed.
   - **The cause is the critic's imperfect precision, NOT non-stationarity.** Freezing
     the critic (`llmcriticfrozen`, a stationary judge) does **not** close the gap —
     it fails identically to the live critic (0.93 vs 0.94). What both share is P=0.94
     (vs the rule's 1.00): ~6% false positives that penalize correct actions. On csops's
     all-fail groups the per-turn penalty is the *only* gradient, so that imprecision
     has nothing to wash it out. **A perfectly-recalling but imperfectly-precise
     self-critic optimizes far worse than a deterministic rule — and making it stationary
     doesn't help.**

4. **No self-rewarding bootstrap.** The training-time critic↔oracle agreement is **flat**
   across 40 iters — csops stably high (P 0.92–1.0, R=1.0), fileops stably low/noisy
   (P 0→1.0, R 0–0.5). The policy improving did **not** sharpen the self-critic; the
   hoped-for virtuous cycle did not appear in these domains. (The frozen-critic ablation
   in finding #3 settled the residual question: the gap is the 6% false-positive
   imprecision, not per-iteration drift — a stationary frozen critic fails identically.)

5. **4B replication — the gap is *regime-specific* (all-fail), and that's on-thesis.**
   Repeating the matrix at 4B (`*_4b` runs) does **not** reproduce the sharp 1.7B gap —
   for an illuminating reason: at 4B the tasks become *solvable*, so the policy escapes
   the all-fail wall and the **outcome channel dominates**, washing out the process-
   channel distinction. csops: 4B solves it (success 1.00 for all of c3/llmcritic/
   c2nodis), and none removes the residual outcome-neutral violation (c3 1.00, c2nodis
   1.18, llmcritic 1.34 — rules mildly better, critic mildly worse, all weak). fileops:
   4B gets it partly solvable (success 0.53→0.84–0.89); both c3 and `llmcritic` cut
   viol/ep 1.48→1.00 — but `llmcritic`'s critic had **recall 0** there, so its harm drop
   rode the *outcome* improvement, not the critic. **Takeaway:** rules ≫ self-critique is
   a phenomenon of the **all-fail regime** (weak policy, process channel = sole
   gradient); once a stronger policy escapes all-fail and outcome reward takes over, the
   distinction fades. This bounds the 1.7B result's scope rather than contradicting it,
   and it's exactly consistent with RLVP's core thesis (process rewards matter most
   precisely on the all-fail groups outcome-only RL can't learn from).

## Result 1 / cell C — tau2 intent (✅ offline, Qwen3-4B)

The symmetric counterpart: a domain where **no cheap verifiable rule reaches the
reward**, so self-critique is the *only* dense channel. τ²-bench airline is exactly this
(FINDINGS §5): the `semantic` RuleTracker (`modify_basic_economy`,
`change_passenger_count`, `payment_not_in_profile`) covers policy *validity*, but the
reward's residual is task *intent*, which rules can't express.

**Method** (`scripts/tau2_cellc_{rollout,critique}.py`, driver `tau2_cellc_run.sh`):
roll out 60 airline episodes (Qwen3-4B + domain policy, `semantic` tracker, user-sim =
Qwen3-4B vLLM), record transcript + outcome + semantic violations; then **blind
self-critique** (same 4B) flags mistakes. The rule oracle here is the semantic tracker.
**Replicated over 3 independent rollout seeds** (60 eps each; ~43–50 intent-miss
episodes per run).

**Results.** Most episodes fail and **the large majority of failures are semantically
*clean*** (rules silent) — so the verifiable rules see almost none of the actual failure
mass; intent is the residual, exactly as §5 predicts. Across the 3 runs, as a
**failure predictor** (mean ± std):

| failure predictor | F1 |
|---|---|
| semantic rules | **0.23 ± 0.06** |
| blind self-critique | **0.63 ± 0.02** |

- **Intent-miss recall** (episodes that failed but are rule-clean, where rules flag 0% by
  construction): blind self-critique flags **0.42 ± 0.01** of them — highly consistent
  across seeds.
- As a failure signal, self-critique (F1 0.63) is **~2.7× the semantic rules** (F1 0.23),
  which are nearly blind to intent failures.

*(Single-run numbers were 60 eps, 49/53 failures clean, recall 0.41, F1 0.61 vs 0.14 —
the 3-seed averages confirm them.)*

| failure predictor | precision | recall | F1 |
|---|---|---|---|
| semantic rules | 1.00 | **0.075** | 0.14 |
| blind self-critique | 0.92 | **0.45** | **0.61** |

- **Intent-miss recall:** on the 49 failed-but-clean episodes (rules flag **0%** by
  construction), the blind self-critic flags a mistake on **41%**. False-flag rate on
  the 7 successes is 29% (small n).
- **As a failure signal**, self-critique (F1 0.61) is **~4× better than the semantic
  rules** (F1 0.14), which are nearly blind to intent failures (R=0.075).

**Offline conclusion.** Where verifiable rules hit their intent ceiling, blind
self-critique supplies a **partial but real diagnostic** signal the rules structurally
cannot — the mirror image of Exp0/Exp1's stateful blind spot. (Bounds: intent recall
0.42, precision-on-failure ≈ base rate — a weak-but-nonzero *detector*.)

### Training test (✅ — and it's a NEGATIVE that sharpens the whole study)

Does that offline intent signal actually *train* a better policy? We ran `llmcritic` vs
`semantic`-c3 vs `outcome` on tau2 (20 iters each, `scripts/tau2_train.py`,
`run_tau2_cellc_*`). Final reward (last-4-iter mean), reward early→late:

| reward channel | reward early→late | verdict |
|---|---|---|
| `outcome` | 0.09 → **0.50** | learns well |
| `semantic`-c3 (rules) | 0.10 → **0.35** | learns, but below outcome (= §5 ceiling) |
| `llmcritic` (self-critique) | 0.01 → **0.00** | **collapses — never learns** |

**Self-critique is a good intent *detector* but a failed intent *reward*.** As a dense
training signal it doesn't just underperform — it **collapses the policy to zero reward**
(flat at ~0 across all 20 iters), worse than outcome-only and worse than the rules. The
noisy per-turn critic penalties actively corrupt the (graded, non-all-fail) outcome
signal rather than complementing it.

**Unified conclusion — self-critique is a poor *training reward* everywhere.** In cell B
(rules domain) it's beaten by a deterministic rule even at equal structure & perfect
recall (the 6% imprecision). In cell C (intent domain) it collapses entirely. Its **only**
demonstrated value across the study is **offline diagnosis** of intent failures rules
can't see. So the honest 2×2 is asymmetric after all: *rules are the better dense
reward wherever they exist; where they don't, self-critique can flag intent failures
offline but cannot (here) be turned into a reward that trains.* That's a stronger,
more defensible "why verifiable rules" than a clean cell-C training win would have been.
(Caveat: tau2 training is short/high-variance — 20 iters, 4 tasks/iter, single seed — so
treat magnitudes as indicative; the qualitative collapse is unambiguous across all iters.)

## Positioning

This is the **RLVP vs. on-policy RLAIF / Self-Rewarding LM** ablation (cf. Yuan et al.
Self-Rewarding LMs; Constitutional AI critique; RLAIF). The claim is not that intrinsic
critique is useless — it is that it has a **structural blind spot** (stateful + masked
norms) exactly where verifiable rules are robust, and conversely is the *only* channel
where norms can't be cheaply specified. The two results are complementary corners of
one map, not a contest.
