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
- **T2 (ceiling):** higher converged score. **VALIDATED (gated)** — on the
  non-saturating gated task, outcome/DAPO/clean-RLVP/prompted all 0.0 vs
  RLVP+mixing 1.0 (reachable vs unreachable). tau2 is the real-benchmark check.

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
| **Auto-derived rules = hand rules** (capstone) | tags+errors: eps50 **320** vs hand **256**, both ~7× vs outcome 2240 |
| **T2 ceiling gap** (gated, non-saturating) | outcome/DAPO/clean-RLVP/prompted **0.0** vs RLVP+mixing **1.0** |
| **Mixing boundary** (when demos are needed) | REDUNDANT on discoverable chains; NECESSARY on hidden-precondition gated (clean RLVP discovery ceiling) |

### REFUTED / SCOPED DOWN
| claim | status |
|---|---|
| T2 ceiling on chain tasks | **REFUTED** — chains are compositionally easy, outcome-only also reaches 1.0. Ceiling test moved to non-saturating tasks. |
| "discharge credit is the hero" (earlier) | corrected — token-channel is the efficiency lever; discharge is secondary (dead-iter reduction) |
| step_cost as a universal bloat fix | harmful on short horizons (final 0.84<1.0); length control is HORIZON-DEPENDENT |
| pure penalties / positive-only rewards | inert / useless (earlier toy-env results, still hold) |
| **RLVP helps outcome UNIVERSALLY** | **REFUTED -> COVERAGE GRADIENT** (tau2). Generic/orthogonal rules HARM (collapse, reward 0). Policy-derived rules (procedural 0.37 AND verifiable-semantic 0.38, peak 0.67) remove the harm but neither beats outcome 0.48-0.60. Gradient SATURATES: rules cover general POLICY (procedure+validity), NOT task INTENT (=the reward). Irreducible ceiling, measured. |

### OPEN / RUNNING
- **Gated ceiling test** — DONE. T2 ceiling gap shown; mixing necessary here (see Validated).
- **tau2 head-to-head** — DONE (honest negative): outcome 0.60 > RLVP 0.375-0.45 with
  generic rules. Scopes the thesis: outcome gains need outcome-instrumental rules.
- auto_rlvp seeds (capstone n=1). tau2 with policy-derived (instrumental) rules = future work.

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

## Infra bug-fixes (real-benchmark pipelines, 2026-06)
Three blocking bugs found + fixed while bringing up the three real benchmarks:
1. **Lean REPL stderr-PIPE deadlock** (leanprove/lean_repl.py): `stderr=PIPE` was
   never drained -> REPL blocks on write once ~64KB accumulates -> readline hangs
   forever. Fixed: `stderr=DEVNULL`. This was the real cause of the "hung at 0% CPU"
   training stalls (NOT CPU starvation, as first hypothesized).
2. **Lean REPL select/readline buffer conflict** (same file): a select-gated
   buffered `readline()` slurps the multi-line reply into Python's text buffer, then
   select polls an empty fd and blocks. Fixed: raw `os.read(fd)` byte accumulation
   with a real per-call timeout (warmup_timeout for 1st cmd, tight per-tactic after).
   Adapter (lean_adapter.py) now catches apply_tactic/start_theorem exceptions and
   scores them as a penalty; trainer's _repl() rebuilds the dead process.
3. **SWE shared-venv editable race** (swegym/swe_env_setup.py + rlvp/swe_adapter.py):
   all concurrent episodes re-pointed ONE shared venv's `pip install -e` target ->
   corruption -> "no episodes" void runs. Fixed: per-slot venv POOL (SWE_VENV_SLOTS,
   default 6), each episode leases a private venv so its editable target is isolated.

Lesson: running all 3 benchmark backends (Lean REPL / Docker / pytest) 6-way parallel
masked these as "contention". Root causes were real bugs, surfaced by concurrency.

## REAL BENCHMARK RESULT #1: Lean theorem proving (clean, 40 iters/arm)
Qwen3-4B LoRA, synthetic Nat/algebra theorems, structural rules (errored-tactic
penalty + goal-progress discharge), anneal@24. RLVP (c3) vs outcome-only GRPO:

| metric            | RLVP | Outcome |
|-------------------|------|---------|
| dead iters (loss=0, all-fail) | **0/40** | **16/40** |
| iters to 0.5 success          | **16**   | **28**   |
| success (last 5)              | **0.80** | 0.717    |
| peak success                  | 1.0      | 1.0      |
| discharges/ep (goal progress) | **1.01** | 0.72     |
| violations/ep                 | 1.48     | 1.33     |

Headline: outcome-only spends 16/40 iters (all early: 2,3,6,8-12,...) doing dead
all-fail updates; RLVP's process channel has gradient on every one -> 1.75x faster
to threshold, higher plateau, more goal progress. Honest caveat: (a) single seed
so far; (b) RLVP violations slightly HIGHER (1.48 vs 1.33) -- annealing fades the
penalty late + RLVP explores more tactics while making more progress; not a
violations win, the win is dead-iter elimination + speed. Result: results/lean_RESULT.json

## REAL BENCHMARK RESULT #2: TerminalBench (clean, 40 iters/arm) -- HARM AXIS
Qwen3-4B LoRA, TerminalBench Docker tasks, structural rules (repeat_error +
blind_destructive penalties). RLVP (c3) vs outcome-only GRPO:

| metric                | RLVP | Outcome |
|-----------------------|------|---------|
| dead iters (all-fail) | 0/40 | 31/40 |
| task success (mean)   | 0.147 | 0.141 (EQUAL, near floor) |
| violations/ep (mean)  | 1.83 | 3.20 |
| violations/ep (last5) | 0.47 | 3.67 |

Headline: task success EQUAL and near floor (4B rarely solves TerminalBench), but
RLVP cuts harmful actions ~7x (3.67->0.47) over training while outcome stays high.
INDEPENDENT-AXIS contribution: RLVP reduces harm without changing outcome.
Result: results/term_RESULT.json

## REAL BENCHMARK #3: SWE-bench/SWE-Gym (dask) -- BOUNDARY, not a clean result
Pipeline runs end-to-end (venv-pool concurrency fix works), but at 4B scale:
 - ALL rewards 0: Qwen3-4B solves 0% of real dask bugs -> NO outcome signal.
 - Only ~2 episodes/iter: most instance setups fail (clean-instance list never
   persisted; ~8 loose-"clean" instances, most fail test_patch/editable setup).
Conclusion: SWE-bench Verified needs a stronger base model for RLVP to show an
outcome gain; at 4B it yields neither outcome nor dense-enough process signal.
Honest boundary. Lean(efficiency)+Terminal(harm) already cover both contributions.

## CORRECTION (SWE boundary RETRACTED): broken test env, not capability
The earlier "SWE 0% at 4B and 30B -> confirmed boundary" is INVALID. Verbose
trace of a 30B oracle+small-patch episode (1-line gold fix, file/line handed to
model) showed every run_tests erroring at `import dask.dataframe`:
  "Dask dataframe requirements are not installed"
Root cause: pin groups use a uniform pandas==1.3.5 (Jun 2021) for ALL dask eras,
but dask 2.25 (Aug 2020) is incompatible with pandas 1.3.x -> dataframe module
fails to import -> ALL dataframe tests error regardless of the model's fix. The
model was engaging correctly (15 test runs). So SWE success was NEVER measured;
0% is an env artifact. FIX: era-appropriate pandas/numpy pins (dask 2.25 -> pandas
~1.1). Until then, NO SWE capability claim (boundary or otherwise) is valid.
miniF2F stands: 1/20 (5%) zero-shot at 30B, real signal.

## SEED-VALIDATED (3 seeds/arm) — the two core results
LEAN (efficiency): dead iters RLVP 0±0 [0,0,0] vs OUTCOME 16±1.6 [16,14,18].
  success 0.886±0.084 vs 0.839±0.093 (overlapping -> NOT a significant success
  win; iters-to-0.5 variance huge -> speed claim NOT robust, retracted).
  Robust claim = dead-iteration elimination.
TERM (harm, independent axis): success EQUAL 0.097±0.069 vs 0.097±0.063 (floor);
  violations/ep RLVP 0.95±0.71 vs OUTCOME 3.99±0.57 (error bars SEPARATE) -> ~4x
  harm reduction at equal outcome. dead iters 0±0 vs 30±0.8.
results/seed_aggregate.json

## CORRECTION-2 (SWE env actually WORKS): it was a venv-build race, not pandas
The pandas-incompat diagnosis (CORRECTION above) was itself wrong. A clean
per-era import test shows dask.dataframe imports OK for ALL 7 eras (48/48
small-patch instances), era_2020 (dask 2.25 + pandas 1.3.5) included. The earlier
0% was episodes racing with HALF-BUILT slot venvs (concurrent first-time builds
during training corrupted/incompleted them) -> "requirements not installed". With
venvs pre-built (sequential), the env is valid. Re-measuring oracle+small-patch at
30B now. SWE is measurable after all.
