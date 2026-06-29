# Prospective prediction: does the criterion predict a held-out domain *before* training?

Every result in the paper is post-hoc: we ran training, then explained the verdict with the
criterion. This experiment inverts the order on a **genuinely held-out domain** — we measure
the cheap probe, **pre-register the predicted verdict from a fixed rule (this file, committed
before any training)**, then train and check. A confirmed prediction is the difference between
"a story that fits five cases" and "a predictive instrument."

## The held-out domain
**The *hard* miniF2F slice**: `load_minif2f(easy_only=False)` minus the easy set = 104
IMO/AIME-level theorems (`aime_*`, `imo_*`, `aimeII_*`, ...). Every prior 30B Lean run used
`easy_only=True`; these theorems were **never trained on and never probed**.

Why this domain makes a *non-obvious* prediction: naively, harder tasks (more all-fail groups)
look like *more* room for a process reward to help. The reachability gate predicts the
**opposite** — if the base policy cannot make partial goal progress on these theorems, the
aligned potential is unreachable (Var_G(Phi)=0, like the SWE/E-C null) and gives no gradient,
*despite* maximal outcome sparsity. This directly tests the "help-where-least-needed" reframe.

## The probe (cheap, no training)
`leanprove/probe_minif2f.py`: base 30B, G=8 rollouts on TASKS held-out theorems, mode=`aligned`
(goal_progress discharge only -- exactly the event the `c3` credit rewards). Reports:
- `mean_var_phi`     = mean within-group Var_G(Phi), Phi = #goal_progress discharges.
- `frac_groups_reachable` = fraction of theorem-groups with >=1 rollout that decreases goals.

## PRE-REGISTERED decision rule (fixed before seeing the probe)
Let R = `frac_groups_reachable`, V = `mean_var_phi`.

- **REACHABLE** if `R >= 0.30` AND `V > 0.02`
    -> PREDICT: aligned `c3` **HELPS** -- reaches success strictly faster than outcome-only
       (>=1 iteration earlier to first nonzero success, or higher final-3 success),
       mirroring the easy-set #1 result.
- **UNREACHABLE** if `R < 0.10` (effectively the SWE/E-C null)
    -> PREDICT: aligned `c3` does **NOT help** -- aligned and outcome-only are within noise
       (both stuck near 0, or |final-3 succ gap| < 0.05). The reachability paradox: maximal
       sparsity, zero benefit.
- **AMBIGUOUS** if neither (0.10 <= R < 0.30, or V <= 0.02 with R >= 0.30)
    -> the criterion declines to make a sharp prediction here; report as inconclusive
       (this itself is honest -- the gate is near its boundary).

## Confirmation (training)
Two arms on the held-out hard set, identical except credit:
`scripts/minif2f_train.py 14 --credit c3  --out hard_aligned_heldout --seed 7 --muon`  (hard set)
`scripts/minif2f_train.py 14 --credit outcome --out hard_outcome_heldout --seed 7 --muon`
(a `--hard` flag is added to the trainer to load the held-out slice.)
Check whether the measured aligned-vs-outcome gap matches the pre-registered branch.

## Outcome log (filled in AFTER, append-only)
- **probe result** (commit a4bf2a3 ruleset, run 2026-06-29 ~01:21Z, 24 held-out theorems,
  G=8): `frac_groups_reachable=0.417`, `mean_var_phi=0.1107`, `base_succ=0.078`,
  `max_phi_seen=3`. The hard slice is hard-but-REACHABLE (unlike the SWE/E-C Phi=0 null):
  ~40% of theorem-groups admit partial goal progress though only 7.8% close.
- **pre-registered branch** (mechanical from rule): R=0.417>=0.30 AND V=0.1107>0.02
  => **REACHABLE => PREDICT aligned `c3` HELPS** (faster to first success / higher final-3
  than outcome-only). NB: this contradicts the naive "harder => unreachable" guess; the
  criterion predicts from the *probe*, not the difficulty.
- **training run v1 (Muon lr=5e-3, pre-registered default)**: aligned `c3` learned fast
  on the held-out hard set -- succ 0.0 -> 0.10 (it2) -> 0.90 (it3) -> 0.96 (it5) -- then the
  Muon update **diverged** at it6 (entropy 0.12->0.90, grad_norm 110+, succ collapse to 0.0).
  An optimization artifact (Muon orthogonalizes the gradient, so grpo's grad-clip=1.0 cannot
  bound its step; lr 5e-3 is too hot once Var_G(Phi) sharpens the landscape at the success
  jump). The early trajectory already supports HELPS, but the late collapse corrupts the
  pre-registered *final-3 succ* metric. Data preserved: results/run_hard_aligned_heldout_v1diverged.
- **DEVIATION (logged, fair)**: lower Muon lr 5e-3 -> **2e-3 for BOTH arms** (the only change,
  applied symmetrically; prediction and metric unchanged). Re-running aligned vs outcome.
- training run v2 result: <pending -- two arms at lr 2e-3>
- verdict (confirmed / refuted / inconclusive): <pending>
