# Lean Theorem-Proving Oracle — Pipeline Status

Gate check: prove a formal-theorem-proving pipeline works end-to-end on this box
for one theorem, as the prerequisite for building RLVP training on it.

**Status: PASS.** A working per-tactic verifiable oracle is running. The Lean
kernel, via the leanprover-community REPL, tells us for any (theorem, tactic)
pair whether the step is valid and what goals remain. 12/12 oracle assertions
pass (`demo_oracle.py`).

All work is under `/home/ubuntu/rlvp/leanprove/`. Nothing outside it was touched.

---

## 1. Toolchain

| Component | Version / status |
|-----------|------------------|
| elan      | 4.2.3 (installed to `leanprove/.elan`, self-contained via `ELAN_HOME`) |
| Lean 4    | **4.31.0** (`leanprover/lean4:v4.31.0`, Release) |
| lake      | bundled with Lean 4.31.0 |
| Lean REPL | `leanprover-community/repl`, **builds clean in ~5 s**, no Mathlib needed |

Disk footprint: **3.0 GB total** (2.8 GB Lean toolchain + 227 MB REPL build).
Build is fast because the REPL itself has no Mathlib dependency.

To use: `export ELAN_HOME=/home/ubuntu/rlvp/leanprove/.elan; export PATH=$ELAN_HOME/bin:$PATH`

---

## 2. REPL build status

Cloned `https://github.com/leanprover-community/repl`. Its `lean-toolchain`
pins `v4.31.0` (matches the stable toolchain installed). `lake build` →
"Build completed successfully (24 jobs)" in ~5 s. Run with `lake exe repl`,
JSON on stdin/stdout, commands separated by blank lines.

---

## 3. miniF2F availability (+ Mathlib cost)

- Source: `https://github.com/yangky11/miniF2F-lean4` — **244 test + 244 valid
  = 488 statements**, all in Lean 4 `:= by sorry` form.
- **Every statement does `import Mathlib`.** Running miniF2F therefore requires
  a full Mathlib build inside a REPL-enabled project: multi-GB cache/build,
  tens of minutes to hours of compile, and meaningful disk on a 90%-full shared
  box. **Deferred as a scaling cost**, not a blocker for the gate.
- Sample miniF2F statement (captured, in `load_theorems.py`):

  ```lean
  theorem aime_1983_p1 (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 ≤ w)
      (h0 : Real.log w / Real.log x = 24) (h1 : Real.log w / Real.log y = 40)
      (h2 : Real.log w / Real.log (x * y * z) = 12) : Real.log w / Real.log z = 60 := by sorry
  ```

- For the gate demonstration we use a **self-contained theorem set** (no
  Mathlib): `2 + 2 = 4` (`rfl`), `n + 0 = n` (`omega`), `a + b = b + a`
  (`omega`), `p ∧ q → q ∧ p`, transitivity of `≤` (`omega`). These exercise the
  full oracle today without paying the Mathlib cost.

---

## 4. Oracle demonstration (the deliverable)

`apply_tactic(proofState, tactic)` distinguishes valid proof steps, proof
completion, and invalid steps. Raw JSON exchange through `lake exe repl`:

**(a) Open a goal** — submit `theorem ... := by sorry`, get a `proofState`:
```
>> {"cmd": "theorem t (n : Nat) : n + 0 = n := by sorry"}
<< {"sorries":[{"proofState":0,"goal":"n : Nat\n⊢ n + 0 = n", ...}], "env":0}
```

**(b) CORRECT tactic** — proof completes, no goals:
```
>> {"tactic": "omega", "proofState": 0}
<< {"proofStatus": "Completed", "proofState": 1, "goals": []}
```

**(c) WRONG tactic** (`rfl` on commutativity) — error reported:
```
>> {"tactic": "rfl", "proofState": 2}
<< {"message": "Lean error:\nTactic `rfl` failed: The left-hand side\n  a + b\n
    is not definitionally equal to the right-hand side\n  b + a ..."}
```

**(d) UNKNOWN tactic** — error reported:
```
>> {"tactic": "ring_nf", "proofState": 2}
<< {"message": "Lean error:\n<input>:1:1: unknown tactic"}
```

**(e) MULTI-STEP process signal** — intermediate goal remains, then completes:
```
>> {"tactic": "obtain ⟨hp, hq⟩ := h", "proofState": 0}
<< {"proofStatus": "Incomplete: open goals remain", "proofState": 1,
    "goals": ["p q : Prop\nhp : p\nhq : q\n⊢ q ∧ p"]}
>> {"tactic": "exact ⟨hq, hp⟩", "proofState": 1}
<< {"proofStatus": "Completed", "proofState": 2, "goals": []}
```

`demo_oracle.py` asserts all of the above across 5 theorems + 1 multi-step
proof: **12 passed, 0 failed**, covering error modes: unknown tactic, parse
error, `rfl` failure, type mismatch.

---

## 5. Interaction interface for the RLVP tool-env

The tool-env wraps `LeanREPL` (in `lean_repl.py`). Per rollout step:

```
agent emits a tactic string
  -> repl.apply_tactic(proof_state, tactic)
  -> {done: bool, error: str|None, goals: list[str], proofState: int|None, latency_ms}
```

Reward / process-signal mapping:

| Oracle return                                  | RL signal |
|-----------------------------------------------|-----------|
| `done == True and error is None`              | **terminal REWARD** — proof complete (goals == []) |
| `error is None and not done` and goals shrank/changed | **process reward** — valid step (discharge), goals advanced |
| `error is None and not done` and goals unchanged | neutral/no-op (e.g. `skip`-like) — optional small penalty |
| `error is not None`                           | **PENALTY** — invalid tactic (unknown / parse / kernel reject) |

The agent always threads the `proofState` returned by the previous step into
the next `apply_tactic` call. `start_theorem(statement)` opens a fresh goal and
returns the initial `proofState`.

**Latency (measured):**
- First call after process start (JIT/warm-up): ~300 ms.
- Steady-state tactic roundtrip on a **persistent** REPL process: **~0.5–8 ms**
  (median ~5 ms; trivial `exact`/`rfl` <1 ms, `omega` 5–8 ms).
- **The REPL process is reused across tactics** — `LeanREPL` keeps one
  long-lived `lake exe repl` subprocess and threads proofState IDs. This is the
  key rollout-speed property: pay the warm-up once, then thousands of cheap
  tactic checks. Each proof can be re-opened cheaply via a new `sorry` command.

---

## 6. Files delivered

| File | Purpose |
|------|---------|
| `lean_repl.py`     | `LeanREPL` — persistent subprocess; `start_theorem`, `apply_tactic(state, tactic) -> {done, error, goals, proofState, latency_ms}`. Run directly for a smoke test. |
| `load_theorems.py` | Self-contained theorem set (proven) + miniF2F metadata/sample (deferred). `load_theorems()`, `minif2f_available()`. |
| `demo_oracle.py`   | End-to-end gate proof: good vs bad tactic per theorem + multi-step. 12/12 pass. |
| `repl/`            | Built Lean REPL (`lake exe repl`). |
| `.elan/`           | Self-contained elan + Lean 4.31.0 toolchain. |

Reproduce:
```bash
export ELAN_HOME=/home/ubuntu/rlvp/leanprove/.elan
export PATH=$ELAN_HOME/bin:$PATH
cd /home/ubuntu/rlvp/leanprove
python3 demo_oracle.py      # -> 12 passed, 0 failed
```

---

## 7. Mathlib build for miniF2F — DONE via `lake exe cache get` (fast path)

**Status: COMPLETE.** `leanprove/mathlib_repl/` is a lake project requiring
Mathlib pinned to **v4.31.0** (same Lean the REPL uses). The frugal path worked:

```
lake update                # resolve mathlib dep + manifest
lake exe cache get         # downloaded 8542 prebuilt Mathlib oleans  <-- KEY TRICK
lake build                 # 8559 jobs, "Build completed successfully"
```

- **`lake exe cache get` worked** — all 8542 olean files downloaded + decompressed
  from the leanprover-community cache. **No slow from-scratch compile was needed.**
- **Wall time: ~53 seconds total** (clone + cache get + build), not the
  feared "tens of minutes to hours." Log: `leanprove/mathlib_build.log`,
  build script `leanprove/mathlib_build.sh` (launched with `nohup ... &`;
  finished cleanly, `MATHLIB_BUILD_DONE rc=0`).
- **Disk cost: ~8 GB** (363 GB free -> 355 GB free). Frugal as hoped.
- `minif2f_available()` now returns **True** (`mathlib_repl/.lake/build` exists).
  To run the real miniF2F benchmark, launch `LeanREPL(repl_dir=...)` against a
  REPL built inside / pointed at this project so `import Mathlib` resolves, then
  stream the 488 `yangky11/miniF2F-lean4` statements. The oracle interface
  (`apply_tactic`) is unchanged.

ETA to miniF2F-ready: **already there** (Mathlib oleans cached). Remaining glue
is wiring a Mathlib-env REPL + downloading the 488 statement files.

---

## 8. RLVP Lean adapter (training-ready) — DONE, smoke gate PASS

Self-contained (no-Mathlib) theorem set + token-exact RLVP adapter, validated
without waiting on Mathlib:

| File | Purpose |
|------|---------|
| `theorems_synth.py` | 15 self-contained multi-step theorems (2-5 tactics: induction, cases, rw, omega, simp, constructor, intro/exact). **15/15 gold sequences reach `done`** via the REPL. |
| `../rlvp/lean_adapter.py` | `LeanRuleTracker` (process signals from the oracle: `errored_tactic` penalty, `goal_progress` discharge, `no_progress` penalty), `run_lean_episode(...)` building the SAME Episode shape as tau2_adapter (token-exact action_spans + turn_violations/discharges + a `LeanShimEnv` with .success/.violations/.discharges/.outcome_reward()/.calls). Reuses `GenServer`. One persistent REPL per episode; fresh proof state per theorem. |
| `../scripts/lean_smoke.py` | Gate (no training): Qwen3-1.7B, 3 rollouts. **PASS** — 3/3 episodes roll out via the REPL oracle, rewards 0/1, rules fire. |
| `../scripts/lean_train.py` | GRPO training loop (Qwen3-4B + LoRA r=32, G=6, credit c3, `build_advantages`/`update_policy` unchanged, micro_token_budget 1024, `--rule-mode/--anneal/--credit/--out`). 1-iter loop verified end-to-end. |

**Smoke output (3 episodes, ~0.9-1.5 s each):**
- `and_swap`: 3 invalid tactics -> 3x `errored_tactic`, reward 0.0.
- `and_comm_iff`: model PROVED it (`intro h` -> `rw [and_comm]` -> `exact h`),
  reward 1.0, `no_progress` + `goal_progress` both fired.
- `or_swap`: 3 invalid tactics -> 3x `errored_tactic`, reward 0.0.

Token bookkeeping verified: action_spans are valid/monotonic/in-bounds; a
scripted-gold rollout reaches reward 1.0 with `goal_progress` on the productive
step. Full GRPO loop (rollout -> build_advantages(c3) -> update_policy -> LoRA
merge/save) runs in ~7 s/iter on 1.7B.
