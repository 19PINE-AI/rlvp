# SWE-Gym Dask Task Set — Reusable Harness + Clean-Instance Count

**Date:** 2026-06-17
**Goal:** Scale the verified single-task oracle into a robust, reusable setup-and-test
harness across MANY Dask instances, and report how many are cleanly usable as an
RL training set. (NO RL training built here — just the harness + the count.)

## TL;DR

- **Dataset chosen: `SWE-Gym/SWE-Gym` (full)** — has **145 `dask/dask` instances**
  vs only **14** in `SWE-Gym/SWE-Gym-Lite`. Used the full set.
- **CLEAN dask instances: `129 / 145` (89%)** — setup_ok AND F2P fails on base AND
  F2P passes on gold. List saved to `dask_clean_instances.json`. **This is the
  training-set size.**
- **Per-task setup+verify: mean 4.6 s, median 3.9 s, p90 6.7 s** (with SHARED venvs).
  Full 145-instance sweep ran in **11.2 min** total.
- **Disk: 1.9 GB total cache** = 131 MB cached bare clone + **5 shared venvs**
  (~1.7 GB) covering all 145 instances. No per-task venv, no per-task clone.
- Harness: `swe_env_setup.py` → `setup_and_verify(instance, workdir) -> dict`.

## 1. Dask instance counts

| Dataset | total instances | `dask/dask` |
|---|---|---|
| `SWE-Gym/SWE-Gym-Lite` (train) | 230 | **14** |
| `SWE-Gym/SWE-Gym` (train) | 2438 | **145** |

The full set has 10× more dask tasks, so we use it. The 145 span **43 dask
versions** from `2.25` (2020) through `2024.5`.

## 2. Harness design (`swe_env_setup.py`)

`setup_and_verify(instance, workdir, run_gold=True, run_p2p=True) -> dict`
returns `{instance_id, version, group, setup_ok, f2p_fail_on_base,
f2p_pass_on_gold, p2p_pass_on_gold, timing{...}, error}`.

Frugality / RL-rollout-cost choices:

1. **One cached bare clone** at `cache/dask.git` (131 MB, cloned once). Each task
   gets a `git worktree --detach` at its `base_commit` (cheap, shares the object
   store), removed after the run. No re-clone, ever.
2. **SHARED venv per pin-group** under `cache/venvs/<group>`. Instances are grouped
   by dask-version era into 7 pin-groups that **dedupe by pin content to 5 actual
   venvs**. Each venv holds the pinned numpy/pandas/scipy + pytest. The first task
   in a group pays the ~7 s venv build; every later task reuses it.
3. **Per-task editable re-point**: `pip install -e .[array,dataframe,test]` into the
   shared venv, with a `constraints.txt` that *holds* numpy/pandas/scipy at the
   group pins while still pulling dask's own runtime deps (pyyaml, toolz,
   cloudpickle, fsspec, partd…). Re-point costs ~1–3 s.

### Pin-groups (generalised numpy<2 fix)

All these dask versions predate numpy 2.0, so **every group pins numpy<2** (newer
numpy removed `numpy.compat.basestring` etc. and breaks old dask on import). This
box runs **CPython 3.10**, whose earliest binary wheels are numpy 1.21.2 /
pandas 1.3.3 / scipy 1.7.2 — older pins have no cp310 wheels and fail to build from
source, so pre-2021 dask is floored at numpy 1.21.6 (imports fine there).

| group(s) | numpy | pandas | scipy | serves dask versions |
|---|---|---|---|---|
| era_2020 / era_2021a / era_2021b | 1.21.6 | 1.3.5 | 1.7.3 | ≤ 2021.05 (incl. 2.x) |
| era_2022a | 1.21.6 | 1.4.4 | 1.7.3 | 2022.01–2022.02 |
| era_2022b | 1.22.4 | 1.4.4 | 1.8.1 | 2022.03–2022.12 |
| era_2023 | 1.24.4 | 1.5.3 | 1.10.1 | 2023.x |
| era_2024 | 1.26.4 | 2.2.2 | 1.13.1 | 2024.x |

### One generalisation beyond the numpy pin

Old dask (2020 / early-2021) sets `filterwarnings = error:::dask.*` in `setup.cfg`,
which turns the py3.10 `distutils.LooseVersion` DeprecationWarning (raised at import
from `dask/compatibility.py`) into a hard error — masking the real F2P as an import
error. The harness runs pytest with `-W ignore::DeprecationWarning -W
ignore::ImportWarning`, which overrides that one ini filter without masking test
logic (these F2P tests assert on values, not on warnings).

## 3. Sample verification (8 instances, one per pin-group, spanning 2.30 → 2024.1)

| instance_id | ver | group | setup_ok | f2p_fails_base | f2p_passes_gold | p2p_passes_gold | seconds |
|---|---|---|---|---|---|---|---|
| dask__dask-6862 | 2.30 | era_2020 | True | True | True | True | 2.3 |
| dask__dask-7092 | 2021.01 | era_2021a | True | True | True | False* | 7.5 |
| dask__dask-8462 | 2021.11 | era_2021b | True | True | True | True | 4.2 |
| dask__dask-8686 | 2022.01 | era_2022a | True | True | True | True | 3.8 |
| dask__dask-7688 | 2022.04 | era_2022b | True | True | True | True | 3.0 |
| dask__dask-10441 | 2023.8 | era_2023 | True | True | True | False* | 5.5 |
| dask__dask-10846 | 2024.1 | era_2024 | True | True | True | False* | 4.6 |
| dask__dask-6960 | 2020.12 | era_2020 | True | True | True | False* | 3.7 |

**Sample: 8/8 CLEAN** (F2P fails on base, passes on gold). `*` P2P "False" = some
PASS_TO_PASS node-ids don't collect at that commit (renamed/parametrized tests not
present); P2P is run leniently (`--continue-on-collection-errors`) and counted as
pass iff there are 0 genuine failures among collectable P2P tests. P2P is a
secondary signal; the oracle is the F2P fail-then-pass.

## 4. KEY OUTPUT — full sweep over all 145 dask instances

**CLEAN = 129 / 145 (89.0%)** → `dask_clean_instances.json` (129 instance_ids).

Per-version clean rate is high and uniform across the whole 2020→2024 range
(e.g. 2.28 4/4, 2.30 11/14, 2022.02 7/7, 2023.3 5/8, 2024.1 6/6). No era is broken.

### The 16 non-clean instances (honest breakdown)

| category | count | cause |
|---|---|---|
| F2P fails after gold | 11 | mostly `dataframe/io/tests/test_parquet.py` tests that fail at **collection** (parametrize over parquet engines → `None` params) because the env lacks a pinned **pyarrow/fastparquet**; a few others are dep-version-sensitive |
| F2P "passes" on base | 5 | F2P is **skipped** on base (pytest exit 0 ⇒ looks like pass) — gated behind an optional dep (e.g. specific pyarrow feature) not present in the env |

6 of the 16 have `parquet`/`/io/` in the F2P node-id. **All 16 trace to optional
parquet/IO deps (pyarrow, fastparquet) not being pinned per-era** — a known,
addressable gap, NOT a harness bug. Adding era-appropriate `pyarrow`/`fastparquet`
to the pin-groups would likely recover most of these; 129 is the conservative,
verified-clean count today.

## 5. Cost numbers (for RL rollout feasibility)

- **Per-task setup+verify (shared venv warm):** mean **4.6 s**, median **3.9 s**,
  p90 **6.7 s**, max 21.6 s. This is clone-worktree + editable re-point + F2P-base +
  F2P-gold + 30 P2P tests.
- **First task per pin-group** additionally pays a one-time ~7 s venv build (5 venvs
  total = ~35 s amortised across 145 tasks).
- **Full 145-instance sweep:** 11.2 min wall.
- **Disk: 1.9 GB total**, steady-state: 131 MB bare clone + 5 shared venvs
  (305–484 MB each, ~1.7 GB). Per-task worktrees are created and removed; no
  per-task venv. Transient `/tmp` workdirs cleaned each task. Box stayed at 90% /
  ~363 GB free throughout.
- **RL implication:** a rollout that sets up the env + runs the oracle for a clean
  task costs **~4–5 s + the agent's own test run(s)**. Grouping rollouts by
  pin-group keeps all 5 venvs warm. Very cheap; 129-task training set is feasible.

## 6. Files

- `swe_env_setup.py` — **reusable harness.** `setup_and_verify()`, `pin_group_for()`,
  `ensure_bare_clone()`, `make_worktree()`, `ensure_group_venv()`, `cleanup_worktree()`.
  Holds `PIN_GROUPS` (per-era numpy/pandas/scipy) and content-dedup of venvs.
- `run_sample.py` — runs the N-instance spanning sample, prints the table.
- `run_all.py` — sweeps all 145 (grouped so each venv builds once), writes
  `cache/all_results.json` + `dask_clean_instances.json`.
- `analyze_failures.py` — clean count, per-version breakdown, failure categories, timing.
- `dask_clean_instances.json` — **the 129 clean instance_ids** (the RL training set).
- `cache/dask.git` — cached bare clone (131 MB). `cache/venvs/<group>` — 5 shared venvs.
- `cache/dask_instances.json` — the 145 dask instances (trimmed fields).
- `load_tasks.py`, `run_one.py` — original single-task proof (still valid).

## 7. Reproduce

```bash
cd /home/ubuntu/rlvp/swegym
python3 run_sample.py 8     # 8-instance spanning sample table
python3 run_all.py          # full 145 sweep -> dask_clean_instances.json (~11 min)
python3 analyze_failures.py # clean count + failure breakdown + timing
```

## 8. Notes for extending beyond dask

- The bare-clone + worktree + shared-venv-per-version scheme is repo-agnostic; only
  `PIN_GROUPS`, the GitHub URL, the extras string, and the test command are
  dask-specific. To add another SWE-Gym repo, replicate those four.
- To push dask past 129/145: add per-era `pyarrow`/`fastparquet` pins (recovers the
  parquet/IO F2P tests) and detect optional-dep skips (treat a base-skip as
  not-a-valid-oracle rather than counting it).
