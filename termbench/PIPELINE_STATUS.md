# TerminalBench Pipeline Status — END-TO-END PROVEN ✅

Date: 2026-06-17
Box: shared devbox, Docker 28.5.2, Python 3.10.12, disk 90% (~363 GB free)
All work under `/home/ubuntu/rlvp/termbench/`. No tbench Docker images/containers left behind.

## TL;DR

A single TerminalBench task (`hello-world`) was proven end-to-end on this box via
**direct Docker** (bypassing the tbench Python harness):
- oracle **PASSES** with the gold solution (2 passed)
- oracle **FAILS** without it (2 failed)  → the oracle is a real verifiable signal, not always-pass
- oracle **PASSES** when driven by an agent bash command  → the RL action interface works

This is the gate for building RLVP training on it. **Gate: PASSED.**

---

## 1. Install / setup

- pip package `terminal-bench` **does not install** here: it requires `python>=3.12`
  (`.python-version` = 3.13); this box has 3.10. `pip install --user terminal-bench`
  → "No matching distribution".
- We **cloned the repo** instead:
  `git clone --depth 1 https://github.com/laude-institute/terminal-bench`
  → `/home/ubuntu/rlvp/termbench/terminal-bench/` (version 0.2.18).
- `uv` is available at `/home/ubuntu/.local/bin/uv` if we later want the full harness
  (it would need a py3.12/3.13 venv: `uv venv --python 3.13`). Not required for the proof.

## 2. Task count & structure

- **241 tasks** under `terminal-bench/original-tasks/<task-id>/`.
- Each task directory contains:
  | file | role |
  |------|------|
  | `task.yaml` | instruction text + metadata (difficulty, category, `parser_name: pytest`, `max_test_timeout_sec`, `run_tests_in_same_shell`) |
  | `Dockerfile` | builds the task image; base sets `WORKDIR /app` |
  | `docker-compose.yaml` | service `client` runs `sh -c "sleep infinity"` |
  | `solution.sh` (or `solution.yaml`) | **gold/reference solution** |
  | `run-tests.sh` | installs pytest via uv, runs the oracle |
  | `tests/test_outputs.py` | the **oracle** (pytest asserts) |

### Chosen task: `hello-world`
- Instruction: *"Create a file called /app/hello.txt. Write \"Hello, world!\" to it."*
- Base image: `ghcr.io/laude-institute/t-bench/python-3-13:20250620` (WORKDIR `/app`).
- `solution.sh`: `echo "Hello, world!" > hello.txt`  (runs in `/app`).
- Oracle `tests/test_outputs.py`: asserts `/app/hello.txt` exists and content `== "Hello, world!"`.
- **Success = all pytest tests PASS.**

## 3. End-to-end proof (exact commands)

Reproduced the harness semantics with `docker build/run/exec/cp`:

```bash
cd terminal-bench/original-tasks/hello-world
docker build -t tbench-hello-world:proof -f Dockerfile .          # build
docker run -d --rm --name HW -w /app tbench-hello-world:proof sh -c "sleep infinity"
# copy oracle in (harness puts it at /tests, sets TEST_DIR=/tests)
docker exec HW mkdir -p /tests
docker cp run-tests.sh HW:/tests/run-tests.sh
docker cp tests/. HW:/tests/

# PHASE 1 — oracle WITHOUT solution  -> FAIL
docker exec -e TEST_DIR=/tests HW bash /tests/run-tests.sh
#   => "2 failed"   (FileNotFoundError: /app/hello.txt)

# PHASE 2 — run gold solution, then oracle  -> PASS
docker exec HW mkdir -p /oracle
docker cp solution.sh HW:/oracle/solution.sh
docker exec -w /app HW bash /oracle/solution.sh
docker exec -e TEST_DIR=/tests HW bash /tests/run-tests.sh
#   => "2 passed"

docker rm -f HW
```

Result table (via `run_one.py`):

| run | action | oracle result |
|-----|--------|---------------|
| 1 | `--solution` (gold) | **SUCCESS=True**  passed=2 failed=0 |
| 2 | none | **SUCCESS=False**  passed=0 failed=2 |
| 3 | `--cmds 'echo "Hello, world!" > hello.txt'` (agent) | **SUCCESS=True**  passed=2 failed=0 |

→ The oracle differentiates correctly. **Verifiable reward signal confirmed.**

## 4. Agent interaction interface (KEY for RLVP)

How an agent interacts with a tbench task (from `terminal_bench/agents/oracle_agent.py`
and `terminal/tmux_session.py`):

- The container runs `sleep infinity`; the harness opens a **tmux shell** inside it.
- **Action**: agent emits a bash command → `session.send_keys(["<cmd>", "Enter"])`.
- **Observation**: `session.capture_pane()` returns the terminal screen text
  (stdout+stderr as rendered); `get_incremental_output()` returns only new output.
- It is **one persistent interactive shell session per episode** (state — cwd, env,
  files — persists across actions). WORKDIR is `/app`.
- **Episode end / reward**: harness copies `run-tests.sh`+`tests/` to `/tests`,
  runs `bash /tests/run-tests.sh` (TEST_DIR=/tests), parses pytest output.
  reward = 1.0 if all tests PASSED else 0.0.
- The gold solution is applied identically: copy `solution.sh` → `/oracle/`,
  `bash /oracle/solution.sh`.

**RLVP tool-env wrapping (recommended):**
```
reset(task_id):   build image, `docker run -d --rm ... sleep infinity`, return instruction
step(bash_cmd):   docker exec -w /app <ctr> bash -lc "<cmd>"  ->  observation = stdout+stderr
                  (or drive a tmux session for interactive/long-running cmds)
score():          copy tests, run run-tests.sh, parse pytest -> {0,1}
close():          docker rm -f <ctr>  (image optional cache or rmi)
```
For most file-ops/scripting tasks the non-interactive `docker exec` path
(used in `run_one.py --cmds`) is sufficient and deterministic. Use the tmux path
only for tasks needing a live TTY / background processes / REPLs.

## 5. Per-task cost (RL rollout cost) — measured on `hello-world`

| metric | value |
|--------|-------|
| image size | **152.7 MB** (base python-3.13-slim + tmux/asciinema) |
| image build | **~5.5 s** first time (pulls/extracts base), **~0.5–1.8 s** when base cached |
| container start | **0.2–0.9 s** |
| oracle run | **~4.6–5.1 s** (dominated by `uv pip install pytest` each run) |
| **total per episode** | **~6–11 s** for this trivial task, mostly oracle setup |

Notes for cost at scale:
- The ~4–5 s oracle overhead is **per-run pytest install via uv**. For RL we can
  **bake pytest into the image once** (custom layer) to cut oracle time to <1 s.
- `hello-world` is best-case. Many of the 241 tasks have **heavy Dockerfiles**
  (e.g. `build-linux-kernel-qemu`, `caffe-cifar-10`, `build-pov-ray`) → multi-GB
  images and minutes-to-build. Pick a small/fast subset for RL.

## 6. Blockers / risks for scaling to many tasks

1. **Python version**: full tbench harness needs py>=3.12; this box default is 3.10.
   → either use `uv venv --python 3.13` for the official runner, OR keep using the
   direct-Docker path in `run_one.py` (no python-version constraint, already proven).
2. **Disk**: box at 90% (363 GB free) and **shared**. Heavy task images are multi-GB.
   → must `--rm` containers, `docker rmi` images after each task, prune aggressively,
   and curate a small-image task subset. Do not bulk-build all 241.
3. **Oracle install overhead**: `uv pip install pytest` every run (~4 s). Bake into
   image for RL throughput.
4. **Per-task heterogeneity**: some solutions are `solution.yaml` (a list of
   TerminalCommands) not `solution.sh`; some tasks need network at build time
   (ghcr/pip/apt). `run_one.py --solution` currently handles `.sh` only (use
   `--cmds` for yaml tasks, or extend to parse solution.yaml).
5. **Network dependency**: builds pull from ghcr.io / apt / astral.sh. Need outbound
   net during build & first oracle run. Air-gapped RL would require pre-baked images.

## 7. Files delivered (all under /home/ubuntu/rlvp/termbench/)

- `terminal-bench/`        — cloned repo (241 tasks in `original-tasks/`)
- `load_tasks.py`          — list tasks / dump one task's metadata (py3.10-safe, no tbench dep)
- `run_one.py`             — build+start container, run solution OR agent bash cmds,
                             run oracle, return pass/fail + cost metrics; auto-cleans
- `PIPELINE_STATUS.md`     — this file

### Quick repro
```bash
cd /home/ubuntu/rlvp/termbench
python3 load_tasks.py                              # -> 241 tasks
python3 run_one.py hello-world --solution          # -> SUCCESS=True
python3 run_one.py hello-world                      # -> SUCCESS=False
python3 run_one.py hello-world --cmds 'echo "Hello, world!" > hello.txt'  # SUCCESS=True
```
