#!/usr/bin/env python3
"""
load_tasks.py -- list / inspect TerminalBench tasks.

TerminalBench (laude-institute/terminal-bench) ships tasks under
<repo>/original-tasks/<task-id>/. Each task dir contains:
    task.yaml          - instruction + metadata (difficulty, timeouts, parser)
    Dockerfile         - builds the task's container image (WORKDIR /app)
    docker-compose.yaml- service def (runs `sleep infinity`)
    solution.sh        - gold/reference solution (bash); some tasks use solution.yaml
    run-tests.sh       - installs pytest via uv, runs the oracle
    tests/             - the oracle (pytest test_outputs.py etc.)

This module does NOT require the terminal-bench package (which needs py>=3.12);
it just reads the task directories, so it runs on this box's py3.10.

Usage:
    python3 load_tasks.py                 # print count + table
    python3 load_tasks.py --id hello-world  # dump one task's details
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).parent / "terminal-bench"
TASKS_DIR = REPO / "original-tasks"


def list_tasks(tasks_dir: Path = TASKS_DIR):
    """Return sorted list of task-id strings (dirs containing a task.yaml)."""
    out = []
    for d in sorted(tasks_dir.iterdir()):
        if d.is_dir() and (d / "task.yaml").exists():
            out.append(d.name)
    return out


def _parse_yaml_lite(text: str) -> dict:
    """Minimal task.yaml reader for the fields we care about (no pyyaml dep
    required). Handles the `instruction: |-` block scalar and simple key: value."""
    meta = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("instruction:") and "|" in line:
            block = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or lines[i].strip() == ""):
                block.append(lines[i][2:] if lines[i].startswith("  ") else lines[i])
                i += 1
            meta["instruction"] = "\n".join(block).strip()
            continue
        if ":" in line and not line.startswith(" ") and not line.startswith("#"):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        i += 1
    return meta


def load_task(task_id: str, tasks_dir: Path = TASKS_DIR) -> dict:
    """Return a dict describing one task: paths + parsed metadata."""
    d = tasks_dir / task_id
    if not (d / "task.yaml").exists():
        raise FileNotFoundError(f"No task '{task_id}' in {tasks_dir}")
    meta = _parse_yaml_lite((d / "task.yaml").read_text())
    sol = d / "solution.sh"
    if not sol.exists():
        sol = d / "solution.yaml"
    return {
        "task_id": task_id,
        "dir": str(d),
        "instruction": meta.get("instruction", ""),
        "difficulty": meta.get("difficulty"),
        "category": meta.get("category"),
        "parser_name": meta.get("parser_name"),
        "max_test_timeout_sec": meta.get("max_test_timeout_sec"),
        "dockerfile": str(d / "Dockerfile"),
        "solution": str(sol),
        "solution_type": sol.suffix.lstrip("."),
        "run_tests": str(d / "run-tests.sh"),
        "tests_dir": str(d / "tests"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="dump one task's details as JSON")
    args = ap.parse_args()

    if args.id:
        print(json.dumps(load_task(args.id), indent=2))
    else:
        tasks = list_tasks()
        print(f"TerminalBench tasks found: {len(tasks)}")
        print(f"Tasks dir: {TASKS_DIR}")
        print("\nFirst 20:")
        for t in tasks[:20]:
            print(" ", t)
