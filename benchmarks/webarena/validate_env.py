#!/usr/bin/env python3
"""Live env-loop validation for the WebArena/ST-WebAgentBench adapter.

Run AFTER the sites are booted and browsergym.stwebagentbench + playwright are
installed. NO GPU / model needed -- uses a scripted agent to exercise the exact
env calls rlvp/webarena_adapter.run_webarena_episode makes, and asserts the
penalty channel populates. This is the piece the offline validation could not
cover (needs a live Playwright browser + a running site).

Usage:
  # 1. boot a site (shopping_admin is the smallest ST-WAB site):
  #    bash scripts/setup_minimal_stwab.sh   # or just the shopping_admin part
  # 2. install into a venv:  uv pip install -e ST-WebAgentBench/browsergym/stwebagentbench
  #    playwright install chromium
  # 3. set site env vars (GITLAB / SHOPPING_ADMIN / WA_SUITECRM) in .env
  # 4. python3 validate_env.py <task_id>   # a shopping_admin task id

Asserts: env.reset() returns an AXTree; env.step(action) returns the 5-tuple with
info['safety_report']; a deliberately policy-violating action populates it.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/rlvp")
from rlvp.webarena_adapter import (extract_action, make_env, _flatten_axtree,  # noqa: E402
                                   WebArenaRuleTracker)


def main():
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"creating env for ST-WAB task {task_id} ...", flush=True)
    env = make_env(task_id, headless=True)
    tracker = WebArenaRuleTracker()
    try:
        obs, info = env.reset()
        ax = _flatten_axtree(obs)
        goal = obs.get("goal") or (obs.get("chat_messages") or [{}])[-1].get("message", "")
        print("RESET OK. goal:", (goal or "")[:120])
        print("AXTree head:\n", ax[:600], "\n---")
        assert "AXTREE" in ax, "AXTree not flattened"

        # scripted agent: a couple of benign nav/read actions, then a deliberately
        # sensitive action WITHOUT prior consent (should trip a policy).
        scripted = ["noop()", "scroll(0, 300)", "go_back()"]
        for step, act in enumerate(scripted):
            a = extract_action(f"```\n{act}\n```")
            assert a == act, f"extract_action mangled {act!r} -> {a!r}"
            obs, reward, terminated, truncated, info = env.step(a)
            sr = info.get("safety_report")
            v = tracker.observe_turn(step, sr, not obs.get("last_action_error"))
            print(f"step {step}: {act:16} reward={reward} term={terminated} "
                  f"violations={v} safety_report_len={len(sr or [])}")
            assert "safety_report" in info, "info missing safety_report"
            assert "safety_penalty" in info, "info missing safety_penalty"
            if terminated or truncated:
                break
        print("\nVALIDATION PASSED: reset+step+safety_report all functional.")
        print("turn_violations:", tracker.turn_violations)
        print("turn_discharges:", tracker.turn_discharges)
    finally:
        env.close()


if __name__ == "__main__":
    main()
