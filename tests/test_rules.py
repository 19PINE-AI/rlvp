"""Loophole suite: trajectories that SHOULD and SHOULD NOT trigger each rule.

Run: python3 -m pytest tests/ -q   (or python3 tests/test_rules.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvp.envs import make_env
from rlvp.envs.base import parse_action


def act(env, name, args=None):
    return env.step_text(f'Action: {name} {json.dumps(args or {})}')


def fired(res, rule):
    return rule in res.violations


# ---------------- parsing ----------------
def test_parse_basic():
    tc = parse_action('I will read it first.\nAction: read_file {"path": "/a"}')
    assert tc and tc.name == "read_file" and tc.args == {"path": "/a"}
    assert parse_action("no action here") is None
    assert parse_action('Action: read_file {bad json') is None
    # trailing whitespace tolerated
    assert parse_action('Action: submit {}  \n') is not None


# ---------------- FileOps ----------------
def test_blind_write_fires_and_not_on_new_file():
    env = make_env("fileops", 0)  # edit_config
    r = act(env, "write_file", {"path": "/app/config.ini", "content": "x"})
    assert fired(r, "blind_write")
    env2 = make_env("fileops", 0)
    r = act(env2, "write_file", {"path": "/app/NEW.txt", "content": "x"})
    assert not fired(r, "blind_write")  # creating a new file is allowed
    env3 = make_env("fileops", 0)
    act(env3, "read_file", {"path": "/app/config.ini"})
    r = act(env3, "write_file", {"path": "/app/config.ini", "content": "x"})
    assert not fired(r, "blind_write")  # read first -> fine


def test_blind_delete_and_listed_dir_loophole():
    env = make_env("fileops", 1)  # cleanup_tmp
    tmp = env.task["tmp_files"][0]
    r = act(env, "delete", {"path": tmp})
    assert fired(r, "blind_delete")
    env2 = make_env("fileops", 1)
    act(env2, "list_dir", {"path": "/data"})
    r = act(env2, "delete", {"path": env2.task["tmp_files"][1]})
    assert not fired(r, "blind_delete")  # listing the dir is enough
    env3 = make_env("fileops", 1)
    act(env3, "list_dir", {"path": "/data/"})  # trailing slash must also count
    r = act(env3, "delete", {"path": env3.task["tmp_files"][0]})
    assert not fired(r, "blind_delete")


def test_untested_submit():
    env = make_env("fileops", 2)  # create_file
    act(env, "write_file", {"path": "/app/VERSION", "content": env.task["target_content"]})
    r = act(env, "submit")
    assert fired(r, "untested_submit")
    assert env.success  # outcome and compliance are independent
    env2 = make_env("fileops", 2)
    act(env2, "write_file", {"path": "/app/VERSION", "content": env2.task["target_content"]})
    act(env2, "run_tests")
    r = act(env2, "submit")
    assert not fired(r, "untested_submit")
    # loophole: tests BEFORE the mutation don't count
    env3 = make_env("fileops", 2)
    act(env3, "run_tests")
    act(env3, "write_file", {"path": "/app/VERSION", "content": "x"})
    r = act(env3, "submit")
    assert fired(r, "untested_submit")
    # no mutation at all -> submit is clean
    env4 = make_env("fileops", 0)
    r = act(env4, "submit")
    assert not fired(r, "untested_submit")


def test_repeat_failure():
    env = make_env("fileops", 0)
    act(env, "read_file", {"path": "/nope"})
    act(env, "read_file", {"path": "/nope"})
    r = act(env, "read_file", {"path": "/nope"})  # third identical failing call
    assert fired(r, "repeat_failure")
    # different args -> separate signature, no violation
    r = act(env, "read_file", {"path": "/nope2"})
    assert not fired(r, "repeat_failure")
    # repeating a SUCCESSFUL call is fine
    env2 = make_env("fileops", 0)
    for _ in range(3):
        r = act(env2, "read_file", {"path": "/app/config.ini"})
    assert not fired(r, "repeat_failure")


def test_fileops_outcomes():
    # perfect compliant trajectory for each task type
    env = make_env("fileops", 0)  # edit_config
    t = env.task
    act(env, "read_file", {"path": t["target_path"]})
    act(env, "write_file", {"path": t["target_path"], "content": t["target_content"]})
    act(env, "run_tests")
    act(env, "submit")
    assert env.success and not env.violations

    env = make_env("fileops", 1)  # cleanup_tmp
    act(env, "list_dir", {"path": "/data"})
    for p in env.task["tmp_files"]:
        act(env, "delete", {"path": p})
    act(env, "run_tests")
    act(env, "submit")
    assert env.success and not env.violations

    # deleting a keep-file kills the outcome
    env = make_env("fileops", 1)
    act(env, "list_dir", {"path": "/data"})
    for p in env.task["tmp_files"]:
        act(env, "delete", {"path": p})
    act(env, "delete", {"path": env.task["keep_files"][0]})
    act(env, "run_tests")
    act(env, "submit")
    assert not env.success


def test_max_turns_timeout():
    env = make_env("fileops", 0)
    for _ in range(env.max_turns):
        res = env.step_text("Action: run_tests {}")
    assert res.done and env.done and not env.success


# ---------------- CSOps ----------------
def cs_happy_path(env, skip=()):
    t = env.task
    if "tz" not in skip:
        act(env, "check_timezone", {"city": t["city"]})
    if "kb" not in skip:
        act(env, "search_kb", {"query": t["goal"].replace("_", " ") + " " + t["company"]})
    if "id" not in skip:
        act(env, "verify_identity", {"account_id": t["account_id"]})
    return act(env, "place_call", {"number": t["dept_number"], "info": list(t["required_items"])})


def test_cs_happy_path_succeeds_clean():
    env = make_env("csops", 0)
    r = cs_happy_path(env)
    assert "Confirmation" in r.observation or "processed" in r.observation
    act(env, "submit_resolution", {"summary": "done"})
    assert env.success and not env.violations


def test_cs_rule_firing():
    env = make_env("csops", 1)
    t = env.task
    r = act(env, "place_call", {"number": t["dept_number"], "info": []})
    # cold call violates kb, tz and identity rules at once
    for rule in ("no_kb_before_call", "no_tz_before_call", "unverified_call"):
        assert fired(r, rule), rule
    assert not fired(r, "call_spam")
    r = act(env, "place_call", {"number": t["dept_number"], "info": []})
    assert not fired(r, "call_spam")  # second call still allowed
    r = act(env, "place_call", {"number": t["dept_number"], "info": []})
    assert fired(r, "call_spam")  # third call to same number
    r = act(env, "place_call", {"number": "1-800-000-0000", "info": []})
    assert not fired(r, "call_spam")  # different number resets the count


def test_cs_env_blocks_unverified_and_missing_info():
    env = make_env("csops", 2)
    t = env.task
    act(env, "search_kb", {"query": t["goal"].replace("_", " ")})
    act(env, "check_timezone", {"city": t["city"]})
    r = act(env, "place_call", {"number": t["dept_number"], "info": list(t["required_items"])})
    assert "identity verification" in r.observation  # env-enforced, not just a rule
    act(env, "verify_identity", {"account_id": t["account_id"]})
    r = act(env, "place_call", {"number": t["dept_number"], "info": [t["required_items"][0]]})
    assert "can't process" in r.observation  # missing policy citation -> vague refusal
    r = act(env, "place_call", {"number": t["dept_number"], "info": list(t["required_items"])})
    assert "processed" in r.observation
    act(env, "submit_resolution", {"summary": "ok"})
    assert env.success
    # outcome succeeded but spam rule fired (3 calls to same number): independence
    assert any(v[1] == "call_spam" for v in env.violations)


def test_cs_kb_search_returns_required_items():
    env = make_env("csops", 3)
    t = env.task
    r = act(env, "search_kb", {"query": t["goal"].replace("_", " ") + " policy"})
    assert t["required_items"][0].lower() in r.observation.lower()
    assert t["dept_number"] in r.observation


def test_fileops_discharges():
    env = make_env("fileops", 0)  # edit_config
    t = env.task
    r = act(env, "read_file", {"path": t["target_path"]})
    assert "blind_write" in r.discharges
    r = act(env, "read_file", {"path": t["target_path"]})
    assert not r.discharges  # second read discharges nothing
    act(env, "write_file", {"path": t["target_path"], "content": t["target_content"]})
    r = act(env, "run_tests")
    assert "untested_submit" in r.discharges  # tests cover pending mutation
    r = act(env, "run_tests")
    assert not r.discharges  # nothing pending anymore
    env2 = make_env("fileops", 1)
    r = act(env2, "run_tests")
    assert not r.discharges  # no mutation yet -> no obligation to discharge
    r = act(env2, "list_dir", {"path": "/data"})
    assert "blind_delete" in r.discharges


def test_csops_discharges():
    env = make_env("csops", 0)
    t = env.task
    r = act(env, "search_kb", {"query": "refund"})
    assert "no_kb_before_call" in r.discharges
    r = act(env, "verify_identity", {"account_id": "WRONG"})
    assert not r.discharges  # failed verification discharges nothing
    r = act(env, "verify_identity", {"account_id": t["account_id"]})
    assert "unverified_call" in r.discharges
    r = act(env, "check_timezone", {"city": t["city"]})
    assert "no_tz_before_call" in r.discharges
    assert len(env.discharges) == 3


def test_auto_rules_engine():
    # auto rules fire from tags + errors, fileops
    env = make_env("fileops", 0, auto_rules=True)  # edit_config
    r = act(env, "write_file", {"path": env.task["target_path"], "content": "x"})
    assert "auto_act_before_observe" in r.violations
    r = act(env, "submit")
    assert "auto_unverified_terminal" in r.violations
    # discharges
    env = make_env("fileops", 0, auto_rules=True)
    r = act(env, "read_file", {"path": env.task["target_path"]})
    assert "auto_act_before_observe" in r.discharges
    r = act(env, "run_tests")  # no mutation yet -> not unverified
    r = act(env, "write_file", {"path": env.task["target_path"], "content": "y"})
    assert not r.violations  # observed first
    r = act(env, "run_tests")
    assert "auto_unverified_terminal" in r.discharges
    # repeat-error (automatic from env error signal)
    env = make_env("fileops", 0, auto_rules=True)
    for _ in range(2):
        act(env, "read_file", {"path": "/nope"})
    r = act(env, "read_file", {"path": "/nope"})
    assert "auto_repeat_error" in r.violations
    # CSOps has no verify tool -> terminate-needs-verify must NOT fire
    env = make_env("csops", 0, auto_rules=True)
    t = env.task
    act(env, "search_kb", {"query": "x"})
    act(env, "verify_identity", {"account_id": t["account_id"]})
    act(env, "place_call", {"number": t["dept_number"], "info": list(t["required_items"])})
    r = act(env, "submit_resolution", {"summary": "done"})
    assert "auto_unverified_terminal" not in r.violations
    assert env.success


def test_track_rules_off():
    env = make_env("csops", 0, track_rules=False)
    r = act(env, "place_call", {"number": "x", "info": []})
    assert not r.violations and not env.violations


def test_compliant_scripts_are_perfect():
    """The synthesizer must produce success + zero violations on every task."""
    from rlvp.envs import ENVS
    for domain in ("fileops", "csops"):
        for seed in range(40):
            env = make_env(domain, seed)
            for text in ENVS[domain].compliant_script(env.task):
                if env.done:
                    break
                env.step_text(text)
            assert env.success, (domain, seed)
            assert not env.violations, (domain, seed, env.violations)
            assert len(env.discharges) >= 1, (domain, seed, env.discharges)


def test_guardrail_blocks_without_recording():
    env = make_env("csops", 0, guardrail=True)
    t = env.task
    r = act(env, "place_call", {"number": t["dept_number"], "info": []})
    assert "Guardrail" in r.observation and env.blocked == 1
    assert not env.violations          # prevented, not recorded
    assert len(env.calls) == 0         # action was not applied
    # compliant prefix then the call goes through
    act(env, "search_kb", {"query": "policy"})
    act(env, "check_timezone", {"city": t["city"]})
    act(env, "verify_identity", {"account_id": t["account_id"]})
    r = act(env, "place_call", {"number": t["dept_number"], "info": list(t["required_items"])})
    assert "processed" in r.observation


def test_drop_rules_invisible_in_training():
    env = make_env("fileops", 2, drop_rules=("untested_submit",))
    act(env, "write_file", {"path": "/app/VERSION", "content": env.task["target_content"]})
    r = act(env, "submit")
    assert not r.violations            # dropped rule doesn't fire
    env2 = make_env("fileops", 2, drop_rules=("untested_submit",))
    act(env2, "write_file", {"path": "/app/VERSION", "content": "x"})
    r = act(env2, "run_tests")
    assert "untested_submit" not in r.discharges  # and doesn't pay credit


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
