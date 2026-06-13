"""Unit tests for advantage construction (no GPU needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvp.grpo import TrainConfig, build_advantages
from rlvp.rollout import Episode


class FakeEnv:
    def __init__(self, success, violations, discharges=()):
        self.success = success
        self.violations = violations  # list of (turn, rule)
        self.discharges = list(discharges)
        self.calls = []

    def outcome_reward(self):
        return 1.0 if self.success else 0.0


def ep(success, viol_turns):
    e = Episode(env=FakeEnv(success, [(t, "r") for t in viol_turns]))
    e.ids = list(range(40))
    e.action_spans = [(0, 10, 0), (20, 30, 1)]
    e.turn_violations = {t: ["r"] for t in viol_turns}
    return e


def test_outcome_centering():
    cfg = TrainConfig(credit="outcome")
    grp = [ep(True, []), ep(False, []), ep(False, [0])]
    out = build_advantages([grp], cfg)
    advs = [a for _, a, _ in out]
    assert abs(sum(advs)) < 1e-9
    assert advs[0] > 0 > advs[1]
    assert all(pt == {} for _, _, pt in out)  # outcome ignores violations


def test_c1_folds_penalties_into_scalar():
    cfg = TrainConfig(credit="c1", lam=0.3)
    grp = [ep(True, [0, 1]), ep(True, [])]
    out = build_advantages([grp], cfg)
    a_viol, a_clean = out[0][1], out[1][1]
    assert a_clean > a_viol  # same outcome, penalties separate them
    assert out[0][2] == {}   # but NO token-level attachment
    # clip: 10 violations doesn't blow up
    grp = [ep(True, [0] * 1), ep(True, [])]


def test_c2_attaches_to_violating_turn_only():
    cfg = TrainConfig(credit="c2", lam=0.3)
    grp = [ep(True, [1]), ep(False, [])]
    out = build_advantages([grp], cfg)
    e0_turns = out[0][2]
    assert e0_turns == {1: -0.3}
    assert out[1][2] == {}
    # outcome channel unaffected by the penalty
    assert abs(out[0][1] - 0.5) < 1e-9 and abs(out[1][1] + 0.5) < 1e-9


def test_discharge_credits():
    cfg = TrainConfig(credit="c2", lam=0.3, beta=0.3)
    e = ep(True, [0])
    e.env.discharges = [(1, "r")]
    e.turn_discharges = {1: ["r"]}
    grp = [e, ep(True, [])]
    out = build_advantages([grp], cfg)
    assert out[0][2] == {0: -0.3, 1: 0.3}  # penalty turn 0, credit turn 1
    # c1 folds both into the scalar
    cfg1 = TrainConfig(credit="c1", lam=0.3, beta=0.3)
    out1 = build_advantages([grp], cfg1)
    # ep0: 1 - 0.3 + 0.3 = 1.0; ep1: 1 - 0 + 0 = 1.0 -> centered to 0, 0
    assert abs(out1[0][1]) < 1e-9 and abs(out1[1][1]) < 1e-9


def test_c2pos_rewards_clean_turns():
    cfg = TrainConfig(credit="c2pos", lam=0.3)
    grp = [ep(True, [0]), ep(True, [])]
    out = build_advantages([grp], cfg)
    assert out[0][2] == {1: 0.3}            # turn 0 violated, only turn 1 paid
    assert out[1][2] == {0: 0.3, 1: 0.3}    # all clean turns paid


def test_script_process_only():
    cfg = TrainConfig(credit="c3", lam=0.25, beta=0.25, script_scalar=False)
    live_win = ep(True, [])
    live_lose = ep(False, [])
    demo = ep(False, [])           # imperfect demo: fails the task
    demo.scripted = True
    demo.env.discharges = [(0, "r")]
    demo.turn_discharges = {0: ["r"]}
    out = build_advantages([[live_win, live_lose, demo]], cfg)
    # baseline over live only: +0.5 / -0.5; demo scalar EXACTLY zero
    assert abs(out[0][1] - 0.5) < 1e-9 and abs(out[1][1] + 0.5) < 1e-9
    assert out[2][1] == 0.0
    assert out[2][2] == {0: 0.25}  # but its discharge still teaches
    # default behavior unchanged: demo participates in the baseline
    cfg_def = TrainConfig(credit="c3", lam=0.25, beta=0.25)
    out_def = build_advantages([[live_win, live_lose, demo]], cfg_def)
    assert out_def[2][1] != 0.0


def test_gigpo_step_groups():
    cfg = TrainConfig(credit="gigpo", lam=0.5, beta=0.5)
    e1, e2 = ep(True, []), ep(False, [1])
    out = build_advantages([[e1, e2]], cfg)
    assert out[0][1] == 0.0 and out[1][1] == 0.0  # no trajectory channel
    # turn 0: rtg e1=1, e2=0-0.5=-0.5 (viol at turn1 counts for t<=1) -> mu=0.25
    assert abs(out[0][2][0] - 0.75) < 1e-9
    assert abs(out[1][2][0] + 0.75) < 1e-9
    # turn 1: e1 rtg=1, e2 rtg=-0.5 -> same split
    assert abs(out[0][2][1] - 0.75) < 1e-9


def test_steptool_succ_calling():
    cfg = TrainConfig(credit="steptool")
    e1, e2 = ep(True, []), ep(False, [])
    e2.turn_errors = {1}
    out = build_advantages([[e1, e2]], cfg)
    assert out[0][2] == {0: 0.2, 1: 0.2}
    assert out[1][2] == {0: 0.2}  # errored turn gets no SuccCalling reward
    assert abs(out[0][1] - 0.5) < 1e-9


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("PASS", k)
    print("credit tests passed")
