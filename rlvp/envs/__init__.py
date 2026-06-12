from . import csops, fileops
from .base import ToolEnv, parse_action

ENVS = {"fileops": fileops, "csops": csops}


def make_env(domain: str, seed: int, track_rules: bool = True,
             drop_rules: tuple = (), guardrail: bool = False) -> ToolEnv:
    mod = ENVS[domain]
    task = mod.make_task(seed)
    cls = fileops.FileOpsEnv if domain == "fileops" else csops.CSOpsEnv
    return cls(task, track_rules=track_rules, drop_rules=drop_rules, guardrail=guardrail)
