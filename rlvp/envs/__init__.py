from . import csops, fileops
from .base import ToolEnv, parse_action

class _Envs(dict):
    def __getitem__(self, k):  # chainN domains use the fileops module
        return super().__getitem__("fileops" if k.startswith("chain") else k)


ENVS = _Envs({"fileops": fileops, "csops": csops})


def make_env(domain: str, seed: int, track_rules: bool = True,
             drop_rules: tuple = (), guardrail: bool = False) -> ToolEnv:
    if domain.startswith("chain"):  # 'chain4' -> 4-stage chained fileops
        n = int(domain[len("chain"):])
        env = fileops.FileOpsEnv(fileops.make_chain_task(seed, n),
                                 track_rules=track_rules, drop_rules=drop_rules,
                                 guardrail=guardrail)
        env.max_turns = 8 + 7 * n
        return env
    mod = ENVS[domain]
    task = mod.make_task(seed)
    cls = fileops.FileOpsEnv if domain == "fileops" else csops.CSOpsEnv
    return cls(task, track_rules=track_rules, drop_rules=drop_rules, guardrail=guardrail)
