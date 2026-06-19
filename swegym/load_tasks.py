"""Load SWE-Gym / SWE-bench tasks.

Returns a list of instance dicts with the fields needed to set up and verify a
single task: instance_id, repo, base_commit, version, problem_statement, patch
(gold), test_patch, FAIL_TO_PASS, PASS_TO_PASS.

Usage:
    from load_tasks import load_tasks, get_instance
    tasks = load_tasks()                      # default SWE-Gym-Lite (230)
    inst  = get_instance("dask__dask-8597")
"""
from datasets import load_dataset

# Preference order: smaller first.
DATASET_CANDIDATES = [
    ("SWE-Gym/SWE-Gym-Lite", "train"),
    ("SWE-Gym/SWE-Gym", "train"),
    ("princeton-nlp/SWE-bench_Verified", "test"),
]


def load_tasks(dataset_name=None, split=None):
    candidates = (
        [(dataset_name, split or "train")] if dataset_name else DATASET_CANDIDATES
    )
    last_err = None
    for name, sp in candidates:
        try:
            ds = load_dataset(name, split=sp)
            return [dict(r) for r in ds]
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load any dataset; last error: {last_err!r}")


def get_instance(instance_id, dataset_name=None, split=None):
    for r in load_tasks(dataset_name, split):
        if r["instance_id"] == instance_id:
            return r
    raise KeyError(instance_id)


if __name__ == "__main__":
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} instances")
    r = tasks[0]
    print("Fields:", list(r.keys()))
    print("Example instance_id:", r["instance_id"], "| repo:", r["repo"])
