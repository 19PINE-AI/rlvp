"""Run the harness over ALL dask instances, grouped by pin-group so each shared
venv is built once. Writes:
  - cache/all_results.json     (full per-instance result dicts)
  - dask_clean_instances.json  (list of clean instance_ids + metadata)

Clean := setup_ok AND f2p_fail_on_base AND f2p_pass_on_gold.
"""
import json
import os
import time

from swe_env_setup import (setup_and_verify, pin_group_for, venv_name_for,
                           cleanup_worktree)

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/tmp/swegym_all"


def main():
    insts = json.load(open(os.path.join(HERE, "cache/dask_instances.json")))
    # order by pin-group so each venv builds once, then is reused
    insts.sort(key=lambda r: (venv_name_for(pin_group_for(r["version"])),
                              r["version"], r["instance_id"]))
    results = []
    t_start = time.time()
    for i, inst in enumerate(insts, 1):
        wd = os.path.join(WORK, inst["instance_id"])
        t0 = time.time()
        try:
            res = setup_and_verify(inst, wd, run_gold=True, run_p2p=True,
                                   p2p_limit=30)
        except Exception as e:  # noqa: BLE001
            res = {"instance_id": inst["instance_id"],
                   "version": inst["version"], "error": f"OUTER:{e}",
                   "setup_ok": False, "f2p_fail_on_base": None,
                   "f2p_pass_on_gold": None, "p2p_pass_on_gold": None,
                   "timing": {}}
        res["seconds"] = round(time.time() - t0, 1)
        res.pop("_log_tail", None)          # keep results file small
        cleanup_worktree(wd)
        clean = bool(res.get("setup_ok") and res.get("f2p_fail_on_base")
                     and res.get("f2p_pass_on_gold"))
        res["clean"] = clean
        results.append(res)
        print(f"[{i}/{len(insts)}] {inst['instance_id']:24s} "
              f"v{res['version']:9s} clean={clean} {res['seconds']}s "
              f"err={res.get('error')}", flush=True)

    json.dump(results, open(os.path.join(HERE, "cache/all_results.json"), "w"),
              indent=2, default=str)
    clean_ids = [r["instance_id"] for r in results if r["clean"]]
    out = {
        "repo": "dask/dask",
        "dataset": "SWE-Gym/SWE-Gym",
        "n_total_dask": len(results),
        "n_clean": len(clean_ids),
        "clean_instance_ids": clean_ids,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(out, open(os.path.join(HERE, "dask_clean_instances.json"), "w"),
              indent=2)
    print(f"\nTOTAL {len(results)} | CLEAN {len(clean_ids)} | "
          f"{round((time.time()-t_start)/60,1)} min")


if __name__ == "__main__":
    main()
