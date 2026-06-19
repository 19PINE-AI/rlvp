#!/usr/bin/env python3
"""Track A: which dask eras actually `import dask.dataframe` on our py3.10 venv?
Sets up ONE instance per era and tests the import (the thing that was silently
failing all tests). Eras that import OK are measurable; old eras that need a
pre-3.10 pandas get excluded (honest limitation, not a fake 0%)."""
import sys, collections
sys.path.insert(0, ".")
import swegym.swe_env_setup as H
from rlvp.swe_adapter import load_small_patch_instances, SweWorktree

insts = load_small_patch_instances(max_changed=8, max_files=1, max_hunks=2)
# one representative instance per pin-group
by_era = {}
for i in insts:
    g = H.pin_group_for(i["version"])
    by_era.setdefault(g, i)

print("testing one instance per era:", sorted(by_era))
results = {}
for era, inst in sorted(by_era.items()):
    wt = SweWorktree(inst, f"/tmp/era_test/{era}")
    wt.setup()
    if not wt.setup_ok:
        results[era] = f"SETUP-FAIL: {wt.setup_error}"
        wt.close(); continue
    # run the import check in the venv
    import subprocess
    r = subprocess.run([wt.py, "-c", "import dask.dataframe; print('OK')"],
                       capture_output=True, text=True, cwd=wt.repo_dir, timeout=120)
    out = (r.stdout + r.stderr).strip().splitlines()
    ok = "OK" in r.stdout
    results[era] = ("IMPORT-OK" if ok else "IMPORT-FAIL: " +
                    (out[-1][:80] if out else "?"))
    print(f"  {era} (dask {inst['version']}): {results[era]}", flush=True)
    wt.close()

print("\n=== ERA IMPORT SUMMARY ===")
ok_eras = [e for e, v in results.items() if v == "IMPORT-OK"]
n_ok = sum(1 for i in insts if H.pin_group_for(i["version"]) in ok_eras)
print("working eras:", ok_eras)
print(f"measurable small-patch instances: {n_ok}/{len(insts)}")
print("SWE ERA TEST DONE")
