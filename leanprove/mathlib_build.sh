#!/usr/bin/env bash
# Persistent Mathlib build for the miniF2F benchmark REPL.
# cache-get path (frugal): download prebuilt oleans instead of compiling.
set -x
export ELAN_HOME=/home/ubuntu/rlvp/leanprove/.elan
export PATH=$ELAN_HOME/bin:$PATH
cd /home/ubuntu/rlvp/leanprove/mathlib_repl || exit 1

echo "=== [$(date)] lake update (resolve mathlib dep + manifest) ==="
lake update || { echo "LAKE_UPDATE_FAILED"; exit 1; }

echo "=== [$(date)] lake exe cache get (download prebuilt Mathlib oleans) ==="
lake exe cache get || echo "CACHE_GET_RETURNED_NONZERO (will still try build)"

echo "=== [$(date)] lake build (compile remaining / verify) ==="
lake build
RC=$?
echo "=== [$(date)] lake build exit code: $RC ==="
echo "MATHLIB_BUILD_DONE rc=$RC"
