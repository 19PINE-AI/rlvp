"""Run one RLVP training variant. Usage:
  python3 scripts/train.py <credit> [iters] [model]
credit: outcome | c1 | c2 | c2pos
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvp.grpo import TrainConfig, train

credit = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 40
model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-4B"

cfg = TrainConfig(model_name=model, credit=credit, iters=iters,
                  gen_batch=48, out_dir=f"results/run_{credit}")
print(f"training {credit} for {iters} iters on {model}", flush=True)
train(cfg)
print("DONE", flush=True)
