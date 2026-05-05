"""Head-to-head benchmark: greedy rollout vs learned rollout.

Loads one of MonteQ's bundled .pkl benchmark Hamiltonians, runs MCTS with
both rollouts at matched iteration budgets, and reports CX count and
wall-clock time.

Example:
    python -m rl_monteq.run_baseline \
        --benchmark Heisen_2_2 \
        --checkpoint rl_monteq/checkpoints/heisen_2_2.pt \
        --sims 50

Note
----
This is a smoke harness. Real comparison numbers should sweep `sims`,
average over multiple seeds, and use held-out benchmarks the policy
network was *not* trained on. See README.md for the full eval protocol.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
MONTEQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "MonteQ"))
if MONTEQ_ROOT not in sys.path:
    sys.path.insert(0, MONTEQ_ROOT)

from src import mcts_module as mq  # noqa: E402
from src import heuristic_module as hm  # noqa: E402

from rl_monteq.learned_rollout import (
    patch_mcts_with_learned_rollout,
    restore_default_rollout,
)


def load_paulis(name: str) -> List[str]:
    """Look for a benchmark .pkl in MonteQ/src/."""
    for ext in (".pkl",):
        path = os.path.join(MONTEQ_ROOT, "src", name + ext)
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "paulis" in obj:
                return obj["paulis"]
            raise ValueError(f"Don't know how to interpret {path}: type={type(obj)}")
    raise FileNotFoundError(f"No benchmark named {name} in {MONTEQ_ROOT}/src/")


def run_one(paulis: List[str], sims: int, mcts_param: float = 2.0 ** 0.5):
    """Run full_circuit once, return (cx_count, depth, size, wall_time)."""
    t0 = time.time()
    circ = mq.full_circuit(
        paulis,
        hm.pair_solve,
        mcts_param,
        sims=sims,
        order_preserving=True,
    )
    t = time.time() - t0
    # full_circuit returns the best QuantumCircuit directly (via best_solution)
    ops = circ.count_ops()
    cx = ops.get("cx", 0)
    return cx, circ.depth(), circ.size(), t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="Heisen_2_2",
                   help="Benchmark name without .pkl, e.g. Heisen_2_2, UCC_2_4, LiH")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample", action="store_true",
                   help="Stochastic rollout sampling (vs argmax).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    paulis = load_paulis(args.benchmark)
    print(f"[bench] {args.benchmark}: {len(paulis)} Pauli strings, "
          f"{len(paulis[0])} qubits, sims={args.sims}")

    # Greedy baseline.
    restore_default_rollout()
    cx_g, d_g, s_g, t_g = run_one(paulis, sims=args.sims)
    print(f"[greedy ] cx={cx_g}  depth={d_g}  size={s_g}  time={t_g:.2f}s")

    # Learned rollout.
    patch_mcts_with_learned_rollout(
        args.checkpoint, sample=args.sample,
        temperature=args.temperature, device=args.device,
    )
    cx_l, d_l, s_l, t_l = run_one(paulis, sims=args.sims)
    print(f"[learned] cx={cx_l}  depth={d_l}  size={s_l}  time={t_l:.2f}s")

    delta = (cx_g - cx_l) / max(1, cx_g) * 100
    print(f"[delta  ] cx improvement: {delta:+.1f}%   "
          f"time overhead: {(t_l - t_g):+.2f}s")

    restore_default_rollout()


if __name__ == "__main__":
    main()
