"""CLI: collect behavior-cloning traces for one or more bundled benchmarks.

Example:
    python -m rl_monteq.collect_traces \
        --benchmark Heisen_2_2 UCC_2_4 \
        --sims 200 \
        --top-frac 0.2 \
        --out-dir rl_monteq/data
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
MONTEQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "MonteQ"))
for path in (PROJECT_ROOT, MONTEQ_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from src import heuristic_module as hm  # noqa: E402 # type: ignore
from .trace_collection import collect_for_hamiltonian


def load_paulis(name: str) -> List[str]:
    path = os.path.join(MONTEQ_ROOT, "src", name + ".pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "paulis" in obj:
        return obj["paulis"]
    raise ValueError(f"Unrecognized benchmark format: {path} -> {type(obj)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", nargs="+", required=True)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--out-dir", default="rl_monteq/data")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.benchmark:
        paulis = load_paulis(name)
        print(f"[collect] {name}: K={len(paulis)} n={len(paulis[0])} sims={args.sims}")
        out_path = os.path.join(args.out_dir, f"{name}.pkl")
        samples = collect_for_hamiltonian(
            paulis,
            heuristic_function=hm.pair_solve,
            sims=args.sims,
            top_frac=args.top_frac,
            out_path=out_path,
        )
        print(f"  -> {len(samples):,} samples -> {out_path}")


if __name__ == "__main__":
    main()
