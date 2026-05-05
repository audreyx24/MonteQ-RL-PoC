"""Sanity-check a collected trace pickle.

Prints:
    - sample count and step-length statistics
    - shape and value-range checks per sample
    - the distribution of `return_cx` (should peak near a small magnitude
      for steps near terminal and be more negative near the root)
    - the unique terminal-CX values seen (should match the spread of
      CNOT counts in the trajectories MCTS found)
    - a few example samples decoded back to readable Pauli strings

Optionally re-runs MonteQ's stock `full_circuit` on the same benchmark
and compares its best CX to the best terminal in the traces. If the two
disagree, the trace collector isn't faithfully tracking MCTS.

Example:
    python -m rl_monteq.inspect_traces \
        --data rl_monteq/data/Heisen_2_2.pkl \
        --benchmark Heisen_2_2 \
        --cross-check --sims 50
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import Counter
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MONTEQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "MonteQ"))
if MONTEQ_ROOT not in sys.path:
    sys.path.insert(0, MONTEQ_ROOT)


def decode_row(xs_row, zs_row, ss_val) -> str:
    out = "-" if ss_val else ""
    for x, z in zip(xs_row, zs_row):
        if x and z:
            out += "Y"
        elif x:
            out += "X"
        elif z:
            out += "Z"
        else:
            out += "I"
    return out


def inspect(samples: List[dict], n_show: int = 3):
    print(f"[inspect] total samples: {len(samples):,}")

    # Shape stats.
    Ks = [s["xs"].shape[0] for s in samples]
    ns = [s["xs"].shape[1] for s in samples]
    print(f"[inspect] K (rows remaining) min/median/max = "
          f"{min(Ks)}/{int(np.median(Ks))}/{max(Ks)}")
    print(f"[inspect] n (qubits) min/max = {min(ns)}/{max(ns)}  "
          f"(should be constant per benchmark)")

    # Value-target distribution.
    rets = np.array([s["return_cx"] for s in samples])
    print(f"[inspect] return_cx (=-remaining_cx): "
          f"min={rets.min():.0f} mean={rets.mean():.1f} max={rets.max():.0f}")
    print("           closer-to-terminal states should have return_cx ~ 0;")
    print("           root states should have the most-negative values.")

    # Terminal-CX spread per trajectory.
    # We can recover this: every step in the same trajectory shares the
    # same `terminal_cx = cx_added_so_far - return_cx`. We don't carry
    # cx_added_so_far in the saved samples, but for steps where
    # `return_cx == 0` we are at the terminal step itself, and for the
    # very first step of any trajectory cx_added_so_far == 0 so
    # terminal_cx == -return_cx.
    # The cleanest signal: histogram of (the most-negative return_cx
    # seen) is the histogram of terminal CX counts across trajectories.
    # That's what the next block does.
    # We bucket by approximate trajectory: every contiguous run of
    # samples with monotonically-non-decreasing return_cx and shrinking
    # K is one trajectory. Lossy but informative.
    terminals = []
    last = None
    cur_min = None
    for s in samples:
        if last is None or s["xs"].shape[0] > last:
            if cur_min is not None:
                terminals.append(-cur_min)
            cur_min = s["return_cx"]
        else:
            cur_min = min(cur_min, s["return_cx"])
        last = s["xs"].shape[0]
    if cur_min is not None:
        terminals.append(-cur_min)
    cnt = Counter(terminals)
    print(f"[inspect] approximate terminal-CX distribution across "
          f"~{len(terminals)} trajectories:")
    for cx in sorted(cnt):
        print(f"             cx={cx:>4}  count={cnt[cx]}")

    # Allowed-mask sanity.
    bad = 0
    for s in samples:
        if s["chosen"] not in s["allowed"]:
            bad += 1
        if not (0 <= s["chosen"] < s["xs"].shape[0]):
            bad += 1
    print(f"[inspect] mask sanity: {bad} samples had chosen out of allowed/range "
          f"(should be 0).")

    # Decode a few samples.
    print(f"\n[inspect] first {n_show} samples decoded:")
    for i, s in enumerate(samples[:n_show]):
        K, n = s["xs"].shape
        chosen = s["chosen"]
        print(f"  sample {i}: K={K} n={n}  chosen_row={chosen}  "
              f"return_cx={s['return_cx']:.0f}")
        for r in range(min(K, 6)):
            mark_chosen = "<-- chosen" if r == chosen else ""
            mark_allowed = " (allowed)" if r in s["allowed"] else ""
            ps = decode_row(s["xs"][r], s["zs"][r], s["ss"][r])
            print(f"     row {r:>3} {ps}{mark_allowed}  {mark_chosen}")
        if K > 6:
            print(f"     ... ({K-6} more rows)")


def cross_check(benchmark: str, sims: int, samples: List[dict]):
    """Run MonteQ.full_circuit directly on the same benchmark and compare."""
    from src import mcts_module as mq
    from src import heuristic_module as hm

    pkl = os.path.join(MONTEQ_ROOT, "src", benchmark + ".pkl")
    with open(pkl, "rb") as f:
        paulis = pickle.load(f)
    if isinstance(paulis, dict):
        paulis = paulis["paulis"]

    print(f"\n[cross-check] running MonteQ.full_circuit({benchmark}, sims={sims}) ...")
    t0 = time.time()
    sol = mq.full_circuit(
        paulis, hm.pair_solve, 2.0 ** 0.5,
        sims=sims, order_preserving=True,
    )
    t = time.time() - t0
    monteq_cx = sol[0]
    print(f"[cross-check] MonteQ best cx={monteq_cx} in {t:.2f}s")

    rets = np.array([s["return_cx"] for s in samples])
    # Approximate best terminal seen during collection.
    # For step at root cx_added_so_far=0, so terminal = -return_cx;
    # the smallest terminal across roots is min(-return_cx) over root samples.
    Ks = np.array([s["xs"].shape[0] for s in samples])
    root_mask = Ks == Ks.max()
    if root_mask.any():
        best_seen = int((-rets[root_mask]).min())
        print(f"[cross-check] best terminal CX across root samples = {best_seen}")
        if best_seen <= monteq_cx:
            print("[cross-check] OK: trace collector saw at least as good a "
                  "trajectory as direct full_circuit (expected, since BC keeps "
                  "the top trajectories).")
        else:
            print("[cross-check] WARNING: direct full_circuit beat the saved "
                  "traces. Either top-frac filtered out the best run, or the "
                  "instrumented rollout diverged from the stock one.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--benchmark", default=None,
                   help="Benchmark name without .pkl. Required for --cross-check.")
    p.add_argument("--cross-check", action="store_true")
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--n-show", type=int, default=3)
    args = p.parse_args()

    with open(args.data, "rb") as f:
        samples = pickle.load(f)
    inspect(samples, n_show=args.n_show)

    if args.cross_check:
        if not args.benchmark:
            raise SystemExit("--cross-check requires --benchmark")
        cross_check(args.benchmark, args.sims, samples)


if __name__ == "__main__":
    main()
