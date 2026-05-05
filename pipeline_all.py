"""Collect traces for every MonteQ benchmark and train on all of them.

Steps
-----
1. Discover every .pkl Hamiltonian file in MonteQ/src/.
2. For each one, run the MCTS trace collector  (skip if already done).
3. Auto-compute k_max (max Pauli count) and n_max (max qubit count)
   from the actual benchmark dims — so the featurizer never crashes
   with "too large" errors regardless of which data you include.
4. Train a single PolicyValueNet on all collected trace files.

The larger benchmarks (Heisen_5_5, Heisen_6_5, UCC_6_12 …) take
longer to collect — each MCTS simulation visits more nodes.  Use
--sims to control the budget per benchmark, or --only to start small.

Usage
-----
    # Collect + train everything (may take a while for large benchmarks):
    python -m rl_monteq.pipeline_all

    # Skip collection; train on whatever is already in rl_monteq/data/:
    python -m rl_monteq.pipeline_all --no-collect

    # Pick specific benchmarks only:
    python -m rl_monteq.pipeline_all --only Heisen_2_2 UCC_2_4 LiH H2O

    # Quick sanity-check run:
    python -m rl_monteq.pipeline_all --sims 50 --epochs 5 --only Heisen_2_2 UCC_2_4

    # Full run with explicit paths:
    python -m rl_monteq.pipeline_all \
        --sims 300 --top-frac 0.2 \
        --epochs 60 --lr 2e-4 \
        --out rl_monteq/checkpoints/all_data.pt \
        --log-file logs/all_data.jsonl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any local imports.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MONTEQ_ROOT  = PROJECT_ROOT / "MonteQ"
MONTEQ_SRC   = MONTEQ_ROOT / "src"

for p in (str(PROJECT_ROOT), str(MONTEQ_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import heuristic_module as hm           # noqa: E402
from rl_monteq.trace_collection import (         # noqa: E402
    collect_for_hamiltonian,
    load_samples,
    save_samples,
)
from rl_monteq.training import train             # noqa: E402


# ---------------------------------------------------------------------------
# Benchmark discovery

def discover_benchmarks() -> List[str]:
    """Return names (no .pkl) of every Hamiltonian in MonteQ/src/."""
    names = sorted(
        p.stem for p in MONTEQ_SRC.glob("*.pkl")
    )
    return names


def load_paulis(name: str) -> List[str]:
    """Load a list of Pauli strings from MonteQ/src/<name>.pkl."""
    path = MONTEQ_SRC / f"{name}.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "paulis" in obj:
        return obj["paulis"]
    raise ValueError(f"Unrecognised benchmark format in {path}: {type(obj)}")


def benchmark_dims(name: str) -> Tuple[int, int]:
    """Return (K, n) — number of Pauli strings and number of qubits."""
    paulis = load_paulis(name)
    return len(paulis), len(paulis[0]) if paulis else 0


# ---------------------------------------------------------------------------
# Collection

def collect_benchmark(
    name: str,
    out_dir: Path,
    sims: int,
    top_frac: float,
    force: bool = False,
) -> Path:
    """Collect traces for one benchmark; return path to the saved .pkl file."""
    out_path = out_dir / f"{name}.pkl"

    if out_path.exists() and not force:
        existing = load_samples(str(out_path))
        print(f"[collect] {name}: already have {len(existing):,} samples — skipping. "
              f"(use --force-collect to redo)")
        return out_path

    paulis = load_paulis(name)
    K, n = len(paulis), len(paulis[0]) if paulis else 0
    print(f"[collect] {name}: K={K} Paulis, n={n} qubits, sims={sims} ...")
    samples = collect_for_hamiltonian(
        paulis,
        heuristic_function=hm.pair_solve,
        sims=sims,
        top_frac=top_frac,
        out_path=str(out_path),
    )
    print(f"  -> {len(samples):,} samples saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Auto k_max / n_max from actual data

def compute_padding(names: List[str]) -> Tuple[int, int]:
    """
    Scan benchmark dims and return (k_max, n_max) large enough for all of them.
    We round up to the next power of two for each dimension so the model can
    be reused on slightly larger inputs without retraining.
    """
    max_K, max_n = 0, 0
    for name in names:
        K, n = benchmark_dims(name)
        max_K = max(max_K, K)
        max_n = max(max_n, n)

    def next_pow2(x: int) -> int:
        p = 1
        while p < x:
            p <<= 1
        return p

    k_max = next_pow2(max_K)
    n_max = next_pow2(max_n)

    print(f"\n[dims] max K={max_K} -> k_max={k_max}  "
          f"max n={max_n} -> n_max={n_max}\n")
    return k_max, n_max


# ---------------------------------------------------------------------------
# Main

def main():
    p = argparse.ArgumentParser(
        description="Collect traces for all MonteQ benchmarks and train on them."
    )

    # --- Collection flags ---
    p.add_argument(
        "--only", nargs="*", metavar="NAME",
        help="Restrict to these benchmark names (default: all in MonteQ/src/).",
    )
    p.add_argument(
        "--no-collect", action="store_true",
        help="Skip trace collection; use whatever .pkl files are already in data-dir.",
    )
    p.add_argument(
        "--force-collect", action="store_true",
        help="Re-collect even if a .pkl already exists in data-dir.",
    )
    p.add_argument(
        "--sims", type=int, default=200,
        help="MCTS simulations per benchmark during collection (default: 200).",
    )
    p.add_argument(
        "--top-frac", type=float, default=0.2,
        help="Fraction of best trajectories to keep per benchmark (default: 0.2).",
    )
    p.add_argument(
        "--data-dir", default="rl_monteq/data",
        help="Directory to read/write collected trace .pkl files (default: rl_monteq/data).",
    )

    # --- Training flags ---
    p.add_argument(
        "--out", default="rl_monteq/checkpoints/all_data.pt",
        help="Path to save the best checkpoint (default: rl_monteq/checkpoints/all_data.pt).",
    )
    p.add_argument(
        "--log-file", default="logs/all_data.jsonl",
        help="Path to write per-epoch metric log (default: logs/all_data.jsonl).",
    )
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--alpha-value",type=float, default=0.5,
                   help="Weight of the value loss term (default: 0.5).")
    p.add_argument("--val-frac",   type=float, default=0.1)
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=2)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument(
        "--max-samples-per-file", type=int, default=5000,
        help="Max training samples to load from each benchmark .pkl "
             "(prevents OOM on large benchmarks; default: 5000). "
             "Set 0 to load everything.",
    )

    # --- k/n override ---
    p.add_argument(
        "--k-max", type=int, default=None,
        help="Override k_max instead of auto-computing it (useful if you know what you need).",
    )
    p.add_argument(
        "--n-max", type=int, default=None,
        help="Override n_max instead of auto-computing it.",
    )

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Decide which benchmarks to include.
    all_names = discover_benchmarks()
    if args.only:
        # Validate names.
        unknown = [n for n in args.only if n not in all_names]
        if unknown:
            print(f"[warn] Unknown benchmark names: {unknown}")
            print(f"       Available: {all_names}")
        names = [n for n in args.only if n in all_names]
    else:
        names = all_names

    print(f"[pipeline] Benchmarks selected ({len(names)}): {names}\n")

    # -----------------------------------------------------------------------
    # 2. Collect traces.
    if not args.no_collect:
        for name in names:
            collect_benchmark(
                name,
                out_dir=data_dir,
                sims=args.sims,
                top_frac=args.top_frac,
                force=args.force_collect,
            )
    else:
        print("[pipeline] --no-collect: skipping trace collection.\n")

    # -----------------------------------------------------------------------
    # 3. Build the list of data files to train on.
    #    Include any .pkl that exists in data_dir whose stem matches a selected
    #    benchmark name.  This means you can mix pre-existing files with newly
    #    collected ones seamlessly.
    data_files = []
    for name in names:
        candidate = data_dir / f"{name}.pkl"
        if candidate.exists():
            data_files.append(str(candidate))
        else:
            print(f"[warn] No trace file for {name} at {candidate} — skipping from training.")

    if not data_files:
        print("[error] No data files found. Run without --no-collect or check --data-dir.")
        sys.exit(1)

    print(f"\n[pipeline] Training on {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  {f}")

    # -----------------------------------------------------------------------
    # 4. Auto-compute k_max / n_max from the benchmarks we're actually using,
    #    unless the user overrode them.
    train_names = [Path(f).stem for f in data_files]

    k_max = args.k_max
    n_max = args.n_max
    if k_max is None or n_max is None:
        auto_k, auto_n = compute_padding(train_names)
        k_max = k_max or auto_k
        n_max = n_max or auto_n
    else:
        print(f"\n[dims] Using manual k_max={k_max}, n_max={n_max}\n")

    # -----------------------------------------------------------------------
    # 5. Train.
    print(f"[pipeline] Starting training — epochs={args.epochs}, lr={args.lr}, "
          f"d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"           k_max={k_max}, n_max={n_max}")
    print(f"           checkpoint -> {args.out}")
    print(f"           metric log  -> {args.log_file}\n")

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    max_samp = args.max_samples_per_file if args.max_samples_per_file > 0 else None

    best_val = train(
        data_paths=data_files,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_value=args.alpha_value,
        val_frac=args.val_frac,
        k_max=k_max,
        n_max=n_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        seed=args.seed,
        device=args.device,
        log_file=args.log_file,
        max_samples_per_file=max_samp,
    )

    print(f"\n[pipeline] Done. Best val_loss={best_val:.4f}")
    print(f"           Checkpoint: {args.out}")
    print(f"           Metric log: {args.log_file}")
    print(f"\nPlot the training curves with:")
    print(f"  python -m rl_monteq.plot_training --log {args.log_file} --out logs/all_data_curves.png")


if __name__ == "__main__":
    main()
