"""Visualise training metrics saved by training.py --log-file.

Usage
-----
    # After a training run logged to logs/run1.jsonl:
    python -m rl_monteq.plot_training --log logs/run1.jsonl

    # Compare several runs:
    python -m rl_monteq.plot_training \
        --log logs/run1.jsonl logs/run2.jsonl \
        --labels "small" "large" \
        --out training_curves.png

Each log file is a JSON-lines file where every line is one epoch:
    {"epoch": 0, "train_loss": ..., "train_policy_loss": ...,
     "train_value_loss": ..., "val_loss": ..., "val_acc1": ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(path: str) -> dict[str, list]:
    """Read a JSON-lines log file; return dict of metric→list-of-values."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    keys = records[0].keys()
    return {k: [r[k] for r in records] for k in keys}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(log_paths: List[str], labels: List[str], out_path: str):
    runs = [load_log(p) for p in log_paths]

    # Use a clean, publication-friendly style that works everywhere.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Training dashboard", fontsize=14, fontweight="bold", y=1.01)

    panels = [
        # (ax, metric_keys, y_label, title)
        (axes[0, 0],
         [("train_loss", "Train"), ("val_loss",   "Val")],
         "Loss",
         "Total loss  (policy + α·value)"),
        (axes[0, 1],
         [("train_policy_loss", "Train policy")],
         "Cross-entropy",
         "Policy loss  (cross-entropy)"),
        (axes[1, 0],
         [("train_value_loss", "Train value")],
         "MSE",
         "Value loss  (MSE)"),
        (axes[1, 1],
         [("val_acc1", "Val acc@1")],
         "Accuracy",
         "Validation top-1 accuracy"),
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric_pairs, ylabel, title in panels:
        for run_idx, (run_data, run_label) in enumerate(zip(runs, labels)):
            base_color = colors[run_idx % len(colors)]
            epochs = run_data["epoch"]

            for line_idx, (metric, line_label) in enumerate(metric_pairs):
                if metric not in run_data:
                    continue
                values = run_data[metric]
                # Slightly desaturate the second line (val) within each run.
                alpha = 1.0 if line_idx == 0 else 0.6
                ls    = "-"  if line_idx == 0 else "--"
                label = f"{run_label} – {line_label}" if len(runs) > 1 else line_label
                ax.plot(epochs, values, color=base_color, alpha=alpha,
                        linestyle=ls, label=label, marker="o" if len(epochs) <= 60 else "")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)

    # Mark best val epoch on the total-loss panel.
    best_ax = axes[0, 0]
    for run_idx, (run_data, run_label) in enumerate(zip(runs, labels)):
        val_losses = run_data["val_loss"]
        best_ep    = run_data["epoch"][val_losses.index(min(val_losses))]
        best_ax.axvline(best_ep, color=colors[run_idx % len(colors)],
                        linestyle=":", alpha=0.7, linewidth=1.2,
                        label=f"{run_label} best epoch ({best_ep})")
    best_ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot_training] saved → {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Plot training curves from one or more JSON-lines log files."
    )
    p.add_argument(
        "--log", nargs="+", required=True, metavar="FILE",
        help="Path(s) to .jsonl log file(s) produced by training.py --log-file.",
    )
    p.add_argument(
        "--labels", nargs="*", metavar="LABEL",
        help="Display name for each log file (defaults to the filename).",
    )
    p.add_argument(
        "--out", default="training_curves.png",
        help="Output image path (PNG/PDF/SVG etc.).",
    )
    args = p.parse_args()

    labels = args.labels or [Path(p).stem for p in args.log]
    if len(labels) < len(args.log):
        labels += [Path(p).stem for p in args.log[len(labels):]]

    plot(args.log, labels, args.out)


if __name__ == "__main__":
    main()
