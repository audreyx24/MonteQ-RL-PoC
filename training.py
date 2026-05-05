"""Behavior-cloning training loop for the MonteQ rollout policy.

Loss = cross_entropy(policy_logits, chosen_action) + alpha * MSE(value, return_cx).

Run as a script:
    python -m rl_monteq.training \
        --data rl_monteq/data/heisen_2_2.pkl \
        --out  rl_monteq/checkpoints/heisen_2_2.pt \
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rl_monteq.featurizer import FeatureConfig, collate_traces
from rl_monteq.network import PolicyValueNet, num_params
from rl_monteq.trace_collection import load_samples


class TraceDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loader(samples: List[dict], config: FeatureConfig, batch_size: int, shuffle: bool):
    ds = TraceDataset(samples)

    def collate(batch):
        return collate_traces(batch, config)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def train(
    data_paths: List[str],
    out_path: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    alpha_value: float = 0.5,
    val_frac: float = 0.1,
    k_max: int = 256,
    n_max: int = 16,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    seed: int = 0,
    device: str = "cpu",
    log_file: Optional[str] = None,
    max_samples_per_file: Optional[int] = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples: List[dict] = []
    for p in data_paths:
        loaded = load_samples(p)
        original_len = len(loaded)
        if max_samples_per_file and original_len > max_samples_per_file:
            random.shuffle(loaded)
            loaded = loaded[:max_samples_per_file]
            print(f"[train] {os.path.basename(p)}: subsampled {max_samples_per_file:,} / {original_len:,}")
        else:
            print(f"[train] {os.path.basename(p)}: {original_len:,} samples")
        samples.extend(loaded)
    if not samples:
        raise RuntimeError("No training samples loaded.")
    print(f"[train] total samples: {len(samples):,}")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    config = FeatureConfig(k_max=k_max, n_max=n_max)

    train_loader = make_loader(train_samples, config, batch_size, shuffle=True)
    val_loader = make_loader(val_samples, config, batch_size, shuffle=False)

    model = PolicyValueNet(
        row_dim=config.row_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    print(f"[train] model params: {num_params(model):,}")
    print(f"[train] train samples: {len(train_samples):,}, val: {len(val_samples):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Set up optional metric log file (JSON lines — one dict per epoch).
    _log_fh = None
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        _log_fh = open(log_file, "w")

    for epoch in range(epochs):
        model.train()
        train_loss = train_pol = train_val = 0.0
        n_batches = 0
        for batch in train_loader:
            rows = batch["rows"].to(device)
            row_mask = batch["row_mask"].to(device)
            action_mask = batch["action_mask"].to(device)
            chosen = batch["chosen"].to(device)
            value_target = batch["value"].to(device)

            logits, value_pred = model(rows, row_mask, action_mask)

            # Policy: cross-entropy over the (already-masked) logits.
            policy_loss = F.cross_entropy(logits, chosen)
            value_loss = F.mse_loss(value_pred, value_target)
            loss = policy_loss + alpha_value * value_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()
            train_pol += policy_loss.item()
            train_val += value_loss.item()
            n_batches += 1

        # Validation.
        model.eval()
        val_loss = val_top1 = val_n = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rows = batch["rows"].to(device)
                row_mask = batch["row_mask"].to(device)
                action_mask = batch["action_mask"].to(device)
                chosen = batch["chosen"].to(device)
                value_target = batch["value"].to(device)

                logits, value_pred = model(rows, row_mask, action_mask)
                policy_loss = F.cross_entropy(logits, chosen)
                value_loss = F.mse_loss(value_pred, value_target)
                val_loss += (policy_loss + alpha_value * value_loss).item() * chosen.size(0)
                val_top1 += (logits.argmax(dim=-1) == chosen).float().sum().item()
                val_n += chosen.size(0)

        val_loss /= max(1, val_n)
        val_top1 /= max(1, val_n)

        avg_train = train_loss / max(1, n_batches)
        avg_pol   = train_pol  / max(1, n_batches)
        avg_val_c = train_val  / max(1, n_batches)

        print(
            f"[epoch {epoch:03d}] "
            f"train={avg_train:.4f} "
            f"(pol={avg_pol:.4f}, "
            f"val={avg_val_c:.4f}) | "
            f"val_loss={val_loss:.4f} acc@1={val_top1:.3f}"
        )

        if _log_fh:
            record = {
                "epoch": epoch,
                "train_loss": avg_train,
                "train_policy_loss": avg_pol,
                "train_value_loss": avg_val_c,
                "val_loss": val_loss,
                "val_acc1": val_top1,
            }
            _log_fh.write(json.dumps(record) + "\n")
            _log_fh.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "row_dim": config.row_dim,
                        "k_max": config.k_max,
                        "n_max": config.n_max,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                out_path,
            )
            print(f"  -> saved checkpoint to {out_path}")

    if _log_fh:
        _log_fh.close()

    return best_val


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Restore a PolicyValueNet from a training checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device)
    cfg = ck["config"]
    model = PolicyValueNet(
        row_dim=cfg["row_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    feat_cfg = FeatureConfig(k_max=cfg["k_max"], n_max=cfg["n_max"])
    return model, feat_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha-value", type=float, default=0.5)
    p.add_argument("--k-max", type=int, default=256)
    p.add_argument("--n-max", type=int, default=16)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write per-epoch metrics as JSON lines "
             "(e.g. logs/run1.jsonl). Used by plot_training.py.",
    )
    args = p.parse_args()

    train(
        data_paths=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_value=args.alpha_value,
        k_max=args.k_max,
        n_max=args.n_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=args.device,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
