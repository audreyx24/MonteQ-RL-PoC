"""Pauli-word featurizer.

Turns a MonteQ ``Cirq_Tableau`` plus the current ``allowed`` action set
into the tensors the policy/value network consumes.

Encoding choice
---------------
Each Pauli string is a row in a K x (2n+1) binary matrix:
    [ x-side (n) | z-side (n) | sign (1) ]

We pad both the row dimension (K -> K_MAX) and qubit dimension (n -> N_MAX)
so the network sees fixed-shape tensors. Unused rows are zeroed and masked.

The action space is the set of row indices ``allowed`` returned by MonteQ's
front-layer logic in the unitary-preserving regime (a subset of [0, K-1]).
We expose this as a length-K_MAX boolean mask.

Per-row feature size is therefore 2 * N_MAX + 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Default padding sizes. Override per-experiment via FeatureConfig.
# These cover all the small benchmarks in the MonteQ paper (UCC_2_4 ~ 12
# Pauli strings, 4 qubits; LiH ~12 qubits; Heisenberg(2,2) 4 qubits & ~14
# strings). Bump them for larger benchmarks.
DEFAULT_K_MAX = 256
DEFAULT_N_MAX = 16


@dataclass
class FeatureConfig:
    k_max: int = DEFAULT_K_MAX
    n_max: int = DEFAULT_N_MAX

    @property
    def row_dim(self) -> int:
        # x-side + z-side + sign
        return 2 * self.n_max + 1


# ---------------------------------------------------------------------------
# Core featurization

def encode_state(
    xs: np.ndarray,
    zs: np.ndarray,
    ss: np.ndarray,
    allowed: Sequence[int],
    config: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a single Pauli word into (rows, row_mask, action_mask).

    Parameters
    ----------
    xs, zs : (K, n) int arrays from ``Cirq_Tableau``.
    ss     : (K,) int array of signs.
    allowed: indices in [0, K) that are currently legal actions
             (front layer in the unitary-preserving regime).
    config : padding configuration.

    Returns
    -------
    rows        : (K_MAX, row_dim) float tensor.
    row_mask    : (K_MAX,) bool tensor; True where a real row exists.
    action_mask : (K_MAX,) bool tensor; True where the index is in `allowed`.
    """
    K, n = xs.shape
    if K > config.k_max:
        raise ValueError(f"Pauli word has {K} rows, k_max={config.k_max}")
    if n > config.n_max:
        raise ValueError(f"Pauli word has {n} qubits, n_max={config.n_max}")

    rows = np.zeros((config.k_max, config.row_dim), dtype=np.float32)
    rows[:K, :n] = xs.astype(np.float32)                 # x-side
    rows[:K, config.n_max : config.n_max + n] = zs.astype(np.float32)
    rows[:K, 2 * config.n_max] = ss.astype(np.float32)   # sign in last slot

    row_mask = np.zeros(config.k_max, dtype=bool)
    row_mask[:K] = True

    action_mask = np.zeros(config.k_max, dtype=bool)
    for i in allowed:
        if 0 <= i < K:
            action_mask[i] = True

    return (
        torch.from_numpy(rows),
        torch.from_numpy(row_mask),
        torch.from_numpy(action_mask),
    )


def encode_tableau(tableau, allowed: Sequence[int], config: FeatureConfig):
    """Convenience wrapper: takes a MonteQ Cirq_Tableau directly."""
    if tableau.row_num == 0:
        # Terminal state. Return all-zero tensors with empty masks.
        rows = torch.zeros((config.k_max, config.row_dim), dtype=torch.float32)
        empty = torch.zeros(config.k_max, dtype=torch.bool)
        return rows, empty, empty
    return encode_state(tableau.xs, tableau.zs, tableau.ss, allowed, config)


# ---------------------------------------------------------------------------
# Batching helper for training

def collate_traces(samples: List[dict], config: FeatureConfig):
    """Collate a list of trace dicts (see trace_collection.py) into batched tensors.

    Each sample dict has:
        xs, zs, ss : np.ndarrays
        allowed    : List[int]
        chosen     : int  (index actually taken from `allowed`)
        return_cx  : float (negative cumulative CX from this state to terminal,
                            so larger == better; matches MonteQ's reward convention)
    """
    B = len(samples)
    rows = torch.zeros((B, config.k_max, config.row_dim), dtype=torch.float32)
    row_mask = torch.zeros((B, config.k_max), dtype=torch.bool)
    action_mask = torch.zeros((B, config.k_max), dtype=torch.bool)
    chosen = torch.zeros(B, dtype=torch.long)
    value = torch.zeros(B, dtype=torch.float32)

    for b, s in enumerate(samples):
        r, rm, am = encode_state(s["xs"], s["zs"], s["ss"], s["allowed"], config)
        rows[b] = r
        row_mask[b] = rm
        action_mask[b] = am
        chosen[b] = int(s["chosen"])
        value[b] = float(s["return_cx"])

    return {
        "rows": rows,
        "row_mask": row_mask,
        "action_mask": action_mask,
        "chosen": chosen,
        "value": value,
    }
