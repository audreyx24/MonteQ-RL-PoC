"""Small policy/value network for MonteQ rollouts.

Architecture
------------
    rows: (B, K_MAX, row_dim)
       -> per-row MLP encoder (row_dim -> d_model)
       -> a couple of Transformer-encoder layers across the K dimension
          (Set Transformer style; permutation-equivariant in row order
          before positional information is added, which is what we want
          since the row order is just a Pauli-word index).
       -> two heads:
            policy head:   linear(d_model -> 1) per row, then masked softmax
                           over `action_mask`.
            value head:    masked-mean pool across rows -> linear -> scalar.

We deliberately keep this small. A PoC just needs to beat "argmin Pauli
weight"; bigger nets are the next iteration's problem.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        row_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.row_dim = row_dim
        self.d_model = d_model

        self.row_embed = nn.Sequential(
            nn.Linear(row_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.policy_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        rows: torch.Tensor,         # (B, K, row_dim)
        row_mask: torch.Tensor,     # (B, K) True for real rows
        action_mask: torch.Tensor,  # (B, K) True for currently legal actions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits, value).

        policy_logits : (B, K) with -inf where action_mask is False (so
                        downstream cross-entropy / softmax over `chosen`
                        is well-defined).
        value         : (B,) scalar prediction of return_cx.
        """
        # Encode each row.
        x = self.row_embed(rows)  # (B, K, d_model)

        # Transformer expects key-padding-mask True for *invalid* positions.
        key_padding = ~row_mask  # (B, K)
        x = self.encoder(x, src_key_padding_mask=key_padding)

        # Policy head.
        logits = self.policy_head(x).squeeze(-1)  # (B, K)
        logits = logits.masked_fill(~action_mask, float("-inf"))

        # Value head: mean-pool over real rows only.
        mask_f = row_mask.float().unsqueeze(-1)  # (B, K, 1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = (x * mask_f).sum(dim=1) / denom  # (B, d_model)
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        return logits, value


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
