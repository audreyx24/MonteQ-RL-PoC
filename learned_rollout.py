"""Drop-in replacement for ``mcts_module.rollout_policy`` that consults a
trained ``PolicyValueNet`` instead of the greedy least-Pauli-weight rule.

Usage
-----
    from rl_monteq.learned_rollout import patch_mcts_with_learned_rollout
    patch_mcts_with_learned_rollout("rl_monteq/checkpoints/heisen_2_2.pt",
                                    sample=False, temperature=1.0)
    # ... now run MonteQ's full_circuit() as usual; MCTS will use the
    # learned rollout for the simulation phase.

The function signature ``rollout_policy(node, function, solution_list, kwargs)``
is preserved so it slots into the existing MCTS loop without further edits.

Modes
-----
sample=False  -> argmax over masked policy logits (deterministic).
sample=True   -> categorical sample with temperature (stochastic; better
                 match for MCTS rollouts since variability across
                 simulations is the whole point).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MONTEQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "MonteQ"))
if MONTEQ_ROOT not in sys.path:
    sys.path.insert(0, MONTEQ_ROOT)

from src import mcts_module as mq  # noqa: E402

from rl_monteq.featurizer import FeatureConfig, encode_state
from rl_monteq.training import load_model


def make_learned_rollout(
    model,
    feat_cfg: FeatureConfig,
    sample: bool = False,
    temperature: float = 1.0,
    device: str = "cpu",
) -> Callable:
    """Build a rollout_policy(node, function, solution_list, kwargs) closure."""

    @torch.no_grad()
    def policy_pick(state, allowed):
        rows, row_mask, action_mask = encode_state(
            state.xs, state.zs, state.ss, allowed, feat_cfg
        )
        rows = rows.unsqueeze(0).to(device)
        row_mask = row_mask.unsqueeze(0).to(device)
        action_mask = action_mask.unsqueeze(0).to(device)
        logits, _ = model(rows, row_mask, action_mask)
        logits = logits.squeeze(0)  # (K_MAX,)

        if not sample:
            return int(torch.argmax(logits).item())

        # Stable softmax: replace -inf with very negative number.
        masked = torch.where(
            action_mask.squeeze(0),
            logits / max(temperature, 1e-6),
            torch.full_like(logits, -1e9),
        )
        probs = torch.softmax(masked, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def learned_rollout_policy(node, function, solution_list, kwargs):
        state = node.state.copy()
        action = node.action.copy()
        action_time = node.action_time
        ndx_list = node.ndx_list.copy()
        allowed = node.untouched.copy()
        dag = None
        if node.dag is not None:
            dag = node.dag.copy()

        while True:
            if state.row_num == 0:
                count = mq.cx_count(action)
                solution = (count, action_time, len(action), action)
                if solution not in solution_list:
                    solution_list.append(solution)
                return count

            # Net picks the next index. Fall back to greedy if the net
            # somehow returns an out-of-allowed index (shouldn't happen
            # with proper masking, but be defensive).
            index = policy_pick(state, allowed)
            if index not in allowed:
                weight_word = state.xs | state.zs
                weight = float("inf")
                index = allowed[0]
                for i, string in enumerate(weight_word):
                    w_sum = int(np.sum(string))
                    if w_sum < weight and i in allowed:
                        weight = w_sum
                        index = i

            start = time.time()
            new_state, actions, new_ndx_list = function(
                state, index, ndx_list, allowed, kwargs
            )
            end = time.time()

            if dag is not None:
                rmv = [i for i in ndx_list if i not in new_ndx_list]
                new_dag = dag.copy()
                new_dag.remove_nodes_from(rmv)
                dag = new_dag
                allowed = [new_ndx_list.index(i) for i in mq.front_layer(dag)]
            else:
                allowed = list(range(new_state.row_num))

            state = new_state
            ndx_list = new_ndx_list
            action_time += end - start
            action += actions

    return learned_rollout_policy


# ---------------------------------------------------------------------------
# Convenience: monkeypatch MonteQ's mcts_module so the existing MCTS loop
# uses the learned rollout. Reversible via restore_default_rollout().

_ORIGINAL_ROLLOUT = mq.rollout_policy


def patch_mcts_with_learned_rollout(
    checkpoint_path: str,
    sample: bool = False,
    temperature: float = 1.0,
    device: str = "cpu",
):
    model, feat_cfg = load_model(checkpoint_path, device=device)
    fn = make_learned_rollout(model, feat_cfg, sample=sample,
                              temperature=temperature, device=device)
    mq.rollout_policy = fn
    return fn


def restore_default_rollout():
    mq.rollout_policy = _ORIGINAL_ROLLOUT
