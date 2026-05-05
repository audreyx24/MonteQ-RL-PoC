"""Collect behavior-cloning training data from MonteQ MCTS runs.

Strategy
--------
Run MonteQ's MCTS at a high iteration budget on a benchmark Hamiltonian.
Capture every rollout trajectory (which `mcts_module.rollout_policy`
already collects in `solution_list`, but as gates only -- we need the
intermediate states too).

Approach: re-implement `rollout_policy` here in instrumented form. The
instrumented version is byte-compatible with the original (same return
value, same effect on `solution_list`) but additionally:
    1. Logs each (state, allowed, chosen) decision it makes.
    2. After reaching terminal, attaches the trajectory's terminal CX
       count as a return-to-go target for every logged step.

Then we cherry-pick the best trajectories (lowest total CX) for
behavior-cloning. The `chosen` action recorded is the same min-Pauli-weight
greedy choice, so the BC target is "imitate the rollout that happened to
hit the best terminal" -- this is the same trick AlphaGo's policy network
used initially before self-play kicked in.

Output format: list of pickled training samples, each a dict matching
the keys consumed by `featurizer.collate_traces`.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# Make MonteQ importable. Layout:
#   <project>/MonteQ/src/{mcts_module,classes,heuristic_module}.py
#   <project>/rl_monteq/trace_collection.py
HERE = os.path.dirname(os.path.abspath(__file__))
MONTEQ_ROOT = os.path.normpath(os.path.join(HERE, "..", "MonteQ"))
if MONTEQ_ROOT not in sys.path:
    sys.path.insert(0, MONTEQ_ROOT)

# Imports below depend on the path append above.
from src import mcts_module as mq  # noqa: E402
from src.classes import Cirq_Tableau, Node  # noqa: E402


@dataclass
class StepRecord:
    xs: np.ndarray
    zs: np.ndarray
    ss: np.ndarray
    allowed: List[int]
    chosen: int            # row index actually implemented
    cx_added_so_far: int   # CX count up through and including this step's action

    # Filled in after the trajectory finishes.
    terminal_cx: Optional[int] = None

    @property
    def return_cx(self) -> float:
        """Negative remaining CX from this state onwards (matches MonteQ's reward)."""
        assert self.terminal_cx is not None, "Trajectory not yet terminated."
        return -float(self.terminal_cx - self.cx_added_so_far)


@dataclass
class Trajectory:
    steps: List[StepRecord] = field(default_factory=list)
    terminal_cx: Optional[int] = None

    def finalize(self, terminal_cx: int):
        self.terminal_cx = terminal_cx
        for s in self.steps:
            s.terminal_cx = terminal_cx


# ---------------------------------------------------------------------------
# Instrumented rollout: a byte-compatible reimplementation of
# mcts_module.rollout_policy that additionally logs each step.

def instrumented_rollout_policy(
    node: Node,
    function: Callable,
    solution_list: list,
    kwargs: dict,
    sink: List[Trajectory],
):
    state = node.state.copy()
    action = node.action.copy()
    action_time = node.action_time
    ndx_list = node.ndx_list.copy()
    allowed = node.untouched.copy()
    dag = None
    if node.dag is not None:
        dag = node.dag.copy()

    cx_so_far = mq.cx_count(action)
    traj = Trajectory()

    while True:
        if state.row_num == 0:
            count = mq.cx_count(action)
            solution = (count, action_time, len(action), action)
            if solution not in solution_list:
                solution_list.append(solution)
            traj.finalize(count)
            sink.append(traj)
            return count

        # Greedy least-Pauli-weight choice (matches stock MonteQ rollout).
        weight_word = state.xs | state.zs
        weight = float("inf")
        index = None
        for i, string in enumerate(weight_word):
            w_sum = int(np.sum(string))
            if w_sum < weight and i in allowed:
                weight = w_sum
                index = i

        # Log BEFORE applying the action.
        traj.steps.append(
            StepRecord(
                xs=state.xs.copy(),
                zs=state.zs.copy(),
                ss=state.ss.copy(),
                allowed=list(allowed),
                chosen=int(index),
                cx_added_so_far=cx_so_far,
            )
        )

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
        cx_so_far = mq.cx_count(action)


def MCTS_with_traces(root_node, function, param, sims, rot_params, checks, kwargs):
    """A trimmed MCTS loop that uses the instrumented rollout and returns the traces."""
    solution_list: list = []
    trajectories: List[Trajectory] = []

    if sims is None:
        sims = len(root_node.untouched)

    while root_node.ni < sims:
        leaf = mq.tree_policy(root_node, function, param, kwargs)
        value = instrumented_rollout_policy(
            leaf, function, solution_list, kwargs, sink=trajectories
        )
        mq.backpropagate(leaf, value)

    return trajectories, solution_list


# ---------------------------------------------------------------------------
# Convenience: filter and serialize.

def best_trajectories(
    trajectories: List[Trajectory], top_frac: float = 0.2
) -> List[Trajectory]:
    """Keep the best `top_frac` of trajectories by terminal CX count."""
    scored = sorted(trajectories, key=lambda t: t.terminal_cx)
    k = max(1, int(len(scored) * top_frac))
    return scored[:k]


def trajectories_to_samples(trajectories: List[Trajectory]) -> List[dict]:
    """Flatten trajectories into a list of training-sample dicts."""
    samples = []
    for t in trajectories:
        for s in t.steps:
            samples.append(
                {
                    "xs": s.xs,
                    "zs": s.zs,
                    "ss": s.ss,
                    "allowed": s.allowed,
                    "chosen": s.chosen,
                    "return_cx": s.return_cx,
                }
            )
    return samples


def save_samples(samples: List[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_samples(path: str) -> List[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Top-level entry: collect samples for one Hamiltonian.

def collect_for_hamiltonian(
    paulis: List[str],
    heuristic_function: Callable,
    sims: int = 200,
    mcts_param: float = 2.0 ** 0.5,
    top_frac: float = 0.2,
    out_path: Optional[str] = None,
    **kwargs,
) -> List[dict]:
    """Build a root node for ``paulis``, run MCTS, return top-trajectory samples.

    ``heuristic_function`` is the implementation heuristic (e.g.
    ``heuristic_module.pair_solve``). Pass extra heuristic args via kwargs,
    matching MonteQ's ``full_circuit`` signature.
    """
    DAG = mq.build_anticommute_dag(paulis)
    tableau = Cirq_Tableau([s for s in paulis])  # Cirq_Tableau strips signs in-place
    root = Node(tableau)
    root.ndx_list = list(range(len(paulis)))
    root.dag = DAG
    root.untouched = [root.ndx_list.index(j) for j in mq.front_layer(DAG)]

    trajectories, _ = MCTS_with_traces(
        root, heuristic_function, mcts_param, sims, None, [], kwargs
    )

    keep = best_trajectories(trajectories, top_frac=top_frac)
    samples = trajectories_to_samples(keep)

    if out_path is not None:
        save_samples(samples, out_path)
    return samples
