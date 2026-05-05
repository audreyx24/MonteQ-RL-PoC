"""RL-augmented rollout policy for MonteQ.

This package adds a learned policy/value network in place of MonteQ's
default greedy least-Pauli-weight rollout. Behavior-cloning targets
are collected by instrumenting MonteQ's MCTS, then a small Set-Transformer
policy is trained and dropped back in as the rollout function.

Modules:
    featurizer        - Cirq_Tableau -> torch tensors with action masks.
    network           - Small policy/value network (per-row encoder + attention).
    trace_collection  - Instrumented MCTS that logs training data to disk.
    training          - Behavior-cloning training loop.
    learned_rollout   - Drop-in replacement for mcts_module.rollout_policy.
    run_baseline      - Head-to-head A/B benchmark vs greedy rollout.
"""
