# rl_monteq — RL-augmented rollout for MonteQ

Proof-of-concept that replaces MonteQ's hand-coded greedy rollout
(`mcts_module.rollout_policy`) with a small learned policy/value network
trained by behavior cloning on traces from MCTS itself.

## Layout

```
rl_monteq/
  featurizer.py        Pauli word + allowed mask -> tensors.
  network.py           PolicyValueNet (per-row MLP -> Transformer -> two heads).
  trace_collection.py  Instrumented MCTS that logs (state, action, return) tuples.
  collect_traces.py    CLI wrapper around trace_collection for bundled benchmarks.
  training.py          Behavior-cloning training loop (CE on policy + MSE on value).
  learned_rollout.py   Drop-in replacement for mcts_module.rollout_policy.
  run_baseline.py      A/B benchmark: greedy vs learned rollout at matched sims.
  data/                (created on first run) collected traces.
  checkpoints/         (created on first run) trained .pt files.
```
## How the integration works

```
MonteQ.full_circuit
   -> MCTS loop
       -> tree_policy (selection + expansion)         <- unchanged
       -> rollout_policy (simulation)                 <- patched
       -> backpropagate                                <- unchanged
```

`learned_rollout.patch_mcts_with_learned_rollout(checkpoint)` rebinds
`mcts_module.rollout_policy` to a closure that, at each step:

1. Encodes the current `Cirq_Tableau` + `allowed` set into tensors.
2. Forward-passes the policy/value net.
3. Picks `argmax` (or samples) over the masked policy logits.
4. Calls the same `function(state, idx, ndx_list, allowed, kwargs)`
   heuristic MonteQ already uses to actually implement the chosen Pauli.
5. Updates the DAG front layer just like the original rollout does.

Reversible via `restore_default_rollout()`.
