# rl_monteq — RL-augmented rollout for MonteQ

Proof-of-concept that replaces MonteQ's hand-coded greedy rollout
(`mcts_module.rollout_policy`) with a small learned policy/value network
trained by behavior cloning on traces from MCTS itself.

Targets the **unitary-preserving** regime (DAG front-layer action set).

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

The MonteQ repo is sitting next to this folder at `../MonteQ/`. Both
`trace_collection` and `learned_rollout` add it to `sys.path` automatically.

## Install

The sandbox this was scaffolded in doesn't have torch or qiskit, so I
couldn't run the end-to-end smoke test. Locally:

```bash
pip install torch numpy qiskit rustworkx qiskit-ibm-runtime qiskit-nature
```

(MonteQ's own README pins `qiskit[visualize]`, `rustworkx`, `numpy`,
`matplotlib`. Add `torch` for this PoC.)

## Workflow

### 1. Collect behavior-cloning traces from MCTS

```bash
python -m rl_monteq.collect_traces \
    --benchmark Heisen_2_2 UCC_2_4 \
    --sims 200 \
    --top-frac 0.2 \
    --out-dir rl_monteq/data
```

This runs MonteQ MCTS at 200 iterations on each named Hamiltonian (the
`.pkl` files bundled in `MonteQ/src/`), keeps the top 20% of rollout
trajectories by terminal CX count, and dumps each step as a training
sample dict to `rl_monteq/data/<bench>.pkl`.

A "step" is a tuple `(xs, zs, ss, allowed, chosen, return_cx)` where
`return_cx` is the negative remaining CX from this state to terminal —
matches MonteQ's existing reward sign convention.

### 2. Train the policy/value net

```bash
python -m rl_monteq.training \
    --data rl_monteq/data/Heisen_2_2.pkl rl_monteq/data/UCC_2_4.pkl \
    --out  rl_monteq/checkpoints/small.pt \
    --epochs 30
```

Loss is `cross_entropy(policy_logits, chosen)` plus
`alpha * MSE(value, return_cx)`. Validation reports top-1 action accuracy
against the BC targets — when this stops improving, your net has learned
to mimic the best rollouts.

Tunable knobs: `--k-max`, `--n-max` for padding sizes (defaults
`256` and `16`), `--d-model`, `--n-heads`, `--n-layers` for capacity.

### 3. A/B benchmark vs greedy rollout

```bash
python -m rl_monteq.run_baseline \
    --benchmark Heisen_2_2 \
    --checkpoint rl_monteq/checkpoints/small.pt \
    --sims 50 \
    --sample
```

Runs `full_circuit` once with the original greedy rollout, once with the
learned rollout, and prints CX count + wall time for each. `--sample`
flips on stochastic action sampling (recommended for actual MCTS use,
since variance across simulations is the point).

For real numbers, sweep `--sims` over `[1, 10, 50, 200]` and average over
several runs on benchmarks the net was *not* trained on. See "Eval
protocol" below.

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

## What this PoC is and isn't

**Is**:
- A behavior-cloning baseline. Learn what a good rollout *should* pick
  by imitating the rollouts that happened to reach the best terminals.
- A drop-in replacement so head-to-head against the greedy rollout is
  one CLI flag away.

**Isn't (yet)**:
- AlphaZero-style. Selection still uses MonteQ's UCT (not PUCT), and
  there's no self-improvement loop. To go that direction:
   - Replace `Node.UCT` with `Q + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
     using the network's policy as the prior `P`.
   - Skip the rollout entirely; bootstrap leaf values from the value head.
   - Iterate: search-with-current-net, retrain on visit-count targets,
     repeat.
- Trained on big benchmarks. Behavior cloning on the smaller
  Hamiltonians (Heisenberg(2,2), UCC_2_4, LiH) is the right starting
  point because MCTS at 200 iterations actually *finds* good
  trajectories there. On NH3 / Fermi-Hub(6,5), the BC targets are
  themselves not great, so you'd want offline RL with advantage
  weighting (or full self-play).

## Eval protocol (for real numbers)

1. **Train benchmarks**: small ones where MCTS@200 sims explores enough
   of the tree (Heisenberg(2,2), Heisenberg(2,3), UCC_2_4, LiH).
2. **Held-out benchmarks**: at least one bigger one where MCTS@200 sims
   is far from optimal (Fermi-Hub(4,3), Heisenberg(3,3), H2O). The
   research question is whether the policy generalizes to states it
   couldn't be trained on, since that's where MCTS-only ran out of
   compute.
3. **Sweep sims** in `[1, 10, 50, 200]`. Plot CX count vs sims for both
   rollouts. The hoped-for shape: learned rollout's curve sits below
   greedy's at every budget, with the gap widest at low sims (because
   each rollout is now informative).
4. **Wall-clock**: per-rollout, the learned rollout is slower (NN
   forward pass per step). The fair comparison is *iso-time*: how much
   CX reduction per second?
5. **Multiple seeds**: MCTS is deterministic in MonteQ but the learned
   rollout with `--sample` is not. Average over 5+ runs per (benchmark,
   sims) cell.

## Known sharp edges / TODOs

- `featurizer.encode_state` raises if `K > k_max` or `n > n_max`. The
  bigger benchmarks in MonteQ's table need bumped padding sizes; this
  is a deliberate fail-loudly choice.
- The instrumented rollout in `trace_collection` deliberately keeps the
  same greedy choice rule so behavior cloning matches MonteQ's existing
  policy. To explore beyond that policy, swap in epsilon-greedy or
  random rollouts during data collection.
- `Node.UCT` selection still consults the running average of greedy
  rollouts. To fully exploit the learned value head, override UCT with
  PUCT (see "isn't yet" above).
- No Pauli-string-permutation augmentation in training. Adding a random
  row permutation per epoch (the network is permutation-equivariant
  before its policy head, so this is free regularization) would help
  generalization to held-out benchmarks.
- Smoke-tested as syntactically valid Python; full end-to-end run
  needs torch + qiskit installed in your environment.

## References

- MonteQ paper (sitting next to this folder as `MonteQ.pdf`). Section
  IV-B describes the rollout being replaced.
- Quantinuum MCTS+RL state-prep paper — same recipe applied to a
  neighboring problem; useful for architecture sanity-check.
- Offline RL for Hamiltonian Simulation (also in this folder) — natural
  next step beyond behavior cloning when targets are noisy.
# MonteQ-RL-PoC
