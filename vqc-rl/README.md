# vqc-rl — deep Q-learning with a variational quantum circuit

`quantum_rl` is the quantum notebook of [`../original/`](../original/) rewritten as a
Python package: the stochastic 4×3 grid world, exact value iteration, and deep
Q-learning with a 4-qubit variational quantum circuit (PennyLane + PyTorch).

It comes with:

* a **GUI** (`gui.py`) to set the hyper-parameters, train, and watch the circuit, its
  weights and its gradients live, with every run saved automatically;
* a **command-line interface** (`python -m quantum_rl`);
* a **test suite**, including a bit-for-bit comparison with the original training cell;
* the **diagnostics** that located why the original notebook did not learn
  ([`diagnostics/README.md`](diagnostics/README.md)).

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The GUI also needs Tk (`sudo apt install python3-tk` on Debian/Ubuntu). Tested with
Python 3.13, numpy 2.5, matplotlib 3.11, torch 2.14, PennyLane 0.45.

## The GUI

```bash
./gui.py
```

`gui.py` relaunches itself with `.venv/bin/python` when that exists, so there is no
need to activate the environment.

**Setup screen.** Every hyper-parameter has a short explanation (hover it, or read the
bar at the bottom). The *Output* card picks the folder the run is saved to; relative
paths are taken from this folder, and the default is a new timestamped folder under
`runs/`.

**Training screen.** A status bar with the episode, MSE against Bellman, loss,
learning rate, epsilon and gradient norm, and four views:

| view | what it shows |
| --- | --- |
| **Curves** | MSE against Bellman, the training loss (raw per episode, faint, with a centred moving average whose window you choose), the learned utilities against Bellman's, learning rate and epsilon. |
| **Policy** | The Bellman policy next to the learned one; learned arrows are green where they agree, red where they do not. |
| **Circuit** | The circuit as it is built: re-uploading of the input, CNOT chain, one `Rot(φ, θ, ω)` per qubit and layer. *Angles* mode shows each parameter as a dial (current value, initial value, distance travelled); *Gradients* mode shows the gradient RMS of the episode on a log gauge. A timeline scrubs or replays the whole training, a grid picks the input state and shows its Q-values, a mini-map and a detail strip follow any gate. |
| **Gradients** | Gradient RMS against time on a log scale, with the vanishing region shaded: whole circuit, per layer (or per qubit of the selected layer), every parameter as a heatmap, and one gate. A side list shows every parameter whose gradient is below a threshold (default `1e-10`), in the last episode or over the whole run. |

Parameters that cannot affect the output (see below) are drawn grey and marked
*no effect*, so they are never mistaken for vanishing gradients.

**Saved automatically** in the output folder, also when training is stopped:

| file | when |
| --- | --- |
| `summary.json` | config and Bellman reference at start, training results at the end |
| `live.png`, `progress.json` | every few seconds during training |
| `bellman_policy.pdf`, `bellman_convergence.pdf` | at start |
| `policy.pdf`, `training_curves.pdf`, `utility_comparison.pdf`, `residuals.pdf` | at the end |
| `params_history.npz` | weights and gradient RMS of every episode |
| `gradient_stats.csv` | per parameter: last and maximum gradient, share of episodes below the threshold |
| `circuit/circuit_angles.*`, `circuit/circuit_gradients.*` | the final circuit, all layers (PNG and PDF) |
| `gradients/all.*`, `gradients/layer_<k>.*` | the Gradients view for the whole circuit and for each layer |
| `gui_curves.png`, `gui_policy.png` | the Curves and Policy views |

`./debug_gui.py` opens the same GUI with one more card: switches that put the bugs of
the original notebook back, to compare them with the fixed agent. The switches in use
are recorded in `summary.json`.

## Command line

```bash
.venv/bin/python -m quantum_rl --episodes 1000 --max-time 40 --lr 0.01 --out-dir runs/example
.venv/bin/python -m quantum_rl --bellman-only --out-dir runs/bellman   # value iteration only
.venv/bin/python -m quantum_rl --notebook --episodes 50                # the original behaviour
.venv/bin/python -m quantum_rl --help
```

`--bug NAME` turns a single notebook bug back on, `--notebook` turns all of them on
(and restores the notebook's learning-rate schedule and γ = 0.95).

## As a library

```python
from quantum_rl import Config, GridWorld, value_iteration, train

cfg = Config(num_games=500, max_time=40, LR_0=0.01, seed=0)
env = GridWorld(cfg)
U_final, _ = value_iteration(env, cfg.gamma, cfg.max_epoch)
result = train(cfg, env, U_ref=U_final)
print(result.policy, result.U, result.mse_u[-1])
```

`train()` accepts two callbacks: `monitor` (every `eval_every` episodes: loss,
learning rate, epsilon, MSE, current utilities and policy) and `game_monitor` (every
episode: weights, gradient RMS, loss). Neither changes the training.

## Package layout

| module | content |
| --- | --- |
| `config.py` | `Config` (every hyper-parameter) and `FixFlags` (the notebook bugs) |
| `environment.py` | `GridWorld`: transitions, rewards, slip model |
| `bellman.py` | value iteration and the greedy policy (the reference) |
| `circuit.py` | the VQC: state encoding, layers, QNode, `variational_classifier` |
| `agent.py` | epsilon-greedy, replay memory, minibatch sampling |
| `train.py` | the deep Q-learning loop → `TrainResult` |
| `analysis.py`, `plotting.py`, `outputs.py` | residuals, figures, saved files |
| `statevector.py` | a small numpy simulator of the same circuit, used by the GUI |
| `gui.py`, `debug_gui.py`, `gui_widgets.py`, `circuit_view.py`, `circuit_plot.py` | the GUI |

## The bugs of the original notebook

Each one is a field of `FixFlags`. `False` (the default) is the fix; `True`, or
`FixFlags.notebook()`, restores the notebook.

| flag | notebook behaviour | effect |
| --- | --- | --- |
| `td_pred_uses_next_state` | the TD prediction is `Q(s′)[a]` instead of `Q(s)[a]` | every action of a state is pulled to the same target: Q stops telling actions apart and the policy collapses. **The main cause.** |
| `epsilon_greedy_inverted` | greedy when `rand() < ε` | 5% exploration at the start, 95% at the end |
| `env_deterministic_transitions` | the sampled slip is overwritten with the intended move | training on a deterministic grid, scoring on the stochastic one |
| `greedy_policy_ignores_reward` | the Bellman policy maximises `Σ P·U`, without reward and discount | only the reference policy plot changes |

Two more problems are not flags but defaults: the notebook's learning-rate schedule
(halving every 3 episodes, dead after 24) is replaced by a much slower one, and its
running averages of the loss are replaced by honest ones.

`Config.notebook()` reproduces the original bit for bit. `tests/test_integration.py`
checks it against `tests/notebook_cell50_reference.py`, a statement-by-statement
transcription of the notebook's training cell, with a tolerance of exactly zero on
losses, returns, final weights, MSE trace and final policy.

## Parameters that never learn

With the architecture of the notebook, 8 of the `12L` rotation angles have an
identically zero gradient, whatever the weights:

* `φ` of the first layer: `RZ(φ)` acts on a computational-basis state (the input
  encoding plus CNOTs), so it only adds a global phase;
* `ω` of the last layer: `RZ(ω)` right before the measurement commutes with `Z`.

Separately, a gradient can be exactly zero in a single episode: a `Rot` on qubit `i`
only reaches the outputs of qubits `≥ i` (the CNOT chain runs one way), so if no
action updated in that episode depends on it, it gets no signal. The GUI distinguishes
the three cases: *no effect*, *no signal* and a genuinely small gradient.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests -q
```

## Performance

A forward pass of the circuit takes a few milliseconds and every training step needs
two or three of them plus a backward pass. The notebook's 20000 episodes × 250 steps
take many hours; a few hundred to a few thousand episodes of 40 steps are enough to
see the agent learn.
