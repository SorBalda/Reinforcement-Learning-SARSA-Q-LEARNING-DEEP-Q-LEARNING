# Visualizing a VQC in Reinforcement Learning (frozen lake environment)

This project studies a four-qubit variational quantum circuit (VQC) as the
action-value function in deep Q-learning on a stochastic 4×3 grid world. Exact
Bellman value iteration provides a reference for learned values and policies.
It is part of an internship done in 2024 in Porto University, under the supervision of Prof. Duarte Magano and Ariel Guerreiro.


## Live visualization

![Circuit, gradients, and policy updating live](vqc-rl/docs/assets/live-training-tabs.gif)

![Training metrics updating in real time](vqc-rl/docs/assets/training-metrics.gif)

The first animation replays real checkpoints while the GUI rotates through
Circuit, Gradients, and Policy. The second follows MSE, TD loss, discounted
reward, learning rate, and epsilon during the same 10,000-episode run.
Structurally zero gradients are excluded from aggregate statistics.

## Quick start

```bash
cd vqc-rl
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./gui.py
```

Tk is required by the GUI. For command-line training, run
`.venv/bin/python -m quantum_rl --help`.

## GUI and saved runs

The GUI visualizes the circuit, parameter trajectories, gradients, policy,
state utilities, rewards, loss, and Bellman-reference error. Each run stores
its configuration and reference values alongside progress, policies, plots,
parameter history, and gradient statistics.

## Reproducibility and original implementation

Notebook-compatibility mode is verified bit-for-bit against the original implementation. The original code had a bug: only for the VQC implementation used `Q(s_f,a)` instead of 
`Q(s_i,a)` in the predicted value in the training loop (the classical training loop was not affected by this bug). In this GUI, this bug is fixed.  The original notebooks are retained in [original/](original/). for historical attribution to the original internship implementation.
