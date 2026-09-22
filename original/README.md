# Original internship code (July 2024)

The notebooks and the report as written during the internship, unchanged.

| file | content |
| --- | --- |
| `Stochastic_frozen_lake_ClassicalReinforcementLearning.ipynb` | The stochastic 4×3 grid world, exact value iteration (Bellman), SARSA, tabular Q-learning and a deep Q-network (PyTorch), each compared with the Bellman solution. |
| `StochFrozenLake_QUANTUM.ipynb` | The same environment with the Q-values approximated by a 4-qubit variational quantum circuit (PennyLane). An exploratory attempt: it does not converge. |
| `Internship_REPORT.pdf` | The technical report of the internship. |

Why the quantum notebook does not converge, and the corrected version, are in
[`../vqc-rl/`](../vqc-rl/README.md).

## Running the notebooks

They were written in Google Colab. They need `numpy matplotlib torch` and, for the
quantum notebook, `pennylane` (its first cells install it). Cells must run top to
bottom: the notebooks share state through globals, and a few functions are
redefined further down.
