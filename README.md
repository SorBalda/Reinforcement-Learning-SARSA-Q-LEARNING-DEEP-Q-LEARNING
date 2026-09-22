# Variational quantum circuit for Q-learning on Frozen Lake Environment

Reinforcement learning on the stochastic 4×3 grid world of Russell & Norvig,
*Artificial Intelligence: A Modern Approach* (chapter 17), with the Q-function
approximated by a **4-qubit variational quantum circuit (VQC)**. Exact value
iteration (Bellman) is the ground truth every method is scored against.

The work started as a summer internship at the Physics Department of the
University of Porto (July 2024, supervisors Duarte Magano and Ariel Guerreiro).

## Repository layout

| folder | what is inside |
| --- | --- |
| [`original/`](original/) | The code written during the internship: two Jupyter notebooks (classical RL and the quantum attempt) and the technical report. Kept exactly as it was. |
| [`vqc-rl/`](vqc-rl/) | The same quantum agent rewritten as a tested Python package, with the bugs of the original training loop found and fixed, a command-line interface and a GUI to train and inspect the circuit live. |

## The problem

```
 y=2 |  8  |  9  | 10  | 11 (+1)
 y=1 |  4  |  5  |  6  |  7 (−1)
     |     |wall |     |
 y=0 |  0  |  1  |  2  |  3
```

Twelve states, four actions. The intended move succeeds with probability 0.8 and
slips to each perpendicular direction with probability 0.1; every step costs a
small living reward `r = −0.004`.

In the quantum agent the state index is encoded in 4 bits, uploaded again before
every layer (`RX(πx)·RZ(πx)` per qubit), followed by a CNOT chain and one
`Rot(φ, θ, ω)` per qubit. The four `⟨Z⟩` expectation values, scaled by 3, are the
Q-values of the four actions.

## What the rewrite found

The original quantum notebook did not learn. The rewrite reproduces it bit for bit
(checked by a test against a line-by-line transcription of the training cell) and
puts each suspicious behaviour behind a switch, which made the causes measurable:

1. **The TD prediction used the wrong state.** `Q(s′, a)` was regressed on a
   target built from `s′` itself, so all four actions of a state converged to the
   same value and the policy collapsed. Replacing the circuit with an exact Q table
   fails in exactly the same way: the problem was the loop, not the circuit.
2. **Epsilon-greedy was inverted**: 5% exploration at the start, 95% at the end.
3. **The agent trained on a deterministic grid** while being scored against the
   stochastic one.
4. **The learning rate collapsed** within the first 24 episodes.

With the fixes and a smaller learning rate than the notebook's 0.2, the circuit does
learn. A 6-layer VQC trained for 8000 episodes (lr 0.1) reaches an MSE of 0.002
against the Bellman utilities and picks the optimal action in 8 of the 9 playable
states. At lr 0.2 the same runs do not converge. The loop analysis and its numbers
are in [`vqc-rl/diagnostics/`](vqc-rl/diagnostics/README.md).

The live gradient view also shows that some parameters can never learn: the `φ` of
the first layer (a phase on a computational-basis state) and the `ω` of the last
layer (it commutes with the `Z` measurement). With `L` layers only `12L − 8` of the
`12L` angles affect the output.

## Quick start

```bash
cd vqc-rl
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./gui.py                                   # graphical interface
.venv/bin/python -m quantum_rl --help      # command line
```

See [`vqc-rl/README.md`](vqc-rl/README.md) for the full documentation.
