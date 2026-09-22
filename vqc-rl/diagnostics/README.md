# Why the quantum network was not training

**Conclusion: the variational circuit is innocent. The training loop had a bug on a
single variable.**

## The experiment

The question "is it the circuit or the loop?" is answered by removing the circuit. In
`loop_ablation.py` the training loop of cell 50 of `StochFrozenLake_QUANTUM.ipynb` is
reproduced verbatim — replay memory, target network synced every `C=200` steps, sampling
with replacement, epsilon decay, terminal handling — but the variational circuit is
replaced by an **exact Q table**.

A table is the perfect approximator: 48 free parameters for 48 values, no expressivity
limit, no barren plateau, no simulation noise, exact gradients. If the loop is sound, the
tabular version must converge to Bellman. If it fails even then, the logic is to blame.

```
python3 diagnostics/loop_ablation.py
```

## Result (1500 episodes × 60 steps, averaged over 5 seeds)

γ = 0.95 — **the value used by the quantum notebook** (cell 11); Bellman: mean `U = 0.7260`

| scenario | MSE vs U | mean V | policy /9 | action spread |
|---|---|---|---|---|
| A — notebook as-is | 0.6177 | −0.0425 | 3.2 | **0.0016** |
| B — epsilon fixed only | 0.6171 | −0.0423 | 3.4 | 0.0016 |
| C — **pred-state fixed only** | **0.0050** | **0.7618** | 6.0 | 0.3606 |
| D — both fixed | 0.0024 | 0.7391 | 7.6 | 0.3561 |

γ = 0.99 — the classical notebook's value (cell 5); Bellman: mean `U = 0.9031`

| scenario | MSE vs U | mean V | policy /9 | action spread |
|---|---|---|---|---|
| A — notebook as-is | 0.9826 | −0.0815 | 2.2 | **0.0025** |
| B — epsilon fixed only | 0.9763 | −0.0788 | 2.4 | 0.0025 |
| C — **pred-state fixed only** | **0.0007** | **0.9106** | 8.0 | 0.3516 |
| D — both fixed | 0.0002 | 0.9033 | 8.4 | 0.3637 |

The loop fails even with the exact approximator, at both discount factors:
**this is not a quantum problem.**

> **Note on the two γ.** The notebooks use different discount factors — `Gamma=0.95` in
> the quantum one, `Gamma=0.99` in the classical one — so their respective Bellman
> `U_final` are not comparable with each other. The ablation runs on both to show the
> diagnosis does not depend on that choice.

## The cause: `pred` indexed on the wrong state

Cell 50 of the quantum notebook:

```python
y    = reward_j + Gamma * max(variational_classifier(var_target_Q_circuit,
                                 angles=decimalToBinaryFixLength(4, s_jf)))
pred = variational_classifier(var_Q_circuit,
                              angles=decimalToBinaryFixLength(4, s_jf))[action_j]
#                                                                 ^^^^
```

`pred` and `y` are **both computed on `s_jf`**, the arrival state. The Bellman update
instead requires `Q(s, a)` on the **departure** state. The classical notebook does it
correctly — `pred = Q(phi_j)[action_j]` — which is exactly why the classical DQN converged
and the quantum version did not.

**Why it is devastating.** The update becomes

```
Q(s', a_j)  ←  R[s'] + γ · max_a Q_target(s', a)
```

where `a_j` is the action taken in the *previous* state, hence effectively random with
respect to `s'`. Over many samples, all four actions of `s'` are driven towards the *same*
target. The fixed point is

```
q = R[s'] + γ·q   ⟹   q = R[s']/(1−γ)
```

identical for every action: −0.08 at γ=0.95, −0.40 at γ=0.99. Q stops telling actions
apart, `argmax` becomes arbitrary and the policy collapses.

The **spread** column measures it directly: `0.0016` against `0.36` once fixed, a factor of
220. That is the difference between "every action has the same value" and an informative Q.
Consistently, `policy 3.2/9` is compatible with pure chance (9 × ¼ = 2.25), and the mean V
converges to the predicted negative value instead of `U ≈ 0.73`.

The reward signal never propagates back along the trajectory: it is rewritten into the
state it came from.

## The second bug: inverted epsilon-greedy

Cell 45:

```python
if train or np.random.rand() < epsilon:
    action = torch.argmax(...)   # GREEDY with probability epsilon
else:
    action = np.random.randint(0, 4)
```

It is greedy *with* probability `epsilon`, not `1−epsilon`. With `epsilon` starting at 0.95
and decaying to 0.05, the agent **explores 5% of the time at the start and 95% at the end**:
it explores when it should exploit and exploits when it should explore, and the replay
memory fills with random trajectories exactly when it should be refining.

On its own it does not explain the failure — scenario B is indistinguishable from A,
because the `pred` bug dominates and masks everything else — but it matters once the first
is fixed: from C to D the policy goes from 6.0 to 7.6 out of 9 (γ=0.95) and the MSE halves.

## Third problem: the learning rate collapses in 24 episodes

`StepLR(step_size=3, gamma=0.5)` is stepped **once per episode**. Simulating PyTorch's
exact scheduler:

| episode | learning rate |
|---|---|
| 0 | 0.2 |
| 10 | 0.025 |
| 24 | 0.00078 — frozen by the `lr > LR_MIN` guard |

Out of 20 000 episodes, **only 12 run at a lr ≥ LR_0/10**. The remaining 19 976 run at
7.8 × 10⁻⁴, i.e. 256× below the initial value. Training is effectively over after the first
25 episodes. It is not the cause of the policy collapse, but it makes recovery impossible.

## Fourth problem: the diagnostics were lying

```python
print(f"... LOSSavg20 = {np.mean(loss_games[:-20:]):.8f} ... "
      f"game_rewardavg20 = {np.mean(games_reward[:-20:]):.4f}")
```

`loss_games[:-20:]` is `loss_games[0:-20]` — **everything except** the last 20 episodes, not
the last 20. As history accumulates this mean becomes ever more inert and flattens any
recent change.

On top of that there are three bugs in the block-averaging of the plots (cell 54), found
during the port: the averaging arrays are allocated with one extra element, so **every loss
and reward curve dives to zero at the right edge**; the final partial bucket is divided by
the full bucket size, so it is underestimated; and the standard-deviation loop iterates
over `j` but indexes with `i`, a leftover variable from the preceding loop. The dashboard
used to judge training was not showing what was happening.

## Priority order

1. **`pred` on `s_j`** instead of `s_jf` — alone it takes the MSE from 0.62 to 0.005
   (×124 at γ=0.95).
2. **Invert the epsilon-greedy test** — takes the policy from 6.0 to 7.6 out of 9.
3. **Fix the scheduler** — a `step_size` on the order of thousands of episodes, or a
   continuous decay, so the lr stays useful throughout.
4. **Fix the averaging windows and the block-averaging** for honest diagnostics.

## What remains to be verified

The ablation proves the loop was broken, not that the 4-qubit, 2-layer circuit (24
parameters for 48 values) suffices to represent the solution. The next step is to rerun the
same ablation with the real VQC, through the `quantum_rl/` package, to separate out the
residual due to circuit capacity.

```
PYTHONPATH=. .venv/bin/python diagnostics/vqc_comparison.py --episodes 600 --max-time 40
```

## Note on structure

`loop_ablation.py` carries a minimal copy of the environment so that it runs with numpy
alone, independently of `quantum_rl/`.
