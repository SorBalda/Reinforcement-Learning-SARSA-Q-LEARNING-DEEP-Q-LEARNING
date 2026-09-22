"""Value iteration and the greedy policy -- the reference baseline.

Faithful port of cells 32 and 34 of ``StochFrozenLake_QUANTUM.ipynb``.
Everything the VQC learns is scored against the ``U_final`` produced here.

Which array do I use?
---------------------
The notebook ends up with three different utility arrays and uses each of them
for a different purpose.  :func:`value_iteration_detailed` returns all three:

``U_time``   shape ``(max_epoch, n_states)``; ``U_time[k]`` is the iterate
             written during sweep ``k`` (row 0 is never written and stays 0).
``U_final``  ``U_time[max_epoch - 1]`` -- the *last* iterate.  This is the
             notebook's ``U_final`` and the one every residual/MSE is measured
             against.  It has ``U_final[goal] == U_final[death] == 0.0``
             because terminal states have no outgoing transition probability.
``U_policy`` the notebook's bare ``U``: the value of ``U_1`` as it was at the
             *start* of the last sweep (i.e. ``U_time[max_epoch - 2]``, one
             iterate behind ``U_final``), with the manual overrides
             ``U[goal] = +1.0`` and ``U[death] = -1.0`` applied afterwards.
             This -- not ``U_final`` -- is what cell 34 feeds to the greedy
             policy, and the overrides are what make the states next to the
             goal/pit prefer it.

:func:`value_iteration` returns the tuple ``(U_final, U_time)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FixFlags
from .environment import GridWorld

__all__ = [
    "BellmanResult",
    "value_iteration",
    "value_iteration_detailed",
    "greedy_policy",
]


@dataclass(frozen=True)
class BellmanResult:
    """The three utility arrays of notebook cell 32 (see the module docstring)."""

    U_final: np.ndarray  # U_time[max_epoch - 1]
    U_time: np.ndarray  # (max_epoch, n_states)
    U_policy: np.ndarray  # U_time[max_epoch - 2] + the manual +1/-1 override


# --------------------------------------------------------------------------
# notebook cell 32
# --------------------------------------------------------------------------
def value_iteration_detailed(
    env: GridWorld, gamma: float, max_epoch: int
) -> BellmanResult:
    """Jacobi-style value iteration, exactly as the notebook runs it.

    Per sweep: ``U = U_1.copy()`` is frozen first and every backup reads from
    that snapshot while writing into ``U_1`` (Jacobi, not Gauss-Seidel).

    NOTE (faithful): the loop runs ``range(max_epoch - 1)`` times, so for
    ``max_epoch=100`` there are only 99 sweeps and row 0 of ``U_time`` stays
    zero.  The backup is
    ``max_a sum_f P(f|i,a) * (gamma*U[f] + R[f])``, i.e. the reward is the
    reward *of the state reached*, taken inside the expectation.
    """
    if max_epoch < 1:
        raise ValueError(f"max_epoch must be >= 1 (got {max_epoch})")

    n = env.n_states
    U = np.zeros(n)
    U_1 = np.zeros(n)
    U_time = np.zeros((max_epoch, n))

    for epochs in range(max_epoch - 1):
        U = U_1.copy()
        for state_i in range(n):
            array_to_find_max = np.zeros(4)
            for action in range(4):
                for state_f in range(n):
                    array_to_find_max[action] += env.transition_probability(
                        state_f, state_i, action
                    ) * (gamma * U[state_f] + env.R[state_f])

            u_update = np.max(array_to_find_max)
            U_1[state_i] = u_update
            U_time[epochs + 1, state_i] = u_update

    # NOTE (faithful): the notebook hardcodes ``U[11] = +1.0`` / ``U[7] = -1.0``
    # AFTER the loop and applies them to ``U`` -- the snapshot taken at the top
    # of the last sweep -- and never to ``U_1`` nor to ``U_time``.  So the
    # overrides are invisible to ``U_final`` and only reach the greedy policy.
    U_policy = U.copy()
    for i in env.alive_indexes:
        U_policy[i] = +1.0
    for i in env.death_indexes:
        U_policy[i] = -1.0

    U_final = U_time[max_epoch - 1, :]
    return BellmanResult(U_final=U_final, U_time=U_time, U_policy=U_policy)


def value_iteration(
    env: GridWorld, gamma: float, max_epoch: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(U_final, U_time)`` -- the short form of :func:`value_iteration_detailed`.

    Use :func:`value_iteration_detailed` when you also need ``U_policy`` (the
    array the notebook's greedy policy is actually computed from).
    """
    res = value_iteration_detailed(env, gamma, max_epoch)
    return res.U_final, res.U_time


# --------------------------------------------------------------------------
# notebook cell 34
# --------------------------------------------------------------------------
def greedy_policy(
    env: GridWorld,
    U: np.ndarray,
    gamma: float,
    *,
    fixes: FixFlags | None = None,
) -> np.ndarray:
    """One-step-lookahead greedy policy, ``argmax_a sum_f P(f|i,a) * U[f]``.

    NOTE (faithful): cell 34 drops both the immediate reward and the discount
    -- it maximises ``sum_f P * U[f]`` where the Bellman backup of cell 32
    maximises ``sum_f P * (gamma*U[f] + R[f])``.  With the notebook's
    ``U_policy`` (whose terminal entries are overridden to +-1) the two happen
    to agree on this grid, but they are not the same operator, and ``gamma``
    is therefore UNUSED unless the fix is enabled.

    Set ``fixes=FixFlags(greedy_policy_ignores_reward=False)`` to use the
    consistent ``sum_f P * (gamma*U[f] + R[f])`` instead.

    Ties are broken by ``np.argmax``, i.e. lowest action index wins -- which is
    why fully blocked/terminal states come out as action 0 ("up").
    """
    if fixes is None:
        fixes = env.cfg.fixes

    n = env.n_states
    policy = np.zeros(n)  # float, like the notebook
    for i in range(n):
        my_sum = np.zeros(4)
        for action in range(0, 4):
            for j in range(n):
                p = env.transition_probability(j, i, action)
                if fixes.greedy_policy_ignores_reward:
                    my_sum[action] += p * U[j]  # notebook
                else:
                    my_sum[action] += p * (gamma * U[j] + env.R[j])
        policy[i] = np.argmax(my_sum)
    return policy
