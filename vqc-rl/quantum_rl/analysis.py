"""Analysis helpers ported from StochFrozenLake_QUANTUM.ipynb.

Origin cells
------------
* notebook cell 36  -> ``action_name``            (``My_action``)
* notebook cell 50  -> ``mse``                    (the ``MAE_value`` accumulator)
* notebook cell 56  -> ``restrict`` / ``correct_indexes`` / ``greedy_from_q_table``
                       (``U_PLOT_DEEP`` + the hardcoded ``index_CORRECT``)
* classical notebook cells 41/47/93 -> ``residuals``
  (the quantum notebook only keeps the MSE; the ``avg +- std`` residual
  statistic lives in the classical companion notebook and uses exactly the
  arithmetic reproduced below.)

This module is deliberately **numpy-only**: no torch, no pennylane.  Values
coming from the VQC are duck-typed through :func:`_as_float`, which understands
anything exposing ``.item()`` (torch tensors, numpy scalars) as well as plain
floats.  That keeps the analysis layer importable and testable in
milliseconds.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "ACTION_NAMES",
    "action_name",
    "correct_indexes",
    "restrict",
    "residuals",
    "mse",
    "greedy_from_q_table",
]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _as_float(x: Any) -> float:
    """Convert a scalar-ish value to a python float.

    Accepts python numbers, numpy scalars and 0-d torch tensors without
    importing torch (we only rely on the ``.item()`` duck-type).
    """
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:  # pragma: no cover - non scalar tensors/arrays
            pass
    return float(x)


def _index_set(values: Iterable[Any]) -> set:
    """Normalise a container of state indexes (possibly numpy ints) to a set of int."""
    if values is None:
        return set()
    return {int(v) for v in values}


# --------------------------------------------------------------------------- #
# notebook cell 36 -- My_action
# --------------------------------------------------------------------------- #
#: action index -> human readable name.  Same mapping as the notebook.
ACTION_NAMES = ("up", "right", "down", "left")


def action_name(a: Any) -> str:
    """notebook cell 36 (``My_action``): convert 0,1,2,3 to up/right/down/left.

    The notebook returns ``None`` for anything outside 0..3 (it is a bare
    if/elif chain with no ``else``).  We keep that behaviour but make it
    explicit instead of accidental.
    """
    idx = int(a)
    if 0 <= idx < len(ACTION_NAMES):
        return ACTION_NAMES[idx]
    return None  # notebook: falls off the end of the if/elif chain


# --------------------------------------------------------------------------- #
# notebook cell 56 -- index_CORRECT
# --------------------------------------------------------------------------- #
#: what ``correct_indexes`` must return for the default 4x3 grid world.
DEFAULT_CORRECT_INDEXES = [0, 1, 2, 3, 4, 6, 8, 9, 10]


def correct_indexes(env) -> List[int]:
    """States kept for the utility analysis: non-terminal and non-obstacle.

    notebook cell 56 (and classical notebook cell 41) hardcode this list as::

        index_CORRECT = [0, 1, 2, 3, 4, 6, 8, 9, 10]

    and build the matching utility vector with the hardcoded slices
    ``U[0:5] + [U[6]] + U[8:11]``.

    DELIBERATE DEVIATION FROM THE NOTEBOOK
    --------------------------------------
    We *derive* the list from the environment (drop alive / death / obstacle
    states) instead of hardcoding it.  The hardcoded slices silently produce
    garbage the moment the grid, the goal or the obstacle move -- they encode
    "state 5 is the obstacle, 7 is death, 11 is the goal" in three separate
    literal slice bounds.  For the default configuration the derived list is
    byte-identical to the notebook's, and we assert exactly that below, so the
    deviation cannot change any reproduced number.
    """
    n_states = int(env.n_states)
    excluded = (
        _index_set(env.alive_indexes)
        | _index_set(env.death_indexes)
        | _index_set(env.obstacle_indexes)
    )
    idx = [i for i in range(n_states) if i not in excluded]

    # Guard-rail for the deviation above: on the default 4x3 board the derived
    # list MUST reproduce the notebook's hardcoded index_CORRECT.
    if n_states == 12 and excluded == {5, 7, 11}:
        assert idx == DEFAULT_CORRECT_INDEXES, (
            "correct_indexes() diverged from the notebook's hardcoded "
            f"index_CORRECT: got {idx}, expected {DEFAULT_CORRECT_INDEXES}"
        )
    return idx


def restrict(arr: Sequence[Any], env) -> List[float]:
    """Restrict a per-state array to :func:`correct_indexes`.

    notebook cell 56::

        U_PLOT_DEEP = []
        for s in range(Nx*Ny):
          if (s not in ALIVE_indexes) and (s not in DEATH_indexes) and (s not in OBSTACLES_indexes):
            U_PLOT_DEEP.append(U_DEEP[s])

    (classical notebook cell 41 writes the same thing as
    ``U[0:5] + [U[6]] + U[8:11]``.)
    """
    return [_as_float(arr[i]) for i in correct_indexes(env)]


# --------------------------------------------------------------------------- #
# residual statistics
# --------------------------------------------------------------------------- #
def residuals(pred_U: Sequence[Any], ref_U: Sequence[Any], env) -> Tuple[float, float, List[float]]:
    """``(avg, std, residuals_list)`` of ``pred_U - ref_U`` over the correct states.

    Faithful port of the classical notebook's residual block (cells 41 / 47 /
    93), which the report quotes as ``avg +- std``::

        std_X = 0.0 ; avg_X = 0.0 ; residuals_X = []
        for j in range(12):
          if (j not in OBSTACLES) and (j not in ALIVE) and (j not in DEATH):
            std_X += (plot_U_X[j] - U_final[j])**2
            avg_X += plot_U_X[j] - U_final[j]
            residuals_X.append(plot_U_X[j] - U_final[j])
        std_X = np.sqrt(std_X/9.0)
        avg_X = avg_X/9.0

    CAVEAT (ported faithfully, not fixed)
    -------------------------------------
    What the notebook calls ``std`` is **not** a standard deviation: it is
    ``sqrt(sum(d**2)/n)``, i.e. an RMS of the residuals about ZERO rather than
    about their mean, and it divides by the count ``n`` rather than ``n-1``.
    ``avg`` and ``std`` are therefore not independent (``std >= |avg|`` always),
    and the error bar printed as "avg +- std" in the report is an RMS
    residual.  We keep the arithmetic exactly as-is because every published
    number depends on it; a corrected sample std would be
    ``sqrt(sum((d-avg)**2)/(n-1))``.

    The only generalisation: the notebook's hardcoded ``/9.0`` becomes
    ``/len(correct_indexes(env))``, which *is* 9 for the default 4x3 board.
    """
    idx = correct_indexes(env)
    n = len(idx)
    if n == 0:
        return 0.0, 0.0, []

    res: List[float] = []
    std_acc = 0.0
    avg_acc = 0.0
    for j in idx:
        d = _as_float(pred_U[j]) - _as_float(ref_U[j])
        std_acc += d * d          # notebook: std_X += (...)**2
        avg_acc += d              # notebook: avg_X += (...)
        res.append(d)

    std = math.sqrt(std_acc / n)  # notebook: np.sqrt(std_X/9.0)  -- RMS about 0
    avg = avg_acc / n             # notebook: avg_X/9.0
    return avg, std, res


def mse(pred_U: Sequence[Any], ref_U: Sequence[Any], env) -> float:
    """Mean squared error of the utilities over the correct states.

    notebook cell 50 (inside the training loop, the variable is misleadingly
    called ``MAE_value`` -- cell 53 even carries the comment
    "#MAE IN REALITY IS MSE")::

        MAE_value = 0.0
        for i in range(12):
          if (i not in ALIVE_indexes) and (i not in DEATH_indexes) and (i not in OBSTACLES_indexes):
            U_tmp = torch.max(torch.tensor(variational_classifier(...)))
            MAE_value += (U_tmp - U_final[i].item())**2/9.0

    Note the division happens per term (``**2/9.0``) rather than once on the
    sum; we keep that so the floating point rounding matches bit-for-bit.
    """
    idx = correct_indexes(env)
    n = len(idx)
    if n == 0:
        return 0.0

    total = 0.0
    for i in idx:
        d = _as_float(pred_U[i]) - _as_float(ref_U[i])
        total += (d * d) / n      # notebook divides each term by 9.0
    return total


# --------------------------------------------------------------------------- #
# notebook cell 56 -- greedy policy / utility out of a Q table
# --------------------------------------------------------------------------- #
def greedy_from_q_table(q_table: Sequence[Sequence[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """``(policy, U)`` from an ``(n_states, n_actions)`` table of Q values.

    notebook cell 56::

        for s in range(Nx*Ny):
          action = torch.argmax(torch.tensor(Q(s, var_Q_circuit)))
          policy_DEEP[s] = action
          U_DEEP[s] = torch.max(torch.tensor(Q(s, var_Q_circuit))).item()

    The notebook evaluates the circuit inside the loop; here the caller passes
    the already-evaluated Q values so this stays torch-free.  ``policy`` is a
    float array, exactly like the notebook's ``np.zeros(Nx*Ny)``, because
    :func:`quantum_rl.plotting.plot_policy` compares it against 0/1/2/3.
    """
    q = np.asarray([[_as_float(v) for v in row] for row in q_table], dtype=float)
    policy = np.zeros(q.shape[0], dtype=float)
    U = np.zeros(q.shape[0], dtype=float)
    for s in range(q.shape[0]):
        policy[s] = float(np.argmax(q[s]))
        U[s] = float(np.max(q[s]))
    return policy, U
