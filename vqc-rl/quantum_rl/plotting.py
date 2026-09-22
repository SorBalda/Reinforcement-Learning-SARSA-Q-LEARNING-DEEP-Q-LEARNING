"""Plotting helpers ported from StochFrozenLake_QUANTUM.ipynb.

Origin cells
------------
* notebook cell 27     -> :func:`plot_policy`
* notebook cell 37     -> :func:`plot_bellman_convergence`
* notebook cells 53/54/55 -> :func:`plot_training_curves`
* notebook cell 56     -> :func:`plot_utility_comparison`
* classical notebook cell 41 -> :func:`plot_residual_hist`

Differences from the notebook that apply to *every* function here
-----------------------------------------------------------------
1. The Agg backend is forced so the module works headless.  The notebook ran
   in Colab with the inline backend.
2. ``plt.show()`` is never called.
3. Output filenames are always explicit ``outpath`` arguments.  The notebook
   hardcodes them -- and ``plot_policy`` in particular builds ``name_file =
   title + ".pdf"`` and then *ignores it*, saving to the literal
   ``"plot_policy.pdf"``, so every policy plot of a session overwrites the
   previous one.  See the note in :func:`plot_policy`.
4. Every function returns its :class:`matplotlib.figure.Figure` and draws on
   explicit Axes objects rather than on pyplot's global "current figure".

This module never imports torch or pennylane: tensors are read through
:func:`quantum_rl.analysis._as_float`, which only needs the ``.item()``
duck-type.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: deviation from the notebook's inline backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import _as_float, correct_indexes, restrict  # noqa: E402

__all__ = [
    "plot_policy",
    "plot_training_curves",
    "plot_utility_comparison",
    "plot_residual_hist",
    "plot_bellman_convergence",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _save(fig, outpath: Optional[str]):
    """Save ``fig`` to ``outpath`` if one was given (creating parent dirs)."""
    if not outpath:
        return
    parent = os.path.dirname(os.path.abspath(outpath))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(outpath)


def _grid_shape(env) -> Tuple[int, int]:
    """``(Nx, Ny)`` of the grid world.

    A grid world object does not have to expose Nx/Ny directly,
    so we look in three places, in order:
      1. ``env.Nx`` / ``env.Ny``
      2. ``env.cfg.Nx`` / ``env.cfg.Ny``
      3. derived from ``neigh_dist``: column 0 ("up") stores the offset ``+Nx``
         for every state that is free to move up, so its maximum is Nx.
    """
    nx = getattr(env, "Nx", None)
    ny = getattr(env, "Ny", None)
    if nx is None or ny is None:
        cfg = getattr(env, "cfg", None)
        if cfg is not None:
            nx = nx if nx is not None else getattr(cfg, "Nx", None)
            ny = ny if ny is not None else getattr(cfg, "Ny", None)

    n_states = int(env.n_states)
    if nx is None:
        neigh = np.asarray(getattr(env, "neigh_dist", np.zeros((n_states, 4))))
        derived = int(np.max(neigh[:, 0])) if neigh.size else 0
        nx = derived if derived > 0 else n_states
    nx = int(nx)
    if ny is None:
        ny = n_states // nx if nx else 1
    return nx, int(ny)


def _cell_reward_text(env, i: int) -> str:
    """Text written inside cell ``i``.

    notebook cell 27 writes ``str(r)`` -- the *global* living reward constant --
    in every free cell.  ``env.R[i] == r`` for exactly those cells, so reading
    it from the environment reproduces the same string while staying correct if
    the reward map ever becomes non-uniform.
    """
    R = getattr(env, "R", None)
    if R is None:
        cfg = getattr(env, "cfg", None)
        return str(getattr(cfg, "r", "")) if cfg is not None else ""
    return str(_as_float(R[i]))


# --------------------------------------------------------------------------- #
# notebook cell 27 -- plot_policy
# --------------------------------------------------------------------------- #
def plot_policy(policy, env, title: str = "my_policy", outpath: Optional[str] = None):
    """Draw a policy on the grid.  Faithful port of notebook cell 27.

    Visual semantics kept identical to the notebook:
      * ``figsize=(6, 5)``, limits ``[-0.5, Nx-0.5] x [-0.5, Ny-0.5]``, no ticks;
      * gray grid lines at every half-integer;
      * obstacle  -> black square marker, markersize 60;
      * goal      -> green star   marker, markersize 60;
      * death     -> red circle   marker, markersize 60;
      * every other cell gets the per-step reward as text at
        ``(x-0.4, y-0.15)`` plus one black arrow for its action
        (0=up, 1=right, 2=down, 3=left), ``head_width=head_length=0.1``;
      * the state index maps to ``x = i % Nx``, ``y = i // Nx``, so y grows
        upward and index 11 (the goal) sits in the top-right corner.

    DEVIATIONS
    ----------
    * ``outpath`` replaces the notebook's ``plt.savefig("plot_policy.pdf")``.
      The notebook computes ``name_file = title + ".pdf"`` and then never uses
      it, so *every* call clobbers the same ``plot_policy.pdf``; that is a bug,
      but it only affects which file is written, never a number, so fixing it
      here cannot change any reproduced result.  Passing ``outpath=None``
      (the default) writes nothing at all.
    * Returns the Figure instead of relying on the inline backend.

    ``policy`` may be a float array (the notebook's ``np.zeros(12)``); the
    action test is a plain equality against 0/1/2/3, exactly as in cell 27, so
    a non-integral entry simply draws no arrow.
    """
    Nx, Ny = _grid_shape(env)
    alive = {int(v) for v in env.alive_indexes}
    death = {int(v) for v in env.death_indexes}
    obstacles = {int(v) for v in env.obstacle_indexes}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-0.5, Nx - 0.5)
    ax.set_ylim(-0.5, Ny - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # build the grid
    for i in range(0, Ny + 1):
        ax.axhline(i - 0.5, color="gray")
    for i in range(0, Nx + 1):
        ax.axvline(i - 0.5, color="gray")

    # obstacles and winning / losing states
    for i in obstacles:
        ax.plot(i % Nx, int(i / Nx), marker="s", markersize=60, color="black")
    for i in alive:
        ax.plot(i % Nx, int(i / Nx), marker="*", markersize=60, color="green")
    for i in death:
        ax.plot(i % Nx, int(i / Nx), marker="o", markersize=60, color="red")

    for i in range(len(policy)):
        if (i in alive) or (i in death) or (i in obstacles):
            continue
        x, y = i % Nx, int(i / Nx)
        ax.text(x - 0.4, y - 0.15, _cell_reward_text(env, i))

        a = policy[i]
        if a == 0:  # up
            ax.arrow(x, y - 0.3, 0, +0.6, head_width=0.1, head_length=0.1, fc="black", ec="black")
        if a == 1:  # right
            ax.arrow(x - 0.3, y, +0.6, 0.0, head_width=0.1, head_length=0.1, fc="black", ec="black")
        if a == 2:  # down
            ax.arrow(x, y + 0.3, 0, -0.6, head_width=0.1, head_length=0.1, fc="black", ec="black")
        if a == 3:  # left
            ax.arrow(x + 0.3, y, -0.6, 0.0, head_width=0.1, head_length=0.1, fc="black", ec="black")

    ax.set_title(title)
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------- #
# notebook cell 37 -- Bellman convergence
# --------------------------------------------------------------------------- #
def plot_bellman_convergence(U_time, env, title: str = "Bellman convergences",
                             outpath: Optional[str] = None):
    """U(state) against the value-iteration step.  notebook cell 37::

        for j in range(12):
          if (j not in OBSTACLES_indexes) and (j not in ALIVE_indexes) and (j not in DEATH_indexes):
            position_str = "(" + str(j%Nx) + "," + str(int(j/Nx)) + ")"
            plt.plot(U_time[:,j], label=position_str)

    Same curves, same "(x,y)" labels.  ``outpath`` replaces the hardcoded
    ``plt.savefig("Bellman convergences.pdf")``.
    """
    Nx, _Ny = _grid_shape(env)
    U_time = np.asarray(U_time)

    fig, ax = plt.subplots()
    for j in correct_indexes(env):
        position_str = "(" + str(j % Nx) + "," + str(int(j / Nx)) + ")"
        ax.plot(U_time[:, j], label=position_str)
    ax.set_xlabel("number of steps")
    ax.set_ylabel("U")
    ax.set_title(title)
    ax.legend()
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------- #
# notebook cell 54 -- block averaging of the reward / loss traces
# --------------------------------------------------------------------------- #
def _block_means(values: Sequence[Any], jump: int, count: int = 0):
    """Faithful port of the block-average loops of notebook cell 54.

    Returns ``(x, means, count)`` where ``count`` is the loop counter *as the
    notebook leaves it* -- cell 54 never resets ``count`` between the reward
    loop, the (dead) std loop and the loss loop, so the loss buckets are
    offset by ``len(games_reward) % jump`` steps.  We thread that state
    through explicitly so the offset is reproduced instead of accidental.

    Two further notebook bugs are reproduced here:
      * ``x`` has ``n_blocks + 1`` points (``range(len(v[::jump]) + 1)``) and
        ``means`` is allocated to match, so the final entry (and the tail of a
        partial bucket) stays at 0 and the plotted curve drops to zero at the
        right edge;
      * a partial final bucket is still divided by the full ``jump``, so it is
        biased low.
    """
    values = list(values)
    n_blocks = len(values[::jump])                       # == ceil(n / jump)
    x = [i / n_blocks for i in range(n_blocks + 1)]      # notebook: +1 -> trailing zero
    means = np.zeros(len(x))

    index = 0
    for i in range(len(values)):
        count += 1
        if index < len(means):
            means[index] += _as_float(values[i]) / jump
        if count == jump:
            index += 1
            count = 0
    return x, means, count


def _block_stds(values: Sequence[Any], jump: int, means: np.ndarray, count: int = 0):
    """Faithful port of the std loop of notebook cell 54 (dead code, kept for ``count``).

    BUG PORTED AS-IS: the loop is ``for j in range(len(games_reward))`` but the
    body reads ``games_reward[i]`` -- ``i`` is the *stale* loop variable left
    over from the preceding mean loop, i.e. always the last sample.  The
    resulting "std" is therefore meaningless.  The notebook has the two lines
    that would plot it commented out, so nothing downstream depends on it; we
    keep it only because it advances ``count``, which *does* shift the loss
    buckets computed afterwards.
    """
    values = list(values)
    stds = np.zeros(len(means))
    stale_i = len(values) - 1  # the notebook's leftover `i`
    index = 0
    for _j in range(len(values)):
        count += 1
        if index < len(stds) and stale_i >= 0:
            d = _as_float(values[stale_i]) - means[index]
            stds[index] += d * d / jump
        if count == jump:
            if index < len(stds):
                stds[index] = np.sqrt(stds[index])
            index += 1
            count = 0
    return stds, count


# --------------------------------------------------------------------------- #
# notebook cells 53 / 54 / 55 -- training curves
# --------------------------------------------------------------------------- #
def plot_training_curves(result, outpath: Optional[str] = None, jump_reward: int = 20,
                         jump_loss: int = 20, show_reward_std: bool = False):
    """Reward / loss / MSE / Q-probe curves of a :class:`~quantum_rl.train.TrainResult`.

    Combines the three analysis cells into one figure with three panels:

    * panel 1 -- notebook cell 54: block-averaged discounted game reward
      (green, left axis) and block-averaged per-game loss (red, log scale,
      right twin axis), both against "epochs of training(relative)" in [0, 1].
      The notebook saved this one as ``LOSS_REWARD.pdf``.
    * panel 2 -- notebook cell 53: the MSE of U against the Bellman reference,
      black, x rescaled to [0, 1).  (The notebook's variable is ``MAE_U`` but
      its own comment says "#MAE IN REALITY IS MSE".)
    * panel 3 -- notebook cell 55: the two probe Q values, ``good_Q`` = Q(s=10,
      right) i.e. the cell left of the goal, and ``bad_Q`` = Q(s=3, up) i.e.
      the cell under the death state.  The notebook plots them with no labels;
      we label them because an unlabelled two-line plot is unreadable.

    ``show_reward_std`` draws the +-std band that cell 54 has commented out.
    It is off by default and the band is computed by the buggy notebook loop
    (see :func:`_block_stds`) -- do not trust it.

    Missing or empty fields degrade gracefully: the corresponding panel just
    says "no data" instead of raising.
    """
    games_reward = list(getattr(result, "games_reward", []) or [])
    loss_games = list(getattr(result, "loss_games", []) or [])
    mse_u = list(getattr(result, "mse_u", []) or [])
    good_Q = list(getattr(result, "good_Q", []) or [])
    bad_Q = list(getattr(result, "bad_Q", []) or [])

    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    ax_rl, ax_mse, ax_q = axes

    # ---- panel 1: reward + loss (notebook cell 54) ------------------------
    count = 0
    if games_reward:
        x_reward, reward_means, count = _block_means(games_reward, jump_reward, count)
        std_reward, count = _block_stds(games_reward, jump_reward, reward_means, count)
        ax_rl.plot(x_reward, reward_means, color="green")
        if show_reward_std:
            ax_rl.plot(x_reward, reward_means + std_reward, color="green", alpha=0.2)
            ax_rl.plot(x_reward, reward_means - std_reward, color="green", alpha=0.2)
    ax_rl.set_xlabel("epochs of training(relative)")
    ax_rl.set_ylabel("Rewards")
    ax_rl.tick_params(axis="y", labelcolor="tab:green")

    ax_loss = ax_rl.twinx()
    if loss_games:
        x_loss, loss_means, count = _block_means(loss_games, jump_loss, count)
        ax_loss.plot(x_loss, loss_means, marker="", linestyle="-", color="red", alpha=0.6)
        # log scale needs strictly positive data; the notebook always had it.
        if np.any(loss_means > 0):
            ax_loss.set_yscale("log")
    ax_loss.set_ylabel("loss (log scale)")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")

    # ---- panel 2: MSE (notebook cell 53) ----------------------------------
    if mse_u:
        y_mse = [_as_float(v) for v in mse_u]
        x_mse = [float(i) / len(y_mse) for i in range(len(y_mse))]
        ax_mse.plot(x_mse, y_mse, label="MSE", color="black")
        ax_mse.legend()
    else:
        ax_mse.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_mse.transAxes)
    ax_mse.set_ylabel("MSE")
    ax_mse.set_xlabel("epochs of training(relative)")

    # ---- panel 3: probe Q values (notebook cell 55) -----------------------
    if good_Q:
        ax_q.plot([_as_float(v) for v in good_Q], label="good Q (s=10, right)")
    if bad_Q:
        ax_q.plot([_as_float(v) for v in bad_Q], label="bad Q (s=3, up)")
    if good_Q or bad_Q:
        ax_q.legend()
    else:
        ax_q.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_q.transAxes)
    ax_q.set_xlabel("MSE evaluation index (one every 20 games)")
    ax_q.set_ylabel("Q value")

    fig.tight_layout()
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------- #
# notebook cell 56 -- utility comparison
# --------------------------------------------------------------------------- #
def plot_utility_comparison(pred_U, env, ref_U=None, label: str = "VQC",
                            ref_label: str = "Bellman", title: Optional[str] = None,
                            outpath: Optional[str] = None):
    """Learned utility vs the Bellman reference, over :func:`correct_indexes`.

    notebook cell 56 ends with ``plt.plot(index_CORRECT, U_PLOT_DEEP)``; the
    classical notebook (cell 41) overlays the Bellman curve with
    ``marker="*", linestyle=":"``.  Same markers here.
    """
    idx = correct_indexes(env)
    fig, ax = plt.subplots()
    ax.plot(idx, restrict(pred_U, env), marker="*", linestyle=":", label=label)
    if ref_U is not None:
        ax.plot(idx, restrict(ref_U, env), marker="*", linestyle=":",
                label=ref_label, color="gray")
    ax.set_xlabel("state index")
    ax.set_ylabel("Utility")
    ax.legend()
    if title:
        ax.set_title(title)
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------- #
# classical notebook cell 41 -- residual histogram
# --------------------------------------------------------------------------- #
def plot_residual_hist(residuals_list: Sequence[float], avg: float, std: float,
                       title: Optional[str] = None, outpath: Optional[str] = None):
    """Histogram of the residuals with the notebook's four hand-picked bin edges.

    classical notebook cell 41::

        plt.hist(residuals, bins=[avg-3*std, avg-std, avg+std, avg+3*std],
                 edgecolor='white', linewidth=5, color="black")
        plt.axvline(avg+std, color="red", label="std")
        plt.axvline(avg-std, color="red")

    CAVEAT (ported faithfully): with only three bins spanning +-3 std, any
    residual outside that window is silently dropped by ``plt.hist``, so the
    counts need not sum to 9.  And since ``std`` is an RMS about zero rather
    than about ``avg`` (see :func:`quantum_rl.analysis.residuals`), the bins
    are not the usual +-1 sigma / +-3 sigma bands.
    """
    edges = [avg - 3 * std, avg - std, avg + std, avg + 3 * std]
    fig, ax = plt.subplots()
    if std > 0:
        ax.hist(list(residuals_list), bins=edges, edgecolor="white",
                linewidth=5, color="black")
        ax.axvline(avg + std, color="red", label="std")
        ax.axvline(avg - std, color="red")
        ax.legend()
    else:
        # degenerate: all residuals identical -> the bin edges collapse.
        ax.hist(list(residuals_list), edgecolor="white", linewidth=5, color="black")
    ax.set_xlabel("residuals")
    ax.set_ylabel("counts")
    if title:
        ax.set_title(title)
    _save(fig, outpath)
    return fig
