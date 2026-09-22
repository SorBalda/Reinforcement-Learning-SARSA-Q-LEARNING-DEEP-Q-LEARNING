"""The 4x3 stochastic grid world (AIMA ch. 17).

Faithful port of cells 13, 16, 17, 19, 21, 23/25, 24 and 30 of
``StochFrozenLake_QUANTUM.ipynb``.  Every function carries the cell it comes
from.  Where the notebook used module-level globals (``Nx``, ``neigh_dist``,
``R``, ``ALIVE_indexes`` ...) the state now lives on :class:`GridWorld`, which
is the only behavioural restructuring: the arithmetic is unchanged.

State indexing (cell 16)::

    i = x + Nx * y          x = i % Nx,  y = i // Nx

    8   9  10  11(+1)
    4   5*  6   7(-1)       5* = obstacle
    0   1   2   3

Actions: ``0=up, 1=right, 2=down, 3=left``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import Config

__all__ = ["prob_action", "GridWorld"]


# --------------------------------------------------------------------------
# rng plumbing
# --------------------------------------------------------------------------
# The notebook calls the legacy module-level functions (``np.random.rand``).
# Passing ``rng=None`` keeps exactly that behaviour, so seeding through
# ``np.random.seed`` / ``quantum_rl.config.apply_seed`` reproduces notebook
# runs.  Passing a ``np.random.Generator`` gives an isolated, modern stream.
def _rand(rng: Optional[np.random.Generator]) -> float:
    """Uniform draw in [0, 1), from the legacy global RNG when ``rng`` is None."""
    if rng is None:
        return float(np.random.rand())
    return float(rng.random())


# --------------------------------------------------------------------------
# notebook cell 21
# --------------------------------------------------------------------------
def prob_action(action: int) -> tuple[list[float], list[int]]:
    """AIMA slip model: 0.8 intended, 0.1 on each perpendicular direction.

    Returns ``(probabilities, ranked_action_indexes)`` where

    * ``probabilities[d]`` is P(move in direction ``d``) for ``d`` in
      ``0=up, 1=right, 2=down, 3=left``;
    * ``ranked_action_indexes`` is ``[most_probable, second, third]``.

    e.g. ``prob_action(0) == ([0.8, 0.1, 0.0, 0.1], [0, 1, 3])``.

    The notebook returned a 2-element *list*; a tuple is returned here --
    unpacking, the only thing the notebook ever did with the result, is
    identical.
    """
    if action == 0:  # up
        return [0.8, 0.1, 0.0, 0.1], [0, 1, 3]
    if action == 1:  # right
        return [0.1, 0.8, 0.1, 0.0], [1, 0, 2]
    if action == 2:  # down
        return [0.0, 0.1, 0.8, 0.1], [2, 1, 3]
    if action == 3:  # left
        return [0.1, 0.0, 0.1, 0.8], [3, 0, 2]
    # DEVIATION: the notebook fell through and returned ``None`` here, which
    # blew up later with an opaque ``TypeError``.  Unreachable for actions 0-3.
    raise ValueError(f"action must be one of 0,1,2,3 (got {action!r})")


class GridWorld:
    """The environment, with the notebook's globals turned into attributes.

    Attributes
    ----------
    n_states : int
    alive_indexes, death_indexes, obstacle_indexes : list[int]
    neigh_dist : np.ndarray, shape (n_states, 4)
        OFFSET table, not a neighbour list: the successor of ``s`` under
        direction ``d`` is ``s + neigh_dist[s, d]``.  An offset of ``0`` means
        "blocked" (grid edge, obstacle, or ``s`` is terminal/obstacle) i.e.
        stay put.  Kept as float, like the notebook, because
        ``transition_probability`` relies on ``int(neigh_dist[...])``.
    R : np.ndarray, shape (n_states,)
        Reward *of arriving in* a state.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.Nx = int(cfg.Nx)
        self.Ny = int(cfg.Ny)
        self.n_states = self.Nx * self.Ny
        self.n_actions = 4

        # ---- notebook cell 13 --------------------------------------------
        self.alive_positions = np.array([list(p) for p in cfg.alive_positions])
        self.death_positions = np.array([list(p) for p in cfg.death_positions])
        self.obstacle_positions = np.array([list(p) for p in cfg.obstacle_positions])

        # ---- notebook cell 17 --------------------------------------------
        self.alive_indexes = [self.position_to_index(p) for p in cfg.alive_positions]
        self.death_indexes = [self.position_to_index(p) for p in cfg.death_positions]
        self.obstacle_indexes = [
            self.position_to_index(p) for p in cfg.obstacle_positions
        ]

        self.neigh_dist = self._build_neigh_dist()
        self.R = self._build_rewards()

    # ------------------------------------------------------------------
    # notebook cell 16
    # ------------------------------------------------------------------
    def position_to_index(self, s: Sequence[int]) -> int:
        """``(x, y) -> x + Nx * y``."""
        x = s[0]
        y = s[1]
        return int(x + self.Nx * y)

    def index_to_position(self, index: int) -> tuple[int, int]:
        """``i -> (i % Nx, i // Nx)``.

        DEVIATION: returns a tuple; the notebook returned ``np.array([x, y])``.
        Nothing downstream did array maths on the result.
        """
        index = int(index)
        x = index % self.Nx
        y = int(index / self.Nx)
        return (x, y)

    # ------------------------------------------------------------------
    # notebook cell 19
    # ------------------------------------------------------------------
    def _build_neigh_dist(self) -> np.ndarray:
        Nx, Ny = self.Nx, self.Ny
        obstacles = self.obstacle_indexes
        neigh_dist = np.zeros((Nx * Ny, 4))
        for index in range(Nx * Ny):
            neigh_dist[index, 0] = Nx
            if index + Nx >= Nx * Ny or (index + Nx in obstacles):
                neigh_dist[index, 0] = 0

            neigh_dist[index, 1] = +1
            if (index + 1) % Nx == 0 or (index + 1 in obstacles):
                neigh_dist[index, 1] = 0

            neigh_dist[index, 2] = -Nx
            if (index - Nx) < 0 or (index - Nx in obstacles):
                neigh_dist[index, 2] = 0

            neigh_dist[index, 3] = -1
            if (index) % Nx == 0 or (index - 1 in obstacles):
                neigh_dist[index, 3] = 0

            if (
                (index in obstacles)
                or (index in self.alive_indexes)
                or (index in self.death_indexes)
            ):
                for neigh in range(0, 4):
                    neigh_dist[index, neigh] = 0
        return neigh_dist

    # ------------------------------------------------------------------
    # notebook cell 24
    # ------------------------------------------------------------------
    def _build_rewards(self) -> np.ndarray:
        R = np.zeros(self.Nx * self.Ny)
        for i in range(self.Nx * self.Ny):
            if i in self.alive_indexes:
                R[i] = +1.0
            elif i in self.death_indexes:
                R[i] = -1.0
            else:
                R[i] = self.cfg.r
            # NOTE (faithful): the obstacle's reward is explicitly zeroed, not
            # set to the living reward ``r``.  Combined with the self-loop of
            # ``transition_probability`` this pins U[obstacle] to exactly 0.
            if i in self.obstacle_indexes:
                R[i] = 0.0
        return R

    # ------------------------------------------------------------------
    # notebook cell 21 (method form)
    # ------------------------------------------------------------------
    def prob_action(self, a: int) -> tuple[list[float], list[int]]:
        return prob_action(a)

    # ------------------------------------------------------------------
    # notebook cell 25 (cell 23 redefined; cell 25 is the effective one)
    # ------------------------------------------------------------------
    def next_position(
        self,
        initial: int,
        action: int,
        rng: Optional[np.random.Generator] = None,
        *,
        stochastic: Optional[bool] = None,
        control: bool = False,
    ) -> int:
        """Sample the successor of ``initial`` under ``action``.

        Inverse-CDF sampling over the three possible outcomes, with the
        notebook's STRICT inequalities (see the fallback note below).

        ``stochastic``
            ``None`` (default) resolves from the config:
            ``not cfg.fixes.env_deterministic_transitions``.  With the default
            flags that is ``False`` == the notebook: cell 25 declares
            ``stochastic=False`` and the training loop of cell 50 calls
            ``Next_position(s_1, a)`` without the keyword, so the sampled
            outcome is discarded and replaced by the deterministic intended
            move.  Set ``cfg.fixes.env_deterministic_transitions = False`` (or
            pass ``stochastic=True``) to actually slip.

        The uniform draw happens unconditionally, *before* the deterministic
        override, exactly as in the notebook -- the RNG stream must be
        consumed identically for seeded runs to match.
        """
        initial = int(initial)
        if stochastic is None:
            stochastic = not self.cfg.fixes.env_deterministic_transitions

        prob, action_indexes = prob_action(action)
        rnd = _rand(rng)
        if control:
            print(
                rnd,
                prob[action_indexes[0]],
                self.neigh_dist[initial, action_indexes[0]],
                prob[action_indexes[0]],
                prob[action_indexes[0]] + prob[action_indexes[1]],
            )

        final = None
        if rnd < prob[action_indexes[0]]:
            final = initial + self.neigh_dist[initial, action_indexes[0]]

        if (
            rnd > prob[action_indexes[0]]
            and rnd < prob[action_indexes[0]] + prob[action_indexes[1]]
        ):
            final = initial + self.neigh_dist[initial, action_indexes[1]]

        if rnd > prob[action_indexes[1]] + prob[action_indexes[0]]:
            final = initial + self.neigh_dist[initial, action_indexes[2]]

        if not stochastic:
            final = initial + self.neigh_dist[initial, action_indexes[0]]

        if final is None:
            # NOTE (latent notebook bug): the three tests above are all STRICT,
            # so the two exact boundary values ``rnd == p0`` and
            # ``rnd == p0 + p1`` are assigned to no branch at all and the
            # notebook raised UnboundLocalError.  With p = (0.8, 0.1, 0.1) this
            # needs a draw exactly equal to a double boundary (probability
            # ~2^-53 per step), hence "unreachable in practice".  The fallback
            # resolves those two points the way a textbook inverse-CDF
            # (``rnd < cumulative``) would, so it cannot change the sampled
            # distribution of any interior value.
            rank = 1 if rnd < prob[action_indexes[0]] + prob[action_indexes[1]] else 2
            final = initial + self.neigh_dist[initial, action_indexes[rank]]

        return int(final)

    # ------------------------------------------------------------------
    # notebook cell 30
    # ------------------------------------------------------------------
    def transition_probability(
        self, s_1: int, s_0: int, a: int, verbose: bool = False
    ) -> float:
        """Analytic ``P(s_1 | s_0, a)``.

        The four directions are accumulated with ``+=``: when several
        directions land on the same successor (typically blocked moves, whose
        offset is 0 and which therefore all "stay put") their probabilities
        ADD.  That is essential for the rows to sum to 1 -- do not rewrite it
        as an if/elif chain.

        NOTE (faithful): the result is forced to 0.0 when ``s_0`` is an ALIVE
        or DEATH state -- terminal states have no outgoing probability at all,
        which is what pins ``U[7] == U[11] == 0`` in value iteration.  The
        obstacle is NOT special-cased: all four of its offsets are 0, so it
        gets ``P(obstacle | obstacle, a) == 1.0``, an absorbing self-loop.
        """
        s_0 = int(s_0)
        s_1 = int(s_1)
        output = 0.0
        action_probabilities, _ = prob_action(a)
        if verbose:
            for j in range(0, 4):
                print(s_1, s_0 + int(self.neigh_dist[s_0, j]), s_0)

        if s_1 == s_0 + int(self.neigh_dist[s_0, 0]):
            if verbose:
                print("Control 0")
            output += action_probabilities[0]

        if s_1 == s_0 + int(self.neigh_dist[s_0, 1]):
            if verbose:
                print("Control 1")
            output += action_probabilities[1]

        if s_1 == s_0 + int(self.neigh_dist[s_0, 2]):
            if verbose:
                print("Control 2")
            output += action_probabilities[2]

        if s_1 == s_0 + int(self.neigh_dist[s_0, 3]):
            if verbose:
                print("Control 3")
            output += action_probabilities[3]

        if (s_0 in self.alive_indexes) or (s_0 in self.death_indexes):
            output = 0.0
        return output

    # ------------------------------------------------------------------
    # notebook cell 25 (second half)
    # ------------------------------------------------------------------
    def random_state(self, rng: Optional[np.random.Generator] = None) -> int:
        """Uniform non-terminal, non-obstacle start state.

        The notebook hardcoded ``int(np.random.rand() * 12)``; ``n_states`` is
        used here and equals 12 for the default config.  Rejection sampling
        (and therefore the number of RNG draws) is unchanged.
        """
        output = int(_rand(rng) * self.n_states)
        while (
            (output in self.obstacle_indexes)
            or (output in self.alive_indexes)
            or (output in self.death_indexes)
        ):
            output = int(_rand(rng) * self.n_states)
        return output

    # ------------------------------------------------------------------
    def is_terminal(self, i: int) -> bool:
        """True for ALIVE/DEATH states.

        Matches the notebook's episode-termination test
        ``(s in ALIVE_indexes) or (s in DEATH_indexes)`` -- the obstacle is
        deliberately NOT terminal there (it is absorbing instead, so an
        episode that walks into it can only end by running out of steps).
        """
        i = int(i)
        return (i in self.alive_indexes) or (i in self.death_indexes)

    # ------------------------------------------------------------------
    # convenience, not from the notebook
    # ------------------------------------------------------------------
    def transition_matrix(self) -> np.ndarray:
        """``P[a, s_0, s_1]`` built from :meth:`transition_probability`.

        Purely a helper for tests/analysis; the ported algorithms all call
        :meth:`transition_probability` directly, like the notebook does.
        """
        P = np.zeros((self.n_actions, self.n_states, self.n_states))
        for a in range(self.n_actions):
            for s0 in range(self.n_states):
                for s1 in range(self.n_states):
                    P[a, s0, s1] = self.transition_probability(s1, s0, a)
        return P
