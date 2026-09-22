"""Configuration objects shared by every module of :mod:`quantum_rl`.

This package is a faithful port of ``StochFrozenLake_QUANTUM.ipynb`` (4x3
stochastic grid world, AIMA ch. 17, Q-values approximated by a 4-qubit
variational quantum circuit).

Design rule
-----------
Every known or suspected bug of the notebook is guarded by a boolean in
:class:`FixFlags`.  **The defaults are the fixes**; :meth:`FixFlags.notebook`
(or :meth:`Config.notebook`) turns every bug back on and reproduces the
notebook bit for bit, so any fix can be A/B'd against the original.

Nothing in this module imports torch or pennylane, so it stays cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FixFlags", "Config", "apply_seed"]


@dataclass(frozen=True)
class FixFlags:
    """``True`` always means "behave like the notebook", ``False`` means fixed.

    Since the root-cause analysis in ``diagnostics/`` the DEFAULTS ARE THE FIXES:
    a bare ``FixFlags()`` is the corrected agent.  Use :meth:`notebook` to get
    the original, bug-for-bug behaviour back -- it is still fully supported and
    still covered by the bit-for-bit parity test.
    """

    # ---- training-loop flags ---------------------------------------------
    # notebook cell 45: ``if train or np.random.rand() < epsilon:`` takes the
    # GREEDY branch when the draw is *below* epsilon, i.e. the exploration /
    # exploitation test is inverted w.r.t. the usual convention.
    # Measured: the agent explores 5% of the time at the start of training and
    # 95% at the end -- exactly backwards.  See diagnostics/README.md.
    epsilon_greedy_inverted: bool = False

    # notebook cell 50: the TD prediction is
    # ``variational_classifier(var_Q_circuit, angles=decimalToBinaryFixLength(4, s_jf))[action_j]``
    # i.e. it is indexed on the *next* state ``s_jf`` instead of the state the
    # action was taken from (``s_j``).  The classical DQN of the sibling
    # notebook uses ``Q(phi_j)``.
    # ROOT CAUSE of the training failure: with pred and target both indexed on
    # s_jf, every action of s' is regressed onto the same target, so Q stops
    # telling actions apart and the policy degenerates.  diagnostics/README.md
    td_pred_uses_next_state: bool = False

    # ---- flags added by the environment / bellman port -------------------
    # notebook cell 25: ``Next_position(initial, action, control=False,
    # stochastic=False)``.  The training loop (cell 50) calls it as
    # ``Next_position(s_1, a)``, i.e. with the default ``stochastic=False``,
    # and the ``if not stochastic:`` line then OVERWRITES the sampled outcome
    # with the deterministic intended move.  So the notebook trains the agent
    # on a *deterministic* grid world while scoring it against a Bellman
    # baseline computed for the *stochastic* one (``transition_probability``
    # always uses the 0.8/0.1/0.1 slip model).  True == notebook.
    # Was debugging scaffolding left in: the agent trains on a DETERMINISTIC
    # grid while being scored against a STOCHASTIC Bellman baseline.
    env_deterministic_transitions: bool = False

    # notebook cell 34: the greedy policy is ``argmax_a sum_j P(j|i,a) * U[j]``
    # -- the immediate reward ``R[j]`` and the discount are dropped, unlike the
    # Bellman backup of cell 32 which maximises ``sum_j P * (gamma*U[j] + R[j])``.
    # True == notebook.
    greedy_policy_ignores_reward: bool = False

    @classmethod
    def notebook(cls) -> "FixFlags":
        """Every flag set to the original notebook behaviour (all bugs on)."""
        return cls(
            epsilon_greedy_inverted=True,
            td_pred_uses_next_state=True,
            env_deterministic_transitions=True,
            greedy_policy_ignores_reward=True,
        )


@dataclass
class Config:
    """All hyper-parameters of the notebook in one place.

    Field values are the notebook's, except ``gamma`` -- see the note below.
    """

    # ---- grid world (notebook cells 11 and 13) ---------------------------
    Nx: int = 4
    Ny: int = 3
    alive_positions: tuple[tuple[int, int], ...] = ((3, 2),)
    death_positions: tuple[tuple[int, int], ...] = ((3, 1),)
    obstacle_positions: tuple[tuple[int, int], ...] = ((1, 1),)

    # ---- MDP / Bellman (notebook cells 11, 24, 32) -----------------------
    r: float = -0.004  # living reward
    # NOTE: cell 11 of the QUANTUM notebook sets ``Gamma=0.95``; the classical
    # notebook and the verified Bellman reference values use 0.99, so that is
    # the default here.  Pass ``Config(gamma=0.95)`` to reproduce the quantum notebook
    # literally (it changes both the Bellman baseline and the TD target).
    gamma: float = 0.99
    max_epoch: int = 100  # value-iteration sweeps

    # ---- training loop (notebook cell 50) --------------------------------
    LR_0: float = 0.2
    LR_MIN: float = 0.001
    num_games: int = 20000
    max_time: int = 250
    epsilon_0: float = 0.95
    epsilon_min: float = 0.05
    N: int = 700  # replay-memory length
    len_batch: int = 1
    C: int = 200  # target-network sync period, in steps

    # ---- variational quantum circuit (notebook cells 40-44) --------------
    deep_layers: int = 2
    n_wires: int = 4
    q_scale: float = 3.0  # PauliZ expectations are multiplied by this

    # ---- reproducibility / fixes -----------------------------------------
    seed: int | None = None
    fixes: FixFlags = field(default_factory=FixFlags)

    # ---- further settings lifted from the notebook ------------------------
    rmsprop_alpha: float = 0.99  # cell 50, torch.optim.RMSprop(alpha=...)
    # cell 50 uses StepLR(step_size=3, gamma=0.5) stepped ONCE PER EPISODE, which
    # halves the lr every 3 episodes: 0.2 -> 7.8e-4 by episode 24, where the
    # ``lr > LR_MIN`` guard freezes it.  Only 12 of 20000 episodes ever run at a
    # usable lr.  Default widened so the lr stays useful; Config.notebook()
    # restores the 3.
    lr_scheduler_step_size: int = 5000
    lr_scheduler_gamma: float = 0.5
    epsilon_decay_denom: float = 2000.0  # eps = max(eps_0 - (m+1)/2000, eps_min)
    eval_every: int = 20  # cell 50, ``if m % 20 == 0 and m > 0``
    weight_init_scale: float = 0.3  # cell 50, 0.3 * np.random.randn(...)

    @classmethod
    def notebook(cls, **overrides) -> "Config":
        """The config that reproduces StochFrozenLake_QUANTUM.ipynb exactly.

        All :class:`FixFlags` set to the notebook behaviour, the cell-50 learning
        rate schedule restored, and ``gamma`` set to the 0.95 of cell 11 (the
        plain ``Config()`` default is 0.99, as in the classical notebook).

        This is what the bit-for-bit parity test builds; keep it faithful.
        """
        base = dict(
            fixes=FixFlags.notebook(),
            lr_scheduler_step_size=3,
            gamma=0.95,
        )
        base.update(overrides)
        return cls(**base)

    @property
    def n_states(self) -> int:
        return self.Nx * self.Ny

    @property
    def n_actions(self) -> int:
        return 4


def apply_seed(seed: int | None) -> None:
    """Seed the legacy global RNGs the notebook relies on.

    The notebook never creates a ``np.random.Generator``: it calls
    ``np.random.rand``/``np.random.randint`` directly, so reproducibility means
    seeding the legacy global state.  ``torch`` and ``random`` are seeded too
    when available.  ``seed=None`` is a no-op.
    """
    if seed is None:
        return
    import random as _random

    import numpy as _np

    _np.random.seed(seed)
    _random.seed(seed)
    try:  # torch is optional for the pure-environment code paths
        import torch as _torch
    except Exception:  # pragma: no cover - torch is installed in .venv
        return
    _torch.manual_seed(seed)
