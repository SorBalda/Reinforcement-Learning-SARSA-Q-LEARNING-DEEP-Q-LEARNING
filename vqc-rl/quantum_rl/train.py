"""Deep Q-learning training loop with a VQC as the Q-function approximator.

Faithful port of ``StochFrozenLake_QUANTUM.ipynb`` **cell 50** (plus the final
policy / utility extraction of **cell 56**).

Everything that looks like a bug is reproduced and put behind a flag in
:class:`~quantum_rl.config.FixFlags` (the defaults are the fixes;
``FixFlags.notebook()`` restores the original behaviour).  The structure of the loop -- in particular *where* each update
happens -- is kept verbatim, because that is as load-bearing as the formulas:

    * the replay memory is PRE-FILLED with ``N`` transitions from a uniformly
      random policy before the first gradient step, using the bare
      ``D.append(...)`` of the notebook (no eviction), so the memory holds
      exactly ``N`` entries when training starts (see
      :class:`~quantum_rl.agent.ReplayMemory` for why this off-by-one matters);
    * a transition is the plain-index list ``[s_i, action, reward, s_f]``
      (the quantum notebook stores raw state indices; the classical notebook
      stores encoded tensors);
    * the eviction is ``append`` then ``if len(D) >= N: D.pop(0)``;
    * the target network is a ``clone().detach()`` snapshot re-taken every ``C``
      *steps* (a counter that runs across episodes, it is NOT reset per game);
    * ``epsilon`` and the LR scheduler are stepped once per GAME, after the
      episode loop, and the scheduler is stepped only while
      ``lr > LR_MIN`` (so the LR freezes just above the floor, it is never
      clamped to it);
    * the MSE / probe-Q evaluation runs on games ``m % 20 == 0 and m > 0``.

Known notebook bugs reproduced here
-----------------------------------
``td_pred_uses_next_state`` (``True`` = notebook)
    The prediction is indexed on the NEXT state::

        pred = variational_classifier(var_Q_circuit,
                                      angles=decimalToBinaryFixLength(4, s_jf))[action_j]

    so the network is trained to make ``Q(s_jf, a_j)`` match a target built from
    ``s_jf`` as well -- the current state ``s_j`` of the sampled transition never
    enters the loss at all.  The classical sibling notebook uses ``Q(phi_j)``.
    Set the flag to ``False`` to index on ``s_j``.

``epsilon_greedy_inverted`` (in :mod:`quantum_rl.agent`, ``True`` = notebook)
    Greedy when ``rand() < epsilon``, so exploration *increases* as epsilon
    decays from 0.95 to 0.05.

``env_deterministic_transitions`` (in :mod:`quantum_rl.environment`,
``True`` = notebook)
    ``Next_position(s_1, a)`` is called without ``stochastic=True``, and the
    trailing ``if not stochastic:`` line overwrites the sampled successor with
    the deterministic intended move.  The agent therefore trains on a
    *deterministic* grid while :mod:`quantum_rl.bellman` scores it against the
    *stochastic* (0.8/0.1/0.1) MDP -- the MSE can never go to zero.

Ported as-is, not flagged (dead code, kept for the RNG stream)
    ``W = np.random.rand(deep_layers, 4, 3) * 1.3`` and its copy: ``W`` is never
    read again (the ``quantum_backprop`` calls are commented out), but it draws
    ``deep_layers * 12`` uniforms from the global numpy RNG before the replay
    pre-fill, so dropping it would change every seeded run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch

from .agent import ReplayMemory, epsilon_greedy, loss_fnc, sample_random_batch
from .analysis import correct_indexes, mse
from .bellman import value_iteration
from .circuit import (DTYPE, decimal_to_binary_fix_length, default_qnode,
                      variational_classifier)
from .config import Config, apply_seed
from .environment import GridWorld

__all__ = [
    "TrainResult",
    "train",
    "epsilon_for_episode",
    "lr_for_optimizer_step",
    "GOOD_PROBE_STATE",
    "BAD_PROBE_STATE",
]


# notebook cell 50: the two hardcoded probe states of the periodic report.
#: state 10 == the cell immediately to the LEFT of the goal; probe action 1 (right).
GOOD_PROBE_STATE = 10
#: state 3 == the cell immediately BELOW the death state; probe action 0 (up).
BAD_PROBE_STATE = 3
_GOOD_PROBE_ACTION = 1
_BAD_PROBE_ACTION = 0

def epsilon_for_episode(cfg: Config, m: int) -> float:
    """Return epsilon after episode ``m`` according to ``cfg``.

    ``epsilon_decay_interval`` is an episode *update* interval: for ``K > 1``
    epsilon remains at ``epsilon_0`` until episode ``K`` has completed, then
    is evaluated at ``floor((m + 1) / K) * K``. The default ``K=1`` keeps the
    original formula exactly. ``epsilon_decay_duration`` is an explicit
    alternative to the legacy denominator and makes all non-constant
    schedules reach ``epsilon_min`` at episode ``duration``.
    """
    mode = str(cfg.epsilon_decay_mode).lower()
    if cfg.epsilon_decay_denom <= 0:
        raise ValueError("epsilon_decay_denom must be > 0 (got %r)" % cfg.epsilon_decay_denom)
    interval = cfg.epsilon_decay_interval
    if isinstance(interval, bool) or int(interval) != interval or interval <= 0:
        raise ValueError("epsilon_decay_interval must be an integer > 0 (got %r)" % interval)
    duration = cfg.epsilon_decay_duration
    if duration is not None and (
        isinstance(duration, bool) or int(duration) != duration or duration <= 0
    ):
        raise ValueError("epsilon_decay_duration must be an integer > 0 or None (got %r)" % duration)
    if mode not in {"linear", "exponential", "cosine", "constant"}:
        raise ValueError("epsilon_decay_mode must be one of linear, exponential, cosine, or constant (got %r)" % cfg.epsilon_decay_mode)
    if mode == "constant":
        return float(cfg.epsilon_0)

    # Keep this path spelled as the pre-interval implementation for exact
    # default parity. In particular, the old linear schedule used
    # ``float(m + 1)`` while cosine/exponential used ``float(m) + 1.0``.
    if duration is None:
        if interval == 1:
            progress = min(max((float(m) + 1.0) / cfg.epsilon_decay_denom, 0.0), 1.0)
            decay_count = float(m + 1)
        else:
            decay_count = float(((int(m) + 1) // int(interval)) * int(interval))
            progress = min(max(decay_count / cfg.epsilon_decay_denom, 0.0), 1.0)
    else:
        elapsed = int(m) + 1
        # A duration need not be a multiple of interval: once its endpoint is
        # reached, force it even if the last update is a partial interval.
        if elapsed >= int(duration):
            decay_count = float(duration)
        else:
            decay_count = float((elapsed // int(interval)) * int(interval))
        progress = min(max(decay_count / float(duration), 0.0), 1.0)
    if mode == "linear":
        if duration is not None:
            # An explicit duration parameterises the full epsilon range, so
            # the endpoint is epsilon_min even when epsilon_0-epsilon_min
            # is not one (the historical denominator did not have this
            # interpretation).
            value = cfg.epsilon_0 + (cfg.epsilon_min - cfg.epsilon_0) * progress
            return float(np.max([value, cfg.epsilon_min]))
        if interval == 1:
            return float(np.max([cfg.epsilon_0 - (float(m + 1) / cfg.epsilon_decay_denom), cfg.epsilon_min]))
        return float(np.max([cfg.epsilon_0 - (decay_count / cfg.epsilon_decay_denom), cfg.epsilon_min]))
    if mode == "cosine":
        weight = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(cfg.epsilon_min + (cfg.epsilon_0 - cfg.epsilon_min) * weight)
    if cfg.epsilon_0 <= 0 or cfg.epsilon_min <= 0:
        raise ValueError("exponential epsilon decay requires epsilon_0 and epsilon_min > 0")
    return float(cfg.epsilon_0 * (cfg.epsilon_min / cfg.epsilon_0) ** progress)


def lr_for_optimizer_step(cfg: Config, optimizer_step: int) -> float:
    """Return the transient-phase LR for a one-based optimizer step.

    Steps ``1..lr_transient_steps`` use ``LR_0`` and step ``N+1`` uses
    ``LR_AFTER_TRANSIENT``. If the two-phase schedule is not configured this
    returns ``LR_0`` for all steps, preserving the existing scheduler path.
    """
    if optimizer_step < 1:
        raise ValueError("optimizer_step must be >= 1 (got %r)" % optimizer_step)
    if (
        cfg.lr_transient_steps > 0
        and (isinstance(cfg.lr_transient_steps, bool)
             or int(cfg.lr_transient_steps) != cfg.lr_transient_steps)
    ):
        raise ValueError("lr_transient_steps must be an integer (got %r)" % cfg.lr_transient_steps)
    if cfg.LR_AFTER_TRANSIENT is None or cfg.lr_transient_steps <= 0:
        return float(cfg.LR_0)
    if optimizer_step <= cfg.lr_transient_steps:
        return float(cfg.LR_0)
    return float(cfg.LR_AFTER_TRANSIENT)



@dataclass
class TrainResult:
    """Everything cell 50 accumulates, plus the cell-56 policy / utility."""

    # ---- main results ------------------------------------------------------
    var_Q_circuit: Any  # final weights, torch tensor (deep_layers, 4, 3)
    loss_games: List[float] = field(default_factory=list)  # mean loss per game
    games_reward: List[float] = field(default_factory=list)  # discounted return
    mse_u: List[float] = field(default_factory=list)  # notebook's MAE_U (is an MSE)
    good_Q: List[float] = field(default_factory=list)  # Q(10, right) every 20 games
    bad_Q: List[float] = field(default_factory=list)  # Q(3, up) every 20 games
    policy: Optional[np.ndarray] = None  # cell 56, argmax_a Q(s, a)
    U: Optional[np.ndarray] = None  # cell 56, max_a Q(s, a)

    # ---- extras (all from cell 50) ----------------------------------------
    best_weights: Any = None  # weights at the lowest MSE seen
    mse_min: float = float("inf")  # notebook's MAE_min
    good_diff_v: List[float] = field(default_factory=list)  # V(10) - U_final[10]
    bad_diff_v: List[float] = field(default_factory=list)  # V(3)  - U_final[3]
    mse_games: List[int] = field(default_factory=list)  # game index of each mse_u
    U_ref: Optional[np.ndarray] = None  # the Bellman U_final scored against
    steps: int = 0  # total environment steps taken
    cfg: Optional[Config] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rand(rng: Optional[Any] = None) -> float:
    """Uniform in [0, 1) -- legacy global numpy RNG when ``rng is None``.

    ``rng=None`` is what reproduces the notebook: it never builds a
    ``np.random.Generator``, so ``np.random.seed(...)`` (via
    :func:`quantum_rl.config.apply_seed`) is what makes a run repeatable.
    """
    if rng is None:
        return float(np.random.rand())
    if hasattr(rng, "random"):
        return float(rng.random())
    return float(rng.random_sample())


def _q_values(weights, s: int, qnode, cfg: Config):
    """``variational_classifier`` on state index ``s`` (cell 43 + cell 40)."""
    return variational_classifier(
        var_Q_circuit=weights,
        angles=decimal_to_binary_fix_length(cfg.n_wires, s),
        qnode=qnode,
        scale=cfg.q_scale,
    )


def _q_values_detached(weights, s: int, qnode, cfg: Config):
    """Forward-only evaluation, used by the periodic probes and by cell 56.

    The notebook writes ``torch.max(torch.tensor(variational_classifier(...)))``
    on the LIVE ``var_Q_circuit``, i.e. it builds an autograd graph and then
    throws it away (and, for ``good_Q_print``/``bad_Q_print``, keeps a reference
    to it in a list for the whole run).  Evaluating on a detached clone is
    bit-for-bit identical on the forward pass and does not retain the graph.
    """
    return _q_values(weights.clone().detach(), s, qnode, cfg)


def _prefill_replay_memory(
    memory: ReplayMemory, env: GridWorld, cfg: Config, rng: Optional[Any]
) -> None:
    """notebook cell 50 -- the ``while True`` random-rollout pre-fill.

    Random restarts, uniformly random actions, episodes cut short on a terminal
    successor, and a hard stop at exactly ``N`` transitions.  Appends bypass
    eviction (the notebook appends to the bare list here), so the memory holds
    ``N`` -- not ``N - 1`` -- entries when training starts.
    """
    count = 0
    while True:
        if count >= cfg.N:
            break
        s_i = env.random_state(rng)
        while True:
            # notebook: `int(np.random.rand()*4)` -- NOT np.random.randint
            action = int(_rand(rng) * cfg.n_actions)
            s_f = env.next_position(s_i, action, rng)
            reward = env.R[s_f]
            memory.prefill_append([s_i, action, reward, s_f])
            count = count + 1
            if count >= cfg.N:
                break
            if env.is_terminal(s_f):
                break
            s_i = s_f


# ---------------------------------------------------------------------------
# notebook cell 50
# ---------------------------------------------------------------------------
def train(
    cfg: Config,
    env: GridWorld,
    qnode=None,
    progress: bool = False,
    *,
    U_ref: Optional[np.ndarray] = None,
    rng: Optional[Any] = None,
    monitor: Optional[Any] = None,
    game_monitor: Optional[Any] = None,
) -> TrainResult:
    """Run the notebook's deep-Q training loop.

    Args:
        cfg: hyper-parameters; ``cfg.seed`` seeds the legacy global RNGs.
        env: the grid world.  ``env.next_position`` honours
            ``cfg.fixes.env_deterministic_transitions``.
        qnode: QNode from :func:`quantum_rl.circuit.make_qnode`; ``None`` uses
            the process-wide default (the notebook's single global device).
        progress: print the notebook's per-game / every-20-games report.
            Faithful, which means it also reproduces the notebook's broken
            ``[:-20:]`` averaging -- use ``monitor`` for an honest live view.
        monitor: optional callable fired every ``cfg.eval_every`` games with a
            dict: game, loss (mean over the window), loss_cum, lr, epsilon,
            mse, steps, weights (a detached clone).  Return value ignored.
        game_monitor: optional callable fired once before the first game
            (``game=-1``, the initial weights, no gradient) and then at the end
            of EVERY game, with a dict: game, weights (numpy copy, shape
            ``(deep_layers, 4, 3)``), grad_rms (same shape: root mean square
            over the game's optimiser steps of each parameter's gradient -- the
            quantity to watch for vanishing gradients), loss (the game's mean
            loss, i.e. the new entry of ``loss_games``; absent for game -1),
            steps.  It only reads
            ``.grad``, so the training is bit-for-bit unchanged.  Completed
            games also include ``discounted_reward``, ``trajectory`` (the
            initial state and every successor), ``actions`` and ``rewards``.
        U_ref: Bellman utilities to score the MSE against.  ``None`` runs
            :func:`quantum_rl.bellman.value_iteration` with ``cfg.gamma`` /
            ``cfg.max_epoch``, which is what the notebook's ``U_final`` is.
        rng: ``None`` (default, = notebook) uses the legacy global numpy RNG.

    Returns:
        :class:`TrainResult`.

    WARNING: the notebook's ``num_games=20000`` x ``max_time=250`` is many hours
    of statevector simulation.  Use small values for smoke tests.
    """
    if cfg.max_time < 1:
        # the notebook would raise ZeroDivisionError on `loss_game/cntr`
        raise ValueError(f"cfg.max_time must be >= 1 (got {cfg.max_time})")
    # Validate schedules before constructing devices or starting training.
    epsilon_for_episode(cfg, 0)
    if (
        isinstance(cfg.lr_transient_steps, bool)
        or int(cfg.lr_transient_steps) != cfg.lr_transient_steps
        or cfg.lr_transient_steps < 0
    ):
        raise ValueError(
            "lr_transient_steps must be an integer >= 0 (got %r)"
            % cfg.lr_transient_steps
        )
    if cfg.LR_AFTER_TRANSIENT is not None and cfg.LR_AFTER_TRANSIENT <= 0:
        raise ValueError(
            "LR_AFTER_TRANSIENT must be > 0 (got %r)" % cfg.LR_AFTER_TRANSIENT
        )
    if cfg.len_batch < 1:
        # the notebook would raise AttributeError on `loss.backward()` (loss stays float)
        raise ValueError(f"cfg.len_batch must be >= 1 (got {cfg.len_batch})")

    # Resolve the QNode BEFORE seeding.  `qml.device('default.qubit', ...)`
    # defaults to `seed="global"`, which draws ONE `np.random.randint` from the
    # legacy global RNG at construction time.  The notebook builds its device in
    # cell 42, i.e. long before the training loop, so that draw must not land
    # inside the seeded stream -- building the device lazily on the first
    # `variational_classifier` call would shift every subsequent action,
    # transition and minibatch index.
    if qnode is None:
        qnode = default_qnode()

    apply_seed(cfg.seed)

    # The Bellman baseline the MSE is measured against (notebook cell 32).
    if U_ref is None:
        U_ref, _ = value_iteration(env, cfg.gamma, cfg.max_epoch)
    U_ref = np.asarray(U_ref, dtype=float)

    gamma = cfg.gamma
    n_action = cfg.n_actions

    # ---- cell 50 preamble --------------------------------------------------
    c = 0  # steps since the last target sync; NOT reset per game
    memory = ReplayMemory(cfg.N)

    mse_u: List[float] = []  # notebook: MAE_U
    good_Q: List[float] = []
    bad_Q: List[float] = []
    good_diff_v: List[float] = []
    bad_diff_v: List[float] = []
    mse_games: List[int] = []

    # DEAD CODE, kept on purpose: `W` is never read again, but these draws shift
    # the global RNG stream and therefore every seeded run downstream.
    W = np.random.rand(cfg.deep_layers, cfg.n_wires, 3) * 1.3
    W_before = W.copy()  # noqa: F841  (notebook keeps it too, also unused)

    _prefill_replay_memory(memory, env, cfg, rng)

    epsilon = cfg.epsilon_0
    loss_games: List[float] = []
    games_reward: List[float] = []

    # notebook: torch.tensor(0.3*np.random.randn(deep_layers,4,3), requires_grad=True).type(dtype)
    # `.type(torch.DoubleTensor)` is a no-op here (numpy randn is already
    # float64), so the tensor stays a leaf and RMSprop can optimise it.
    var_init_circuit = torch.tensor(
        cfg.weight_init_scale * np.random.randn(cfg.deep_layers, cfg.n_wires, 3),
        device="cpu",
        requires_grad=True,
    ).type(DTYPE)
    if not var_init_circuit.is_leaf or not var_init_circuit.requires_grad:  # pragma: no cover
        raise RuntimeError("var_Q_circuit must be a trainable leaf tensor")

    var_Q_circuit = var_init_circuit
    var_target_Q_circuit = var_Q_circuit.clone().detach()

    opt = torch.optim.RMSprop(
        [var_Q_circuit],
        lr=cfg.LR_0,
        alpha=cfg.rmsprop_alpha,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma
    )
    # The default path below intentionally remains the notebook's exact
    # per-episode StepLR sequence.  The optional path delays scheduler ticks
    # until the transient has completed and starts from LR_AFTER_TRANSIENT.
    transient_enabled = (
        cfg.LR_AFTER_TRANSIENT is not None and cfg.lr_transient_steps > 0
    )
    transient_complete = not transient_enabled
    mse_min = 1000.0  # notebook: MAE_min
    best_weights = None
    total_steps = 0
    s_2 = 0

    if game_monitor is not None:
        game_monitor({"game": -1,
                      "weights": var_Q_circuit.detach().numpy().copy(),
                      "grad_rms": None, "steps": 0,
                      "lr": float(opt.param_groups[0]["lr"]),
                      "lr_phase": "transient" if transient_enabled else "scheduled",
                      "transient_boundary": False})

    # ---- the game loop -----------------------------------------------------
    for m in range(cfg.num_games):
        grad_sq = np.zeros(var_Q_circuit.shape) if game_monitor is not None else None
        s_1 = env.random_state(rng)
        reward_history: List[float] = []
        # Observational playback data; this introduces no RNG calls.
        episode_trajectory: List[int] = [int(s_1)]
        episode_actions: List[int] = []
        episode_rewards: List[float] = []
        game_reward = 0.0
        cntr = 0
        loss_game = 0.0
        # True only for the episode containing the N-th transient step.  That
        # episode is deliberately not counted by StepLR.
        transient_boundary_this_game = False

        for _time in range(cfg.max_time):
            a = epsilon_greedy(
                var_Q_circuit=var_Q_circuit,
                epsilon=epsilon,
                n_action=n_action,
                s=s_1,
                qnode=qnode,
                cfg=cfg,
                rng=rng,
            )
            s_2 = env.next_position(s_1, a, rng)
            reward = env.R[s_2]
            episode_actions.append(int(a))
            episode_trajectory.append(int(s_2))
            episode_rewards.append(float(reward))
            memory.append([s_1, a, reward, s_2])  # append + `if len(D)>=N: D.pop(0)`
            s_1 = s_2

            # ---- sample a minibatch from D (cell 47) ----------------------
            batch = sample_random_batch(memory.memory, cfg.len_batch, rng=rng)
            loss = 0.0
            for j in range(cfg.len_batch):
                s_j, action_j, reward_j, s_jf = (
                    batch[j][0],
                    batch[j][1],
                    batch[j][2],
                    batch[j][3],
                )

                target_q = _q_values(var_target_Q_circuit, s_jf, qnode, cfg)
                if env.is_terminal(s_jf):
                    # notebook: `reward_j + max(...) * 0.0` -- the target net is
                    # still evaluated, the product just zeroes it out.
                    y = reward_j + max(target_q) * 0.0
                else:
                    y = reward_j + gamma * max(target_q)

                # BUG (flagged): the notebook indexes the prediction on s_jf,
                # the NEXT state, not on the state the action was taken from.
                pred_state = s_jf if cfg.fixes.td_pred_uses_next_state else s_j
                pred = _q_values(var_Q_circuit, pred_state, qnode, cfg)[action_j]

                loss = loss + loss_fnc(pred, y) / cfg.len_batch

            # Set the LR immediately before each optimiser step.  Thus the
            # first N steps use LR_0 and the next step starts at the post-
            # transient LR, with no off-by-one at an episode boundary.
            if transient_enabled and not transient_complete:
                step_lr = lr_for_optimizer_step(cfg, total_steps + 1)
                if opt.param_groups[0]["lr"] != step_lr:
                    opt.param_groups[0]["lr"] = step_lr

            opt.zero_grad()
            loss.backward()
            if grad_sq is not None and var_Q_circuit.grad is not None:
                grad_sq += var_Q_circuit.grad.detach().numpy() ** 2
            opt.step()

            loss_game += loss.item()
            reward_history.append(float(reward))
            cntr += 1
            c += 1
            total_steps += 1
            if transient_enabled and not transient_complete and total_steps >= cfg.lr_transient_steps:
                # Switch right after step N, so a possible step N+1 uses the
                # post-transient value even when N ends an episode.
                opt.param_groups[0]["lr"] = float(cfg.LR_AFTER_TRANSIENT)
                transient_complete = True
                transient_boundary_this_game = True
            if c >= cfg.C:
                var_target_Q_circuit = var_Q_circuit.clone().detach()
                c = 0

            if env.is_terminal(s_2):
                break

        # ---- end of game: discounted return, epsilon, LR ------------------
        for esponent in range(len(reward_history)):
            game_reward += reward_history[esponent] * gamma**esponent
        games_reward.append(game_reward)
        loss_games.append(loss_game / cntr)
        epsilon = epsilon_for_episode(cfg, m)
        if progress:
            print(cntr)

        # notebook: step the scheduler only while the LR is still above the
        # floor -- so the LR settles just ABOVE LR_MIN and is never clamped to it.
        # In two-phase mode the boundary episode is not a scheduler episode.
        if (not transient_enabled or (transient_complete and not transient_boundary_this_game)):
            if opt.param_groups[0]["lr"] > cfg.LR_MIN:
                scheduler.step()
                if transient_enabled:
                    # The configured two-phase schedule is user-facing: its
                    # halving must approach, but never cross, LR_MIN. Keep the
                    # legacy/default path untouched for notebook parity.
                    for group in opt.param_groups:
                        group["lr"] = max(float(group["lr"]), float(cfg.LR_MIN))

        if game_monitor is not None:
            game_monitor({"game": m,
                          "weights": var_Q_circuit.detach().numpy().copy(),
                          "grad_rms": np.sqrt(grad_sq / max(cntr, 1)),
                          "loss": float(loss_games[-1]),
                          "discounted_reward": float(game_reward),
                          "trajectory": episode_trajectory.copy(),
                          "actions": episode_actions.copy(),
                          "rewards": episode_rewards.copy(),
                          "steps": int(total_steps),
                          "lr": float(opt.param_groups[0]["lr"]),
                          "lr_phase": (
                              "transient" if transient_enabled and not transient_complete
                              else ("post_transient" if transient_enabled else "scheduled")
                          ),
                          "transient_boundary": bool(transient_boundary_this_game)})

        # ---- periodic evaluation (cell 50, `if m % 20 == 0 and m > 0`) -----
        if m % cfg.eval_every == 0 and m > 0:
            good_q_values = _q_values_detached(var_Q_circuit, GOOD_PROBE_STATE, qnode, cfg)
            good_v = torch.max(good_q_values).item()
            good_diff_v.append(good_v - float(U_ref[GOOD_PROBE_STATE]))
            good_q_print = float(good_q_values[_GOOD_PROBE_ACTION].item())

            bad_q_values = _q_values_detached(var_Q_circuit, BAD_PROBE_STATE, qnode, cfg)
            bad_v = torch.max(bad_q_values).item()
            bad_diff_v.append(bad_v - float(U_ref[BAD_PROBE_STATE]))
            bad_q_print = float(bad_q_values[_BAD_PROBE_ACTION].item())
            bad_q2_print = float(bad_q_values[_GOOD_PROBE_ACTION].item())

            # notebook: MAE_value += (max_a Q(i,a) - U_final[i])**2 / 9.0 over
            # the non-terminal, non-obstacle states (`analysis.mse` is that sum,
            # with /9.0 generalised to /len(correct_indexes)).
            U_pred = np.zeros(env.n_states)
            # the current policy comes out of the same Q vector already
            # computed for the utilities: the argmax costs no extra evaluation
            policy_pred = np.zeros(env.n_states, dtype=int)
            for i in correct_indexes(env):
                q_i = _q_values_detached(var_Q_circuit, i, qnode, cfg)
                U_pred[i] = torch.max(q_i).item()
                policy_pred[i] = int(torch.argmax(q_i).item())
            mse_value = mse(U_pred, U_ref, env)

            mse_u.append(mse_value)
            mse_games.append(m)
            good_Q.append(good_q_print)
            bad_Q.append(bad_q_print)

            if progress:
                print(
                    "GOOD Q (left to the green):", good_q_print,
                    ", BAD Q (under the red)", bad_q_print,
                    ", BAD Q 2(left of the red)", bad_q2_print,
                )
                print("GOOD V (left to the green):", good_v, ", BAD V (under the red)", bad_v)
                print("MSE_value=", mse_value)
                print("Learning rate:", opt.param_groups[0]["lr"], "MSE=", mse_value)
                print(
                    f"GAME = {m}, LOSSavg20 = {np.mean(loss_games[:-20:]):.8f}, "
                    f"number of steps done in last= {cntr}, "
                    f"last_episode's_last_state = {s_2:.0f}, "
                    f"epsilon = {epsilon:.4f}, "
                    f"game_rewardavg20 = {np.mean(games_reward[:-20:]):.4f}"
                )

            if monitor is not None:
                window = loss_games[-cfg.eval_every:] or [float("nan")]
                monitor({
                    "game": m,
                    "loss": float(np.mean(window)),
                    "loss_cum": float(np.mean(loss_games)) if loss_games else float("nan"),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "lr_phase": (
                        "transient" if transient_enabled and not transient_complete
                        else ("post_transient" if transient_enabled else "scheduled")
                    ),
                    "transient_boundary": bool(transient_boundary_this_game),
                    "epsilon": float(epsilon),
                    "mse": float(mse_value),
                    "steps": int(total_steps),
                    "weights": var_Q_circuit.clone().detach(),
                    # already computed for the MSE above -- free to hand out
                    "U": U_pred.copy(),
                    "policy": policy_pred.copy(),
                    "U_ref": np.asarray(U_ref, dtype=float).copy(),
                })

            # notebook: `if m > 0:` (always true inside this branch)
            if mse_value < mse_min:
                best_weights = var_Q_circuit.clone().detach()
                mse_min = mse_value

        if progress and m == 0:
            print(loss_games[0])

    # ---- cell 56: greedy policy and utility from the final weights ---------
    policy = np.zeros(env.n_states)
    U = np.zeros(env.n_states)
    for s in range(env.n_states):
        q_s = _q_values_detached(var_Q_circuit, s, qnode, cfg)
        policy[s] = int(torch.argmax(q_s).item())
        U[s] = float(torch.max(q_s).item())

    return TrainResult(
        var_Q_circuit=var_Q_circuit,
        loss_games=loss_games,
        games_reward=games_reward,
        mse_u=mse_u,
        good_Q=good_Q,
        bad_Q=bad_Q,
        policy=policy,
        U=U,
        best_weights=best_weights,
        mse_min=mse_min,
        good_diff_v=good_diff_v,
        bad_diff_v=bad_diff_v,
        mse_games=mse_games,
        U_ref=U_ref,
        steps=total_steps,
        cfg=cfg,
    )
