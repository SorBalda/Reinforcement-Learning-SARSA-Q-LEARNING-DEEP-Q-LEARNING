"""Integration tests for the :mod:`quantum_rl` package.

Anchored on the verified ground truth of ``StochFrozenLake_QUANTUM.ipynb``:

    * state indexing ``i = x + Nx*y``; 11 = goal (+1), 7 = death (-1), 5 = obstacle;
    * actions 0=up, 1=right, 2=down, 3=left; slip 0.8 / 0.1 / 0.1;
    * ``neigh_dist[s, a]`` is an OFFSET added to ``s``; offset 0 means blocked;
    * value iteration at ``r=-0.004, gamma=0.99, max_epoch=100`` gives, over
      ``correct_indexes == [0,1,2,3,4,6,8,9,10]``, the nine values asserted below.

Everything that touches the VQC is deliberately tiny -- statevector simulation
costs ~4 ms per forward pass, so the whole file must stay well under a minute.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_rl.agent import ReplayMemory, epsilon_greedy, sample_random_batch
from quantum_rl.analysis import correct_indexes, mse, residuals, restrict
from quantum_rl.bellman import greedy_policy, value_iteration, value_iteration_detailed
from quantum_rl.circuit import (DTYPE, decimal_to_binary_fix_length, make_qnode,
                                variational_classifier)
from quantum_rl.config import Config, FixFlags, apply_seed
from quantum_rl.environment import GridWorld
from quantum_rl.train import TrainResult, epsilon_for_episode, lr_for_optimizer_step, train

# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
CORRECT = [0, 1, 2, 3, 4, 6, 8, 9, 10]
U_ANCHOR = [
    0.910414, 0.894075, 0.876564, 0.759436, 0.929002,
    0.865424, 0.945782, 0.964872, 0.982105,
]
GOAL, DEATH, OBSTACLE = 11, 7, 5


@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def env(cfg):
    return GridWorld(cfg)


@pytest.fixture(scope="module")
def bellman(env, cfg):
    return value_iteration_detailed(env, cfg.gamma, cfg.max_epoch)


@pytest.fixture(scope="module")
def qnode():
    """One QNode for the whole module -- building a device is not free."""
    return make_qnode()


def tiny_cfg(**kwargs):
    """A Config small enough for a fast VQC smoke run."""
    base = dict(num_games=2, max_time=3, N=20, max_epoch=20, len_batch=1, C=200, seed=1234)
    base.update(kwargs)
    return Config(**base)


# --------------------------------------------------------------------------- #
# environment: indexing, neigh_dist, transitions
# --------------------------------------------------------------------------- #
def test_index_mapping_round_trip(env):
    """``i = x + Nx*y`` round-trips for every state."""
    for i in range(env.n_states):
        x, y = env.index_to_position(i)
        assert 0 <= x < env.Nx and 0 <= y < env.Ny
        assert i == x + env.Nx * y
        assert env.position_to_index((x, y)) == i


def test_special_state_indexes(env):
    assert env.alive_indexes == [GOAL]
    assert env.death_indexes == [DEATH]
    assert env.obstacle_indexes == [OBSTACLE]
    assert env.is_terminal(GOAL) and env.is_terminal(DEATH)
    # faithful quirk: the obstacle is absorbing, but NOT terminal
    assert not env.is_terminal(OBSTACLE)


def test_neigh_dist_blocking(env):
    """Offset 0 == blocked (grid edge, obstacle wall, or terminal state)."""
    nd = env.neigh_dist
    assert nd.shape == (env.n_states, 4)

    # bottom row (y == 0) cannot go down; top row (y == Ny-1) cannot go up
    for i in range(env.n_states):
        x, y = env.index_to_position(i)
        if y == 0:
            assert nd[i, 2] == 0, f"state {i} should be blocked downwards"
        if y == env.Ny - 1:
            assert nd[i, 0] == 0, f"state {i} should be blocked upwards"
        if x == 0:
            assert nd[i, 3] == 0, f"state {i} should be blocked leftwards"
        if x == env.Nx - 1:
            assert nd[i, 1] == 0, f"state {i} should be blocked rightwards"

    # every offset lands inside the grid, and no state can walk INTO the
    # obstacle (the obstacle itself is excluded: it is fully blocked, so its
    # own offsets are 0 and it "stays" on itself -- an absorbing self-loop).
    for i in range(env.n_states):
        for a in range(4):
            j = i + int(nd[i, a])
            assert 0 <= j < env.n_states
            if i != OBSTACLE:
                assert j != OBSTACLE, f"neigh_dist[{i},{a}] walks into the obstacle"

    # the obstacle and the two terminal states are fully blocked
    for i in (OBSTACLE, DEATH, GOAL):
        assert list(nd[i]) == [0, 0, 0, 0]

    # state 1 sits right below the obstacle (index 5) -> up is blocked
    assert nd[1, 0] == 0
    # state 0 can go right (+1) and up (+Nx)
    assert nd[0, 1] == 1 and nd[0, 0] == env.Nx


def test_transition_rows_sum_to_one(env):
    """Rows sum to 1.0 for every non-terminal state, 0.0 for 7 and 11."""
    for s0 in range(env.n_states):
        for a in range(4):
            row = sum(env.transition_probability(s1, s0, a) for s1 in range(env.n_states))
            expected = 0.0 if s0 in (DEATH, GOAL) else 1.0
            assert row == pytest.approx(expected, abs=1e-12), (
                f"row (s0={s0}, a={a}) sums to {row}, expected {expected}"
            )


def test_slip_model(env):
    """0.8 intended / 0.1 each perpendicular, from a state with no blocking."""
    # state 4 == (0, 1): up -> 8, right -> 5? no: right is blocked by the obstacle
    # use state 8 == (0, 2): right -> 9 (0.8), up blocked (0.1), down -> 4 (0.1)
    assert env.transition_probability(9, 8, 1) == pytest.approx(0.8)
    assert env.transition_probability(4, 8, 1) == pytest.approx(0.1)
    # the blocked "up" perpendicular has offset 0 -> it stays put
    assert env.transition_probability(8, 8, 1) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Bellman
# --------------------------------------------------------------------------- #
def test_bellman_anchor_values(bellman, env):
    got = restrict(bellman.U_final, env)
    assert correct_indexes(env) == CORRECT
    assert got == pytest.approx(U_ANCHOR, abs=1e-5)


def test_bellman_terminal_and_obstacle_are_zero(bellman):
    """Before the manual +-1 override: U[5] = U[7] = U[11] = 0."""
    assert bellman.U_final[OBSTACLE] == 0.0
    assert bellman.U_final[DEATH] == 0.0
    assert bellman.U_final[GOAL] == 0.0


def test_bellman_greedy_policy_avoids_death(bellman, env, cfg):
    """No state adjacent to the death cell points at it."""
    policy = greedy_policy(env, bellman.U_policy, cfg.gamma)
    assert policy.shape == (env.n_states,)
    assert set(int(a) for a in policy) <= {0, 1, 2, 3}

    # for every state, the greedy action must not be the one whose intended
    # move lands on the death state
    for i in range(env.n_states):
        if env.is_terminal(i) or i == OBSTACLE:
            continue
        a = int(policy[i])
        assert i + int(env.neigh_dist[i, a]) != DEATH, (
            f"greedy policy walks state {i} straight into the death state"
        )

    # and the state below the goal is steered towards it
    assert int(policy[10]) == 1  # (2,2) -> right -> goal


def test_value_iteration_returns_a_tuple(env, cfg):
    U_final, U_time = value_iteration(env, cfg.gamma, cfg.max_epoch)
    assert U_time.shape == (cfg.max_epoch, env.n_states)
    # faithful quirk: only max_epoch-1 sweeps run, so row 0 stays zero
    assert np.all(U_time[0] == 0.0)
    assert np.allclose(U_final, U_time[cfg.max_epoch - 1])


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def test_residuals_and_mse_are_zero_against_self(bellman, env):
    avg, std, res = residuals(bellman.U_final, bellman.U_final, env)
    assert avg == 0.0 and std == 0.0 and res == [0.0] * len(CORRECT)
    assert mse(bellman.U_final, bellman.U_final, env) == 0.0


def test_mse_matches_the_notebook_formula(bellman, env):
    pred = np.asarray(bellman.U_final) + 1.0
    # the notebook divides EACH term by 9.0 -> 9 * (1/9) == 1.0
    assert mse(pred, bellman.U_final, env) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# circuit
# --------------------------------------------------------------------------- #
def test_decimal_to_binary_fix_length():
    assert list(decimal_to_binary_fix_length(4, 9)) == [1, 0, 0, 1]
    assert list(decimal_to_binary_fix_length(4, 0)) == [0, 0, 0, 0]
    assert list(decimal_to_binary_fix_length(4, 11)) == [1, 0, 1, 1]


def test_circuit_output_shape_and_gradient(qnode):
    """4 Q-values, float64, and ``loss.backward()`` must reach the weights."""
    import torch

    np.random.seed(0)
    weights = torch.tensor(
        0.3 * np.random.randn(2, 4, 3), device="cpu", requires_grad=True
    ).type(DTYPE)
    assert weights.is_leaf and weights.requires_grad

    q = variational_classifier(
        var_Q_circuit=weights, angles=decimal_to_binary_fix_length(4, 9),
        qnode=qnode, scale=3.0,
    )
    assert isinstance(q, torch.Tensor)
    assert q.shape == (4,)
    assert q.dtype == torch.float64
    assert q.requires_grad

    q.sum().backward()
    assert weights.grad is not None
    assert weights.grad.shape == weights.shape
    assert torch.isfinite(weights.grad).all()
    assert float(weights.grad.norm()) > 0.0

    # action selection must NOT build a graph
    detached = variational_classifier(
        var_Q_circuit=weights.clone().detach(),
        angles=decimal_to_binary_fix_length(4, 9), qnode=qnode, scale=3.0,
    )
    assert not detached.requires_grad
    assert torch.allclose(detached, q.detach())


def test_q_scale_is_linear(qnode):
    import torch

    weights = torch.zeros(2, 4, 3, dtype=torch.float64)
    a = variational_classifier(weights, decimal_to_binary_fix_length(4, 3), qnode, scale=1.0)
    b = variational_classifier(weights, decimal_to_binary_fix_length(4, 3), qnode, scale=3.0)
    assert torch.allclose(b, 3.0 * a)


# --------------------------------------------------------------------------- #
# agent / replay memory
# --------------------------------------------------------------------------- #
def test_replay_memory_prefill_saturates_at_maxlen():
    """The notebook's pre-fill bypasses eviction -> exactly N entries."""
    m = ReplayMemory(5)
    for i in range(5):
        m.prefill_append([i, 0, 0.0, i])
    assert len(m) == 5
    assert m.memory[0][0] == 0


def test_replay_memory_append_evicts_oldest():
    """``append`` then ``if len(D) >= N: D.pop(0)`` -> steady state N after a prefill."""
    m = ReplayMemory(5)
    for i in range(5):
        m.prefill_append([i, 0, 0.0, i])
    m.append([99, 0, 0.0, 99])
    assert len(m) == 5
    assert m.memory[0][0] == 1  # the oldest was dropped
    assert m.memory[-1][0] == 99


def test_replay_memory_append_only_saturates_at_maxlen_minus_one():
    """Faithful off-by-one: filling from empty with ``append`` stops at N-1."""
    m = ReplayMemory(5)
    for i in range(50):
        m.append([i, 0, 0.0, i])
    assert len(m) == 4


def test_sample_random_batch_uses_one_uniform_per_item():
    D = [[i, 0, 0.0, i] for i in range(10)]
    np.random.seed(0)
    batch = sample_random_batch(D, 3)
    np.random.seed(0)
    expected = [D[int(np.random.rand() * len(D))] for _ in range(3)]
    assert batch == expected


def test_epsilon_greedy_branches(qnode, cfg):
    """Notebook: greedy when rand() < epsilon.  Fixed: the other way round."""
    import torch

    weights = torch.zeros(2, 4, 3, dtype=torch.float64)
    cfg = Config(fixes=FixFlags.notebook())   # notebook branch
    fixed = Config()                          # default == fixed branch

    # epsilon = 1.0 -> notebook branch is always greedy, fixed branch always random
    np.random.seed(0)
    greedy_actions = [epsilon_greedy(weights, 1.0, 4, s, qnode, cfg) for s in range(12)]
    assert len(set(greedy_actions)) >= 1
    for a in greedy_actions:
        assert a in (0, 1, 2, 3)

    np.random.seed(0)
    a_notebook = epsilon_greedy(weights, 1.0, 4, 9, qnode, cfg)
    np.random.seed(0)
    a_fixed = epsilon_greedy(weights, 0.0, 4, 9, qnode, fixed)
    assert a_notebook == a_fixed  # eps=1 (notebook) == eps=0 (fixed) == always greedy

    # train=True short-circuits and consumes NO random draw
    np.random.seed(0)
    before = np.random.rand()
    np.random.seed(0)
    epsilon_greedy(weights, 0.0, 4, 9, qnode, cfg, train=True)
    assert np.random.rand() == before


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_train_smoke_two_episodes(env, qnode):
    """End-to-end: 2 episodes x 3 steps must produce a complete TrainResult."""
    import torch

    c = tiny_cfg(num_games=2, max_time=3)
    e = GridWorld(c)
    result = train(c, e, qnode=qnode)

    assert isinstance(result, TrainResult)
    assert isinstance(result.var_Q_circuit, torch.Tensor)
    assert result.var_Q_circuit.shape == (c.deep_layers, 4, 3)
    assert torch.isfinite(result.var_Q_circuit).all()

    assert len(result.loss_games) == 2
    assert len(result.games_reward) == 2
    assert all(np.isfinite(v) for v in result.loss_games)
    assert all(v >= 0.0 for v in result.loss_games)  # MSE loss

    assert result.policy.shape == (e.n_states,)
    assert set(int(a) for a in result.policy) <= {0, 1, 2, 3}
    assert result.U.shape == (e.n_states,)
    # Q-values are 3 * <PauliZ> -> every utility lies in [-3, 3]
    assert np.all(np.abs(result.U) <= 3.0 + 1e-9)

    assert 0 < result.steps <= c.num_games * c.max_time
    assert result.U_ref is not None and result.U_ref.shape == (e.n_states,)

    # the weights actually moved
    assert not torch.allclose(result.var_Q_circuit, result.var_Q_circuit * 0)


def test_train_updates_weights_and_records_mse(qnode):
    """With eval_every=1 the periodic probe block runs and fills mse_u/good_Q/bad_Q."""
    c = tiny_cfg(num_games=3, max_time=2, eval_every=1)
    e = GridWorld(c)
    result = train(c, e, qnode=qnode)

    assert len(result.mse_u) == 2  # games m=1 and m=2 (m>0 and m%1==0)
    assert result.mse_games == [1, 2]
    assert len(result.good_Q) == 2 and len(result.bad_Q) == 2
    assert all(np.isfinite(v) for v in result.mse_u)
    assert result.mse_min == pytest.approx(min(result.mse_u))
    assert result.best_weights is not None


def test_train_is_deterministic_under_a_fixed_seed(qnode):
    """Same seed -> identical losses, rewards and final weights."""
    import torch

    c = tiny_cfg(num_games=2, max_time=3, seed=4321)
    e = GridWorld(c)
    a = train(c, e, qnode=qnode)
    b = train(c, e, qnode=qnode)

    assert a.loss_games == pytest.approx(b.loss_games)
    assert a.games_reward == pytest.approx(b.games_reward)
    assert np.array_equal(a.policy, b.policy)
    assert torch.equal(a.var_Q_circuit, b.var_Q_circuit)

    # ...and with the implicit default QNode too.  Building a `default.qubit`
    # device draws one np.random.randint from the global RNG (seed="global"), so
    # this would fail if `train` resolved the QNode lazily after seeding: the
    # first call would pay the draw and the second (cached) one would not.
    d = train(c, e)
    e2 = train(c, e)
    assert torch.equal(d.var_Q_circuit, e2.var_Q_circuit)
    assert torch.equal(a.var_Q_circuit, d.var_Q_circuit)


def test_different_seeds_diverge(qnode):
    import torch

    a = train(tiny_cfg(seed=1), GridWorld(tiny_cfg(seed=1)), qnode=qnode)
    b = train(tiny_cfg(seed=2), GridWorld(tiny_cfg(seed=2)), qnode=qnode)
    assert not torch.equal(a.var_Q_circuit, b.var_Q_circuit)


def test_env_determinism_is_reproducible_under_a_seed(env):
    """The environment RNG stream itself is seedable."""
    apply_seed(99)
    first = [env.next_position(env.random_state(), a) for a in [0, 1, 2, 3] * 3]
    apply_seed(99)
    second = [env.next_position(env.random_state(), a) for a in [0, 1, 2, 3] * 3]
    assert first == second


def test_td_pred_flag_changes_the_trajectory(qnode):
    """The td_pred_uses_next_state fix must actually change training."""
    import torch

    notebook = tiny_cfg(num_games=2, max_time=3, seed=7,
                        fixes=FixFlags.notebook())
    fixed = tiny_cfg(num_games=2, max_time=3, seed=7)   # default == fixed
    a = train(notebook, GridWorld(notebook), qnode=qnode)
    b = train(fixed, GridWorld(fixed), qnode=qnode)
    assert not torch.equal(a.var_Q_circuit, b.var_Q_circuit)


def test_train_rejects_degenerate_configs(env):
    with pytest.raises(ValueError):
        train(tiny_cfg(max_time=0), env)
    with pytest.raises(ValueError):
        train(tiny_cfg(len_batch=0), env)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_bellman_only_runs(capsys):
    from quantum_rl.__main__ import main

    assert main(["--bellman-only", "--no-plots"]) == 0
    out = capsys.readouterr().out
    assert "value iteration" in out
    assert "0.910414" in out


def test_cli_fix_flags_are_applied():
    from quantum_rl.__main__ import build_parser, make_config

    args = build_parser().parse_args(
        ["--episodes", "2", "--max-time", "2", "--bug", "td_pred_uses_next_state"]
    )
    c = make_config(args)
    assert c.fixes.td_pred_uses_next_state is True     # bug re-enabled
    assert c.fixes.epsilon_greedy_inverted is False    # default: fixed
    assert c.num_games == 2 and c.max_time == 2

    nb = make_config(build_parser().parse_args(["--notebook"]))
    assert nb.fixes == FixFlags.notebook()
    assert nb.lr_scheduler_step_size == 3 and nb.gamma == 0.95


def test_cli_writes_plots(tmp_path):
    from quantum_rl.__main__ import main

    out = tmp_path / "run"
    assert main(["--episodes", "2", "--max-time", "2", "--memory", "20",
                 "--seed", "3", "--out-dir", str(out)]) == 0
    for name in ("bellman_policy.pdf", "bellman_convergence.pdf", "policy.pdf",
                 "training_curves.pdf", "utility_comparison.pdf", "residuals.pdf",
                 "summary.json"):
        assert (out / name).exists(), f"{name} was not written"


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #
def test_every_fix_flag_defaults_to_the_fix():
    import dataclasses

    flags = FixFlags()
    for f in dataclasses.fields(flags):
        assert getattr(flags, f.name) is False, (
            f"{f.name} must default to False (= fixed behaviour)"
        )
    nb = FixFlags.notebook()
    for f in dataclasses.fields(nb):
        assert getattr(nb, f.name) is True


# --------------------------------------------------------------------------- #
# differential test against the literal notebook cell-50 transcription
# --------------------------------------------------------------------------- #
def test_train_reproduces_notebook_cell_50_bit_for_bit(env):
    """``train()`` must equal a statement-for-statement copy of cell 50.

    This is the test that makes the "faithful port" claim meaningful: the
    reference in ``tests/notebook_cell50_reference.py`` is the notebook loop
    copied verbatim, with its own device and its own transcribed
    ``variational_classifier`` / ``epsilon_greedy`` / ``Next_position`` /
    ``sample_random_batch``.  The tolerance is exactly ZERO on every
    accumulated quantity -- losses, discounted returns, the final weights and
    the cell-56 policy / utilities.

    Regression guarded: ``qml.device('default.qubit', ...)`` defaults to
    ``seed="global"`` and draws one ``np.random.randint`` when it is built, so
    a QNode created lazily *after* ``apply_seed`` silently desynchronises the
    entire run.  ``train()`` builds it before seeding; this test fails loudly if
    that ever regresses.
    """
    import torch

    import notebook_cell50_reference as nb

    num_games, max_time, N, seed = 6, 5, 25, 20240921
    c = Config.notebook(num_games=num_games, max_time=max_time, N=N, seed=seed)
    e = GridWorld(c)
    U_final, _ = value_iteration(e, c.gamma, c.max_epoch)

    ref = nb.run(e, U_final, num_games, max_time, N, seed, Gamma=c.gamma)
    got = train(c, e, U_ref=U_final)

    assert got.loss_games == ref["loss_games"]
    assert got.games_reward == ref["games_reward"]
    assert np.array_equal(got.policy, ref["policy"])
    assert np.array_equal(got.U, ref["U"])
    assert torch.equal(got.var_Q_circuit, ref["var_Q_circuit"])
    assert got.mse_u == ref["MAE_U"]
    assert got.good_Q == ref["good_Q"]
    assert got.bad_Q == ref["bad_Q"]
    assert got.good_diff_v == ref["good_diff_V"]
    assert got.bad_diff_v == ref["bad_diff_V"]


# --------------------------------------------------------------------------- #
# numpy simulator used by the GUI circuit view
# --------------------------------------------------------------------------- #
def test_statevector_matches_the_qnode():
    """The GUI computes Q(s, a) with a hand-written simulator: it must be the
    same circuit as the PennyLane QNode, for every state and several depths."""
    import torch

    from quantum_rl.circuit import (decimal_to_binary_fix_length, make_qnode,
                                    variational_classifier)
    from quantum_rl.statevector import q_values

    qnode = make_qnode()
    rng = np.random.default_rng(0)
    for layers in (1, 2, 5):
        W = rng.normal(0.0, 1.5, (layers, 4, 3))
        for s in range(12):
            ref = variational_classifier(
                torch.tensor(W), angles=decimal_to_binary_fix_length(4, s),
                qnode=qnode, scale=3.0).detach().numpy()
            assert np.allclose(q_values(W, s, 3.0), ref, atol=1e-12)


def test_dead_parameters_have_no_effect():
    """phi of layer 1 and omega of the last layer never change <Z>: the GUI
    greys them out instead of reporting a vanishing gradient."""
    from quantum_rl.circuit_view import dead_mask
    from quantum_rl.statevector import expvals, state_bits

    rng = np.random.default_rng(1)
    W = rng.normal(0.0, 1.0, (3, 4, 3))
    dead = dead_mask(3)
    for l in range(3):
        for i in range(4):
            for j in range(3):
                Wp = W.copy()
                Wp[l, i, j] += 0.7
                moved = max(np.abs(expvals(Wp, state_bits(s))
                                   - expvals(W, state_bits(s))).max()
                            for s in range(12))
                assert (moved < 1e-12) == dead[l, i, j], (l, i, j, moved)

# epsilon schedules
@pytest.mark.parametrize("mode", ["linear", "exponential", "cosine"])
def test_epsilon_decay_reaches_minimum_and_is_monotone(mode):
    c = Config(epsilon_decay_mode=mode, epsilon_decay_denom=10.0)
    values = [epsilon_for_episode(c, m) for m in range(12)]
    assert values[-1] == pytest.approx(c.epsilon_min)
    assert all(a >= b for a, b in zip(values, values[1:]))

def test_epsilon_constant_and_default_linear_parity():
    c = Config(epsilon_decay_denom=10.0)
    assert epsilon_for_episode(c, 0) == float(np.max([c.epsilon_0 - 1.0 / c.epsilon_decay_denom, c.epsilon_min]))
    constant = Config(epsilon_decay_mode="constant", epsilon_decay_denom=10.0)
    assert [epsilon_for_episode(constant, m) for m in (0, 10, 100)] == [constant.epsilon_0] * 3

@pytest.mark.parametrize("kwargs", [{"epsilon_decay_mode": "unknown"}, {"epsilon_decay_denom": 0}, {"epsilon_decay_denom": -1}])
def test_epsilon_decay_configuration_is_validated(kwargs):
    with pytest.raises(ValueError, match="epsilon_decay"):
        epsilon_for_episode(Config(**kwargs), 0)


@pytest.mark.parametrize("mode", ["linear", "exponential", "cosine"])
def test_epsilon_interval_and_explicit_duration(mode):
    stepped = Config(epsilon_decay_mode=mode, epsilon_decay_denom=10.0, epsilon_decay_interval=3)
    values = [epsilon_for_episode(stepped, m) for m in range(7)]
    assert values[:2] == [stepped.epsilon_0, stepped.epsilon_0]
    assert values[2] == pytest.approx(values[3])

    # Duration is an endpoint in episodes, even when it is not divisible by K.
    duration = Config(epsilon_decay_mode=mode, epsilon_decay_duration=7, epsilon_decay_interval=3)
    assert epsilon_for_episode(duration, 6) == pytest.approx(duration.epsilon_min)
    assert epsilon_for_episode(duration, 7) == pytest.approx(duration.epsilon_min)

def test_epsilon_interval_and_duration_are_validated():
    with pytest.raises(ValueError, match="epsilon_decay"):
        epsilon_for_episode(Config(epsilon_decay_interval=0), 0)
    with pytest.raises(ValueError, match="epsilon_decay"):
        epsilon_for_episode(Config(epsilon_decay_duration=0), 0)

def test_transient_lr_plan_has_exact_step_boundary():
    cfg = Config(LR_0=0.2, LR_AFTER_TRANSIENT=0.05, lr_transient_steps=3)
    assert [lr_for_optimizer_step(cfg, step) for step in range(1, 7)] == [0.2, 0.2, 0.2, 0.05, 0.05, 0.05]

def test_transient_lr_plan_defaults_to_initial_lr():
    cfg = Config(LR_0=0.2, lr_transient_steps=12)
    assert [lr_for_optimizer_step(cfg, step) for step in (1, 12, 13)] == [0.2, 0.2, 0.2]


def test_transient_lr_does_not_reset_after_scheduler_halving(qnode):
    c = tiny_cfg(
        num_games=4, max_time=1, eval_every=1,
        LR_0=0.2, LR_AFTER_TRANSIENT=0.05, lr_transient_steps=1,
        lr_scheduler_step_size=1, lr_scheduler_gamma=0.5, LR_MIN=1e-8,
    )
    seen = []
    train(c, GridWorld(c), qnode=qnode, monitor=lambda data: seen.append(data["lr"]))
    # Game 0 is the boundary and is not reported by the periodic monitor.
    # Every later episode halves the post-transient value; it must not jump
    # back to LR_AFTER_TRANSIENT before the next optimiser step.
    assert seen == pytest.approx([0.025, 0.0125, 0.00625])

def test_transient_lr_configuration_is_validated():
    c = tiny_cfg(lr_transient_steps=-1)
    with pytest.raises(ValueError, match="lr_transient_steps"):
        train(c, GridWorld(c))
    c = tiny_cfg(LR_AFTER_TRANSIENT=0.0)
    with pytest.raises(ValueError, match="LR_AFTER_TRANSIENT"):
        train(c, GridWorld(c))


def test_transient_lr_halving_clamps_at_lr_min(qnode):
    c = tiny_cfg(
        num_games=4, max_time=1, eval_every=1,
        LR_0=0.2, LR_AFTER_TRANSIENT=0.05, lr_transient_steps=1,
        lr_scheduler_step_size=1, lr_scheduler_gamma=0.5, LR_MIN=0.02,
    )
    seen = []
    train(c, GridWorld(c), qnode=qnode, monitor=lambda data: seen.append(data["lr"]))
    assert seen == pytest.approx([0.025, 0.02, 0.02])
