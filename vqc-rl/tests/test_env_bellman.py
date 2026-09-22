"""Tests for quantum_rl.config / environment / bellman.

Anchors come from two places:

* the values verified independently for ``gamma=0.99`` (the default of
  :class:`quantum_rl.config.Config`), and
* the outputs stored inside ``StochFrozenLake_QUANTUM.ipynb`` itself, which
  were produced with ``Gamma=0.95`` (cells 32, 34, 35, 37).  Reproducing those
  digit for digit is the real proof that the port is faithful.

Run with:  ./.venv/bin/python -m pytest tests/test_env_bellman.py
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_rl.bellman import greedy_policy, value_iteration, value_iteration_detailed
from quantum_rl.config import Config, FixFlags
from quantum_rl.environment import GridWorld, prob_action

CORRECT = [0, 1, 2, 3, 4, 6, 8, 9, 10]  # non-terminal, non-obstacle

# notebook cell 37 / verified reference, gamma = 0.99
U_FINAL_099 = [
    0.910414, 0.894075, 0.876564, 0.759436, 0.929002,
    0.865424, 0.945782, 0.964872, 0.982105,
]
# notebook cell 32 stored output, Gamma = 0.95 (the `U` array, after the
# manual +1/-1 override)
U_NOTEBOOK_095 = np.array([
    0.71087955, 0.66205982, 0.62375952, 0.40934501, 0.76901356, 0.0,
    0.68775684, -1.0, 0.82486972, 0.89138264, 0.95528939, 1.0,
])
# notebook cell 35 stored output
POLICY_NOTEBOOK_095 = np.array([0., 3., 3., 3., 0., 0., 0., 0., 1., 1., 1., 0.])


@pytest.fixture
def env():
    return GridWorld(Config())


@pytest.fixture
def nb_env():
    """Grid world with the original notebook behaviour (deterministic moves)."""
    return GridWorld(Config.notebook())


# ---------------------------------------------------------------- config ---
def test_default_flags_are_the_fixes():
    """Since the root-cause analysis the defaults are CORRECTED, not notebook."""
    import dataclasses
    f = FixFlags()
    for fld in dataclasses.fields(f):
        assert getattr(f, fld.name) is False, f"{fld.name} must default to the fix"


def test_notebook_flags_restore_every_bug():
    import dataclasses
    f = FixFlags.notebook()
    for fld in dataclasses.fields(f):
        assert getattr(f, fld.name) is True, f"{fld.name} must be True in notebook()"


def test_config_defaults():
    c = Config()
    assert (c.Nx, c.Ny, c.n_states, c.n_actions) == (4, 3, 12, 4)
    assert c.r == -0.004 and c.gamma == 0.99 and c.max_epoch == 100
    assert c.num_games == 20000 and c.max_time == 250 and c.N == 700
    assert c.deep_layers == 2 and c.n_wires == 4 and c.q_scale == 3.0
    assert c.seed is None and isinstance(c.fixes, FixFlags)


# ----------------------------------------------------------- environment ---
def test_index_mapping(env):
    assert env.position_to_index([2, 1]) == 6
    assert env.index_to_position(6) == (2, 1)
    assert env.index_to_position(11) == (3, 2)
    for i in range(env.n_states):
        assert env.position_to_index(env.index_to_position(i)) == i


def test_special_indexes(env):
    assert env.alive_indexes == [11]
    assert env.death_indexes == [7]
    assert env.obstacle_indexes == [5]
    assert env.is_terminal(11) and env.is_terminal(7)
    # the obstacle is deliberately NOT terminal in the notebook
    assert not env.is_terminal(5)
    assert not any(env.is_terminal(i) for i in CORRECT)


def test_rewards(env):
    assert env.R[11] == +1.0
    assert env.R[7] == -1.0
    assert env.R[5] == 0.0  # obstacle is explicitly zeroed, not `r`
    for i in CORRECT:
        assert env.R[i] == -0.004


def test_neigh_dist_table(env):
    expected = np.array([
        [4, 1, 0, 0],
        [0, 1, 0, -1],
        [4, 1, 0, -1],
        [4, 0, 0, -1],
        [4, 0, -4, 0],
        [0, 0, 0, 0],   # obstacle
        [4, 1, -4, 0],
        [0, 0, 0, 0],   # death
        [0, 1, -4, 0],
        [0, 1, 0, -1],
        [0, 1, -4, -1],
        [0, 0, 0, 0],   # goal
    ], dtype=float)
    assert np.array_equal(env.neigh_dist, expected)


def test_prob_action():
    assert prob_action(0) == ([0.8, 0.1, 0.0, 0.1], [0, 1, 3])
    assert prob_action(1) == ([0.1, 0.8, 0.1, 0.0], [1, 0, 2])
    assert prob_action(2) == ([0.0, 0.1, 0.8, 0.1], [2, 1, 3])
    assert prob_action(3) == ([0.1, 0.0, 0.1, 0.8], [3, 0, 2])
    for a in range(4):
        p, _ = prob_action(a)
        assert sum(p) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        prob_action(4)


def test_transition_rows_sum(env):
    for s0 in range(env.n_states):
        for a in range(4):
            total = sum(env.transition_probability(s1, s0, a) for s1 in range(12))
            expected = 0.0 if s0 in (7, 11) else 1.0
            assert total == pytest.approx(expected), (s0, a)


def test_obstacle_is_an_absorbing_self_loop(env):
    for a in range(4):
        assert env.transition_probability(5, 5, a) == pytest.approx(1.0)


def test_transition_accumulates_blocked_directions(env):
    # from state 0, "up" (offset +4) succeeds w.p. 0.8; right -> 1 w.p. 0.1;
    # left is blocked (offset 0) so its 0.1 lands back on state 0 itself.
    assert env.transition_probability(4, 0, 0) == pytest.approx(0.8)
    assert env.transition_probability(1, 0, 0) == pytest.approx(0.1)
    assert env.transition_probability(0, 0, 0) == pytest.approx(0.1)
    # from state 3, "down": down and right are both blocked -> 0.8 + 0.1 add up
    assert env.transition_probability(3, 3, 2) == pytest.approx(0.9)


def test_next_position_notebook_is_deterministic(nb_env):
    # notebook: stochastic=False, the sampled outcome is overwritten
    env = nb_env
    np.random.seed(0)
    assert {env.next_position(0, 0) for _ in range(300)} == {4}
    assert env.next_position(6, 0) == 10
    assert env.next_position(3, 1) == 3  # blocked by the grid edge -> stay


def test_next_position_consumes_one_draw_even_when_deterministic(env):
    # RNG-stream fidelity: the notebook draws rnd before the override, so a
    # seeded run must consume exactly one uniform per call in both modes.
    for stochastic in (False, True):
        np.random.seed(5)
        for _ in range(50):
            env.next_position(0, 1, stochastic=stochastic)
        after = np.random.rand()
        np.random.seed(5)
        np.random.rand(50)
        assert after == np.random.rand()


def test_next_position_slip_distribution():
    cfg = Config(fixes=FixFlags(env_deterministic_transitions=False))
    env = GridWorld(cfg)
    np.random.seed(42)
    n = 20000
    draws = [env.next_position(0, 0) for _ in range(n)]
    freq = {k: draws.count(k) / n for k in (4, 1, 0)}
    assert freq[4] == pytest.approx(0.8, abs=0.02)  # up
    assert freq[1] == pytest.approx(0.1, abs=0.02)  # slip right
    assert freq[0] == pytest.approx(0.1, abs=0.02)  # slip left -> blocked, stay
    assert set(draws) == {0, 1, 4}


def test_next_position_generator_path(env):
    rng = np.random.default_rng(0)
    out = [env.next_position(0, 0, rng, stochastic=True) for _ in range(500)]
    assert set(out) <= {0, 1, 4}
    # an explicit Generator must not touch the legacy global stream
    np.random.seed(11)
    before = np.random.rand()
    np.random.seed(11)
    env.next_position(0, 0, np.random.default_rng(3), stochastic=True)
    assert np.random.rand() == before


def test_random_state(env):
    np.random.seed(3)
    out = {env.random_state() for _ in range(2000)}
    assert out == set(CORRECT)


# --------------------------------------------------------------- bellman ---
def test_value_iteration_anchor_gamma_099(env):
    U_final, U_time = value_iteration(env, 0.99, 100)
    assert U_time.shape == (100, 12)
    assert np.all(U_time[0] == 0.0)  # loop runs range(max_epoch-1), row 0 unwritten
    assert np.array_equal(U_final, U_time[99])
    got = [U_final[i] for i in CORRECT]
    assert got == pytest.approx(U_FINAL_099, abs=1e-5)


def test_terminal_and_obstacle_utilities_are_zero(env):
    U_final, _ = value_iteration(env, 0.99, 100)
    assert U_final[5] == 0.0   # obstacle self-loop with R = 0
    assert U_final[7] == 0.0   # no outgoing transition probability
    assert U_final[11] == 0.0


def test_manual_override_only_touches_the_policy_array(env):
    res = value_iteration_detailed(env, 0.99, 100)
    assert res.U_policy[11] == +1.0
    assert res.U_policy[7] == -1.0
    assert res.U_final[11] == 0.0 and res.U_final[7] == 0.0
    # U_policy is the iterate BEFORE U_final (Jacobi snapshot of the last sweep)
    lagged = res.U_policy.copy()
    lagged[11] = res.U_time[98][11]
    lagged[7] = res.U_time[98][7]
    assert np.array_equal(lagged, res.U_time[98])


def test_reproduces_notebook_stored_output_gamma_095(nb_env):
    env = nb_env
    res = value_iteration_detailed(env, 0.95, 100)
    assert res.U_policy == pytest.approx(U_NOTEBOOK_095, abs=5e-9)
    nb_final = U_NOTEBOOK_095.copy()
    nb_final[7] = 0.0
    nb_final[11] = 0.0
    assert res.U_final == pytest.approx(nb_final, abs=5e-9)
    policy = greedy_policy(env, res.U_policy, 0.95)
    assert np.array_equal(policy, POLICY_NOTEBOOK_095)


def test_greedy_policy_lookahead_sums_match_notebook_cell_34(env):
    res = value_iteration_detailed(env, 0.95, 100)
    expected = {
        0: [0.75250479, 0.67763717, 0.70599757, 0.71669295],
        3: [-0.69668955, 0.26841051, 0.43078646, 0.43994212],
        6: [0.73300720, -0.64209511, 0.46778330, 0.70811036],
        10: [0.95336978, 0.96430462, 0.73934373, 0.87741073],
    }
    for i, exp in expected.items():
        mine = [
            sum(env.transition_probability(j, i, a) * res.U_policy[j] for j in range(12))
            for a in range(4)
        ]
        assert mine == pytest.approx(exp, abs=5e-8)


def test_greedy_policy_reward_fix_changes_state_3_at_gamma_095(env):
    res = value_iteration_detailed(env, 0.95, 100)
    fixed = greedy_policy(env, res.U_policy, 0.95,
                          fixes=FixFlags(greedy_policy_ignores_reward=False))
    assert fixed[3] == 2.0 and POLICY_NOTEBOOK_095[3] == 3.0
    assert fixed[6] == 3.0 and POLICY_NOTEBOOK_095[6] == 0.0


def test_value_iteration_rejects_zero_epochs(env):
    with pytest.raises(ValueError):
        value_iteration(env, 0.99, 0)
