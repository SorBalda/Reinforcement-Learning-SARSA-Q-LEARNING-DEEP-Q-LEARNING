"""Attributing the blame: is it the quantum circuit or the training loop?

The decisive experiment is a CONTROL: take the training loop of the notebook
StochFrozenLake_QUANTUM.ipynb (cell 50) verbatim and replace the approximator
alone -- the variational circuit -- with an exact Q table.

A Q table is the perfect approximator: 48 free parameters for 48 values, no
expressivity limit, no barren plateau, no simulation noise, exact gradients.  If
the loop converges to the Bellman solution in tabular form, then the logic is
sound and the circuit is to blame.  If it fails IN TABULAR FORM TOO, the logic is
to blame and the circuit is innocent.

Two suspects, found by reading cell 50 and cell 45, are ablated:

  td_pred_uses_next_state
      Il notebook scrive
          pred = variational_classifier(var_Q_circuit,
                                        angles=decimalToBinaryFixLength(4, s_jf))[action_j]
      indexing the prediction on s_jf, the ARRIVAL state, while the target is
      computed on s_jf as well.  The classical notebook (DQN) uses Q(phi_j), the
      DEPARTURE state, instead.  True = notebook behaviour.

  epsilon_greedy_inverted
      Il notebook (cella 45) scrive
          if train or np.random.rand() < epsilon:  -> argmax (greedy)
          else:                                    -> azione casuale
      that is, it is GREEDY with probability epsilon.  With epsilon starting at
      0.95 and decaying to 0.05, the agent explores 5% of the time at the start
      and 95% at the end: exactly backwards.  True = notebook behaviour.

Usage:
    ../.venv/bin/python -m diagnostics.loop_ablation
    python3 diagnostics/loop_ablation.py          # needs neither torch nor pennylane

NOTE on duplication: this module carries a minimal copy of the environment so it
stays runnable with numpy alone and independent of the quantum_rl/ package.  Once
quantum_rl/environment.py and bellman.py are stable, this file should be
refactored to import them and the copy deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------- #
# Stochastic 4x3 grid world (AIMA ch. 17) -- minimal copy, see the note above
# --------------------------------------------------------------------------- #

NX, NY = 4, 3
N_STATES = NX * NY
N_ACTIONS = 4
GOAL, DEATH, OBSTACLE = 11, 7, 5
LIVING_REWARD = -0.004

#: WARNING: the two notebooks use DIFFERENT discount factors.
#: StochFrozenLake_QUANTUM.ipynb cella 11          -> Gamma = 0.95
#: Stochastic_frozen_lake_...Learning.ipynb cella 5 -> Gamma = 0.99
#: The ablation runs on both to show the diagnosis does not depend on gamma.
GAMMA_QUANTUM = 0.95
GAMMA_CLASSICAL = 0.99
GAMMA = GAMMA_QUANTUM

#: the states the notebook analyses: neither terminal nor obstacle
CORRECT_INDEXES = [s for s in range(N_STATES) if s not in (GOAL, DEATH, OBSTACLE)]

#: slip model: 0.8 in the intended direction, 0.1 for each perpendicular one
_SLIP = {
    0: ([0.8, 0.1, 0.0, 0.1], [0, 1, 3]),  # su
    1: ([0.1, 0.8, 0.1, 0.0], [1, 0, 2]),  # destra
    2: ([0.0, 0.1, 0.8, 0.1], [2, 1, 3]),  # giu'
    3: ([0.1, 0.0, 0.1, 0.8], [3, 0, 2]),  # sinistra
}


def build_neigh_dist() -> np.ndarray:
    """OFFSET table: next = s + neigh_dist[s, a].  Offset 0 = blocked move."""
    nd = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    for s in range(N_STATES):
        nd[s, 0] = 0 if (s + NX >= N_STATES or s + NX == OBSTACLE) else NX
        nd[s, 1] = 0 if ((s + 1) % NX == 0 or s + 1 == OBSTACLE) else 1
        nd[s, 2] = 0 if (s - NX < 0 or s - NX == OBSTACLE) else -NX
        nd[s, 3] = 0 if (s % NX == 0 or s - 1 == OBSTACLE) else -1
        if s in (OBSTACLE, GOAL, DEATH):
            nd[s, :] = 0
    return nd


NEIGH_DIST = build_neigh_dist()

REWARD = np.full(N_STATES, LIVING_REWARD)
REWARD[GOAL] = 1.0
REWARD[DEATH] = -1.0
REWARD[OBSTACLE] = 0.0


def _set_living_reward(value: float) -> None:
    """Rebuild the reward array for a different living reward."""
    global LIVING_REWARD
    LIVING_REWARD = value
    REWARD[:] = value
    REWARD[GOAL] = 1.0
    REWARD[DEATH] = -1.0
    REWARD[OBSTACLE] = 0.0


def is_terminal(s: int) -> bool:
    return s in (GOAL, DEATH)


def next_position(s: int, a: int, rng: np.random.Generator) -> int:
    """Sampled transition function (inverse CDF over the 3 possible outcomes)."""
    prob, idx = _SLIP[a]
    u = rng.random()
    c0 = prob[idx[0]]
    c1 = c0 + prob[idx[1]]
    if u < c0:
        d = idx[0]
    elif u < c1:
        d = idx[1]
    else:
        d = idx[2]
    return int(s + NEIGH_DIST[s, d])


def transition_probability(s1: int, s0: int, a: int) -> float:
    """Analytic P(s1 | s0, a).  Blocked directions add up on the same s0."""
    if s0 in (GOAL, DEATH):
        return 0.0
    prob, _ = _SLIP[a]
    return float(sum(prob[d] for d in range(N_ACTIONS) if s1 == s0 + NEIGH_DIST[s0, d]))


def random_state(rng: np.random.Generator) -> int:
    return int(rng.integers(N_STATES))


# --------------------------------------------------------------------------- #
# Bellman baseline (ground truth)
# --------------------------------------------------------------------------- #


def value_iteration(gamma: float = GAMMA, max_epoch: int = 500) -> np.ndarray:
    """Synchronous (Jacobi) value iteration, reward-on-arrival formulation."""
    u_next = np.zeros(N_STATES)
    for _ in range(max_epoch - 1):
        u = u_next.copy()
        for s0 in range(N_STATES):
            u_next[s0] = max(
                sum(
                    transition_probability(s1, s0, a) * (gamma * u[s1] + REWARD[s1])
                    for s1 in range(N_STATES)
                )
                for a in range(N_ACTIONS)
            )
    return u_next


def greedy_policy(u: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    pol = np.zeros(N_STATES, dtype=int)
    for s0 in range(N_STATES):
        scores = [
            sum(
                transition_probability(s1, s0, a) * (gamma * u[s1] + REWARD[s1])
                for s1 in range(N_STATES)
            )
            for a in range(N_ACTIONS)
        ]
        pol[s0] = int(np.argmax(scores))
    return pol


# --------------------------------------------------------------------------- #
# The notebook training loop, with a tabular Q
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoopConfig:
    """The two flags under test.  True = notebook behaviour."""

    td_pred_uses_next_state: bool = True
    epsilon_greedy_inverted: bool = True

    num_games: int = 1500
    max_time: int = 60
    epsilon_0: float = 0.95
    epsilon_min: float = 0.05
    epsilon_decay_denom: float = 2000.0  # eps = max(eps0 - (m+1)/denom, eps_min)
    replay_size: int = 700  # N
    len_batch: int = 1
    target_sync_steps: int = 200  # C, counted in STEPS as in the notebook
    lr: float = 0.05  # tabular analogue of the optimiser step
    gamma: float = GAMMA


def select_action(q: np.ndarray, s: int, epsilon: float, cfg: LoopConfig,
                  rng: np.random.Generator) -> int:
    """The epsilon_greedy of cell 45, with the inverted branch selectable."""
    if cfg.epsilon_greedy_inverted:
        greedy = rng.random() < epsilon          # notebook: greedy WITH prob. epsilon
    else:
        greedy = rng.random() >= epsilon         # conventional: greedy with prob. 1-epsilon
    return int(np.argmax(q[s])) if greedy else int(rng.integers(N_ACTIONS))


def train_tabular(cfg: LoopConfig, seed: int) -> np.ndarray:
    """Reproduce cell 50 with a Q table in place of the variational circuit."""
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    q_target = q.copy()

    # pre-fill the replay memory with random transitions (as the notebook does)
    memory: list[tuple[int, int, float, int]] = []
    while len(memory) < cfg.replay_size:
        s = random_state(rng)
        while len(memory) < cfg.replay_size:
            a = int(rng.integers(N_ACTIONS))
            s_f = next_position(s, a, rng)
            memory.append((s, a, float(REWARD[s_f]), s_f))
            if is_terminal(s_f):
                break
            s = s_f

    epsilon = cfg.epsilon_0
    step_counter = 0

    for m in range(cfg.num_games):
        s_1 = random_state(rng)
        for _ in range(cfg.max_time):
            a = select_action(q, s_1, epsilon, cfg, rng)
            s_2 = next_position(s_1, a, rng)
            memory.append((s_1, a, float(REWARD[s_2]), s_2))
            if len(memory) >= cfg.replay_size:
                memory.pop(0)
            s_1 = s_2

            # minibatch sampled WITH replacement, like sample_random_batch
            for _ in range(cfg.len_batch):
                s_j, a_j, r_j, s_jf = memory[int(rng.random() * len(memory))]
                if is_terminal(s_jf):
                    y = r_j                       # the notebook multiplies the max by 0.0
                else:
                    y = r_j + cfg.gamma * float(np.max(q_target[s_jf]))
                # THE CRUCIAL POINT: which state the prediction is indexed on
                s_pred = s_jf if cfg.td_pred_uses_next_state else s_j
                q[s_pred, a_j] += 2.0 * cfg.lr * (y - q[s_pred, a_j])

            step_counter += 1
            if step_counter >= cfg.target_sync_steps:
                q_target = q.copy()
                step_counter = 0

            if is_terminal(s_2):
                break

        epsilon = max(cfg.epsilon_0 - (m + 1) / cfg.epsilon_decay_denom, cfg.epsilon_min)

    return q


# --------------------------------------------------------------------------- #
# Metrics and presentation
# --------------------------------------------------------------------------- #


def evaluate(q: np.ndarray, u_ref: np.ndarray, pol_ref: np.ndarray) -> dict:
    v = q.max(axis=1)
    pol = q.argmax(axis=1)
    idx = CORRECT_INDEXES
    return {
        "mse": float(np.mean((v[idx] - u_ref[idx]) ** 2)),
        "v_mean": float(np.mean(v[idx])),
        "policy_match": int(sum(pol[s] == pol_ref[s] for s in idx)),
        "action_spread": float(np.mean(q[idx].max(axis=1) - q[idx].min(axis=1))),
    }


SCENARIOS = [
    ("A  notebook as-is", dict(td_pred_uses_next_state=True, epsilon_greedy_inverted=True)),
    ("B  epsilon fixed only", dict(td_pred_uses_next_state=True, epsilon_greedy_inverted=False)),
    ("C  pred-state fixed only", dict(td_pred_uses_next_state=False, epsilon_greedy_inverted=True)),
    ("D  both fixed", dict(td_pred_uses_next_state=False, epsilon_greedy_inverted=False)),
]
SEEDS = [0, 1, 2, 3, 4]


def run_ablation(gamma: float, label: str, budget: dict | None = None) -> None:
    u_ref = value_iteration(gamma=gamma)
    pol_ref = greedy_policy(u_ref, gamma=gamma)
    idx = CORRECT_INDEXES

    print("=" * 78)
    print("gamma = %.2f  (%s)" % (gamma, label))
    print("=" * 78)
    print("Bellman U over the 9 states :", np.round(u_ref[idx], 4).tolist())
    print("mean U                      : %.4f" % float(np.mean(u_ref[idx])))
    print("theoretical fixed point of scenario A, R/(1-gamma) : %+.3f"
          % (LIVING_REWARD / (1 - gamma)))
    print()
    header = "%-30s %10s %10s %10s %10s" % ("scenario", "MSE vs U", "mean V", "policy/9", "spread_a")
    print(header)
    print("-" * len(header))

    for name, flags in SCENARIOS:
        cfg = replace(LoopConfig(), gamma=gamma, **(budget or {}), **flags)
        runs = [evaluate(train_tabular(cfg, s), u_ref, pol_ref) for s in SEEDS]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        print("%-30s %10.4f %10.4f %10.1f %10.4f"
              % (name, agg["mse"], agg["v_mean"], agg["policy_match"], agg["action_spread"]))
    print()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reward", "-r", type=float, default=None,
                    help="living reward: what every non-terminal state pays, "
                         "including staying put on a blocked move (default: %g)"
                         % LIVING_REWARD)
    ap.add_argument("--gamma", type=float, default=None,
                    help="run a single discount instead of both 0.95 and 0.99")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-time", type=int, default=None)
    a = ap.parse_args()

    if a.reward is not None:
        _set_living_reward(a.reward)
    budget = {}
    if a.episodes is not None:
        budget["num_games"] = a.episodes
    if a.max_time is not None:
        budget["max_time"] = a.max_time

    print()
    print("ABLATION -- same notebook loop, TABULAR Q")
    print("(exact approximator: no expressivity limit, nothing quantum)")
    shown = replace(LoopConfig(), **budget)
    print("%d episodes x %d steps, averaged over %d seeds"
          % (shown.num_games, shown.max_time, len(SEEDS)))
    print()

    print("living reward r = %g" % LIVING_REWARD)
    print()

    if a.gamma is not None:
        run_ablation(a.gamma, "chosen on the command line", budget)
    else:
        run_ablation(GAMMA_QUANTUM, "value from the QUANTUM notebook, cell 11", budget)
        run_ablation(GAMMA_CLASSICAL, "value from the CLASSICAL notebook, cell 5", budget)

    print("Legend:")
    print("  MSE vs U   mean squared error of V=max_a Q against Bellman (0 = perfect)")
    print("  mean V     mean learned value over the 9 states")
    print("  policy/9   how many of the 9 states got the Bellman-optimal action (9 = perfect)")
    print("  spread_a   mean max_a Q - min_a Q: if ~0 the Q cannot tell actions apart")
    print("             and the policy is degenerate (argmax over identical values)")
    print("  chance     policy expected from pure chance: 9/4 = 2.25")


if __name__ == "__main__":
    main()
