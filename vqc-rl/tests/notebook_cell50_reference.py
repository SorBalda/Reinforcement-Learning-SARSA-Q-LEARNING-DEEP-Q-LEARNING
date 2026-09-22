"""LITERAL transcription of ``StochFrozenLake_QUANTUM.ipynb`` cells 25/40-50/56.

This module exists so that :mod:`quantum_rl.train` can be checked *numerically*,
not just structurally: :func:`run` is the notebook's training loop copied
statement for statement, and ``tests/test_integration.py`` asserts that
``quantum_rl.train.train`` reproduces it bit for bit.

It deliberately does NOT import from :mod:`quantum_rl.train`, :mod:`quantum_rl.agent`
or :mod:`quantum_rl.circuit` -- those are the modules under test.  The grid-world
tables (``neigh_dist``, ``R``, the special-state indexes) *are* taken from
:class:`~quantum_rl.environment.GridWorld`, because that port was already
differentially verified against the notebook's own stored outputs and copying
cells 19/24 again here would only test the copy.

Forced deviations, both of them unavoidable and numerically neutral:

    * ``num_games`` / ``max_time`` / ``N`` are arguments -- the notebook's
      20000 x 250 is hours of statevector simulation;
    * ``from pennylane import numpy as np`` becomes plain ``numpy``.  The values
      of ``decimalToBinaryFixLength`` are bit-identical either way; the
      ``pennylane.numpy`` version additionally marks the *input angles* as
      trainable and emits a UserWarning.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# NOTE: the device is built at IMPORT time, exactly like notebook cell 42.
# `qml.device('default.qubit', ...)` defaults to seed="global", which draws one
# np.random.randint from the legacy global RNG; doing it here (and not inside
# `run`, after the seeding) is what the notebook does and what keeps the RNG
# stream comparable.
dtype = torch.DoubleTensor
dev = qml.device("default.qubit", wires=4)


# --------------------------------------------------------------------------- #
# notebook cell 21
# --------------------------------------------------------------------------- #
def prob_action(action):
    if action == 0:
        return [0.8, 0.1, 0.0, 0.1], [0, 1, 3]
    if action == 1:
        return [0.1, 0.8, 0.1, 0.0], [1, 2, 0]
    if action == 2:
        return [0.0, 0.1, 0.8, 0.1], [2, 3, 1]
    if action == 3:
        return [0.1, 0.0, 0.1, 0.8], [3, 0, 2]


# --------------------------------------------------------------------------- #
# notebook cells 40-44
# --------------------------------------------------------------------------- #
def decimalToBinaryFixLength(_length, _decimal):
    binNum = bin(int(_decimal))[2:]
    outputNum = [int(item) for item in binNum]
    if len(outputNum) < _length:
        outputNum = np.concatenate(
            (np.zeros((_length - len(outputNum),)), np.array(outputNum))
        )
    else:
        outputNum = np.array(outputNum)
    return outputNum


def statepreparation(a):
    for ind in range(len(a)):
        qml.RX(np.pi * a[ind], wires=ind)
        qml.RZ(np.pi * a[ind], wires=ind)


def layer(W):
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)


@qml.qnode(dev, interface="torch")
def circuit(weights, angles=None):
    for W in weights:
        statepreparation(angles)
        layer(W)
    return [qml.expval(qml.PauliZ(ind)) for ind in range(4)]


def variational_classifier(var_Q_circuit, angles=None):
    weights = var_Q_circuit
    raw_output = circuit(weights, angles=angles)
    for i in range(len(raw_output)):
        raw_output[i] = raw_output[i] * 3.0
    return raw_output


# --------------------------------------------------------------------------- #
# notebook cells 45 / 47 / 48
# --------------------------------------------------------------------------- #
def epsilon_greedy(var_Q_circuit, epsilon, n_action, s, train=False):
    if train or np.random.rand() < ((epsilon)):
        action = torch.argmax(
            torch.tensor(
                variational_classifier(
                    var_Q_circuit=var_Q_circuit.clone().detach(),
                    angles=decimalToBinaryFixLength(4, s),
                )
            )
        )
        action = action.item()
    else:
        action = np.random.randint(0, 4)
    return action


def sample_random_batch(D, batch_len):
    output = []
    for i in range(batch_len):
        random_index = int(np.random.rand() * len(D))
        output.append(D[random_index])
    return output


loss_fnc = nn.MSELoss()


# --------------------------------------------------------------------------- #
# notebook cell 50 (+ cell 25 for Next_position / random_state, + cell 56)
# --------------------------------------------------------------------------- #
def run(env, U_final, num_games, max_time, N, seed, Gamma=0.99):
    """The cell-50 loop verbatim.  ``env`` only supplies the grid-world tables."""
    neigh_dist = np.asarray(env.neigh_dist, dtype=int)
    R = np.asarray(env.R, dtype=float)
    ALIVE_indexes = list(env.alive_indexes)
    DEATH_indexes = list(env.death_indexes)
    OBSTACLES_indexes = list(env.obstacle_indexes)
    Nx, Ny = env.Nx, env.Ny

    # ---- cell 25 ---------------------------------------------------------
    def Next_position(initial, action, control=False, stochastic=False):
        prob, action_indexes = prob_action(action)
        rnd = np.random.rand()
        if rnd < prob[action_indexes[0]]:
            final = initial + neigh_dist[initial, action_indexes[0]]
        if (rnd > prob[action_indexes[0]]
                and rnd < prob[action_indexes[0]] + prob[action_indexes[1]]):
            final = initial + neigh_dist[initial, action_indexes[1]]
        if rnd > prob[action_indexes[1]] + prob[action_indexes[0]]:
            final = initial + neigh_dist[initial, action_indexes[2]]
        if not stochastic:
            final = initial + neigh_dist[initial, action_indexes[0]]
        return int(final)

    def random_state():
        output = int(np.random.rand() * 12)
        while ((output in OBSTACLES_indexes) or (output in ALIVE_indexes)
               or (output in DEATH_indexes)):
            output = int(np.random.rand() * 12)
        return output

    # ---- cell 50 ---------------------------------------------------------
    np.random.seed(seed)
    import random as _random

    _random.seed(seed)
    torch.manual_seed(seed)

    LR_0 = 0.2
    LR_MIN = 0.001
    epsilon_0 = 0.95
    len_batch = 1
    epsilon_min = 0.05
    C = 200
    c = 0
    count = 0
    D = []

    MAE_U = []
    good_Q = []
    bad_Q = []
    deep_layers = 2
    W = np.random.rand(deep_layers, 4, 3) * 1.3  # dead, but consumes RNG
    W_before = W.copy()  # noqa: F841

    while True:
        if count >= N:
            break
        s_i = random_state()
        while True:
            action = int(np.random.rand() * 4)
            s_f = Next_position(s_i, action)
            reward = R[s_f]
            D.append([s_i, action, reward, s_f])
            count = count + 1
            if count >= N:
                break
            if (s_f in ALIVE_indexes) or (s_f in DEATH_indexes):
                break
            s_i = s_f

    epsilon = epsilon_0
    loss_games = []
    games_reward = []
    good_diff_V = []
    bad_diff_V = []
    var_init_circuit = torch.tensor(
        0.3 * np.random.randn(deep_layers, 4, 3), device="cpu", requires_grad=True
    ).type(dtype)
    var_Q_circuit = var_init_circuit
    var_target_Q_circuit = var_Q_circuit.clone().detach()

    opt = torch.optim.RMSprop([var_Q_circuit], lr=LR_0, alpha=0.99, eps=1e-08,
                              weight_decay=0, momentum=0, centered=False)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    MAE_min = 1000.0
    best_weights = None

    for m in range(num_games):
        s_1 = random_state()
        reward_history = []
        game_reward = 0.0
        cntr = 0
        loss_game = 0
        for time in range(max_time):
            a = epsilon_greedy(var_Q_circuit=var_Q_circuit, epsilon=epsilon,
                               n_action=4, s=s_1)
            s_2 = Next_position(s_1, a)
            reward = R[s_2]
            D.append([s_1, a, reward, s_2])
            s_1 = s_2
            if len(D) >= N:
                D.pop(0)

            batch = sample_random_batch(D, len_batch)
            loss = 0.0
            for j in range(len_batch):
                s_j, action_j, reward_j, s_jf = (batch[j][0], batch[j][1],
                                                 batch[j][2], batch[j][3])
                if (s_jf in ALIVE_indexes) or (s_jf in DEATH_indexes):
                    y = reward_j + max(variational_classifier(
                        var_Q_circuit=var_target_Q_circuit,
                        angles=decimalToBinaryFixLength(4, s_jf))) * 0.0
                else:
                    y = reward_j + Gamma * max(variational_classifier(
                        var_Q_circuit=var_target_Q_circuit,
                        angles=decimalToBinaryFixLength(4, s_jf)))
                pred = variational_classifier(
                    var_Q_circuit=var_Q_circuit,
                    angles=decimalToBinaryFixLength(4, s_jf))[action_j]
                loss += loss_fnc(pred, y) / len_batch

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_game += loss.item()
            reward_history.append(reward)
            cntr += 1
            c += 1
            if c >= C:
                var_target_Q_circuit = var_Q_circuit.clone().detach()
                c = 0
            if (s_2 in ALIVE_indexes) or (s_2 in DEATH_indexes):
                break

        for esponent in range(len(reward_history)):
            game_reward += reward_history[esponent] * Gamma**esponent
        games_reward.append(game_reward)
        loss_games.append(loss_game / cntr)
        epsilon = np.max([epsilon_0 - (float(m + 1) / 2000), epsilon_min])

        if opt.param_groups[0]["lr"] > LR_MIN:
            scheduler.step()
        if m % 20 == 0 and m > 0:
            good_Q_values = variational_classifier(
                var_Q_circuit=var_Q_circuit, angles=decimalToBinaryFixLength(4, 10))
            good_V = torch.max(torch.tensor(good_Q_values)).item()
            good_diff_V.append(good_V - U_final[10])
            good_Q_print = good_Q_values[1]

            bad_Q_values = variational_classifier(
                var_Q_circuit=var_Q_circuit, angles=decimalToBinaryFixLength(4, 3))
            bad_V = torch.max(torch.tensor(bad_Q_values)).item()
            bad_diff_V.append(bad_V - U_final[3])
            bad_Q_print = bad_Q_values[0]

            MAE_value = 0.0
            for i in range(12):
                if ((i not in ALIVE_indexes) and (i not in DEATH_indexes)
                        and (i not in OBSTACLES_indexes)):
                    U_tmp = torch.max(torch.tensor(variational_classifier(
                        var_Q_circuit=var_Q_circuit,
                        angles=decimalToBinaryFixLength(4, i))))
                    MAE_value += (U_tmp - U_final[i].item()) ** 2 / 9.0
            MAE_U.append(MAE_value)
            good_Q.append(good_Q_print)
            bad_Q.append(bad_Q_print)
            if m > 0:
                if MAE_value < MAE_min:
                    best_weights = var_Q_circuit.clone().detach()
                    MAE_min = MAE_value
        cntr = 0
        loss_game = 0.0

    # ---- cell 56 ---------------------------------------------------------
    policy_DEEP = np.zeros(Nx * Ny)
    U_DEEP = np.zeros(Nx * Ny)
    for s in range(Nx * Ny):
        policy_DEEP[s] = torch.argmax(torch.tensor(variational_classifier(
            var_Q_circuit=var_Q_circuit, angles=decimalToBinaryFixLength(4, s))))
        U_DEEP[s] = torch.max(torch.tensor(variational_classifier(
            var_Q_circuit=var_Q_circuit,
            angles=decimalToBinaryFixLength(4, s)))).item()

    return dict(
        var_Q_circuit=var_Q_circuit,
        loss_games=loss_games,
        games_reward=games_reward,
        MAE_U=[float(v) for v in MAE_U],
        good_Q=[float(v.item()) for v in good_Q],
        bad_Q=[float(v.item()) for v in bad_Q],
        good_diff_V=good_diff_V,
        bad_diff_V=bad_diff_V,
        MAE_min=float(MAE_min),
        best_weights=best_weights,
        policy=policy_DEEP,
        U=U_DEEP,
        D=D,
    )
