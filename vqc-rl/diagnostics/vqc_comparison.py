"""Comparison on the REAL VQC: the notebook as it was vs every fix enabled.

`loop_ablation.py` proves with a tabular Q that the training loop was broken
regardless of the approximator.  This script closes the loop using the real
4-qubit variational circuit, through the `quantum_rl/` package.

Configurations compared:

  A  notebook  every FixFlag True (original behaviour) and deterministic
               transitions, i.e. `stochastic=False` as in cell 25
  B  fixed     every FixFlag False + STOCHASTIC environment + a sane learning
               rate schedule

The scheduler is not covered by a FixFlag because it is not a logic bug but a
hyper-parameter choice: `StepLR(step_size=3, gamma=0.5)` stepped once per episode
halves the lr every 3 episodes and drops it below LR_MIN within 24.  Here it is
widened in proportion to the episode budget.

Usage:
    .venv/bin/python diagnostics/vqc_comparison.py --episodes 800 --max-time 40
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from quantum_rl.analysis import correct_indexes
from quantum_rl.bellman import greedy_policy, value_iteration
from quantum_rl.config import Config, FixFlags
from quantum_rl.environment import GridWorld
from quantum_rl.train import train


def build(notebook: bool, episodes: int, max_time: int, gamma: float, seed: int,
          layers: int | None = None, eval_every: int | None = None) -> Config:
    """The notebook config (notebook=True) or the fully corrected one."""
    fixes = FixFlags(
        epsilon_greedy_inverted=notebook,
        td_pred_uses_next_state=notebook,
        env_deterministic_transitions=notebook,
        greedy_policy_ignores_reward=notebook,
    )
    cfg = Config(gamma=gamma, seed=seed, fixes=fixes)
    overrides = dict(num_games=episodes, max_time=max_time)
    if layers is not None:
        overrides["deep_layers"] = layers
    if eval_every is not None:
        overrides["eval_every"] = eval_every
    if not notebook:
        # sane scheduler: one halving every ~quarter of the run instead of
        # every 3 episodes, so the lr stays useful all the way through
        overrides["lr_scheduler_step_size"] = max(1, episodes // 4)
    return dataclasses.replace(cfg, **overrides)


def report(name: str, cfg: Config, env: GridWorld, u_ref: np.ndarray,
           pol_ref: np.ndarray, verbose: bool = False,
           show_weights: bool = False) -> dict:
    idx = correct_indexes(env)
    def _monitor(info):
        w = info["weights"]
        print("    game %5d | loss %9.5f | lr %8.6f | eps %.3f | MSE %8.4f | steps %6d"
              % (info["game"], info["loss"], info["lr"], info["epsilon"],
                 info["mse"], info["steps"]))
        if show_weights:
            flat = w.reshape(-1).tolist()
            print("      |W| = %.4f   W = [%s]"
                  % (float(w.norm()), ", ".join("%+.3f" % x for x in flat)))

    t0 = time.time()
    res = train(cfg, env, U_ref=u_ref, monitor=_monitor if verbose else None)
    elapsed = time.time() - t0

    v = np.asarray(res.U, dtype=float)
    pol = np.asarray(res.policy, dtype=int)

    # spread between actions: the diagnostic metric for degeneracy
    from quantum_rl.circuit import (decimal_to_binary_fix_length, make_qnode,
                                    variational_classifier)
    qnode = make_qnode(cfg.n_wires)
    spreads = []
    for s in idx:
        angles = decimal_to_binary_fix_length(cfg.n_wires, s)
        q = variational_classifier(res.var_Q_circuit, angles, qnode, scale=cfg.q_scale)
        q = np.array([float(x) for x in q])
        spreads.append(float(q.max() - q.min()))

    out = {
        "name": name,
        "mse": float(np.mean((v[idx] - u_ref[idx]) ** 2)),
        "v_mean": float(np.mean(v[idx])),
        "policy_match": int(sum(pol[s] == pol_ref[s] for s in idx)),
        "spread": float(np.mean(spreads)),
        "elapsed": elapsed,
    }
    print("%-12s  MSE=%9.4f  mediaV=%8.4f  policy=%d/%d  spread=%7.4f  (%.0fs)"
          % (name, out["mse"], out["v_mean"], out["policy_match"], len(idx),
             out["spread"], elapsed))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--max-time", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.95)  # value from the quantum notebook
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=None,
                   help="VQC depth (notebook: 2)")
    p.add_argument("--eval-every", type=int, default=None,
                   help="how often to print the monitor, in episodes (notebook: 20)")
    p.add_argument("--progress", action="store_true",
                   help="print loss / lr / epsilon / MSE while training")
    p.add_argument("--weights", action="store_true",
                   help="with --progress, also print the circuit weights")
    a = p.parse_args()

    mk = lambda nb: build(nb, a.episodes, a.max_time, a.gamma, a.seed,
                          a.layers, a.eval_every)
    cfg_ref = mk(True)
    env = GridWorld(cfg_ref)
    u_ref, _ = value_iteration(env, a.gamma, cfg_ref.max_epoch)
    pol_ref = greedy_policy(env, u_ref, a.gamma)
    idx = correct_indexes(env)

    print("REAL VQC -- 4 qubits, %d layers, %d episodes x %d steps, gamma=%.2f, seed=%d"
          % (cfg_ref.deep_layers, a.episodes, a.max_time, a.gamma, a.seed))
    print("Bellman (stochastic) over the 9 states: mean U = %.4f" % float(np.mean(u_ref[idx])))
    print("-" * 88)

    rows = [
        report("A notebook", mk(True), env, u_ref, pol_ref, a.progress, a.weights),
        report("B fixed", mk(False), env, u_ref, pol_ref, a.progress, a.weights),
    ]

    print("-" * 88)
    a_, b_ = rows
    if b_["mse"] > 0:
        print("MSE improvement: %.1fx" % (a_["mse"] / b_["mse"]))
    print("spread between actions: %.4f -> %.4f  (~0 = degenerate policy)"
          % (a_["spread"], b_["spread"]))
    print("correct policy:    %d/%d -> %d/%d  (pure chance: %.2f)"
          % (a_["policy_match"], len(idx), b_["policy_match"], len(idx), len(idx) / 4))


if __name__ == "__main__":
    main()
