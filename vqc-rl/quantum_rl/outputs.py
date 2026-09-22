"""Writing a run to disk: ``summary.json`` and the analysis plots.

Shared by the CLI (:mod:`quantum_rl.__main__`) and the GUI, so a run saved from
either one produces the same files:

* ``bellman_policy.pdf``, ``bellman_convergence.pdf`` -- the reference;
* ``policy.pdf``, ``training_curves.pdf``, ``utility_comparison.pdf``,
  ``residuals.pdf`` -- the trained VQC against it;
* ``summary.json`` -- config, Bellman reference and training results.

Plots go through :mod:`quantum_rl.plotting`, i.e. pyplot on the Agg backend.
"""

from __future__ import annotations

import dataclasses
import json
import os

from .analysis import correct_indexes, residuals, restrict


def jsonable(value):
    """Best-effort conversion of numpy / torch values to plain JSON types."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def base_summary(cfg, env, bellman, policy_bellman) -> dict:
    """Config + Bellman reference: the part of ``summary.json`` every run has."""
    return {
        "config": {k: jsonable(v) for k, v in dataclasses.asdict(cfg).items()},
        "correct_indexes": correct_indexes(env),
        "bellman": {
            "U_final": jsonable(bellman.U_final),
            "U_final_correct": restrict(bellman.U_final, env),
            "policy": [int(a) for a in policy_bellman],
        },
    }


def training_summary(result, bellman, env, elapsed: float) -> dict:
    """The ``"training"`` section of ``summary.json`` for a finished run."""
    avg, std, res = residuals(result.U, bellman.U_final, env)
    return {
        "seconds": elapsed,
        "steps": result.steps,
        "loss_games": jsonable(result.loss_games),
        "games_reward": jsonable(result.games_reward),
        "mse_u": jsonable(result.mse_u),
        "mse_games": jsonable(result.mse_games),
        "mse_min": None if result.mse_min == float("inf") else result.mse_min,
        "good_Q": jsonable(result.good_Q),
        "bad_Q": jsonable(result.bad_Q),
        "policy": [int(a) for a in result.policy],
        "U": jsonable(result.U),
        "residual_avg": avg,
        "residual_rms": std,
        "residuals": res,
    }


def write_bellman_plots(out_dir: str, env, bellman, policy_bellman) -> None:
    from .plotting import plot_bellman_convergence, plot_policy

    plot_policy(policy_bellman, env, title="Bellman policy",
                outpath=os.path.join(out_dir, "bellman_policy.pdf"))
    plot_bellman_convergence(bellman.U_time, env,
                             outpath=os.path.join(out_dir, "bellman_convergence.pdf"))


def write_training_plots(out_dir: str, env, bellman, result) -> None:
    from .plotting import (plot_policy, plot_residual_hist,
                           plot_training_curves, plot_utility_comparison)

    avg, std, res = residuals(result.U, bellman.U_final, env)
    plot_policy(result.policy, env, title="VQC policy",
                outpath=os.path.join(out_dir, "policy.pdf"))
    plot_training_curves(result, outpath=os.path.join(out_dir, "training_curves.pdf"))
    plot_utility_comparison(result.U, env, ref_U=bellman.U_final,
                            title="VQC vs Bellman utilities",
                            outpath=os.path.join(out_dir, "utility_comparison.pdf"))
    plot_residual_hist(res, avg, std, title="VQC residuals",
                       outpath=os.path.join(out_dir, "residuals.pdf"))


def write_json(path: str, data) -> None:
    """Write ``data`` as JSON atomically: a reader never sees a half file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)
