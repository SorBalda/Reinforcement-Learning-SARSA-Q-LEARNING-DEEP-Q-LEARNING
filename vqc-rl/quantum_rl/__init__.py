"""quantum_rl -- a runnable port of ``StochFrozenLake_QUANTUM.ipynb``.

Every notebook bug is behind a :class:`~quantum_rl.config.FixFlags` switch: the
defaults are the fixes, and ``Config.notebook()`` reproduces the notebook bit
for bit.

Imports are lazy so that ``import quantum_rl`` (or using only the environment
and the Bellman baseline) does not drag in torch / pennylane::

    from quantum_rl import Config, GridWorld, value_iteration
    cfg = Config()
    env = GridWorld(cfg)
    U_final, U_time = value_iteration(env, cfg.gamma, cfg.max_epoch)
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# public name -> submodule that defines it
_LAZY: dict[str, str] = {
    # config
    "Config": "config",
    "FixFlags": "config",
    "apply_seed": "config",
    # environment
    "GridWorld": "environment",
    "prob_action": "environment",
    # bellman
    "value_iteration": "bellman",
    "value_iteration_detailed": "bellman",
    "greedy_policy": "bellman",
    "BellmanResult": "bellman",
    # circuit (pennylane + torch)
    "decimal_to_binary_fix_length": "circuit",
    "state_preparation": "circuit",
    "layer": "circuit",
    "make_qnode": "circuit",
    "default_qnode": "circuit",
    "variational_classifier": "circuit",
    # agent
    "ReplayMemory": "agent",
    "epsilon_greedy": "agent",
    "sample_random_batch": "agent",
    # analysis
    "correct_indexes": "analysis",
    "restrict": "analysis",
    "residuals": "analysis",
    "mse": "analysis",
    "action_name": "analysis",
    "greedy_from_q_table": "analysis",
    # plotting (matplotlib)
    "plot_policy": "plotting",
    "plot_training_curves": "plotting",
    "plot_utility_comparison": "plotting",
    "plot_bellman_convergence": "plotting",
    "plot_residual_hist": "plotting",
    # training
    "train": "train",
    "TrainResult": "train",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache, so the import cost is paid once
    return value


def __dir__() -> list[str]:
    return list(__all__)
