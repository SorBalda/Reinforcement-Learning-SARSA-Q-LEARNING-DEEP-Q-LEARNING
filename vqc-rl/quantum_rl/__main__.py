"""Command-line entry point: ``python -m quantum_rl``.

Runs the Bellman baseline (notebook cells 32-37) and, unless ``--bellman-only``
is given, the VQC deep-Q training loop (cell 50) on top of it, then writes every
plot of the notebook's analysis section into ``--out-dir``.

Examples
--------
::

    # Bellman value iteration only -- fast, no torch / pennylane needed
    python -m quantum_rl --bellman-only --out-dir runs/bellman

    # tiny end-to-end smoke run
    python -m quantum_rl --episodes 5 --max-time 5 --seed 0 --out-dir runs/smoke

    # A/B a fix against the notebook behaviour (repeat --fix to combine)
    python -m quantum_rl --episodes 5 --max-time 5 \
        --fix td_pred_uses_next_state --fix epsilon_greedy_inverted

WARNING: the notebook's own settings (20000 games x 250 steps) are many hours of
statevector simulation.  ``--episodes`` / ``--max-time`` default to small values
here on purpose; pass ``--full`` to use the notebook's.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from typing import List, Optional

from .config import Config, FixFlags

#: every FixFlag now defaults to False == the FIXED behaviour; ``--bug NAME``
#: turns one back on and ``--notebook`` restores all of them.
FIX_FLAG_NAMES = [f.name for f in dataclasses.fields(FixFlags)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quantum_rl",
        description=(
            "Port of StochFrozenLake_QUANTUM.ipynb: 4x3 stochastic grid world, "
            "Bellman value iteration + deep Q-learning with a 4-qubit VQC."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="available --fix flags:\n  " + "\n  ".join(FIX_FLAG_NAMES),
    )
    p.add_argument("--episodes", type=int, default=5,
                   help="number of training games (notebook: 20000). default: %(default)s")
    p.add_argument("--max-time", type=int, default=5,
                   help="max steps per game (notebook: 250). default: %(default)s")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the legacy global numpy/torch RNGs (default: unseeded)")
    p.add_argument("--out-dir", default="runs/latest",
                   help="directory for the plots and summary.json. default: %(default)s")
    p.add_argument("--fix", action="append", default=[], metavar="FLAG",
                   choices=FIX_FLAG_NAMES,
                   help="turn OFF one FixFlag (opt into that fix). Applied ON TOP "
                        "of --notebook, so e.g. `--notebook --fix "
                        "env_deterministic_transitions` runs the notebook on the "
                        "stochastic grid. Repeatable.")
    p.add_argument("--bug", action="append", default=[], metavar="FLAG",
                   choices=FIX_FLAG_NAMES,
                   help="re-enable one notebook bug (set the FixFlag back to True). "
                        "Repeatable. See the list below.")
    p.add_argument("--notebook", action="store_true",
                   help="reproduce StochFrozenLake_QUANTUM.ipynb exactly: every bug "
                        "on, the cell-50 lr schedule, gamma=0.95.")
    p.add_argument("--bellman-only", action="store_true",
                   help="only run value iteration; skip the VQC training entirely")
    p.add_argument("--list-fixes", action="store_true",
                   help="print the FixFlag names with their current value and exit")

    tweak = p.add_argument_group("hyper-parameter overrides")
    tweak.add_argument("--gamma", type=float, default=None,
                       help="discount (default 0.99; the quantum notebook used 0.95)")
    tweak.add_argument("--reward", "-r", type=float, default=None, dest="r",
                       help="living reward, i.e. what every non-terminal state pays "
                            "-- including staying put on a blocked move (notebook: "
                            "-0.004). Changes the Bellman baseline too, since R feeds "
                            "value iteration as well as training.")
    tweak.add_argument("--memory", type=int, default=None, dest="N",
                       help="replay-memory length N (notebook: 700)")
    tweak.add_argument("--batch", type=int, default=None, dest="len_batch",
                       help="minibatch size (notebook: 1)")
    tweak.add_argument("--layers", type=int, default=None, dest="deep_layers",
                       help="VQC depth (notebook: 2)")
    tweak.add_argument("--lr", type=float, default=None, dest="LR_0",
                       help="initial RMSprop learning rate (notebook: 0.2)")
    tweak.add_argument("--lr-step", default=None, dest="lr_scheduler_step_size",
                       metavar="N|auto",
                       help="halve the learning rate every N games (notebook: 3, which "
                            "kills the lr by game 24). 'auto' = episodes//4, i.e. four "
                            "halvings spread over the run. default: %d"
                            % Config().lr_scheduler_step_size)
    tweak.add_argument("--lr-gamma", type=float, default=None,
                       dest="lr_scheduler_gamma",
                       help="multiplier applied at each lr step (notebook: 0.5)")
    tweak.add_argument("--lr-min", type=float, default=None, dest="LR_MIN",
                       help="the scheduler stops once the lr drops to this (notebook: 0.001)")
    tweak.add_argument("--target-sync", type=int, default=None, dest="C",
                       help="target-network sync period in steps (notebook: 200)")
    tweak.add_argument("--eval-every", type=int, default=None,
                       help="MSE/probe evaluation period in games (notebook: 20)")
    tweak.add_argument("--full", action="store_true",
                       help="use the notebook's 20000 games x 250 steps (HOURS of compute)")

    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="live monitor every --eval-every games: loss (honest window "
                        "average), lr, epsilon, MSE vs Bellman, steps. ON by default; "
                        "use --no-progress to silence it.")
    p.add_argument("--weights", action="store_true",
                   help="with --progress, also print the VQC weights and their norm")
    p.add_argument("--notebook-report", action="store_true",
                   help="print the notebook's own per-game report instead. Faithful, "
                        "so it reproduces its broken [:-20:] averaging too.")
    p.add_argument("--no-plots", action="store_true", help="do not write any figure")
    p.add_argument("--live", action=argparse.BooleanOptionalAction, default=True,
                   help="every --eval-every games, redraw <out-dir>/live.png: MSE, "
                        "loss, learned utilities vs Bellman, and lr/epsilon. Open it "
                        "in any auto-refreshing image viewer to watch training. ON by "
                        "default; use --no-live to skip the redraws.")
    return p


def build_fixes(args: argparse.Namespace) -> FixFlags:
    """Base, then per-flag overrides on top.

    Base is :meth:`FixFlags.notebook` with ``--notebook`` (every bug on) and the
    corrected defaults otherwise.  ``--bug NAME`` then turns one bug back ON and
    ``--fix NAME`` turns one OFF, so ``--notebook --fix env_deterministic_transitions``
    is "the notebook, but on the stochastic grid".
    """
    import dataclasses

    base = FixFlags.notebook() if getattr(args, "notebook", False) else FixFlags()
    d = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    d.update({name: True for name in getattr(args, "bug", [])})
    d.update({name: False for name in getattr(args, "fix", [])})
    return FixFlags(**d)


def make_config(args: argparse.Namespace) -> Config:
    """Turn parsed CLI args into a :class:`~quantum_rl.config.Config`."""
    fixes = build_fixes(args)

    kwargs = dict(fixes=fixes, seed=args.seed)
    if getattr(args, "notebook", False):
        kwargs["lr_scheduler_step_size"] = 3
        kwargs["gamma"] = 0.95
    if args.full:
        kwargs["num_games"] = 20000
        kwargs["max_time"] = 250
    else:
        kwargs["num_games"] = args.episodes
        kwargs["max_time"] = args.max_time

    for name in ("gamma", "r", "N", "len_batch", "deep_layers", "LR_0", "C",
                 "eval_every", "lr_scheduler_gamma", "LR_MIN"):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value

    # --lr-step accepts an int or the literal "auto" (= a quarter of the run,
    # so the lr halves four times over the episodes actually being run).
    step = getattr(args, "lr_scheduler_step_size", None)
    if step is not None:
        if str(step).lower() == "auto":
            kwargs["lr_scheduler_step_size"] = max(1, kwargs["num_games"] // 4)
        else:
            try:
                kwargs["lr_scheduler_step_size"] = int(step)
            except ValueError:
                raise SystemExit(f"--lr-step must be an integer or 'auto' (got {step!r})")
        if kwargs["lr_scheduler_step_size"] < 1:
            raise SystemExit("--lr-step must be >= 1")
    return Config(**kwargs)




class LivePlotter:
    """Redraw ``live.png`` on every monitor tick.

    Four panels: MSE vs Bellman, training loss, the learned utilities against the
    Bellman baseline (the panel that actually tells you whether it is learning),
    and the lr / epsilon schedules.  Written to a temp file and moved into place
    so a viewer never reads a half-written PNG.
    """

    def __init__(self, path, correct):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plt = plt
        self.path = path
        self.correct = list(correct)
        self.games, self.mse, self.loss, self.lr, self.eps = [], [], [], [], []
        self.U = None
        self.U_ref = None

    def __call__(self, info):
        self.record(info)
        self.draw()

    def record(self, info):
        """Keep the numbers of one tick without redrawing (see :meth:`draw`)."""
        self.games.append(info["game"])
        self.mse.append(info["mse"])
        self.loss.append(info["loss"])
        self.lr.append(info["lr"])
        self.eps.append(info["epsilon"])
        if info.get("U") is not None:
            self.U = info["U"]
        if info.get("U_ref") is not None:
            self.U_ref = info["U_ref"]

    def draw(self):
        if not self.games:
            return
        plt = self._plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        g = self.games

        ax[0][0].plot(g, self.mse, color="crimson")
        ax[0][0].set_title("MSE vs Bellman")
        ax[0][0].set_xlabel("game"); ax[0][0].set_ylabel("MSE")
        if self.mse and min(self.mse) > 0:
            ax[0][0].set_yscale("log")
        ax[0][0].grid(alpha=.3)

        ax[0][1].plot(g, self.loss, color="black")
        ax[0][1].set_title("training loss (window mean)")
        ax[0][1].set_xlabel("game"); ax[0][1].grid(alpha=.3)

        a = ax[1][0]
        if self.U is not None and self.U_ref is not None:
            idx = self.correct
            a.plot(idx, [self.U_ref[i] for i in idx], marker="o", linestyle="--",
                   color="steelblue", label="Bellman")
            a.plot(idx, [self.U[i] for i in idx], marker="*", linestyle="-",
                   color="darkorange", label="VQC")
            a.legend()
        a.set_title("utilities  max$_a$ Q(s,a)")
        a.set_xlabel("state"); a.set_ylabel("U"); a.grid(alpha=.3)

        a = ax[1][1]
        a.plot(g, self.lr, color="seagreen", label="lr")
        a.set_xlabel("game"); a.set_ylabel("lr", color="seagreen")
        if self.lr and min(self.lr) > 0:
            a.set_yscale("log")
        t = a.twinx()
        t.plot(g, self.eps, color="purple", label="epsilon")
        t.set_ylabel("epsilon", color="purple")
        a.set_title("schedules"); a.grid(alpha=.3)

        fig.suptitle("game %d   MSE %.4f   loss %.4f"
                     % (g[-1], self.mse[-1], self.loss[-1]))
        fig.tight_layout()
        tmp = self.path + ".tmp.png"
        fig.savefig(tmp, dpi=110)
        plt.close(fig)
        os.replace(tmp, self.path)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_fixes:
        fixes = build_fixes(args)
        print("FixFlags (True == notebook behaviour, False == fixed; "
              "defaults are now the FIXES):")
        for name in FIX_FLAG_NAMES:
            print(f"  {name:<34} {getattr(fixes, name)}")
        return 0

    cfg = make_config(args)
    out_dir = os.path.abspath(args.out_dir)
    if not args.no_plots:
        os.makedirs(out_dir, exist_ok=True)

    # imports are local so that --help / --list-fixes stay instant and
    # --bellman-only never touches torch or pennylane
    from .analysis import correct_indexes, residuals, restrict
    from .bellman import greedy_policy, value_iteration_detailed
    from .environment import GridWorld
    from .outputs import (base_summary, training_summary, write_bellman_plots,
                          write_json, write_training_plots)

    env = GridWorld(cfg)

    print(f"grid {cfg.Nx}x{cfg.Ny}  gamma={cfg.gamma}  r={cfg.r}  seed={cfg.seed}")
    print("fixes: " + ", ".join(f"{n}={getattr(cfg.fixes, n)}" for n in FIX_FLAG_NAMES))

    # ---- Bellman baseline (cells 32-37) -----------------------------------
    t0 = time.time()
    bellman = value_iteration_detailed(env, cfg.gamma, cfg.max_epoch)
    policy_bellman = greedy_policy(env, bellman.U_policy, cfg.gamma)
    print(f"\nvalue iteration: {cfg.max_epoch} epochs in {time.time() - t0:.2f}s")
    print(f"  U_final (states {correct_indexes(env)}):")
    print("    " + "  ".join(f"{v:.6f}" for v in restrict(bellman.U_final, env)))
    print(f"  greedy policy: {[int(a) for a in policy_bellman]}")

    summary = base_summary(cfg, env, bellman, policy_bellman)

    if not args.no_plots:
        write_bellman_plots(out_dir, env, bellman, policy_bellman)

    # ---- VQC training (cell 50) -------------------------------------------
    if not args.bellman_only:
        from .train import train  # pulls in torch + pennylane

        print(f"\ntraining: {cfg.num_games} games x {cfg.max_time} steps, "
              f"N={cfg.N}, batch={cfg.len_batch}, layers={cfg.deep_layers}")
        live = None
        if args.live and not args.no_plots:
            live = LivePlotter(os.path.join(out_dir, "live.png"), correct_indexes(env))
            print(f"  live plot: {live.path}")

        def _monitor(info):
            if live is not None:
                live(info)
            print("  game %6d | loss %9.5f | lr %8.6f | eps %.3f | MSE %8.4f | steps %7d"
                  % (info["game"], info["loss"], info["lr"], info["epsilon"],
                     info["mse"], info["steps"]))
            if args.weights:
                w = info["weights"]
                print("    |W| = %.4f  W = [%s]"
                      % (float(w.norm()),
                         ", ".join("%+.3f" % x for x in w.reshape(-1).tolist())))

        t0 = time.time()
        result = train(cfg, env, progress=args.notebook_report,
                       U_ref=bellman.U_final,
                       monitor=_monitor if (args.progress or args.live) else None)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.2f}s ({result.steps} steps)")

        avg, std, res = residuals(result.U, bellman.U_final, env)
        print(f"  final U (states {correct_indexes(env)}):")
        print("    " + "  ".join(f"{v:.6f}" for v in restrict(result.U, env)))
        print(f"  learned policy: {[int(a) for a in result.policy]}")
        print(f"  residual avg = {avg:.6f}, RMS = {std:.6f}")
        if result.mse_u:
            print(f"  MSE: first = {result.mse_u[0]:.6f}, last = {result.mse_u[-1]:.6f}, "
                  f"min = {result.mse_min:.6f}")
        else:
            print("  MSE: never evaluated "
                  f"(needs > {cfg.eval_every} games, got {cfg.num_games})")

        summary["training"] = training_summary(result, bellman, env, elapsed)

        if not args.no_plots:
            write_training_plots(out_dir, env, bellman, result)

    if not args.no_plots:
        summary_path = os.path.join(out_dir, "summary.json")
        write_json(summary_path, summary)
        print(f"\nwrote plots + summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
