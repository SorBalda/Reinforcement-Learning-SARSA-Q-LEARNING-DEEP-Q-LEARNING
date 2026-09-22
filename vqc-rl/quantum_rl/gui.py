"""Graphical interface: set the parameters, press Start, watch it train.

Launch::

    .venv/bin/python -m quantum_rl.gui

The interface has TWO SCREENS that swap inside the same window, never visible
at the same time:

* **Setup** -- only the parameter form, full width, laid out as cards across
  three columns.  Every field carries a help text, both as a tooltip and in the
  info bar at the bottom. The validated setup (including output folder, plot
  preferences and debug flags) is stored atomically and restored at the next
  launch. The ``Output`` card picks the run folder and ``Start`` launches it.
* **Training** -- only the plots, full window, with a compact status bar
  (episode, moving-average MSE/loss/reward, lr, epsilon, non-zero gradient
  RMS, total parameters and zero-gradient parameters) and the ``Stop``,
  ``Save plot``, ``Episode replay`` and ``Back`` buttons. Replay animates the
  latest completed episode in the grid and refreshes as training continues.  A segmented control switches between
  four views: the curves, the current policy, the live variational circuit
  with its weights and gradients (:mod:`quantum_rl.circuit_view`), and the
  gradients against time -- whole circuit, per layer, every parameter and the
  selected gate, on a log scale with the vanishing zone shaded.  The
  Gradients view has a layer selector (chips, arrow keys or a click on the
  heatmap) that focuses all four plots on one layer, and a side list of every
  parameter whose gradient is below a threshold (default 1e-10), in the last
  episode or over the whole run. Exact and structural zeros are counted but
  explicitly excluded from gradient curves, RMS aggregates and heatmaps. The
  Curves view overlays one episode-based moving average on raw MSE, loss and
  discounted reward, and shows Bellman and learned VQC utilities live; sparse
  MSE samples still use a window measured in episodes.

Saving is automatic, into the folder chosen in the setup:

* at start: ``bellman_policy.pdf``, ``bellman_convergence.pdf`` and a first
  ``summary.json`` (config + Bellman reference);
* during training, at most every ``IO_EVERY`` seconds: ``live.png`` and
  ``progress.json`` (curves plus latest episode) and ``last_episode.json``, so
  a crash or a closed window never loses the whole run;
* at the end: the same files as the CLI (``policy.pdf``,
  ``training_curves.pdf``, ``utility_comparison.pdf``, ``residuals.pdf``, the
  full ``summary.json``) plus ``params_history.npz`` (weights and gradient RMS
  of every episode), ``gradient_stats.csv`` (per parameter: last and max
  gradient RMS, share of episodes below the list threshold, structural zero),
  ``gui_curves.png`` / ``gui_policy.png``, and the final state drawn from the
  data (whatever view is open): ``circuit/circuit_angles`` and
  ``circuit/circuit_gradients`` (all layers, last episode) and
  ``gradients/all`` plus ``gradients/layer_<k>`` (the four gradient plots for
  the whole circuit and for each layer), each as ``.png`` and ``.pdf``.  A
  run interrupted with Stop still gets ``summary.json`` (marked
  ``"stopped": true``), ``params_history.npz`` and all the final plots.

This GUI always trains the corrected agent (``FixFlags()``): the switches that
bring back the notebook bugs live only in :mod:`quantum_rl.debug_gui`.

Training runs on a separate thread: the window stays responsive and the Stop
button interrupts at the end of the current episode.  Redraws are rate-limited
(a matplotlib redraw holds the GIL and would slow the training down), and the
final plots are written by a background thread, so Back, a new run or closing
the window work right away after Stop.  Tkinter is not thread-safe, so the
training thread never touches a widget -- it only communicates through a
``queue.Queue`` that the main loop drains with ``after()``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import warnings
import tkinter as tk
from dataclasses import replace
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from .__main__ import LivePlotter
from .analysis import correct_indexes
from .bellman import greedy_policy, value_iteration_detailed
from .circuit_plot import save_final_circuit
from .circuit_view import (GRAD_HI, GRAD_LO, PARAM_COLORS, PARAM_NAMES,
                           GRAD_ZERO, CircuitView, ParamHistory, dead_mask)
from .config import Config, FixFlags
from .environment import GridWorld
from .episode_view import EpisodeView
from .outputs import (base_summary, jsonable, training_summary,
                      write_bellman_plots, write_json, write_training_plots)
from .train import train
from .gui_widgets import (AMBER, BG, BORDER, BORDER_HI, CARD, CARD_HI, CYAN,
                          FAINT,
                          FIELD, FONTS, GREEN, MUTED, PANEL, PINK, RED, TEXT,
                          VIOLET,
                          NeonButton, Tooltip, _mix, _round_rect)

#: relative output folders are resolved against the project folder, not the
#: current directory, so ``runs/...`` lands in the same place however the GUI
#: was launched
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAST_SETUP_PATH = os.path.join(PROJECT_ROOT, ".vqc_gui_last_setup.json")

# --------------------------------------------------------------------------- #
# parameter metadata: (attribute, label, type, help text)
# --------------------------------------------------------------------------- #

GROUPS = [
    ("Environment", [
        ("gamma", "Discount factor  γ", float,
         "How much the future counts. 0.99 is a long horizon, 0.95 is the value used by "
         "the quantum notebook. Changing it also changes the Bellman reference."),
        ("r", "Living reward", float,
         "What every non-terminal state pays, including staying put when a move is "
         "blocked by a wall. More negative = more hurry to reach the goal."),
    ]),
    ("Duration", [
        ("num_games", "Episodes", int,
         "How many games to play. The notebook uses 20000, which takes hours."),
        ("max_time", "Steps per episode", int,
         "Cap on the steps of one game, which ends early anyway if it reaches "
         "the +1 or the -1."),
    ]),
    ("Circuit", [
        ("deep_layers", "Circuit layers", int,
         "Depth of the VQC. Each layer is 12 parameters (4 qubits x 3). "
         "At 4 layers the parameters match the 48 Q-values to represent."),
        ("q_scale", "Q scale factor", float,
         "The 4 PauliZ measurements live in [-1,1] and are multiplied by "
         "this: it fixes the representable range of the Q-values."),
    ]),
    ("Learning", [
        ("LR_0", "Transient learning rate", float,
         "RMSprop step size during the initial transient. It may be deliberately "
         "high to move quickly before the stationary schedule begins."),
        ("lr_transient_steps", "Transient duration (optimizer steps)", int,
         "Number of RMSprop optimizer steps for which the transient learning rate "
         "is held fixed. The core reports the corresponding episode boundary."),
        ("LR_AFTER_TRANSIENT", "Post-transient learning rate", float,
         "Learning rate immediately after the transient. This is also the starting "
         "value for the subsequent halving schedule."),
        ("lr_scheduler_step_size", "Halve the lr every N episodes", int,
         "Number of episodes between multiplicative halvings after the transient. "
         "The value is never reduced below LR min."),
        ("LR_MIN", "Minimum learning rate", float,
         "Below this threshold the scheduler stops reducing the lr."),
        ("len_batch", "Minibatch size", int,
         "How many transitions to average per step. The notebook uses 1, i.e. "
         "plain SGD: raising it cuts gradient noise but costs time."),
        ("N", "Replay memory length", int,
         "How many past transitions to keep in memory."),
        ("C", "Sync the target net every N steps", int,
         "How often the target network copies the weights of the trained one."),
        ("epsilon_0", "Initial epsilon", float,
         "Exploration rate at the start of training."),
        ("epsilon_min", "Final epsilon", float,
         "Exploration rate at the end of training."),
        ("epsilon_decay_mode", "Epsilon decay", ("choice", ("linear", "exponential", "cosine", "constant")),
         "Schedule used to move epsilon from its initial to its final value."),
        ("epsilon_decay_duration", "Decay duration (episodes)", int,
         "Number of episodes over which epsilon moves from epsilon_0 to epsilon_min. "
         "At this duration it reaches epsilon_min exactly."),
        ("epsilon_decay_interval", "Update every N episodes", int,
         "Epsilon is updated only at this episode interval. Larger values make the "
         "schedule a staircase while preserving its configured duration."),
    ]),
    ("Execution", [
        ("seed", "Random seed", int,
         "Fix it to make the run reproducible. Leave empty to stay unseeded."),
        ("eval_every", "Evaluate and redraw every N episodes", int,
         "How often to compute the MSE and redraw the plots."),
    ]),
]

#: saner starting values than the notebook's, as in the previous GUI
DEFAULT_OVERRIDES = {
    "num_games": 1000,
    "max_time": 40,
    "LR_0": 0.01,
    "lr_transient_steps": 100,
    "LR_AFTER_TRANSIENT": 0.005,
    "lr_scheduler_step_size": 250,
    "epsilon_decay_duration": 1800,
    "epsilon_decay_interval": 1,
    "seed": 1,
}

#: which setup column each group lands in, and with which accent colour
COLUMN_OF = {
    "Environment": (0, CYAN),
    "Duration": (0, CYAN),
    "Circuit": (0, VIOLET),
    "Learning": (1, VIOLET),
    "Execution": (1, CYAN),
}
#: the Output card (and, in the debug GUI, the bug switches) go here
SIDE_COLUMN = 2


def _default_out_dir() -> str:
    return os.path.join("runs", time.strftime("gui_%Y%m%d_%H%M%S"))


def _resolve_out_dir(raw: str) -> str:
    path = os.path.expanduser(raw.strip())
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return os.path.normpath(path)


class StopTraining(Exception):
    """Raised by the monitor to interrupt training from the Stop button."""


# --------------------------------------------------------------------------- #


class App(tk.Tk):
    WINDOW_TITLE = "Quantum RL  ·  VQC training on the 4x3 grid world"

    def __init__(self):
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.configure(bg=BG)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1700, max(1080, sw - 200))
        h = min(1010, max(700, sh - 220))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(960, 620)

        self._init_fonts()
        self._init_style()

        self.vars: dict[str, tuple[tk.StringVar, type]] = {}
        # output folder: the default is a fresh timestamped one, renewed on
        # every return to the setup unless the user typed their own
        self._auto_out = _default_out_dir()
        self.out_var = tk.StringVar(value=self._auto_out)
        self.out_dir: str | None = None   # resolved folder of the current run
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.exporter: threading.Thread | None = None
        self.stop_flag = threading.Event()

        self.history = {"game": [], "mse": [], "loss": [], "lr": [], "eps": []}
        # one snapshot per episode of the weights and their gradient RMS,
        # shared by the Circuit and Gradients views
        self.params = ParamHistory()
        self._grads_drawn = 0.0
        self.grad_layer = None    # layer the Gradients view focuses on; None = all
        self.U = None
        self.U_ref = None
        self.policy = None        # current argmax_a Q(s,a), from the monitor
        self.policy_ref = None    # Bellman's optimal policy
        self.correct = None
        self.view = "curve"       # overview views: curve, policy, circuit, grads, focus
        self.focus_metric = 0       # MSE, loss, reward, utilities, schedule
        self._lr_transition_game = None
        self._run_reward = None   # living reward of the current run
        self.last_episode = None
        self.episode_view = None
        self.log_mse = tk.BooleanVar(value=True)
        # the loss spans decades and is noisy episode to episode: log by default
        self.log_loss = tk.BooleanVar(value=True)
        # per-episode loss (from game_monitor), independent of eval_every
        self.loss_ep = {"game": [], "loss": []}
        self.reward_ep = {"game": [], "reward": []}
        self.loss_avg_n = 20      # moving-average window, in episodes; 1 = off
        self._curves_drawn = 0.0
        self._loss_new = False
        self._tick_pending = False
        self._last_redraw = 0.0
        self._redraw_gap = self.REDRAW_EVERY
        self._last_circuit = 0.0
        self._last_game_at = 0.0
        self.total_games = 0
        self.screen = "setup"

        self.status = tk.StringVar(value="ready")
        self.hint = tk.StringVar(value=self._HINT_DEFAULT)

        self.frame_setup = self._build_setup()
        self.frame_train = self._build_training()
        self._load_last_setup()
        self.show_setup()
        self.after(100, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Closing the window while training must stop it cleanly.

        The worker thread is a daemon: if the interpreter tears down while that
        thread is inside PennyLane/torch, the shutdown ends in a core dump.  Raise
        the stop flag and give it a moment to leave the monitor.
        """
        self.stop_flag.set()
        if self.episode_view is not None and self.episode_view.winfo_exists():
            self.episode_view.close()
        # the window goes away immediately; the threads still writing the run
        # (the worker on Stop, the final-plot exporter) get time to finish
        self.withdraw()
        worker = getattr(self, "worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=30.0)
        exporter = getattr(self, "exporter", None)
        if exporter is not None and exporter.is_alive():
            exporter.join(timeout=120.0)
        self.destroy()

    # ------------------------------------------------------------- stile ---
    _HINT_DEFAULT = ("Hover a parameter to read what it does. "
                     "Then press Start.")

    def _init_fonts(self):
        """Font sizes expressed in PIXELS (negative size).

        Points are not reliable: on high-density displays Tk applies its own
        ``tk scaling``, and a font asked for at 13 points can end up LARGER than
        the system font.  Here we measure how many pixels the system font really
        occupies and derive every size from that unit.
        """
        families = tkfont.families()
        fam = "Noto Sans" if "Noto Sans" in families else "DejaVu Sans"
        mono = "DejaVu Sans Mono" if "DejaVu Sans Mono" in families else "TkFixedFont"

        probe = tkfont.Font(family=fam, size=-20)
        per_px = (probe.metrics("linespace") or 28) / 20.0
        ref = tkfont.nametofont("TkDefaultFont").metrics("linespace") or 22
        # Slightly generous baseline: scientific labels and live values must be
        # readable on laptop/high-DPI displays without requiring zooming.
        unit = max(14, min(24, int(round(ref / max(per_px, 0.8)))))

        def px(k):
            return -max(9, int(round(unit * k)))

        FONTS.update({
            "title": (fam, px(1.85), "bold"),
            "sub": (fam, px(0.86)),
            "card": (fam, px(0.88), "bold"),
            "body": (fam, px(0.96)),
            "small": (fam, px(0.88)),
            "tip": (fam, px(0.88)),
            "tip_b": (fam, px(0.88), "bold"),
            "btn": (fam, px(1.02), "bold"),
            "btn_s": (fam, px(0.90), "bold"),
            "mono": (mono, px(0.92)),
            "mono_b": (mono, px(1.08), "bold"),
            "chip": (fam, px(0.80), "bold"),
        })

    def _init_style(self):
        st = ttk.Style(self)
        if "clam" in st.theme_names():
            st.theme_use("clam")
        st.configure("Q.Vertical.TScrollbar", background=BORDER,
                     troughcolor=PANEL, bordercolor=PANEL, arrowcolor=MUTED,
                     darkcolor=BORDER, lightcolor=BORDER, relief="flat")
        st.map("Q.Vertical.TScrollbar", background=[("active", BORDER_HI)])

    # ------------------------------------------------------- setup screen ---
    def _build_setup(self) -> tk.Frame:
        root = tk.Frame(self, bg=BG)

        self._build_header(root)

        # scrollable body (usually everything fits; the bar is a safety net)
        mid = tk.Frame(root, bg=BG)
        mid.pack(fill="both", expand=True)
        canvas = tk.Canvas(mid, bg=BG, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview,
                            style="Q.Vertical.TScrollbar")
        body = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        self._bar_shown = False

        def _sync(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # the scrollbar only shows up when it is really needed
            need = body.winfo_reqheight() > canvas.winfo_height() + 2
            if need != self._bar_shown:
                self._bar_shown = need
                if need:
                    bar.pack(side="right", fill="y")
                else:
                    bar.pack_forget()

        body.bind("<Configure>", _sync)
        # the inner frame follows the canvas width (otherwise the fields get
        # clipped on the right) and, when there is room to spare, its height,
        # so the cards stay vertically centred instead of leaving a gap below
        def _fit(event):
            canvas.itemconfigure(win, width=event.width,
                                 height=max(event.height,
                                            body.winfo_reqheight()))
            _sync()

        canvas.bind("<Configure>", _fit)
        self._form_canvas = canvas
        self._form_body = body

        # mouse wheel: on Linux Tk delivers it as Button-4 / Button-5
        def _wheel(event):
            if self.screen != "setup":
                return
            num = getattr(event, "num", None)
            if num == 4:
                canvas.yview_scroll(-3, "units")
            elif num == 5:
                canvas.yview_scroll(3, "units")
            elif getattr(event, "delta", 0):
                canvas.yview_scroll(-1 * (event.delta // 40), "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, _wheel)

        self._build_cards(body)
        self._build_hint_bar(root)
        self._build_footer(root)
        root.bind_all("<Return>", lambda _e: self.start() if self.screen == "setup"
                      else None)
        return root

    def _build_header(self, parent):
        cv = tk.Canvas(parent, height=104, bg=PANEL, highlightthickness=0, bd=0)
        cv.pack(fill="x")
        cv.bind("<Configure>", lambda e: self._draw_header(cv, e.width, e.height))
        return cv

    def _draw_header(self, cv, w, h):
        cv.delete("all")
        # background with a very faint horizontal gradient
        steps = 60
        for i in range(steps):
            x0 = i * w / steps
            cv.create_rectangle(x0, 0, x0 + w / steps + 1, h, width=0,
                                fill=_mix("#080e1c", "#111c35", i / steps))
        # a faint grid of dots
        for gx in range(20, w - 20, 26):
            for gy in range(18, h - 22, 26):
                cv.create_oval(gx, gy, gx + 1.6, gy + 1.6, width=0,
                               fill=_mix(PANEL, CYAN, 0.16))
        # title
        cv.create_text(34, h // 2 - 14, anchor="w", text="QUANTUM  RL",
                       fill=TEXT, font=FONTS["title"])
        cv.create_text(36, h // 2 + 18, anchor="w",
                       text="4-qubit variational circuit  ·  stochastic "
                            "4×3 grid  ·  Bellman reference",
                       fill=FAINT, font=FONTS["sub"])
        # decoration: three tilted orbits, each with its electron
        self._draw_orbits(cv, w - 92, h // 2 - 2, 48, 17)
        # glowing line at the bottom, from cyan to violet
        for i in range(steps):
            x0 = i * w / steps
            cv.create_rectangle(x0, h - 2, x0 + w / steps + 1, h, width=0,
                                fill=_mix(CYAN, VIOLET, i / steps))

    @staticmethod
    def _draw_orbits(cv, cx, cy, a, b):
        import math
        for k, (ang, col) in enumerate(((0, CYAN), (60, VIOLET), (120, CYAN))):
            th = math.radians(ang)
            pts = []
            for i in range(49):
                t = 2 * math.pi * i / 48
                x, y = a * math.cos(t), b * math.sin(t)
                pts += [cx + x * math.cos(th) - y * math.sin(th),
                        cy + x * math.sin(th) + y * math.cos(th)]
            cv.create_line(*pts, fill=_mix(PANEL, col, 0.55), width=1, smooth=True)
            t = 2 * math.pi * (0.12 + 0.3 * k)
            x, y = a * math.cos(t), b * math.sin(t)
            px = cx + x * math.cos(th) - y * math.sin(th)
            py = cy + x * math.sin(th) + y * math.cos(th)
            cv.create_oval(px - 3.5, py - 3.5, px + 3.5, py + 3.5, width=0,
                           fill=col)
        cv.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, width=0,
                       fill=_mix(CYAN, "#ffffff", 0.5))

    # -- parameter cards ----------------------------------------------------
    def _build_cards(self, body):
        cols = []
        grid = tk.Frame(body, bg=BG)
        grid.pack(fill="x", expand=True, padx=22, pady=18)
        for c in range(3):
            grid.columnconfigure(c, weight=1, uniform="col")
            f = tk.Frame(grid, bg=BG)
            f.grid(row=0, column=c, sticky="nsew", padx=9)
            cols.append(f)

        d = Config()
        for title, fields in GROUPS:
            col, accent = COLUMN_OF[title]
            inner = self._card(cols[col], title, accent)
            inner.columnconfigure(0, weight=1)
            for row, (name, label, kind, help_text) in enumerate(fields):
                self._field(inner, row, name, label, kind, help_text, d, accent)

        self._build_side_cards(cols[SIDE_COLUMN])

    def _build_side_cards(self, col):
        """Third column.  :class:`~quantum_rl.debug_gui.DebugApp` extends it."""
        self._build_output_card(col)

    _OUT_HELP = ("Folder the run is saved to, created if missing. A relative path "
                 "is taken from the project folder (where gui.py is). Everything "
                 "is written there automatically: live.png and progress.json "
                 "during training, then "
                 "summary.json, params_history.npz (weights and gradients of "
                 "every episode), gradient_stats.csv, the PDF plots, the final "
                 "circuit (circuit/) and every gradient plot (gradients/, one "
                 "per layer) at the end "
                 "(also when you press Stop).")

    def _build_output_card(self, col):
        inner = self._card(col, "Output", GREEN,
                           note="saved automatically, during and after training")
        show = lambda _e: self.hint.set(self._OUT_HELP)

        tk.Label(inner, text="Save folder", bg=CARD, fg=TEXT, anchor="w",
                 font=FONTS["body"]).pack(fill="x")
        row = tk.Frame(inner, bg=CARD)
        row.pack(fill="x", pady=(4, 0))
        cage = tk.Frame(row, bg=BORDER)
        cage.pack(side="left", fill="x", expand=True)
        entry = tk.Entry(cage, textvariable=self.out_var, bg=FIELD, fg=TEXT,
                         insertbackground=GREEN, relief="flat", bd=0,
                         font=FONTS["mono"], highlightthickness=0,
                         selectbackground=_mix(FIELD, GREEN, 0.45),
                         selectforeground=TEXT)
        entry.pack(fill="x", padx=1, pady=1, ipady=4, ipadx=6)
        entry.bind("<FocusIn>", lambda _e: cage.configure(bg=GREEN))
        entry.bind("<FocusOut>", lambda _e: cage.configure(bg=BORDER))
        entry.bind("<FocusIn>", show, add="+")
        browse = NeonButton(row, "Browse", self._browse_out_dir, accent=GREEN,
                            kind="ghost", height=38, bg=CARD,
                            font=FONTS["btn_s"], pad=14)
        browse.pack(side="left", padx=(8, 0))

        self.out_preview = tk.Label(inner, bg=CARD, fg=FAINT, anchor="w",
                                    justify="left", font=FONTS["small"],
                                    wraplength=380)
        self.out_preview.pack(fill="x", pady=(6, 0))
        self._auto_wrap(self.out_preview, inner, pad=12)
        self.out_var.trace_add("write", lambda *_: self._update_out_preview())
        self._update_out_preview()

        for w in (inner, entry, browse, self.out_preview):
            w.bind("<Enter>", show, add="+")
        Tooltip(entry, self._OUT_HELP, title="output folder")

    def _update_out_preview(self):
        raw = self.out_var.get().strip()
        if not raw:
            self.out_preview.configure(text="→ choose a folder", fg=RED)
            return
        path = _resolve_out_dir(raw)
        busy = os.path.isdir(path) and bool(os.listdir(path))
        self.out_preview.configure(
            text="→ " + path + ("\n   not empty: files will be overwritten"
                               if busy else ""),
            fg=AMBER if busy else FAINT)

    def _browse_out_dir(self):
        start = _resolve_out_dir(self.out_var.get() or "runs")
        while start and not os.path.isdir(start):
            start = os.path.dirname(start)
        path = filedialog.askdirectory(initialdir=start or PROJECT_ROOT,
                                       mustexist=False,
                                       title="Folder to save the run to")
        if path:
            self.out_var.set(path)

    @staticmethod
    def _auto_wrap(label: tk.Label, ruler: tk.Widget, pad: int = 12):
        """Fit ``wraplength`` to the width of ``ruler``.

        The measurement is NOT taken from the label itself: doing so triggers a
        feedback loop (wrap changes -> height changes -> another <Configure>)
        that inflates the form to thousands of pixels tall.  ``ruler`` is the
        container, whose width does not depend on the label height.
        """

        def resize(event):
            want = max(160, event.width - pad)
            if abs(want - int(label.cget("wraplength"))) > 10:
                label.configure(wraplength=want)

        ruler.bind("<Configure>", resize, add="+")

    def _card(self, parent, title, accent, note=""):
        """Card: thin frame, accent bar and upper-case title."""
        shell = tk.Frame(parent, bg=BORDER)
        shell.pack(fill="x", pady=(0, 14))
        inner = tk.Frame(shell, bg=CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        head = tk.Frame(inner, bg=CARD)
        head.pack(fill="x", padx=16, pady=(12, 2))
        tk.Frame(head, bg=accent, width=3, height=15).pack(side="left",
                                                           padx=(0, 9))
        tk.Label(head, text=title.upper(), bg=CARD, fg=accent,
                 font=FONTS["card"]).pack(side="left")
        if note:
            nl = tk.Label(inner, text=note, bg=CARD, fg=FAINT, anchor="w",
                          justify="left", font=FONTS["small"], wraplength=600)
            nl.pack(fill="x", padx=16, pady=(0, 2))
            self._auto_wrap(nl, inner, pad=40)

        holder = tk.Frame(inner, bg=CARD)
        holder.pack(fill="both", expand=True, padx=16, pady=(4, 12))
        return holder

    def _field(self, holder, row, name, label, kind, help_text, defaults, accent):
        line = tk.Frame(holder, bg=CARD)
        line.grid(row=row, column=0, sticky="ew", pady=3)
        line.columnconfigure(0, weight=1)

        lbl = tk.Label(line, text=label, bg=CARD, fg=TEXT, anchor="w",
                       justify="left", font=FONTS["body"], wraplength=900)
        lbl.grid(row=0, column=0, sticky="w")

        fallback = {"epsilon_decay_mode": "linear",
                    "epsilon_decay_duration": 1800,
                    "epsilon_decay_interval": 1,
                    "lr_transient_steps": 100,
                    "LR_AFTER_TRANSIENT": 0.005}.get(name, "")
        default = DEFAULT_OVERRIDES.get(name, getattr(defaults, name, fallback))
        var = tk.StringVar(value="" if default is None else str(default))
        # Choices remain strings in self.vars, so the normal setup persistence
        # path applies without a special case.
        is_choice = isinstance(kind, tuple) and kind[:1] == ("choice",)
        converter = str if is_choice else kind
        self.vars[name] = (var, converter)

        cage = tk.Frame(line, bg=BORDER)
        cage.grid(row=0, column=1, sticky="e", padx=(12, 0))
        if is_choice:
            entry = ttk.Combobox(cage, textvariable=var,
                                 values=kind[1], state="readonly", width=15,
                                 font=FONTS["mono"])
        else:
            entry = tk.Entry(cage, textvariable=var, width=9, justify="right",
                             bg=FIELD, fg=TEXT, insertbackground=accent,
                             relief="flat", bd=0, font=FONTS["mono"],
                             disabledbackground=FIELD, highlightthickness=0,
                             selectbackground=_mix(FIELD, accent, 0.45),
                             selectforeground=TEXT)
        entry.pack(padx=1, pady=1, ipady=4, ipadx=6)
        entry.bind("<FocusIn>", lambda _e, c=cage, a=accent: c.configure(bg=a))
        entry.bind("<FocusOut>", lambda _e, c=cage: c.configure(bg=BORDER))

        # when the window narrows the label wraps instead of being clipped:
        # no parameter must ever become unreadable
        def wrap(event, lbl=lbl, cage=cage):
            avail = max(90, event.width - cage.winfo_reqwidth() - 20)
            if abs(avail - int(lbl.cget("wraplength"))) > 8:
                lbl.configure(wraplength=avail)

        line.bind("<Configure>", wrap, add="+")

        show = lambda _e, t=help_text: self.hint.set(t)
        for w in (line, lbl, entry):
            w.bind("<Enter>", show, add="+")
        entry.bind("<FocusIn>", show, add="+")
        Tooltip(lbl, help_text, title=name)
        Tooltip(entry, help_text, title=name)

    # -- hint bar and footer ------------------------------------------------
    def _build_hint_bar(self, parent):
        wrap = tk.Frame(parent, bg=BORDER)
        wrap.pack(fill="x", padx=22, pady=(4, 0))
        inner = tk.Frame(wrap, bg=PANEL)
        inner.pack(fill="x", padx=1, pady=1)
        tk.Label(inner, text="ⓘ", bg=PANEL, fg=CYAN,
                 font=FONTS["body"]).pack(side="left", padx=(14, 10), pady=10)
        lab = tk.Label(inner, textvariable=self.hint, bg=PANEL, fg=MUTED,
                       anchor="w", justify="left", wraplength=1100,
                       font=FONTS["small"], height=2)
        lab.pack(side="left", fill="x", expand=True, padx=(0, 14))
        self._auto_wrap(lab, inner, pad=64)
        inner.bind("<Enter>", lambda _e: self.hint.set(self._HINT_DEFAULT))

    def _build_footer(self, parent):
        foot = tk.Frame(parent, bg=BG)
        foot.pack(fill="x", padx=22, pady=(14, 18))
        tk.Label(foot, bg=BG, fg=FAINT, font=FONTS["small"], justify="left",
                 anchor="w",
                 text="The defaults are already a healthy run "
                      "(1000 episodes, lr 0.01).\nLeave the seed empty for an "
                      "unseeded run.").pack(side="left")
        self.btn_start = NeonButton(foot, "▶   START  TRAINING", self.start,
                                    accent=CYAN, kind="primary", height=54,
                                    bg=BG, font=FONTS["btn"], pad=40)
        self.btn_start.pack(side="right")
        return foot

    # ---------------------------------------------------- training screen ---
    def _build_training(self) -> tk.Frame:
        root = tk.Frame(self, bg=BG)

        top = tk.Frame(root, bg=PANEL)
        top.pack(fill="x")
        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        left = tk.Frame(top, bg=PANEL)
        left.pack(side="left", padx=(18, 0), pady=13)
        self.btn_back = NeonButton(left, "←  Back", self.back, accent=MUTED,
                                   kind="ghost", height=40, bg=PANEL,
                                   font=FONTS["btn_s"], pad=18)
        self.btn_back.pack(side="left")
        tk.Label(left, text="TRAINING", bg=PANEL, fg=TEXT,
                 font=FONTS["card"]).pack(side="left", padx=(20, 0))
        self.run_label = tk.Label(left, text="", bg=PANEL, fg=FAINT,
                                  font=FONTS["small"])
        self.run_label.pack(side="left", padx=(12, 0))

        right = tk.Frame(top, bg=PANEL)
        right.pack(side="right", padx=(0, 18), pady=13)
        self.btn_save = NeonButton(right, "Save plot", self.save,
                                   accent=VIOLET, kind="ghost", height=40,
                                   bg=PANEL, font=FONTS["btn_s"], pad=18)
        self.btn_save.pack(side="right", padx=(10, 0))
        self.btn_replay = NeonButton(right, "▶  Episode replay",
                                       self.open_episode_replay, accent=CYAN,
                                       kind="ghost", height=40, bg=PANEL,
                                       font=FONTS["btn_s"], pad=18)
        self.btn_replay.pack(side="right", padx=(10, 0))
        self.btn_stop = NeonButton(right, "■  Stop", self.stop, accent=RED,
                                   kind="ghost", height=40, bg=PANEL,
                                   font=FONTS["btn_s"], pad=18)
        self.btn_stop.pack(side="right")
        self.btn_stop.config(state="disabled")

        # chip row with the numbers that matter
        chips = tk.Frame(root, bg=BG)
        chips.pack(fill="x", padx=18, pady=(14, 6))
        self.chip_vals = {}
        chip_specs = (("game", "EPISODE", CYAN),
                      ("mse", "MSE AVG", CYAN),
                      ("loss", "LOSS AVG", AMBER),
                      ("reward", "REWARD AVG", GREEN),
                      ("lr", "LR", GREEN), ("eps", "EPSILON", VIOLET),
                      ("grad", "‖∇‖ RMS (NON-ZERO)", PINK),
                      ("params", "PARAMETERS", CYAN),
                      ("zero_grad", "ZERO ∇ (LAST)", RED))
        for column in range(5):
            chips.columnconfigure(column, weight=1, uniform="metric")
        for i, (key, name, col) in enumerate(chip_specs):
            shell = tk.Frame(chips, bg=BORDER)
            shell.grid(row=i // 5, column=i % 5, sticky="ew",
                       padx=(0, 10), pady=(0, 7))
            box = tk.Frame(shell, bg=CARD)
            box.pack(fill="both", expand=True, padx=1, pady=1)
            tk.Label(box, text=name, bg=CARD, fg=FAINT,
                     font=FONTS["chip"]).pack(anchor="w", padx=14, pady=(8, 0))
            v = tk.StringVar(value="—")
            tk.Label(box, textvariable=v, bg=CARD, fg=col,
                     font=FONTS["mono_b"]).pack(anchor="w", padx=14, pady=(0, 9))
            self.chip_vals[key] = v

        # slim progress bar
        self.prog = tk.Canvas(root, height=5, bg=PANEL, highlightthickness=0,
                              bd=0)
        self.prog.pack(fill="x", padx=18, pady=(2, 8))
        self.prog.bind("<Configure>", lambda _e: self._draw_progress())
        self._progress = 0.0

        # control row: which view, and the log scales
        ctl = tk.Frame(root, bg=BG)
        ctl.pack(fill="x", padx=18, pady=(0, 6))

        self._view_btns = {}
        seg = tk.Frame(ctl, bg=_mix(BG, "#ffffff", .05), bd=0,
                       highlightthickness=1, highlightbackground=BORDER_HI)
        seg.pack(side="left")
        for key, label in (("curve", "  Overview  "), ("focus", "  Focus  "),
                           ("policy", "  Policy  "), ("circuit", "  Circuit  "),
                           ("grads", "  Gradients  ")):
            b = tk.Label(seg, text=label, bg=_mix(BG, "#ffffff", .05), fg=MUTED,
                         font=FONTS["small"], padx=10, pady=5, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda _e, k=key: self.set_view(k))
            self._view_btns[key] = b

        # the log toggles only make sense for the curves: hidden elsewhere
        self._log_box = tk.Frame(ctl, bg=BG)
        for var, text in ((self.log_mse, "log MSE"), (self.log_loss, "log loss")):
            cb = tk.Checkbutton(self._log_box, text=text, variable=var,
                                command=self._redraw, bg=BG, fg=MUTED,
                                selectcolor=_mix(BG, "#ffffff", .10),
                                activebackground=BG, activeforeground=TEXT,
                                font=FONTS["small"], bd=0, highlightthickness=0,
                                cursor="hand2")
            cb.pack(side="left", padx=(14, 0))

        # one episode-based moving average shared by MSE, loss and reward
        tk.Label(self._log_box, text="AVG", bg=BG, fg=FAINT,
                 font=FONTS["chip"]).pack(side="left", padx=(24, 8))
        seg = tk.Frame(self._log_box, bg=_mix(BG, "#ffffff", .05),
                       highlightthickness=1, highlightbackground=BORDER_HI)
        seg.pack(side="left")
        self._avg_btns = {}
        for n in self._AVG_PRESETS:
            b = tk.Label(seg, text="  off  " if n == 1 else "  %d  " % n,
                         font=FONTS["small"], padx=2, pady=5, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda _e, k=n: self.set_loss_avg(k))
            self._avg_btns[n] = b
        cage = tk.Frame(self._log_box, bg=BORDER)
        cage.pack(side="left", padx=(8, 0))
        self.avg_var = tk.StringVar(value=str(self.loss_avg_n))
        ent = tk.Entry(cage, textvariable=self.avg_var, width=5, justify="right",
                       bg=FIELD, fg=TEXT, insertbackground=AMBER, relief="flat",
                       bd=0, font=FONTS["mono"], highlightthickness=0)
        ent.pack(padx=1, pady=1, ipady=3, ipadx=4)
        ent.bind("<Return>", self._avg_typed)
        ent.bind("<FocusOut>", self._avg_typed)
        tk.Label(self._log_box, text="episodes", bg=BG, fg=FAINT,
                 font=FONTS["small"]).pack(side="left", padx=(6, 0))
        Tooltip(ent, "Moving average shared by MSE, loss and reward. Each point is the "
                     "mean of the N episodes around it (at the live edge, of "
                     "the last N/2 available). The raw series stay "
                     "in the background, faint. Type any N and press Enter.",
                title="shared moving average")
        self._paint_avg_btns()

        # Focus mode: one large, readable scientific panel.  A native Tk Scale
        # is used as the QSlider-equivalent control so it remains dependency-free.
        self._focus_box = tk.Frame(ctl, bg=BG)
        tk.Label(self._focus_box, text="METRIC", bg=BG, fg=FAINT,
                 font=FONTS["chip"]).pack(side="left", padx=(20, 8))
        self.focus_name = tk.StringVar(value="MSE")
        tk.Label(self._focus_box, textvariable=self.focus_name, bg=BG, fg=CYAN,
                 font=FONTS["body"]).pack(side="left", padx=(0, 8))
        self.focus_slider = tk.Scale(
            self._focus_box, from_=0, to=4, orient="horizontal", showvalue=False,
            resolution=1, length=220, width=18, sliderlength=24,
            bg=BG, troughcolor=BORDER, activebackground=CYAN,
            highlightthickness=0, bd=0, fg=TEXT,
            command=self._focus_changed)
        self.focus_slider.set(self.focus_metric)
        self.focus_slider.pack(side="left", padx=(0, 8))
        tk.Label(self._focus_box, text="← drag to inspect one curve →", bg=BG,
                 fg=FAINT, font=FONTS["small"]).pack(side="left")

        # layer selector of the Gradients view: "All" plus one chip per layer,
        # rebuilt at every start (the depth is a setup parameter)
        self._layer_box = tk.Frame(ctl, bg=BG)
        tk.Label(self._layer_box, text="LAYER", bg=BG, fg=FAINT,
                 font=FONTS["chip"]).pack(side="left", padx=(22, 8))
        self._layer_seg = tk.Frame(self._layer_box, bg=_mix(BG, "#ffffff", .05),
                                   highlightthickness=1,
                                   highlightbackground=BORDER_HI)
        self._layer_seg.pack(side="left")
        tk.Label(self._layer_box, text="←/→ to step · click the heatmap to pick",
                 bg=BG, fg=FAINT, font=FONTS["small"]).pack(side="left",
                                                           padx=(12, 0))
        self._layer_btns = {}
        self._build_layer_chips(Config().deep_layers)
        self.bind_all("<Left>", lambda _e: self._step_grad_layer(-1), add="+")
        self.bind_all("<Right>", lambda _e: self._step_grad_layer(1), add="+")

        # the plots take all the remaining space
        holder = tk.Frame(root, bg=BORDER)
        holder.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        self.fig = Figure(figsize=(12, 7), dpi=100, facecolor=BG)
        # Curves: metrics on top; utilities get the wide lower panel.
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=(1, 1.15),
                      width_ratios=(1, 1, .72), hspace=.35, wspace=.3)
        self.ax = [self.fig.add_subplot(gs[0, 0]),
                   self.fig.add_subplot(gs[0, 1]),
                   self.fig.add_subplot(gs[0, 2]),
                   self.fig.add_subplot(gs[1, 0:2]),
                   self.fig.add_subplot(gs[1, 2])]
        # The epsilon twin axis is created once, so redraws do not stack ghosts.
        self.ax_eps = self.ax[4].twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        tkw = self.canvas.get_tk_widget()
        tkw.configure(bg=BG, highlightthickness=0, bd=0)
        tkw.pack(fill="both", expand=True, padx=1, pady=1)
        self._holder = holder
        self._tkw_curve = tkw

        # Focus figure/canvas: rebuilt as one large axis whenever the slider
        # changes, while the overview retains its five-panel live dashboard.
        self.fig_focus = Figure(figsize=(14, 8), dpi=100, facecolor=BG)
        self.ax_focus = self.fig_focus.add_subplot(1, 1, 1)
        self.canvas_focus = FigureCanvasTkAgg(self.fig_focus, master=holder)
        self._tkw_focus = self.canvas_focus.get_tk_widget()
        self._tkw_focus.configure(bg=BG, highlightthickness=0, bd=0)

        # policy figure, in the same holder: only one of the two is shown
        self.fig_pol = Figure(figsize=(12, 7), dpi=100, facecolor=BG)
        self.ax_pol = [self.fig_pol.add_subplot(1, 2, i + 1) for i in range(2)]
        self.canvas_pol = FigureCanvasTkAgg(self.fig_pol, master=holder)
        self._tkw_pol = self.canvas_pol.get_tk_widget()
        self._tkw_pol.configure(bg=BG, highlightthickness=0, bd=0)

        # live circuit, drawn on a Tk canvas of its own
        self.circuit = CircuitView(holder, self.params,
                                   on_select=self._on_gate_select)

        # gradients against time, with the list of the near-zero ones beside
        self._grads_frame = tk.Frame(holder, bg=BG)
        self._build_low_grad_panel(self._grads_frame)
        self.fig_grad = Figure(figsize=(12, 7), dpi=100, facecolor=BG)
        self.ax_grad = [self.fig_grad.add_subplot(2, 2, i + 1) for i in range(4)]
        self.canvas_grad = FigureCanvasTkAgg(self.fig_grad,
                                             master=self._grads_frame)
        self._tkw_grad = self.canvas_grad.get_tk_widget()
        self._tkw_grad.configure(bg=BG, highlightthickness=0, bd=0)
        self._tkw_grad.pack(side="left", fill="both", expand=True)
        self._grad_cbar = None
        self.canvas_grad.mpl_connect("button_press_event", self._on_grad_click)

        bar = tk.Frame(root, bg=PANEL)
        bar.pack(fill="x")
        tk.Label(bar, textvariable=self.status, bg=PANEL, fg=MUTED, anchor="w",
                 font=FONTS["small"]).pack(fill="x", padx=18, pady=9)

        self.set_view("curve")
        self._redraw()
        return root

    # -------------------------------------------------------------- views ---
    def set_view(self, which: str):
        """Show one view at a time: overview, focus, policy, circuit or gradients."""
        self.view = which
        for key, btn in self._view_btns.items():
            on = (key == which)
            btn.configure(fg=CYAN if on else MUTED,
                          bg=_mix(BG, CYAN, .18) if on else _mix(BG, "#ffffff", .05))
        widgets = {"curve": self._tkw_curve, "focus": self._tkw_focus,
                   "policy": self._tkw_pol, "circuit": self.circuit,
                   "grads": self._grads_frame}
        for w in widgets.values():
            w.pack_forget()
        widgets[which].pack(fill="both", expand=True, padx=1, pady=1)
        if which == "curve":
            self._log_box.pack(side="left")
        else:
            self._log_box.pack_forget()
        if which == "focus":
            self._focus_box.pack(side="left")
        else:
            self._focus_box.pack_forget()
        if which == "grads":
            self._layer_box.pack(side="left")
        else:
            self._layer_box.pack_forget()
        if which == "circuit":
            self.update_idletasks()
            self.circuit.refresh(force=True)
        self._redraw()

    _FOCUS_NAMES = ("MSE", "loss", "reward", "utilities", "schedule")

    def _focus_changed(self, value):
        self.focus_metric = max(0, min(4, int(float(value))))
        self.focus_name.set(self._FOCUS_NAMES[self.focus_metric])
        if self.view == "focus":
            self._redraw_focus()

    def _draw_transition_boundary(self, ax):
        """Mark the configured transient boundary when it is known."""
        game = self._lr_transition_game
        if game is not None and game > 0:
            ax.axvline(game, color=AMBER, lw=1.5, ls=":", alpha=.9,
                       label="transient → post-transient")
            ax.text(game, .98, "transient", transform=ax.get_xaxis_transform(),
                    color=AMBER, fontsize=10, ha="right", va="top")

    def _redraw_focus(self):
        """Render exactly one large metric panel selected by the slider."""
        if not hasattr(self, "ax_focus"):
            return
        a = self.ax_focus
        a.clear()
        h = self.history
        n = self.loss_avg_n
        idx = self.focus_metric
        if idx == 0:
            g, v, color, title = h["game"], h["mse"], CYAN, "MSE against Bellman"
            self._apply_yscale(a, v, self.log_mse.get())
        elif idx == 1:
            g, v, color, title = self.loss_ep["game"], self.loss_ep["loss"], AMBER, "training loss"
            self._apply_yscale(a, v, self.log_loss.get())
        elif idx == 2:
            g, v, color, title = self.reward_ep["game"], self.reward_ep["reward"], GREEN, "discounted reward"
        elif idx == 3:
            g = self.correct or []
            if self.U is not None and self.correct:
                a.plot(g, [self.U_ref[i] for i in g], "o--", color=CYAN,
                       lw=2.2, ms=8, label="Bellman (exact)")
                a.plot(g, [self.U[i] for i in g], "*-", color=AMBER,
                       lw=2.4, ms=14, label="VQC (learned)")
                a.legend(fontsize=12, facecolor=_mix(BG, "#ffffff", .07),
                         edgecolor=BORDER_HI, labelcolor=TEXT)
            self._style_axes(a, "state utilities  ·  maxₐ Q(s,a)", "state", "utility")
            self.fig_focus.subplots_adjust(left=.08, right=.97, bottom=.10, top=.93)
            self.canvas_focus.draw_idle()
            return
        else:
            g, v, color, title = h["game"], h["lr"], GREEN, "learning-rate and exploration schedule"
            a.plot(g, v, color=color, lw=2.4, label="learning rate")
            if h["eps"]:
                a.plot(h["game"], h["eps"], color=VIOLET, lw=2.2, ls="--", label="epsilon")
            self._draw_transition_boundary(a)
            a.legend(fontsize=12, facecolor=_mix(BG, "#ffffff", .07),
                     edgecolor=BORDER_HI, labelcolor=TEXT)
            self._style_axes(a, title, "episodes")
            self.fig_focus.subplots_adjust(left=.08, right=.97, bottom=.10, top=.93)
            self.canvas_focus.draw_idle()
            return
        if g:
            a.plot(g, v, color=color, lw=1.0, alpha=.24)
            if n > 1:
                a.plot(g, self._moving_avg(g, v, n), color=color, lw=2.8,
                       label="moving average (%d episodes)" % n)
            else:
                a.plot(g, v, color=color, lw=2.2, label="per episode")
            a.legend(fontsize=12, facecolor=_mix(BG, "#ffffff", .07),
                     edgecolor=BORDER_HI, labelcolor=TEXT)
        self._style_axes(a, title, "episodes")
        self.fig_focus.subplots_adjust(left=.08, right=.97, bottom=.10, top=.93)
        self.canvas_focus.draw_idle()

    def open_episode_replay(self):
        """Open (or refresh) an animated replay of the latest completed episode."""
        if not self.last_episode:
            messagebox.showinfo("Episode replay",
                                "No completed episode is available yet.")
            return
        if self.episode_view is None or not self.episode_view.winfo_exists():
            meta = getattr(self, "_episode_env", {})
            self.episode_view = EpisodeView(
                self, self.last_episode, nx=meta.get("nx", 4),
                ny=meta.get("ny", 3),
                obstacle_indexes=meta.get("obstacles", ()),
                alive_indexes=meta.get("alive", ()),
                death_indexes=meta.get("death", ()))
        else:
            self.episode_view.set_episode(self.last_episode)
            self.episode_view.deiconify()
            self.episode_view.lift()
        self.episode_view.replay()

    def _draw_progress(self):
        cv = self.prog
        cv.delete("all")
        w = cv.winfo_width() or 1
        h = int(cv.cget("height"))
        _round_rect(cv, 0, 0, w, h, h / 2, fill=_mix(PANEL, BORDER, .8),
                    outline="")
        frac = max(0.0, min(1.0, self._progress))
        if frac > 0:
            end = max(h, w * frac)
            steps = 40
            for i in range(steps):
                x0 = i * end / steps
                cv.create_rectangle(x0, 0, x0 + end / steps + 1, h, width=0,
                                    fill=_mix(CYAN, VIOLET, i / steps))

    # --------------------------------------------------------- matplotlib ---
    def _style_axes(self, a, title, xlabel=None, ylabel=None):
        a.set_facecolor(_mix(BG, "#ffffff", 0.045))
        for side, sp in a.spines.items():
            sp.set_color(BORDER_HI)
            sp.set_linewidth(0.8)
            if side in ("top", "right"):
                sp.set_visible(False)
        a.tick_params(colors=MUTED, labelsize=11, length=4, width=0.9)
        a.grid(True, color=BORDER, alpha=0.75, linewidth=0.6)
        a.set_axisbelow(True)
        a.set_title(title, color=TEXT, fontsize=14, pad=12, loc="left", fontweight="semibold")
        if xlabel:
            a.set_xlabel(xlabel, color=FAINT, fontsize=11)
        if ylabel:
            a.set_ylabel(ylabel, color=FAINT, fontsize=11)

    @staticmethod
    def _tick_text(v, _pos=None):
        """A readable number whatever the order of magnitude.

        ``ScalarFormatter`` on a log axis rounds small values down to zero: here
        the number of decimals is chosen from the value itself.
        """
        if v == 0:
            return "0"
        av = abs(v)
        if av >= 100:
            return "%.0f" % v
        if av >= 10:
            return "%.1f" % v
        if av >= 1:
            return "%.2f" % v
        if av >= 0.01:
            return "%.3f" % v
        return "%.1e" % v

    def _apply_yscale(self, ax, values, want_log: bool):
        """Linear or logarithmic y scale, with the ticks ALWAYS labelled.

        On a log scale with data spanning less than a decade, the default
        LogLocator finds no MAJOR tick and the axis is left without numbers --
        exactly what happened to the MSE panel.  So the locator is told to
        consider subdivisions and, when the data fits inside one decade, the
        minor ticks get labelled too.
        """
        vals = [v for v in values if v is not None and np.isfinite(v)]
        fmt = FuncFormatter(self._tick_text)
        if want_log and vals and min(vals) > 0:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0,), numticks=12))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1),
                           numticks=12))
            ax.yaxis.set_major_formatter(fmt)
            decades = len({int(np.floor(np.log10(v))) for v in vals})
            # within a single decade there would be at most one major tick:
            # label the minor ones too, otherwise the axis stays mute
            ax.yaxis.set_minor_formatter(fmt if decades < 2 else NullFormatter())
            ax.tick_params(axis="y", which="both", colors=MUTED, labelsize=9)
        else:
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(fmt)
            ax.tick_params(axis="y", which="both", colors=MUTED, labelsize=9)

    # -------------------------------------------------------- policy live ---
    _ARROW = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # su destra giu sx

    def _draw_policy_grid(self, ax, policy, title, reference=None):
        """4x3 grid in the style of the notebook's ``plot_policy``.

        Same symbols: green star on the +1, red circle on the -1, dark square on
        the obstacle, the living reward written in every cell and one arrow per
        chosen action.  When ``reference`` is given, arrows are green where they
        match that policy and red where they do not.
        """
        nx, ny = 4, 3
        ax.set_facecolor(_mix(BG, "#ffffff", 0.045))
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_color(BORDER_HI); sp.set_linewidth(0.8)
        for i in range(ny + 1):
            ax.axhline(i - 0.5, color=BORDER, lw=1.0)
        for i in range(nx + 1):
            ax.axvline(i - 0.5, color=BORDER, lw=1.0)

        # special cells, as in the notebook
        ax.plot(5 % nx, 5 // nx, marker="s", ms=46, color="#0d1526",
                markeredgecolor=BORDER_HI, markeredgewidth=1.2)
        ax.plot(11 % nx, 11 // nx, marker="*", ms=48, color=GREEN)
        ax.plot(7 % nx, 7 // nx, marker="o", ms=42, color=RED)

        # the living reward of each playable state, top-left in its cell
        r_txt = ("%g" % self._run_reward) if self._run_reward is not None else ""
        for i in (self.correct or []):
            ax.text(i % nx - 0.44, i // nx + 0.33, r_txt, color=FAINT,
                    fontsize=8, ha="left", va="center")

        if policy is not None:
            for i in (self.correct or []):
                x, y = i % nx, i // nx
                dx, dy = self._ARROW.get(int(policy[i]), (0, 0))
                col = CYAN
                if reference is not None:
                    col = GREEN if int(policy[i]) == int(reference[i]) else RED
                ax.arrow(x - dx * 0.28, y - dy * 0.28, dx * 0.52, dy * 0.52,
                         head_width=0.15, head_length=0.14, fc=col, ec=col,
                         lw=2.2, length_includes_head=True)

        ax.set_title(title, color=TEXT, fontsize=12.5, pad=10, loc="left")

    def _redraw_policy(self):
        for a in self.ax_pol:
            a.clear()
        self._draw_policy_grid(self.ax_pol[0], self.policy_ref,
                               "Bellman  (optimal)")
        n_ok = 0
        if self.policy is not None and self.policy_ref is not None:
            n_ok = sum(int(self.policy[i]) == int(self.policy_ref[i])
                       for i in (self.correct or []))
        tot = len(self.correct or [])
        self._draw_policy_grid(
            self.ax_pol[1], self.policy,
            "VQC  (learned)" + (f"   {n_ok}/{tot} correct" if tot else ""),
            reference=self.policy_ref)
        self.fig_pol.tight_layout(pad=2.0)
        self.canvas_pol.draw_idle()

    def _redraw(self):
        view = getattr(self, "view", "curve")
        if view == "policy":
            self._redraw_policy()
        elif view == "grads":
            self._redraw_grads()
        elif view == "curve":
            self._redraw_curves()
        elif view == "focus":
            self._redraw_focus()

    # ---------------------------------------------------------- gradients ---
    #: red (vanishing) -> amber -> green (healthy), same scale as the circuit
    _GRAD_CMAP = LinearSegmentedColormap.from_list(
        "grad", [RED, AMBER, GREEN])

    @staticmethod
    def _log_bins(games, values, nbins=400):
        """Geometric mean of ``values`` in at most ``nbins`` bins of episodes.

        Gradients are noisy episode to episode; binning in log space keeps the
        trend readable at any run length and never draws 20000 points.
        """
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v) & (v > 0)
        g, v = np.asarray(games)[ok], np.log10(v[ok])
        if len(v) == 0:
            return np.array([]), np.array([])
        k = max(1, int(np.ceil(len(v) / nbins)))
        n = len(v) // k * k
        if n == 0:
            return g, 10 ** v
        gb = g[:n].reshape(-1, k).mean(axis=1)
        vb = v[:n].reshape(-1, k).mean(axis=1)
        if n < len(v):
            gb = np.append(gb, g[n:].mean())
            vb = np.append(vb, v[n:].mean())
        return gb, 10 ** vb

    def _vanish_band(self, a):
        """Shade the region under 10**GRAD_LO: gradients there are vanishing.

        The y range always includes the threshold (for context) but stops at
        1e-9: an exact zero on one episode must not squash the whole axis.
        """
        a.relim()
        a.autoscale_view()
        lo, hi = a.get_ylim()
        lo = max(min(lo, 10 ** GRAD_LO / 10), 1e-9)
        hi = max(hi, 10 ** GRAD_LO * 10)
        a.set_ylim(lo, hi)
        a.axhspan(lo, 10 ** GRAD_LO, color=RED, alpha=0.08, lw=0)
        a.axhline(10 ** GRAD_LO, color=RED, lw=0.8, ls=":", alpha=0.7)
        a.text(0.995, 10 ** GRAD_LO, "vanishing ", transform=a.get_yaxis_transform(),
               ha="right", va="top", color=RED, fontsize=8.5, alpha=0.8)

    # -- layer selection -------------------------------------------------------
    def _build_layer_chips(self, n_layers: int):
        for b in self._layer_btns.values():
            b.destroy()
        self._layer_btns = {}
        self._n_layers = int(n_layers)
        if self.grad_layer is not None and self.grad_layer >= self._n_layers:
            self.grad_layer = None
        for key in [None] + list(range(self._n_layers)):
            b = tk.Label(self._layer_seg, text="  All  " if key is None
                         else "  %d  " % (key + 1), font=FONTS["small"],
                         padx=4, pady=5, cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda _e, k=key: self.set_grad_layer(k))
            self._layer_btns[key] = b
        self._paint_layer_chips()

    def _paint_layer_chips(self):
        for key, b in self._layer_btns.items():
            on = key == self.grad_layer
            b.configure(fg=CYAN if on else MUTED,
                        bg=_mix(BG, CYAN, .18) if on else _mix(BG, "#ffffff", .05))

    def set_grad_layer(self, layer):
        """Focus the Gradients view on one layer (``None`` = the whole circuit)."""
        self.grad_layer = layer
        self._paint_layer_chips()
        self._redraw_grads()

    def _step_grad_layer(self, k: int):
        """Arrow keys: All -> 1 -> 2 ... and back, only on the Gradients view."""
        if self.view != "grads" or self.screen != "training":
            return
        if isinstance(self.focus_get(), (tk.Entry, ttk.Treeview)):
            return    # arrows belong to the widget being typed in / browsed
        seq = [None] + list(range(self._n_layers))
        i = seq.index(self.grad_layer) if self.grad_layer in seq else 0
        self.set_grad_layer(seq[max(0, min(len(seq) - 1, i + k))])

    def _on_gate_select(self, gate):
        """Pinning a gate in the Circuit view brings its layer into focus."""
        if gate is not None:
            self.grad_layer = gate[0]
            self._paint_layer_chips()
        self._redraw_grads()

    def _on_grad_click(self, event):
        """Click on the heatmap: pick a layer, or inside a layer, pin a gate."""
        if event.inaxes is not self.ax_grad[2] or event.ydata is None:
            return
        row = int(round(event.ydata))
        if self.grad_layer is None:
            layer = row // 12
            if 0 <= layer < self.params.n_layers:
                self.set_grad_layer(layer)
        elif 0 <= row < 12:
            gate = (self.grad_layer, row // 3)
            if self.circuit.selected != gate:    # _select toggles
                self.circuit._select(gate)

    # -- near-zero gradients list ---------------------------------------------
    _LOW_HELP = ("Parameters whose gradient RMS is below the threshold. "
                 "'Last episode' looks at the newest episode only; 'Whole run' "
                 "lists those that never rose above it. 'no effect' = "
                 "structurally zero (layer-1 φ, last-layer ω); an exact 0 in "
                 "one episode usually means no updated action had that gate in "
                 "its light cone. Follows the layer selector; click a row to "
                 "pin the gate. Exact/structural zero gradients are excluded from "
                 "every curve, RMS aggregate and heatmap.")

    def _build_low_grad_panel(self, parent):
        shell = tk.Frame(parent, bg=BORDER)
        shell.pack(side="right", fill="y", padx=(10, 0))
        box = tk.Frame(shell, bg=CARD, width=400)
        box.pack(fill="both", expand=True, padx=1, pady=1)
        box.pack_propagate(False)

        head = tk.Frame(box, bg=CARD)
        head.pack(fill="x", padx=14, pady=(12, 4))
        tk.Frame(head, bg=PINK, width=3, height=15).pack(side="left", padx=(0, 9))
        tk.Label(head, text="GRADIENTS BELOW THRESHOLD", bg=CARD, fg=PINK,
                 font=FONTS["card"]).pack(side="left")

        row = tk.Frame(box, bg=CARD)
        row.pack(fill="x", padx=14, pady=(6, 2))
        tk.Label(row, text="|∇| <", bg=CARD, fg=TEXT,
                 font=FONTS["body"]).pack(side="left")
        cage = tk.Frame(row, bg=BORDER)
        cage.pack(side="left", padx=(8, 0))
        self.low_thr = tk.StringVar(value="1e-10")
        ent = tk.Entry(cage, textvariable=self.low_thr, width=8, justify="right",
                       bg=FIELD, fg=TEXT, insertbackground=PINK, relief="flat",
                       bd=0, font=FONTS["mono"], highlightthickness=0)
        ent.pack(padx=1, pady=1, ipady=3, ipadx=4)
        ent.bind("<Return>", lambda _e: self._fill_low_grads())
        ent.bind("<FocusOut>", lambda _e: self._fill_low_grads())

        self.low_scope = "last"
        self._scope_btns = {}
        seg = tk.Frame(row, bg=_mix(CARD, "#ffffff", .05), highlightthickness=1,
                       highlightbackground=BORDER_HI)
        seg.pack(side="right")
        for key, label in (("last", " Last episode "), ("run", " Whole run ")):
            b = tk.Label(seg, text=label, font=FONTS["small"], padx=4, pady=3,
                         cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda _e, k=key: self._set_low_scope(k))
            self._scope_btns[key] = b
        self._set_low_scope("last", fill=False)

        self.low_summary = tk.Label(box, text="", bg=CARD, fg=MUTED, anchor="w",
                                    justify="left", font=FONTS["small"])
        self.low_summary.pack(fill="x", padx=14, pady=(6, 6))
        tk.Label(box, text="ZERO AND STRUCTURAL GRADIENTS ARE EXCLUDED FROM "
                           "CURVES, RMS AND HEATMAP", bg=CARD, fg=AMBER,
                 justify="left", wraplength=360,
                 font=FONTS["chip"]).pack(fill="x", padx=14, pady=(0, 8))

        st = ttk.Style(self)
        rowh = max(22, int(self.tk.call("font", "metrics", FONTS["mono"],
                                        "-linespace")) + 8)
        st.configure("Q.Treeview", background=CARD, fieldbackground=CARD,
                     foreground=TEXT, bordercolor=CARD, borderwidth=0,
                     rowheight=rowh, font=FONTS["mono"])
        st.configure("Q.Treeview.Heading", background=PANEL, foreground=MUTED,
                     bordercolor=BORDER, relief="flat", font=FONTS["chip"])
        st.map("Q.Treeview", background=[("selected", _mix(CARD, AMBER, .25))],
               foreground=[("selected", TEXT)])
        st.map("Q.Treeview.Heading", background=[("active", CARD_HI)])

        frame = tk.Frame(box, bg=CARD)
        frame.pack(fill="both", expand=True, padx=(14, 6), pady=(0, 8))
        cols = ("param", "last", "max", "below", "note")
        tv = ttk.Treeview(frame, columns=cols, show="headings",
                          style="Q.Treeview", selectmode="browse")
        for c, text, w, anchor in (("param", "parameter", 88, "w"),
                                   ("last", "last", 62, "e"),
                                   ("max", "max run", 62, "e"),
                                   ("below", "% ep.", 50, "e"),
                                   ("note", "", 90, "w")):
            tv.heading(c, text=text, anchor=anchor)
            tv.column(c, width=w, minwidth=40, anchor=anchor,
                      stretch=(c == "note"))
        sb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview,
                           style="Q.Vertical.TScrollbar")
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tv.pack(side="left", fill="both", expand=True)
        tv.tag_configure("dead", foreground=FAINT)
        tv.tag_configure("zero", foreground=RED)
        tv.tag_configure("low", foreground=AMBER)
        tv.bind("<<TreeviewSelect>>", self._low_grad_pick)
        self.low_tree = tv

        for w in (ent, self.low_summary):
            Tooltip(w, self._LOW_HELP, title="near-zero gradients")
        tk.Label(box, text="grey: no effect · red: ≈0, no signal · amber: small",
                 bg=CARD, fg=FAINT, font=FONTS["small"], anchor="w").pack(
                     fill="x", padx=14, pady=(0, 10))

    def _set_low_scope(self, key, fill=True):
        self.low_scope = key
        for k, b in self._scope_btns.items():
            on = k == key
            b.configure(fg=PINK if on else MUTED,
                        bg=_mix(CARD, PINK, .18) if on else _mix(CARD, "#ffffff", .05))
        if fill:
            self._fill_low_grads()

    def _low_threshold(self) -> float:
        try:
            thr = float(self.low_thr.get())
            if thr > 0:
                return thr
        except ValueError:
            pass
        self.low_thr.set("1e-10")
        return 1e-10

    def _grad_stats(self, thr: float, hp=None):
        """One dict per parameter: last / max gradient RMS, share of episodes
        below ``thr``, structural flag.  Raw values (zeros kept)."""
        hp = hp or self.params
        L = hp.n_layers
        G = hp.G[1:]
        n = len(G)
        if not n:
            return []
        dead = dead_mask(L)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gmax = np.nanmax(G, axis=0)
        below = np.mean(G < thr, axis=0)
        out = []
        for l in range(L):
            for i in range(4):
                for j in range(3):
                    out.append({"layer": l, "qubit": i, "param": j,
                                "last": float(G[-1, l, i, j]),
                                "max": float(gmax[l, i, j]),
                                "frac_below": float(below[l, i, j]),
                                "structural": bool(dead[l, i, j])})
        return out

    def _fill_low_grads(self):
        tv = self.low_tree
        thr = self._low_threshold()
        stats = self._grad_stats(thr)
        lay = self.grad_layer
        if lay is not None:
            stats = [d for d in stats if d["layer"] == lay]
        key = "last" if self.low_scope == "last" else "max"
        hits = [d for d in stats if d[key] < thr]
        # structural zeros first, then the ones that are low most often
        hits.sort(key=lambda d: (not d["structural"], -d["frac_below"], d[key]))
        keep = tv.selection()
        tv.delete(*tv.get_children())
        for d in hits:
            iid = "%d:%d:%d" % (d["layer"], d["qubit"], d["param"])
            if d["structural"]:
                tag, note = "dead", "no effect"
            elif d[key] < GRAD_ZERO:
                # rounding noise around an exact zero: in that episode no
                # updated action had this gate in its light cone
                tag, note = "zero", "≈0, no signal"
            else:
                tag, note = "low", ""
            tv.insert("", "end", iid=iid, tags=(tag,), values=(
                "L%d q%d %s" % (d["layer"] + 1, d["qubit"], PARAM_NAMES[d["param"]]),
                self._fmt_small(d["last"]), self._fmt_small(d["max"]),
                "%.0f%%" % (100 * d["frac_below"]), note))
        for iid in keep:
            if tv.exists(iid):
                tv.selection_set(iid)
        where = "all layers" if lay is None else "layer %d" % (lay + 1)
        n_struct = sum(d["structural"] for d in hits)
        if not stats:
            txt = "no gradients yet"
        else:
            txt = ("%d of %d parameters (%s)  ·  %s\n%d structural, %d from "
                   "training" % (len(hits), len(stats), where,
                                 "newest episode" if key == "last"
                                 else "never above the threshold",
                                 n_struct, len(hits) - n_struct))
        self.low_summary.config(text=txt)

    @staticmethod
    def _fmt_small(v: float) -> str:
        if not np.isfinite(v):
            return "—"
        return "0" if v == 0 else "%.0e" % v

    def _low_grad_pick(self, _e=None):
        sel = self.low_tree.selection()
        if not sel:
            return
        l, i, _j = (int(x) for x in sel[0].split(":"))
        if self.circuit.selected != (l, i):
            self.circuit._select((l, i))

    def write_grad_stats(self, path: str, thr=None, hp=None):
        """``gradient_stats.csv``: every parameter, with the threshold in use."""
        thr = self._low_threshold() if thr is None else thr
        stats = self._grad_stats(thr, hp)
        if not stats:
            return
        with open(path, "w") as fh:
            fh.write("layer,qubit,param,last_grad_rms,max_grad_rms,"
                     "frac_episodes_below_%g,below_now,structural_zero\n" % thr)
            for d in stats:
                fh.write("%d,%d,%s,%.6e,%.6e,%.4f,%d,%d\n" % (
                    d["layer"] + 1, d["qubit"], "phi theta omega".split()[d["param"]],
                    d["last"], d["max"], d["frac_below"], d["last"] < thr,
                    d["structural"]))

    def _redraw_grads(self):
        """The Gradients view: is the gradient signal dying, and where?"""
        if getattr(self, "view", "") != "grads":
            return
        self._grads_drawn = time.time()
        # nanmean over the all-NaN "no effect" rows warns on every redraw
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._draw_grads()
            self._fill_low_grads()

    def _draw_grads(self, fig=None, axes=None, hp=None, layer="current",
                    sel="current", hints=True):
        """Draw the four gradient plots.

        With no arguments it redraws the on-screen view: the layer selector and
        the gate pinned in the Circuit view decide what is shown.  The
        end-of-run export passes its own ``fig``/``axes`` (a figure that is not
        attached to Tk, so it can be drawn off the main thread), a snapshot
        ``hp`` of the history, an explicit ``layer``, no pinned gate and
        ``hints=False`` to keep the "click here" tips out of the saved titles.
        """
        interactive = fig is None
        fig = fig or self.fig_grad
        axes = axes or self.ax_grad
        hp = hp or self.params
        if interactive and self._grad_cbar is not None:
            # the colorbar goes first: removing it gives its room back to the
            # heatmap axes, which clear() would otherwise leave dangling
            self._grad_cbar.remove()
            self._grad_cbar = None
        for a in axes:
            a.clear()
        games = hp.games[1:]             # row 0 = initial weights, no gradient
        L = hp.n_layers
        # structurally dead parameters (identically zero gradient) would drag
        # every norm down and fake a vanishing gradient: they become NaN
        G = hp.G[1:].copy()
        G[:, dead_mask(L)] = np.nan
        # an exact zero means "no signal this episode" (light cone), not a
        # vanishing gradient: leave it out as well
        G[np.abs(G) < GRAD_ZERO] = np.nan

        if layer == "current":
            layer = self.grad_layer
        if sel == "current":
            sel = self.circuit.selected
        lay = layer if (layer is not None and layer < L) else None
        n = len(games)

        def rms(block):
            """RMS over every axis but time; NaN (no effect / no signal) skipped."""
            return np.sqrt(np.nanmean(block.reshape(n, -1) ** 2, axis=1))

        def layer_color(l):
            return _mix(CYAN, VIOLET, l / max(L - 1, 1))

        def legend(a, ncol=1):
            leg = a.legend(fontsize=8.5, ncol=ncol,
                           facecolor=_mix(BG, "#ffffff", .07),
                           edgecolor=BORDER_HI, labelcolor=TEXT)
            leg.get_frame().set_linewidth(0.8)

        # 1. whole circuit, with the focused layer on top of it
        a = axes[0]
        if n:
            tot = rms(G)
            gb, vb = self._log_bins(games, tot)
            if lay is None:
                a.plot(games, tot, color=PINK, lw=0.6, alpha=0.35)
                a.plot(gb, vb, color=PINK, lw=2.0)
            else:
                a.plot(gb, vb, color=PINK, lw=1.2, alpha=0.45,
                       label="whole circuit")
                gl, vl = self._log_bins(games, rms(G[:, lay]))
                a.plot(gl, vl, color=layer_color(lay), lw=2.2,
                       label="layer %d" % (lay + 1))
                legend(a)
            self._vanish_band(a)
        self._style_axes(a, "‖∇W‖ RMS, whole circuit" if lay is None else
                         "‖∇W‖ RMS, layer %d vs whole circuit" % (lay + 1),
                         "episodes")
        a.set_yscale("log")

        # 2. all layers, or the four qubits of the focused layer
        a = axes[1]
        if n and lay is None:
            for l in range(L):
                gb, vb = self._log_bins(games, rms(G[:, l]))
                if len(gb):
                    a.plot(gb, vb, lw=1.7, label="layer %d" % (l + 1),
                           color=layer_color(l))
            self._vanish_band(a)
            if L <= 8:
                legend(a, ncol=2 if L > 4 else 1)
        elif n:
            for i in range(4):
                gb, vb = self._log_bins(games, rms(G[:, lay, i]))
                if len(gb):
                    hi = sel == (lay, i)
                    a.plot(gb, vb, lw=2.6 if hi else 1.6, label="q%d" % i,
                           color=(CYAN, GREEN, AMBER, VIOLET)[i])
            self._vanish_band(a)
            legend(a, ncol=4)
        self._style_axes(a, "‖∇W‖ RMS per layer" if lay is None else
                         "layer %d · per qubit" % (lay + 1), "episodes")
        a.set_yscale("log")

        # 3. heatmap: every parameter, or the 12 of the focused layer
        a = axes[2]
        self._style_axes(a, "every parameter  (log₁₀ gradient RMS · grey = no "
                            "effect / no signal)" if lay is None else
                         "layer %d  (log₁₀ gradient RMS%s)" % (
                             lay + 1, " · click a row to pin its gate" if hints
                             else ""), "episodes")
        a.grid(False)
        if n:
            block = G if lay is None else G[:, lay:lay + 1]
            flat = block.reshape(n, -1)
            lg = np.log10(np.clip(flat, 1e-9, None))   # NaN stays NaN: grey
            # at most ~300 time bins: one column per bin
            k = max(1, n // 300)
            m = n // k * k
            img = np.nanmean(lg[:m].reshape(-1, k, lg.shape[1]), axis=1).T
            rows = img.shape[0]
            im = a.imshow(img, aspect="auto", origin="upper",
                          cmap=self._GRAD_CMAP, vmin=GRAD_LO, vmax=GRAD_HI,
                          interpolation="nearest",
                          extent=(games[0], games[m - 1] + 1, rows - 0.5, -0.5))
            if lay is None:
                for l in range(1, L):
                    a.axhline(l * 12 - 0.5, color=BG, lw=1.2)
                a.set_yticks([l * 12 + 5.5 for l in range(L)])
                a.set_yticklabels(["L%d" % (l + 1) for l in range(L)])
            else:
                for i in range(1, 4):
                    a.axhline(i * 3 - 0.5, color=BG, lw=1.6)
                a.set_yticks(range(12))
                a.set_yticklabels(["q%d %s" % (r // 3, PARAM_NAMES[r % 3])
                                   for r in range(12)], fontsize=8.5)
            a.set_facecolor(_mix(BG, "#ffffff", 0.09))   # NaN = no effect
            cbar = fig.colorbar(im, ax=a, pad=0.01, fraction=0.04)
            cbar.ax.tick_params(colors=MUTED, labelsize=8)
            cbar.outline.set_edgecolor(BORDER_HI)
            if interactive:
                self._grad_cbar = cbar
            if sel is not None and (lay is None or sel[0] == lay):
                y = (sel[0] * 12 if lay is None else 0) + sel[1] * 3 - 0.5
                a.axhspan(y, y + 3, fill=False, ec=AMBER, lw=1.4)

        # 4. the pinned gate (if it is in view), else phi/theta/omega averaged
        a = axes[3]
        if sel is not None and (lay is None or sel[0] == lay):
            l, i = sel
            title = "layer %d · q%d  —  φ θ ω" % (l + 1, i)
            series = [G[:, l, i, j] for j in range(3)]
        else:
            where = "the whole circuit" if lay is None else "layer %d" % (lay + 1)
            title = "φ / θ / ω averaged over %s" % where
            if hints:
                title += "  (pin a gate to follow it)"
            src = G if lay is None else G[:, lay:lay + 1]
            series = [rms(src[:, :, :, j]) if n else [] for j in range(3)]
        for j, ser in enumerate(series):
            gb, vb = self._log_bins(games, ser)
            if len(gb):
                a.plot(gb, vb, lw=1.7, color=PARAM_COLORS[j],
                       label=PARAM_NAMES[j])
        if n:
            self._vanish_band(a)
            legend(a, ncol=3)
        self._style_axes(a, title, "episodes")
        a.set_yscale("log")
        for a in (axes[0], axes[1], axes[3]):
            a.yaxis.set_major_formatter(FuncFormatter(self._tick_text))
            a.yaxis.set_minor_formatter(NullFormatter())

        if not len(games):
            axes[0].text(0.5, 0.5, "gradients appear after the first "
                                 "episode", transform=axes[0].transAxes,
                                 ha="center", color=FAINT)
        fig.tight_layout(pad=2.0)
        if interactive:
            self.canvas_grad.draw_idle()

    # -- shared episode-based moving average -----------------------------------
    _AVG_PRESETS = (1, 10, 20, 50, 100, 500)

    def _paint_avg_btns(self):
        for n, b in self._avg_btns.items():
            on = n == self.loss_avg_n
            b.configure(fg=AMBER if on else MUTED,
                        bg=_mix(BG, AMBER, .18) if on else _mix(BG, "#ffffff", .05))

    def set_loss_avg(self, n: int):
        self.loss_avg_n = max(1, int(n))
        self.avg_var.set(str(self.loss_avg_n))
        self._paint_avg_btns()
        self._update_loss_chip()
        if self.view == "curve":
            self._redraw_curves()

    def _avg_typed(self, _e=None):
        try:
            n = int(float(self.avg_var.get()))
        except ValueError:
            n = self.loss_avg_n
        if n != self.loss_avg_n:
            self.set_loss_avg(n)
        else:
            self.avg_var.set(str(self.loss_avg_n))

    @staticmethod
    def _moving_avg(games, values, n: int) -> np.ndarray:
        """Centred average in an episode-width window (also for sparse MSE)."""
        x = np.asarray(games, dtype=float)
        v = np.asarray(values, dtype=float)
        if n <= 1 or len(v) == 0:
            return v
        half = n / 2.0
        cumulative = np.concatenate([[0.0], np.cumsum(v)])
        lo = np.searchsorted(x, x - half, side="left")
        hi = np.searchsorted(x, x + half, side="right")
        return (cumulative[hi] - cumulative[lo]) / np.maximum(hi - lo, 1)

    def _window_mean(self, games, values):
        if not values:
            return None
        if self.loss_avg_n <= 1:
            return float(values[-1])
        last = games[-1]
        selected = [v for g, v in zip(games, values)
                    if g >= last - self.loss_avg_n + 1]
        return float(np.mean(selected)) if selected else None

    def _update_metric_chips(self):
        for key, data, value_key in (("mse", self.history, "mse"),
                                     ("loss", self.loss_ep, "loss"),
                                     ("reward", self.reward_ep, "reward")):
            value = self._window_mean(data["game"], data[value_key])
            if value is not None:
                self.chip_vals[key].set("%.4f" % value)

    def _update_loss_chip(self):
        self._update_metric_chips()

    def _redraw_curves(self):
        self._curves_drawn = time.time()
        self._loss_new = False
        h = self.history
        self.ax_eps.clear()
        for a in self.ax:
            a.clear()

        a = self.ax[0]
        mg, mv = h["game"], h["mse"]
        n = self.loss_avg_n
        if n > 1:
            a.plot(mg, mv, color=CYAN, lw=0.8, alpha=0.22, label="evaluations")
            ma = self._moving_avg(mg, mv, n)
            a.plot(mg, ma, color=CYAN, lw=2.0, label="mean over %d episodes" % n)
            title = "MSE against Bellman · moving average over %d episodes" % n
        else:
            ma = np.asarray(mv)
            a.plot(mg, mv, color=CYAN, lw=1.2)
            title = "MSE against Bellman · no averaging"
        if len(mg):
            a.plot(mg[-1:], ma[-1:], "o", color=CYAN, ms=5)
        self._style_axes(a, title, "episodes")
        self._apply_yscale(a, mv, self.log_mse.get())

        # loss: every episode, raw and faint, with the moving average on top
        a = self.ax[1]
        g, lv = self.loss_ep["game"], self.loss_ep["loss"]
        n = self.loss_avg_n
        if n > 1:
            a.plot(g, lv, color=AMBER, lw=0.8, alpha=0.22, label="per episode")
            avg = self._moving_avg(g, lv, n)
            a.plot(g, avg, color=AMBER, lw=2.0, label="mean of %d episodes" % n)
            if len(g):
                a.plot(g[-1:], avg[-1:], "o", color=AMBER, ms=5)
            title = "training loss  ·  moving average over %d episodes" % n
            if len(g):
                leg = a.legend(fontsize=9, facecolor=_mix(BG, "#ffffff", .07),
                               edgecolor=BORDER_HI, labelcolor=TEXT,
                               loc="upper right")
                leg.get_frame().set_linewidth(0.8)
        else:
            a.plot(g, lv, color=AMBER, lw=1.2)
            title = "training loss  ·  per episode (no averaging)"
        self._style_axes(a, title, "episodes")
        self._apply_yscale(a, lv, self.log_loss.get())

        a = self.ax[2]
        rg, rv = self.reward_ep["game"], self.reward_ep["reward"]
        if n > 1:
            a.plot(rg, rv, color=GREEN, lw=0.8, alpha=0.22,
                   label="discounted reward per episode")
            ra = self._moving_avg(rg, rv, n)
            a.plot(rg, ra, color=GREEN, lw=2.0,
                   label="mean over %d episodes" % n)
            title = "discounted reward · moving average over %d episodes" % n
        else:
            ra = np.asarray(rv)
            a.plot(rg, rv, color=GREEN, lw=1.2)
            title = "discounted reward · per episode (no averaging)"
        if len(rg):
            a.plot(rg[-1:], ra[-1:], "o", color=GREEN, ms=5)
        self._style_axes(a, title, "episodes")

        # Utilities: exact Bellman values versus the learned VQC, state by state.
        a = self.ax[3]
        if self.U is not None and self.correct:
            idx = self.correct
            a.plot(idx, [self.U_ref[i] for i in idx], marker="o", ls="--",
                   color=CYAN, lw=1.6, ms=6, label="Bellman (exact)")
            a.plot(idx, [self.U[i] for i in idx], marker="*", ls="-",
                   color=AMBER, lw=1.8, ms=11, label="VQC (learned)")
            leg = a.legend(fontsize=10, facecolor=_mix(BG, "#ffffff", .07),
                           edgecolor=BORDER_HI, labelcolor=TEXT)
            leg.get_frame().set_linewidth(0.8)
        self._style_axes(a, "utilities  max$_a$ Q(s,a)", "state")

        a = self.ax[4]
        a.plot(h["game"], h["lr"], color=GREEN, lw=2.0, label="learning rate")
        self._style_axes(a, "learning rate and epsilon", "episodes")
        self._draw_transition_boundary(a)
        a.set_ylabel("learning rate", color=GREEN, fontsize=10)
        a.tick_params(axis="y", colors=GREEN)
        if h["lr"] and min(h["lr"]) > 0:
            a.set_yscale("log")
        t = self.ax_eps
        # clear() also resets the twin axis side: it must be put back on the
        # right, otherwise the epsilon label lands on top of the lr one
        t.yaxis.set_label_position("right")
        t.yaxis.set_ticks_position("right")
        t.plot(h["game"], h["eps"], color=VIOLET, lw=1.7, ls="--")
        t.set_ylabel("epsilon", color=VIOLET, fontsize=10)
        t.tick_params(axis="y", colors=VIOLET, labelsize=9.5)
        t.set_facecolor("none")
        for sp in t.spines.values():
            sp.set_visible(False)
        if self._lr_transition_game is not None:
            leg = a.legend(fontsize=9.5, facecolor=_mix(BG, "#ffffff", .07),
                           edgecolor=BORDER_HI, labelcolor=TEXT,
                           loc="upper right")
            leg.get_frame().set_linewidth(.8)

        self.fig.subplots_adjust(left=.065, right=.93, bottom=.08, top=.94,
                                 hspace=.38, wspace=.35)
        self.canvas.draw_idle()

    # ------------------------------------------------------------ screens ---
    def show_setup(self):
        # a fresh default folder for the next run, so it never overwrites the
        # previous one -- unless the user typed a folder of their own
        if self.out_var.get() == self._auto_out:
            self._auto_out = _default_out_dir()
            self.out_var.set(self._auto_out)
        self._update_out_preview()
        self.frame_train.pack_forget()
        self.frame_setup.pack(fill="both", expand=True)
        self.screen = "setup"

    def show_training(self):
        self.frame_setup.pack_forget()
        self.frame_train.pack(fill="both", expand=True)
        self.screen = "training"

    def back(self):
        """Back to the parameters; if training is running, ask before stopping it."""
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(
                    "Training in progress",
                    "Training is still running.\n"
                    "Stop it and go back to the parameters?"):
                return
            self.stop()
        self.show_setup()

    # ------------------------------------------------------------- config ---
    def _read_config(self) -> Config:
        kw = {}
        for name, (var, kind) in self.vars.items():
            raw = var.get().strip()
            if raw == "":
                if name == "seed":
                    kw["seed"] = None
                continue
            try:
                kw[name] = kind(raw)
            except ValueError:
                raise ValueError(f"'{raw}' is not a valid value for {name}")

        # Keep the GUI usable with older Config objects during upgrades; new
        # controls are accepted automatically as soon as the core exposes them.
        from dataclasses import fields
        known = {f.name for f in fields(Config)}
        return replace(Config(fixes=self._fixes()),
                       **{k: v for k, v in kw.items() if k in known})

    def _extra_setup_state(self):
        """Subclass hook for setup controls not stored in ``self.vars``."""
        return {}

    def _apply_extra_setup_state(self, _data):
        """Inverse subclass hook used while restoring the last setup."""

    def _setup_snapshot(self):
        return {
            "version": 1,
            "parameters": {name: var.get() for name, (var, _kind) in self.vars.items()},
            "output": self.out_var.get(),
            "preferences": {
                "average_episodes": self.loss_avg_n,
                "log_mse": bool(self.log_mse.get()),
                "log_loss": bool(self.log_loss.get()),
            },
            **self._extra_setup_state(),
        }

    def _save_last_setup(self):
        """Persist a validated Start configuration atomically."""
        try:
            write_json(LAST_SETUP_PATH, self._setup_snapshot())
        except (OSError, TypeError, ValueError):
            pass  # preferences must never prevent a training run

    def _load_last_setup(self):
        """Restore the previous Start configuration; ignore absent/bad JSON."""
        try:
            with open(LAST_SETUP_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return
            for name, value in data.get("parameters", {}).items():
                if name in self.vars:
                    self.vars[name][0].set("" if value is None else str(value))
            if isinstance(data.get("output"), str) and data["output"].strip():
                self.out_var.set(data["output"].strip())
            prefs = data.get("preferences", {})
            if isinstance(prefs, dict):
                self.log_mse.set(bool(prefs.get("log_mse", self.log_mse.get())))
                self.log_loss.set(bool(prefs.get("log_loss", self.log_loss.get())))
                try:
                    self.set_loss_avg(int(prefs.get("average_episodes",
                                                    self.loss_avg_n)))
                except (TypeError, ValueError):
                    pass
            self._apply_extra_setup_state(data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    def _fixes(self) -> FixFlags:
        """Always the corrected agent.  Only the debug GUI overrides this."""
        return FixFlags()

    def _prepare_out_dir(self) -> str | None:
        """Resolve and create the output folder; ``None`` if the user backs out."""
        raw = self.out_var.get().strip()
        if not raw:
            messagebox.showerror("Output folder", "Choose a folder to save the run to.")
            return None
        path = _resolve_out_dir(raw)
        if os.path.isdir(path) and os.listdir(path):
            if not messagebox.askyesno(
                    "Output folder not empty",
                    f"{path}\n\nalready contains files. Files with the same "
                    "name will be overwritten.\nContinue?"):
                return None
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Output folder", f"Cannot create {path}:\n{e}")
            return None
        return path

    # --------------------------------------------------------------- run ---
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._read_config()
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return
        out_dir = self._prepare_out_dir()
        if out_dir is None:
            return
        self.out_dir = out_dir
        self._save_last_setup()

        self.history = {k: [] for k in self.history}
        self.loss_ep = {"game": [], "loss": []}
        self.reward_ep = {"game": [], "reward": []}
        self.params.clear()
        self.circuit.reset(q_scale=cfg.q_scale)
        self._build_layer_chips(cfg.deep_layers)
        self.U = self.U_ref = None
        self.total_games = max(1, int(cfg.num_games))
        self._progress = 0.0
        for v in self.chip_vals.values():
            v.set("—")
        self.chip_vals["params"].set(str(cfg.deep_layers * 4 * 3))
        self.chip_vals["zero_grad"].set("—")
        self.stop_flag.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        # The transient is measured in optimiser steps, not episodes. Keep the
        # boundary unset until the core monitor reports the actual episode in
        # which it occurs; this avoids drawing a misleading converted x-value.
        self._lr_transition_game = None
        eps_duration = getattr(cfg, "epsilon_decay_duration",
                               getattr(cfg, "epsilon_decay_denom", "—"))
        eps_interval = getattr(cfg, "epsilon_decay_interval", 1)
        self.run_label.config(
            text=("%d ep · %d layers · transient lr %g → %g for %d optimizer steps · "
                  "ε %s / %s ep (every %s)   →  %s")
                 % (cfg.num_games, cfg.deep_layers, cfg.LR_0,
                    getattr(cfg, "LR_AFTER_TRANSIENT", cfg.LR_0),
                    getattr(cfg, "lr_transient_steps", 0),
                    cfg.epsilon_decay_mode, eps_duration, eps_interval,
                    os.path.basename(os.path.normpath(out_dir))))
        self.status.set("building the environment and the Bellman reference...")
        self.show_training()
        self._draw_progress()
        self._redraw()

        self.worker = threading.Thread(target=self._run, args=(cfg, out_dir),
                                       daemon=True)
        self.worker.start()

    def _run(self, cfg: Config, out_dir: str):
        """Runs on the worker thread: NEVER touch a widget here, only the queue.

        All the file writing of a run happens here too (pyplot on Agg, never
        touched by the main thread), except the two GUI views, which belong
        to the Tk figures and are saved by :meth:`_save_views`.
        """
        history = {"game": [], "mse": [], "loss": [], "lr": [], "epsilon": [],
                   "steps": []}
        # per-episode weights and gradient RMS, written as params_history.npz
        p_games, p_W, p_G = [], [], []
        summary = None
        t0 = time.time()
        try:
            env = GridWorld(cfg)
            bellman = value_iteration_detailed(env, cfg.gamma, cfg.max_epoch)
            U_ref = bellman.U_final
            pol_ref = greedy_policy(env, U_ref, cfg.gamma)
            self._run_reward = cfg.r
            self.queue.put(("setup", {
                "correct": correct_indexes(env),
                "U_ref": np.asarray(U_ref, dtype=float),
                "policy_ref": np.asarray(pol_ref, dtype=int),
                "episode_env": {
                    "nx": env.Nx, "ny": env.Ny,
                    "obstacles": env.obstacle_indexes,
                    "alive": env.alive_indexes, "death": env.death_indexes,
                },
            }))

            summary = base_summary(cfg, env, bellman, pol_ref)
            write_json(os.path.join(out_dir, "summary.json"), summary)
            write_bellman_plots(out_dir, env, bellman, pol_ref)
            live = LivePlotter(os.path.join(out_dir, "live.png"),
                               correct_indexes(env))
            last = {"info": None, "episode": None, "io": 0.0}

            def write_progress():
                """live.png + progress.json.  Both cost more the longer the run
                (a pyplot redraw, the whole history rewritten), so they are
                written at most every IO_EVERY seconds, and once at the end."""
                info = last["info"]
                episode = last["episode"]
                last["io"] = time.time()
                if episode is not None:
                    write_json(os.path.join(out_dir, "last_episode.json"), episode)
                if info is None:
                    return
                live.draw()
                write_json(os.path.join(out_dir, "progress.json"), {
                    **history,
                    "U": jsonable(info.get("U")),
                    "policy": jsonable(info.get("policy")),
                    "weights": jsonable(p_W[-1] if p_W else None),
                    "grad_rms": jsonable(p_G[-1] if p_G else None),
                    "last_episode": episode,
                })

            def monitor(info):
                if self.stop_flag.is_set():
                    raise StopTraining
                self.queue.put(("tick", info))
                for k in history:
                    history[k].append(info[k])
                live.record(info)
                last["info"] = info
                if time.time() - last["io"] >= self.IO_EVERY:
                    write_progress()

            def game_monitor(info):
                # every episode: also the quickest place to honour Stop
                if self.stop_flag.is_set():
                    raise StopTraining
                p_games.append(info["game"])
                p_W.append(info["weights"])
                p_G.append(info["grad_rms"])
                if info["game"] >= 0 and info.get("trajectory"):
                    episode = {key: jsonable(info.get(key)) for key in
                               ("game", "discounted_reward", "trajectory",
                                "actions", "rewards", "steps")}
                    last["episode"] = episode
                if time.time() - last["io"] >= self.IO_EVERY:
                    write_progress()
                self.queue.put(("game", info))

            t0 = time.time()
            res = train(cfg, env, U_ref=U_ref, monitor=monitor,
                        game_monitor=game_monitor)
            elapsed = time.time() - t0

            write_progress()
            summary["training"] = training_summary(res, bellman, env, elapsed)
            summary["training"]["stopped"] = False
            summary["training"]["weights_final"] = jsonable(p_W[-1])
            summary["training"]["last_episode"] = last["episode"]
            self._write_params(out_dir, p_games, p_W, p_G)
            write_training_plots(out_dir, env, bellman, res)
            write_json(os.path.join(out_dir, "summary.json"), summary)
            self.queue.put(("done", {
                "policy": np.asarray(res.policy, dtype=int),
                "policy_ref": np.asarray(pol_ref, dtype=int),
                "steps": res.steps,
            }))
        except StopTraining:
            # keep what was learned so far: the curves up to the last evaluation
            try:
                write_progress()
            except Exception:  # noqa: BLE001 - best effort, summary comes next
                pass
            if summary is not None:
                summary["training"] = {"stopped": True,
                                       "seconds": time.time() - t0,
                                       "history": history,
                                       "weights_final": jsonable(
                                           p_W[-1] if p_W else None),
                                       "last_episode": last["episode"]}
                try:
                    write_json(os.path.join(out_dir, "summary.json"), summary)
                    self._write_params(out_dir, p_games, p_W, p_G)
                except OSError as e:
                    self.queue.put(("error", f"saving failed: {e}"))
                    return
            self.queue.put(("stopped", None))
        except Exception as e:  # pragma: no cover - percorso di errore interattivo
            self.queue.put(("error", f"{type(e).__name__}: {e}"))

    #: seconds between two writes of live.png / progress.json during a run
    IO_EVERY = 10.0
    #: minimum seconds between two matplotlib redraws while training
    REDRAW_EVERY = 1.0

    @staticmethod
    def _write_params(out_dir, games, W, G):
        """``params_history.npz``: games (n,), weights and grad_rms (n, L, 4, 3).

        Row 0 is the initial weights (game -1), whose grad_rms is NaN.
        """
        if not games:
            return
        nan = np.full_like(W[0], np.nan)
        np.savez_compressed(
            os.path.join(out_dir, "params_history.npz"),
            games=np.asarray(games), weights=np.stack(W),
            grad_rms=np.stack([nan if g is None else g for g in G]))

    def stop(self):
        self.stop_flag.set()
        self.status.set("stopping at the next evaluation...")

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="training.png",
            initialdir=self.out_dir or PROJECT_ROOT)
        if not path:
            return
        try:
            if self.view == "circuit":
                path = self._save_circuit(path)
            else:
                fig = {"policy": self.fig_pol, "grads": self.fig_grad,
                       "focus": self.fig_focus}.get(self.view, self.fig)
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
            self.status.set(f"saved to {path}")
        except Exception as e:  # noqa: BLE001 - reported, never fatal
            messagebox.showerror("Save failed", str(e))

    def _save_circuit(self, path: str) -> str:
        """The circuit canvas as PNG (through ghostscript) or, failing that, EPS.

        Tk can only export a canvas as PostScript; Pillow turns it into a PNG
        when ghostscript is installed.  Returns the path actually written.
        """
        self.circuit.refresh(force=True)
        cv = self.circuit.cv
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 50 or h < 50:
            raise RuntimeError("open the Circuit view once to lay it out")
        eps = os.path.splitext(path)[0] + ".eps"
        cv.postscript(file=eps, colormode="color", width=w, height=h,
                      pagewidth=w, pageheight=h)
        if not path.lower().endswith(".png"):
            return eps
        try:
            from PIL import Image
            with Image.open(eps) as im:
                im.load(scale=3)
                im.convert("RGB").save(path)
            os.remove(eps)
            return path
        except Exception:  # noqa: BLE001 - no ghostscript: keep the EPS
            return eps

    def _save_views(self) -> str:
        """End of run (or Stop): save the final state of every view.

        Here, on the Tk thread, only the two figures that ARE the window
        (``gui_curves.png``, ``gui_policy.png``: fast).  Everything else --
        ``gradients/all`` and ``gradients/layer_<k>``, ``circuit/circuit_angles``
        and ``circuit/circuit_gradients`` (each .png and .pdf) and
        ``gradient_stats.csv`` -- is drawn from a snapshot of the data on a
        background thread, so the window stays responsive: Back, a new run or
        closing it all work at once.  Returns a note for the status bar.
        """
        if not self.out_dir:
            return ""
        problems = []
        try:
            self._redraw_curves()
            self._redraw_policy()
            for fig, name in ((self.fig, "gui_curves.png"),
                              (self.fig_pol, "gui_policy.png")):
                fig.savefig(os.path.join(self.out_dir, name), dpi=150,
                            facecolor=fig.get_facecolor())
        except Exception as e:  # noqa: BLE001 - reported, never fatal
            problems.append(f"views: {e}")

        # everything the thread needs is copied now: a new run may start (and
        # clear the live history) while it is still writing
        args = (self.out_dir, self.params.snapshot(), self.circuit.q_scale,
                self._low_threshold())
        self.exporter = threading.Thread(target=self._export_final, args=args,
                                         daemon=True)
        self.exporter.start()
        note = "   ·   saving the final plots to %s ..." % self.out_dir
        if problems:
            note += "  (failed: %s)" % "; ".join(problems)
        return note

    def _export_final(self, out_dir, hp, q_scale, thr):
        """Background thread: never touches a widget, reports through the queue."""
        problems = []
        try:
            self._save_final_gradients(out_dir, hp)
        except Exception as e:  # noqa: BLE001
            problems.append(f"gradients: {e}")
        try:
            save_final_circuit(out_dir, hp, q_scale)
        except Exception as e:  # noqa: BLE001
            problems.append(f"circuit: {e}")
        try:
            self.write_grad_stats(os.path.join(out_dir, "gradient_stats.csv"),
                                  thr=thr, hp=hp)
        except OSError as e:
            problems.append(f"gradient_stats.csv: {e}")
        self.queue.put(("exported", (out_dir, problems)))

    def _save_final_gradients(self, out_dir, hp):
        """The Gradients view once for the whole circuit and once per layer,
        on figures of their own (not the on-screen one), with no gate pinned."""
        if hp.n < 2:
            return
        folder = os.path.join(out_dir, "gradients")
        os.makedirs(folder, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for layer in [None] + list(range(hp.n_layers)):
                fig = Figure(figsize=(16, 9), dpi=100, facecolor=BG)
                axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
                self._draw_grads(fig, axes, hp, layer=layer, sel=None,
                                 hints=False)
                name = "all" if layer is None else "layer_%d" % (layer + 1)
                for ext in ("png", "pdf"):
                    fig.savefig(os.path.join(folder, "%s.%s" % (name, ext)),
                                dpi=150, facecolor=BG)

    # ------------------------------------------------------------- queue ---
    def _drain(self):
        # a time budget per call and ONE redraw per batch: if the worker
        # produces faster than matplotlib draws, the backlog must never keep
        # the Tk loop busy (the window would stop answering Back and the X)
        deadline = time.time() + 0.08
        ticked = False
        try:
            while time.time() < deadline:
                kind, payload = self.queue.get_nowait()

                if kind == "exported":
                    out_dir, problems = payload
                    msg = ("final plots saved to %s" % out_dir if not problems
                           else "final plots in %s, but failed: %s"
                           % (out_dir, "; ".join(problems)))
                    if self.worker is None or not self.worker.is_alive():
                        self.status.set(msg)
                    continue

                if kind == "game":
                    game = payload["game"]
                    if payload.get("transient_boundary") and game >= 0:
                        self._lr_transition_game = game
                    self.params.append(game, payload["weights"],
                                       payload["grad_rms"])
                    g = payload["grad_rms"]
                    if g is not None:
                        zero = np.abs(g) < GRAD_ZERO
                        structural = dead_mask(g.shape[0])
                        valid = ~(zero | structural)
                        rms = (float(np.sqrt(np.mean(np.asarray(g)[valid] ** 2)))
                               if np.any(valid) else float("nan"))
                        self.chip_vals["grad"].set("%.1e" % rms)
                        self.chip_vals["params"].set(str(g.size))
                        self.chip_vals["zero_grad"].set(str(int(np.sum(zero))))
                    self._last_game_at = time.time()
                    if game >= 0:
                        self.chip_vals["game"].set("%d / %d" %
                                                   (game, self.total_games))
                    if game >= 0 and payload.get("trajectory"):
                        self.last_episode = {
                            key: jsonable(payload.get(key)) for key in
                            ("game", "discounted_reward", "trajectory",
                             "actions", "rewards", "steps")}
                        if (self.episode_view is not None and
                                self.episode_view.winfo_exists()):
                            self.episode_view.set_episode(self.last_episode)
                            self.episode_view.replay()
                    if payload.get("loss") is not None:
                        self.loss_ep["game"].append(game)
                        self.loss_ep["loss"].append(payload["loss"])
                    if payload.get("discounted_reward") is not None and game >= 0:
                        self.reward_ep["game"].append(game)
                        self.reward_ep["reward"].append(
                            payload["discounted_reward"])
                    self._update_metric_chips()
                    self._loss_new = True
                    self.circuit.mark_dirty()
                    continue

                if kind == "setup":
                    self.correct = payload["correct"]
                    self.policy_ref = payload["policy_ref"]
                    self.policy = None
                    self.U_ref = payload["U_ref"]
                    self._episode_env = payload["episode_env"]
                    self.circuit.set_reference(self.policy_ref, self.U_ref)
                    self.status.set("training in progress...")

                elif kind == "tick":
                    h = self.history
                    h["game"].append(payload["game"])
                    h["mse"].append(payload["mse"])
                    h["loss"].append(payload["loss"])
                    h["lr"].append(payload["lr"])
                    h["eps"].append(payload["epsilon"])
                    if payload.get("U") is not None:
                        self.U = payload["U"]
                    if payload.get("U_ref") is not None:
                        self.U_ref = payload["U_ref"]
                    if payload.get("policy") is not None:
                        self.policy = payload["policy"]
                    if payload.get("transition_game") is not None:
                        self._lr_transition_game = payload["transition_game"]
                    elif payload.get("transient_boundary") and game >= 0:
                        self._lr_transition_game = game
                    self.chip_vals["game"].set("%d / %d" % (payload["game"],
                                                            self.total_games))
                    self._update_metric_chips()
                    self.chip_vals["lr"].set("%.5f" % payload["lr"])
                    self.chip_vals["eps"].set("%.3f" % payload["epsilon"])
                    self._progress = payload["game"] / float(self.total_games)
                    self._draw_progress()
                    self.status.set(
                        "episode %d   MSE %.4f   loss %.4f   lr %.5f   ε %.3f"
                        % (payload["game"], payload["mse"], payload["loss"],
                           payload["lr"], payload["epsilon"]))
                    ticked = True

                elif kind in ("done", "stopped", "error"):
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    if kind == "error":
                        self.status.set("error: " + payload)
                        messagebox.showerror("Error during training", payload)
                    elif kind == "stopped":
                        self.status.set("stopped by the user" + self._save_views())
                    else:
                        self._progress = 1.0
                        self._draw_progress()
                        self.policy = payload["policy"]
                        self.policy_ref = payload["policy_ref"]
                        ok = int(sum(payload["policy"][s] == payload["policy_ref"][s]
                                     for s in (self.correct or [])))
                        self.status.set(
                            "done: %d steps, policy correct on %d/%d states"
                            % (payload["steps"], ok, len(self.correct or []))
                            + self._save_views())
                    self._redraw()
        except queue.Empty:
            pass
        # Redraws are rate-limited, not event-driven: a matplotlib redraw holds
        # the GIL for a good fraction of a second, and redrawing on every
        # episode slowed the training thread down four times.
        now = time.time()
        if ticked:
            self._tick_pending = True
        if now - self._last_redraw >= self._redraw_gap and (
                self._tick_pending
                or (self.view == "grads" and self.params.n
                    and self._grads_drawn < self._last_game_at)
                or (self.view == "curve" and self._loss_new)):
            self._tick_pending = False
            self._redraw()
            self.update_idletasks()   # draw_idle renders here: time it too
            # adaptive: wait at least 3x what the redraw cost, so the GUI never
            # takes more than ~25% of the time away from the training thread
            self._last_redraw = time.time()
            self._redraw_gap = max(self.REDRAW_EVERY,
                                   3.0 * (self._last_redraw - now))
        if self.view == "circuit" and now - self._last_circuit >= 0.25:
            self._last_circuit = now
            self.circuit.refresh()
        self.after(150, self._drain)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
