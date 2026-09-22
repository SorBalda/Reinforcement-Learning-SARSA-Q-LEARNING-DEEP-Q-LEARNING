"""Live view of the variational circuit, its weights and its gradients.

The circuit is drawn on a plain Tk canvas (crisp text at any size, cheap to
redraw several times a second), exactly as :func:`quantum_rl.circuit.make_qnode`
builds it: for every layer the input is re-uploaded (``RX(pi x_i) RZ(pi x_i)``),
then the CNOT chain 0->1->2->3 and one ``Rot(phi, theta, omega)`` per wire; the
four ``<Z_i>`` times ``q_scale`` are the Q-values of the four actions.

What the view offers:

* **Angles / Gradients** switch.  In *Angles* every Rot parameter is a dial:
  the needle is the current value, the shaded wedge is how far it has moved
  since the initial weights.  In *Gradients* the same slot becomes a ring
  gauge of the per-episode gradient RMS on a log scale, red when it is
  vanishing, green when it is healthy.
* **Timeline**: one snapshot per episode.  The slider scrubs through the
  history, ``Play`` replays it, ``LIVE`` sticks to the newest one.
* **Input state**: pick a grid cell on the right; the encoding gates show its
  bits and the output panel shows ``Q(s, a)`` for the displayed snapshot,
  computed with :mod:`quantum_rl.statevector` (so it is exact for ANY past
  snapshot, not only for the evaluation episodes).
* **Layers slider**, shown only when the circuit does not fit at a readable
  size: gates never shrink below legibility, the view scrolls instead.
* **Mini-map** of all the parameters (or gradients), which also navigates, and
  a **detail strip** with the history of the selected gate (click a gate to pin
  it; with nothing pinned, it shows the whole-circuit trends).
"""

from __future__ import annotations

import math
import tkinter as tk
import warnings

import numpy as np

from .gui_widgets import (AMBER, BG, BORDER, BORDER_HI, CARD, CARD_HI, CYAN,
                          FAINT, FONTS, GREEN, MUTED, RED, TEXT, VIOLET,
                          NeonButton, QSlider, _mix, _round_rect)
from .statevector import q_values, state_bits

#: one colour per Rot parameter, used everywhere (dials, mini-map, sparklines)
PARAM_COLORS = (CYAN, AMBER, VIOLET)
PARAM_NAMES = ("φ", "θ", "ω")
ACTION_ARROWS = ("↑", "→", "↓", "←")
ACTION_NAMES = ("up", "right", "down", "left")

#: log10 range of the gradient gauges: below GRAD_LO it is flat red
GRAD_LO, GRAD_HI = -6.0, -1.0
#: below this a gradient is an exact zero, not a small one: in that episode no
#: updated action had the gate in its light cone (a Rot on wire i only reaches
#: <Z_j> for j >= i, and fewer the later the layer).  Shown grey, "no signal".
GRAD_ZERO = 1e-12

GRID_NX, GRID_NY = 4, 3
GOAL, DEATH, WALL = 11, 7, 5


def dead_mask(n_layers: int) -> np.ndarray:
    """Parameters that cannot change any ``<Z_i>``, whatever the weights.

    Two structural zeros of this architecture, verified by finite differences:

    * phi of layer 1: the first ``RZ(phi)`` of ``Rot`` acts on a computational
      basis state (``RX(0 or pi)`` on ``|0>``, then CNOTs), i.e. it only adds a
      global phase;
    * omega of the last layer: the final ``RZ(omega)`` commutes with the
      ``PauliZ`` measurement.

    Their gradient is identically zero.  That is NOT a vanishing gradient, so
    they are drawn grey and left out of every gradient norm.
    """
    m = np.zeros((n_layers, 4, 3), dtype=bool)
    if n_layers:
        m[0, :, 0] = True
        m[-1, :, 2] = True
    return m


DEAD_WHY = {0: "RZ(φ) on a basis state is a global phase",
            2: "RZ(ω) right before a Z measurement commutes with it"}


def grad_color(g: float, bg: str = CARD) -> str:
    """Red (vanishing) -> amber -> green (healthy), on a log scale."""
    if g is None or not np.isfinite(g) or g < GRAD_ZERO:
        return _mix(bg, MUTED, 0.35)    # no data / no signal: neutral
    t = (math.log10(g) - GRAD_LO) / (GRAD_HI - GRAD_LO)
    t = min(max(t, 0.0), 1.0)
    if t < 0.5:
        return _mix(RED, AMBER, t * 2)
    return _mix(AMBER, GREEN, (t - 0.5) * 2)


def grad_frac(g: float) -> float:
    if g is None or not np.isfinite(g) or g < GRAD_ZERO:
        return 0.0
    return min(max((math.log10(g) - GRAD_LO) / (GRAD_HI - GRAD_LO), 0.0), 1.0)


def fmt_grad(g: float) -> str:
    """Compact scientific notation that fits under a gauge: ``3e-4``."""
    if g is None or not np.isfinite(g):
        return "—"
    if abs(g) < GRAD_ZERO:
        return "0"
    e = int(math.floor(math.log10(abs(g))))
    m = g / 10 ** e
    if round(m) >= 10:
        m, e = m / 10, e + 1
    return "%de%d" % (round(m), e)


def angle_color(v: float, bg: str = CARD) -> str:
    """Diverging colour for an angle wrapped to (-pi, pi]: cyan +, violet -."""
    w = (v + math.pi) % (2 * math.pi) - math.pi
    t = min(abs(w) / math.pi, 1.0) ** 0.7
    return _mix(bg, CYAN if w >= 0 else VIOLET, 0.12 + 0.78 * t)


class ParamHistory:
    """Per-episode snapshots of the weights and of their gradient RMS.

    Growing numpy buffers, so reading the whole history (for the sparklines and
    the gradient plots) is a slice, not a ``np.stack`` of thousands of arrays.
    Row 0 is usually the initial weights (``game == -1``, gradient NaN).
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self._games = np.zeros(0, dtype=int)
        self._W = None
        self._G = None

    def append(self, game: int, W, G):
        W = np.asarray(W, dtype=float)
        if self._W is None or self._W.shape[1:] != W.shape:
            self._games = np.zeros(256, dtype=int)
            self._W = np.zeros((256,) + W.shape)
            self._G = np.zeros((256,) + W.shape)
            self.n = 0
        if self.n == len(self._games):
            cap = 2 * self.n
            self._games = np.resize(self._games, cap)
            self._W = np.concatenate([self._W, np.zeros_like(self._W)])
            self._G = np.concatenate([self._G, np.zeros_like(self._G)])
        self._games[self.n] = game
        self._W[self.n] = W
        self._G[self.n] = np.nan if G is None else np.asarray(G, dtype=float)
        self.n += 1

    def snapshot(self) -> "ParamHistory":
        """An independent copy, safe to read from another thread."""
        h = ParamHistory()
        if self._W is not None:
            h.n = self.n
            h._games = self.games.copy()
            h._W = self.W.copy()
            h._G = self.G.copy()
        return h

    @property
    def games(self) -> np.ndarray:
        return self._games[:self.n]

    @property
    def W(self) -> np.ndarray:
        return self._W[:self.n] if self._W is not None else np.zeros((0, 0, 4, 3))

    @property
    def G(self) -> np.ndarray:
        return self._G[:self.n] if self._G is not None else np.zeros((0, 0, 4, 3))

    @property
    def n_layers(self) -> int:
        return 0 if self._W is None else self._W.shape[1]


class CircuitView(tk.Frame):
    """The "Circuit" screen of the training window."""

    def __init__(self, master, history: ParamHistory, *, on_select=None):
        super().__init__(master, bg=BG)
        self.hist = history
        self.on_select = on_select
        self.q_scale = 3.0
        self.policy_ref = None
        self.U_ref = None
        self.mode = "angles"          # "angles" | "grads"
        self.idx = -1                 # snapshot shown; -1 = newest
        self.live = True
        self.state = 0                # input state of the encoding / Q panel
        self.layer0 = 0               # first visible layer
        self.n_fit = 1
        self.selected = None          # pinned gate (layer, wire)
        self.hover = None             # gate under the mouse
        self._playing = False
        self._dirty = True

        self._build()

    # ------------------------------------------------------------ building --
    def _build(self):
        # -- toolbar: display mode + timeline --------------------------------
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", pady=(2, 6))

        self._mode_btns = {}
        seg = tk.Frame(bar, bg=_mix(BG, "#ffffff", .05), highlightthickness=1,
                       highlightbackground=BORDER_HI)
        seg.pack(side="left")
        for key, label in (("angles", "  Angles  "), ("grads", "  Gradients  ")):
            b = tk.Label(seg, text=label, font=FONTS["small"], padx=8, pady=5,
                         cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda _e, k=key: self.set_mode(k))
            self._mode_btns[key] = b

        self.btn_play = NeonButton(bar, "Replay", self.toggle_play,
                                   accent=CYAN, kind="ghost", height=34, bg=BG,
                                   font=FONTS["btn_s"], pad=14)
        self.btn_play.pack(side="left", padx=(18, 8))
        self.slider = QSlider(bar, self._on_scrub, accent=CYAN, bg=BG,
                              height=30, width=300)
        self.slider.pack(side="left", fill="x", expand=True, padx=(4, 10))
        self.ep_label = tk.Label(bar, text="", bg=BG, fg=TEXT, width=17,
                                 anchor="w", font=FONTS["mono"])
        self.ep_label.pack(side="left")
        self.live_lbl = tk.Label(bar, text="", bg=BG, font=FONTS["btn_s"],
                                 padx=10, pady=4, cursor="hand2")
        self.live_lbl.pack(side="left", padx=(6, 0))
        self.live_lbl.bind("<Button-1>", lambda _e: self.go_live())

        # -- middle: circuit canvas + side panel -----------------------------
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="both", expand=True)

        side_shell = tk.Frame(mid, bg=BORDER)
        side_shell.pack(side="right", fill="y", padx=(10, 0))
        side = tk.Frame(side_shell, bg=CARD)
        side.pack(fill="both", expand=True, padx=1, pady=1)
        self.side = tk.Canvas(side, width=290, bg=CARD, highlightthickness=0,
                              bd=0)
        self.side.pack(fill="both", expand=True)
        self.side.bind("<Configure>", lambda _e: self._draw_side())
        self.side.bind("<Button-1>", self._side_click)

        left = tk.Frame(mid, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        shell = tk.Frame(left, bg=BORDER)
        shell.pack(fill="both", expand=True)
        self.cv = tk.Canvas(shell, bg=_mix(BG, "#ffffff", 0.02),
                            highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True, padx=1, pady=1)
        self.cv.bind("<Configure>", lambda _e: self._draw_circuit())
        self.cv.bind("<Motion>", self._cv_motion)
        self.cv.bind("<Leave>", self._cv_leave)
        self.cv.bind("<Button-1>", self._cv_click)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.cv.bind(seq, self._cv_wheel)

        # layer scroller: packed only when the circuit does not fit
        self.layer_row = tk.Frame(left, bg=BG)
        self.layer_lbl = tk.Label(self.layer_row, text="", bg=BG, fg=MUTED,
                                  font=FONTS["small"], width=24, anchor="w")
        self.layer_lbl.pack(side="left")
        self.layer_slider = QSlider(self.layer_row, self._on_layer_scroll,
                                    accent=VIOLET, bg=BG, height=26)
        self.layer_slider.pack(side="left", fill="x", expand=True)
        self._layer_row_shown = False

        # the info line: exact numbers of whatever is under the mouse
        self.info = tk.Label(left, text="", bg=BG, fg=MUTED, anchor="w",
                             justify="left", font=FONTS["mono"])
        self.info.pack(fill="x", pady=(5, 0))

        # -- bottom: mini-map + detail strip ---------------------------------
        bottom = tk.Frame(self, bg=BG, height=150)
        bottom.pack(fill="x", pady=(8, 0))
        bottom.pack_propagate(False)
        mshell = tk.Frame(bottom, bg=BORDER)
        mshell.pack(side="left", fill="y")
        self.mini = tk.Canvas(mshell, width=360, bg=CARD, highlightthickness=0,
                              bd=0, cursor="hand2")
        self.mini.pack(fill="both", expand=True, padx=1, pady=1)
        self.mini.bind("<Configure>", lambda _e: self._draw_minimap())
        self.mini.bind("<Button-1>", self._mini_click)
        dshell = tk.Frame(bottom, bg=BORDER)
        dshell.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.detail = tk.Canvas(dshell, bg=CARD, highlightthickness=0, bd=0)
        self.detail.pack(fill="both", expand=True, padx=1, pady=1)
        self.detail.bind("<Configure>", lambda _e: self._draw_detail())

        self.set_mode("angles", redraw=False)
        self._update_timeline()

    # ----------------------------------------------------------------- API --
    def reset(self, q_scale=3.0, policy_ref=None, U_ref=None):
        self.q_scale = float(q_scale)
        self.policy_ref = policy_ref
        self.U_ref = U_ref
        self.idx, self.live = -1, True
        self.layer0 = 0
        self.selected = self.hover = None
        self._stop_play()
        self._dirty = True
        self.refresh(force=True)

    def set_reference(self, policy_ref=None, U_ref=None):
        self.policy_ref, self.U_ref = policy_ref, U_ref
        self._dirty = True

    def mark_dirty(self):
        self._dirty = True

    def refresh(self, force=False):
        """Redraw if something changed and the view is on screen."""
        if not (self._dirty or force):
            return
        if not self.winfo_ismapped() and not force:
            return
        self._dirty = False
        self._update_timeline()
        self._draw_all()

    def set_mode(self, mode, redraw=True):
        self.mode = mode
        for key, b in self._mode_btns.items():
            on = key == mode
            b.configure(fg=CYAN if on else MUTED,
                        bg=_mix(BG, CYAN, .18) if on else _mix(BG, "#ffffff", .05))
        if redraw:
            self._draw_all()

    def go_live(self):
        self._stop_play()
        self.live, self.idx = True, -1
        self._update_timeline()
        self._draw_all()

    def toggle_play(self):
        if self._playing:
            self._stop_play()
            return
        if self.hist.n < 2:
            return
        if self.live or self._cur() >= self.hist.n - 1:
            self.idx = 0              # replay from the start
        self.live = False
        self._playing = True
        self.btn_play.config(text="Pause")
        self._play_tick()

    # ------------------------------------------------------------ timeline --
    def _cur(self) -> int:
        n = self.hist.n
        if n == 0:
            return -1
        return n - 1 if (self.live or self.idx < 0) else min(self.idx, n - 1)

    def _stop_play(self):
        self._playing = False
        self.btn_play.config(text="Replay")

    def _play_tick(self):
        if not self._playing:
            return
        n = self.hist.n
        step = max(1, n // 240)      # a full replay takes ~10 s at any length
        self.idx = min(self._cur() + step, n - 1)
        if self.idx >= n - 1:
            self._stop_play()
            self.live = True
        self._update_timeline()
        self._draw_all()
        if self._playing:
            self.after(40, self._play_tick)

    def _on_scrub(self, v):
        self._stop_play()
        self.live = v >= self.hist.n - 1 and not self.slider.dragging
        self.idx = v
        self._update_timeline()
        self._draw_all()

    def _update_timeline(self):
        n = self.hist.n
        self.slider.set_range(0, max(0, n - 1))
        cur = self._cur()
        if not self.slider.dragging:
            self.slider.set(max(cur, 0))
        if cur < 0:
            self.ep_label.config(text="no data yet")
        else:
            g = int(self.hist.games[cur])
            self.ep_label.config(text="initial weights" if g < 0
                                 else "episode %d" % g)
        if self.live:
            self.live_lbl.config(text="● LIVE", fg=GREEN,
                                 bg=_mix(BG, GREEN, 0.14))
        else:
            self.live_lbl.config(text="○ go live", fg=MUTED,
                                 bg=_mix(BG, "#ffffff", 0.05))

    # ---------------------------------------------------------- selections --
    def _select(self, gate):
        self.selected = None if gate == self.selected else gate
        if self.selected is not None:
            self._ensure_visible(self.selected[0])
        if self.on_select:
            self.on_select(self.selected)
        self._draw_all()

    def _set_hover(self, gate):
        if gate != self.hover:
            self.hover = gate
            self._draw_circuit()
            self._draw_detail()
            self._update_info()

    def _ensure_visible(self, layer):
        if layer < self.layer0:
            self.layer0 = layer
        elif layer >= self.layer0 + self.n_fit:
            self.layer0 = layer - self.n_fit + 1

    def _on_layer_scroll(self, v):
        self.layer0 = v
        self._draw_circuit()
        self._draw_minimap()

    # -------------------------------------------------------------- layout --
    def _geometry(self):
        """Every size of the circuit drawing, derived from the canvas size.

        Gates have a minimum readable size; when the layers do not fit at
        that size only ``n_fit`` of them are drawn and the layer slider
        scrolls through the rest.
        """
        W = max(self.cv.winfo_width(), 200)
        H = max(self.cv.winfo_height(), 200)
        L = max(self.hist.n_layers, 1)
        g = {"W": W, "H": H, "left": 92, "right": 150, "top": 44, "bottom": 14}
        lane = (H - g["top"] - g["bottom"]) / 4.0
        lane = max(62.0, min(lane, 132.0))
        g["lane"] = lane
        bh = lane - 12
        g["bh"] = bh
        r = max(8.0, min((bh - 38) / 2.0, 19.0))
        g["r"] = r
        m = self.tk.call("font", "measure", FONTS["mono"], "-0.00")
        col = max(2 * r + 14, m + 8)
        g["col"] = col
        g["rot_w"] = 3 * col + 12
        g["enc_w"] = max(40.0, min(lane * 0.5, 56.0))
        g["cx_w"] = max(24.0, min(lane * 0.3, 34.0))
        g["gap"] = 12
        g["pad"] = 26
        g["layer_w"] = (g["enc_w"] + g["gap"] + 3 * g["cx_w"] + g["gap"]
                        + g["rot_w"] + g["pad"])
        avail = W - g["left"] - g["right"]
        self.n_fit = max(1, min(L, int(avail // g["layer_w"])))
        self.layer0 = max(0, min(self.layer0, L - self.n_fit))
        used = self.n_fit * g["layer_w"]
        g["x0"] = g["left"] + max(0.0, (avail - used) / 2.0)
        g["ys"] = [g["top"] + lane * (i + 0.5) for i in range(4)]
        return g

    def _layer_row(self, L):
        need = self.n_fit < L
        if need != self._layer_row_shown:
            self._layer_row_shown = need
            if need:
                self.layer_row.pack(fill="x", pady=(6, 0), before=self.info)
            else:
                self.layer_row.pack_forget()
        if need:
            self.layer_slider.set_range(0, L - self.n_fit)
            self.layer_slider.set(self.layer0)
            self.layer_lbl.config(
                text="layers %d–%d of %d" % (self.layer0 + 1,
                                              self.layer0 + self.n_fit, L))

    # ------------------------------------------------------------- drawing --
    def _draw_all(self):
        self._draw_circuit()
        self._draw_side()
        self._draw_minimap()
        self._draw_detail()
        self._update_info()

    def _snapshot(self):
        cur = self._cur()
        if cur < 0:
            return None, None, None, None
        W = self.hist.W[cur]
        G = self.hist.G[cur]
        W0 = self.hist.W[0]
        prev = self.hist.W[cur - 1] if cur > 0 else W
        return W, G, W0, prev

    def _draw_circuit(self):
        cv = self.cv
        cv.delete("all")
        L = self.hist.n_layers
        if L == 0:
            cv.create_text(cv.winfo_width() / 2, cv.winfo_height() / 2,
                           text="the circuit appears when training starts",
                           fill=FAINT, font=FONTS["body"])
            self._layer_row(1)
            return
        g = self._geometry()
        self._layer_row(L)
        W, G, W0, prev = self._snapshot()
        bits = state_bits(self.state)
        ys = g["ys"]
        lane = g["lane"]
        wire_col = _mix(BG, BORDER_HI, 0.9)
        x_end = g["W"] - g["right"] + 18

        # activity of each gate since the previous snapshot, normalised on the
        # most active one: it lights up the gates that are learning right now
        self._dead = dead_mask(L)
        act = np.linalg.norm(W - prev, axis=2)
        act = act / act.max() if act.max() > 0 else act

        # wires first, everything else sits on top of them
        for i, y in enumerate(ys):
            cv.create_line(g["left"] - 22, y, x_end, y, fill=wire_col, width=2)
            cv.create_text(16, y - 9, anchor="w", text="q%d" % i, fill=TEXT,
                           font=FONTS["mono_b"])
            cv.create_text(16, y + 12, anchor="w", text="x%d = %d" % (i, bits[i]),
                           fill=CYAN if bits[i] else FAINT, font=FONTS["small"])
        if self.layer0 > 0:
            cv.create_text(g["left"] - 8, g["top"] - 24, anchor="w",
                           text="◀ %d more" % self.layer0, fill=VIOLET,
                           font=FONTS["small"])
        rest = L - self.layer0 - self.n_fit
        if rest > 0:
            cv.create_text(x_end - 6, g["top"] - 24, anchor="e",
                           text="%d more ▶" % rest, fill=VIOLET,
                           font=FONTS["small"])

        top_y = g["top"] - 6
        bot_y = ys[-1] + lane / 2 + 2
        for k in range(self.n_fit):
            l = self.layer0 + k
            xl = g["x0"] + k * g["layer_w"]
            x_enc = xl
            x_cx = x_enc + g["enc_w"] + g["gap"]
            x_rot = x_cx + 3 * g["cx_w"] + g["gap"]
            x_r = x_rot + g["rot_w"]

            _round_rect(cv, xl - 8, top_y, x_r + 8, bot_y, 12,
                        fill=_mix(BG, "#ffffff", 0.035 if l % 2 else 0.055),
                        outline=_mix(BG, BORDER_HI, 0.5))
            cv.create_text(xl, top_y - 10, anchor="sw", text="LAYER %d" % (l + 1),
                           fill=MUTED, font=FONTS["chip"])
            cv.create_text(x_enc + g["enc_w"] / 2, top_y + 11, text="encode",
                           fill=FAINT, font=FONTS["chip"])
            cv.create_text(x_cx + 1.5 * g["cx_w"], top_y + 11, text="entangle",
                           fill=FAINT, font=FONTS["chip"])
            cv.create_text(x_rot + g["rot_w"] / 2, top_y + 11,
                           text="Rot(φ, θ, ω)", fill=FAINT, font=FONTS["chip"])

            # -- data re-uploading: RX(pi x) RZ(pi x), identity when x = 0
            eh = min(lane * 0.56, 50)
            for i, y in enumerate(ys):
                on = bool(bits[i])
                _round_rect(cv, x_enc, y - eh / 2, x_enc + g["enc_w"], y + eh / 2,
                            7, fill=_mix(CARD, CYAN, 0.22) if on else CARD,
                            outline=CYAN if on else _mix(CARD, FAINT, 0.7),
                            dash=() if on else (3, 3), width=1,
                            tags=("enc", "enc:%d" % i))
                cv.create_text(x_enc + g["enc_w"] / 2, y - eh / 2 + 10,
                               text="RxRz", fill=MUTED if on else FAINT,
                               font=FONTS["chip"], tags=("enc", "enc:%d" % i))
                cv.create_text(x_enc + g["enc_w"] / 2, y + 6,
                               text="π" if on else "0",
                               fill=TEXT if on else FAINT, font=FONTS["mono_b"],
                               tags=("enc", "enc:%d" % i))

            # -- CNOT staircase 0->1, 1->2, 2->3
            for c in range(3):
                x = x_cx + (c + 0.5) * g["cx_w"]
                y1, y2 = ys[c], ys[c + 1]
                cv.create_line(x, y1, x, y2 + 11, fill=VIOLET, width=2,
                               tags=("cnot", "cnot:%d" % c))
                cv.create_oval(x - 5, y1 - 5, x + 5, y1 + 5, fill=VIOLET,
                               outline="", tags=("cnot", "cnot:%d" % c))
                cv.create_oval(x - 11, y2 - 11, x + 11, y2 + 11,
                               fill=_mix(BG, "#ffffff", 0.05), outline=VIOLET,
                               width=2, tags=("cnot", "cnot:%d" % c))
                cv.create_line(x - 11, y2, x + 11, y2, fill=VIOLET, width=2,
                               tags=("cnot", "cnot:%d" % c))
                cv.create_line(x, y2 - 11, x, y2 + 11, fill=VIOLET, width=2,
                               tags=("cnot", "cnot:%d" % c))

            # -- Rot gates
            for i, y in enumerate(ys):
                self._draw_rot(g, l, i, x_rot, y, W[l, i], W0[l, i], G[l, i],
                               act[l, i])

        self._draw_measure(g, W, bits)

    def _draw_rot(self, g, l, i, x, y, w, w0, grad, act):
        cv = self.cv
        tag = "gate:%d:%d" % (l, i)
        bh, r, col = g["bh"], g["r"], g["col"]
        sel = self.selected == (l, i)
        hov = self.hover == (l, i)
        if sel:
            outline, width = AMBER, 2
        elif self.mode == "angles":
            outline, width = _mix(BORDER_HI, GREEN, 0.85 * float(act)), 1
        else:
            live = grad[~self._dead[l, i]]
            gm = np.nanmean(live) if np.isfinite(live).any() else np.nan
            outline, width = _mix(BORDER_HI, grad_color(gm), 0.6), 1
        _round_rect(cv, x, y - bh / 2, x + g["rot_w"], y + bh / 2, 9,
                    fill=CARD_HI if hov else CARD, outline=outline, width=width,
                    tags=("gate", tag))
        for j in range(3):
            xc = x + 6 + col * (j + 0.5)
            yl = y - bh / 2 + 10
            yc = yl + 8 + r
            dead = self._dead[l, i, j]
            cv.create_text(xc, yl, text=PARAM_NAMES[j],
                           fill=FAINT if dead else PARAM_COLORS[j],
                           font=FONTS["chip"], tags=("gate", tag))
            if dead:
                # no effect on the output: say so instead of a scary red gauge
                cv.create_oval(xc - r, yc - r, xc + r, yc + r,
                               fill=_mix(CARD, BG, 0.5), outline=BORDER,
                               dash=(2, 3), tags=("gate", tag))
                cv.create_line(xc - r * 0.5, yc, xc + r * 0.5, yc, fill=FAINT,
                               width=2, tags=("gate", tag))
                cv.create_text(xc, y + bh / 2 - 9, text="n/a", fill=FAINT,
                               font=FONTS["mono"], tags=("gate", tag))
                continue
            if self.mode == "angles":
                self._dial(xc, yc, r, w[j], w0[j], PARAM_COLORS[j], tag)
                txt, fg = "%+.2f" % w[j], TEXT
            else:
                self._gauge(xc, yc, r, grad[j], tag)
                txt, fg = fmt_grad(grad[j]), grad_color(grad[j])
            cv.create_text(xc, y + bh / 2 - 9, text=txt, fill=fg,
                           font=FONTS["mono"], tags=("gate", tag))

    def _dial(self, xc, yc, r, v, v0, color, tag):
        """Angle dial: 0 at twelve o'clock, positive clockwise.

        The wedge spans from the initial value to the current one: how far this
        parameter has travelled during training.
        """
        cv = self.cv
        cv.create_oval(xc - r, yc - r, xc + r, yc + r, fill=_mix(CARD, BG, 0.5),
                       outline=_mix(CARD, color, 0.45), tags=("gate", tag))
        d = v - v0
        if abs(d) > 1e-3:
            ext = -math.degrees(max(min(d, 2 * math.pi - 1e-3), -2 * math.pi + 1e-3))
            cv.create_arc(xc - r + 2, yc - r + 2, xc + r - 2, yc + r - 2,
                          start=90 - math.degrees(v0), extent=ext,
                          style="pieslice", fill=_mix(CARD, color, 0.32),
                          outline="", tags=("gate", tag))
        cv.create_line(xc, yc - r, xc, yc - r + 3, fill=MUTED,
                       tags=("gate", tag))
        cv.create_line(xc, yc, xc + r * math.sin(v0) * 0.9,
                       yc - r * math.cos(v0) * 0.9, fill=_mix(CARD, MUTED, 0.6),
                       width=1, dash=(2, 2), tags=("gate", tag))
        cv.create_line(xc, yc, xc + r * math.sin(v), yc - r * math.cos(v),
                       fill=color, width=2.5, capstyle="round",
                       tags=("gate", tag))
        cv.create_oval(xc - 2.5, yc - 2.5, xc + 2.5, yc + 2.5, fill=color,
                       outline="", tags=("gate", tag))

    def _gauge(self, xc, yc, r, gval, tag):
        """Ring gauge of a gradient RMS on a log scale (270 degree sweep)."""
        cv = self.cv
        box = (xc - r + 2, yc - r + 2, xc + r - 2, yc + r - 2)
        cv.create_arc(*box, start=225, extent=-270, style="arc",
                      outline=_mix(CARD, BORDER_HI, 0.9), width=5,
                      tags=("gate", tag))
        f = grad_frac(gval)
        if f > 0:
            cv.create_arc(*box, start=225, extent=-270 * f, style="arc",
                          outline=grad_color(gval), width=5, tags=("gate", tag))
        elif np.isfinite(gval) and gval >= GRAD_ZERO:
            cv.create_text(xc, yc, text="!", fill=RED, font=FONTS["mono_b"],
                           tags=("gate", tag))

    def _draw_measure(self, g, W, bits):
        """Meters at the end of the wires, with the Q-value of each action."""
        cv = self.cv
        q = q_values(W, self.state, self.q_scale)
        best = int(np.argmax(q))
        ref = None if self.policy_ref is None else int(self.policy_ref[self.state])
        xm = g["W"] - g["right"] + 22
        for a, y in enumerate(g["ys"]):
            _round_rect(cv, xm, y - 15, xm + 36, y + 15, 6, fill=CARD,
                        outline=BORDER_HI, tags=("meas", "meas:%d" % a))
            cv.create_arc(xm + 7, y - 7, xm + 29, y + 15, start=20, extent=140,
                          style="arc", outline=MUTED, width=1.5,
                          tags=("meas", "meas:%d" % a))
            t = math.radians(90 - 60 * q[a] / self.q_scale)
            cv.create_line(xm + 18, y + 4, xm + 18 + 13 * math.cos(t),
                           y + 4 - 13 * math.sin(t), fill=TEXT, width=1.5,
                           tags=("meas", "meas:%d" % a))
            is_best = a == best
            col = TEXT
            if is_best:
                col = GREEN if (ref is None or ref == a) else RED
            cv.create_text(xm + 46, y - 8, anchor="w",
                           text="%s %s" % (ACTION_ARROWS[a], ACTION_NAMES[a]),
                           fill=col if is_best else MUTED, font=FONTS["small"],
                           tags=("meas", "meas:%d" % a))
            cv.create_text(xm + 46, y + 9, anchor="w", text="%+.3f" % q[a],
                           fill=col, font=FONTS["mono_b" if is_best else "mono"],
                           tags=("meas", "meas:%d" % a))

    # -- side panel: input state picker + Q bars ---------------------------
    def _side_geom(self):
        w = max(self.side.winfo_width(), 200)
        cell = min(52, (w - 40) / GRID_NX)
        gx = (w - cell * GRID_NX) / 2
        return w, cell, gx, 40

    def _draw_side(self):
        cv = self.side
        cv.delete("all")
        w, cell, gx, gy = self._side_geom()
        cv.create_text(18, 18, anchor="w", text="INPUT STATE", fill=CYAN,
                       font=FONTS["card"])
        for s in range(GRID_NX * GRID_NY):
            x, y = s % GRID_NX, s // GRID_NX
            x0 = gx + x * cell
            y0 = gy + (GRID_NY - 1 - y) * cell
            special = {GOAL: ("+1", GREEN), DEATH: ("−1", RED), WALL: ("", FAINT)}
            fill = CARD_HI
            if s == WALL:
                fill = _mix(CARD, BG, 0.7)
            elif s in special:
                fill = _mix(CARD, special[s][1], 0.18)
            on = s == self.state
            cv.create_rectangle(x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2,
                                fill=_mix(fill, CYAN, 0.25) if on else fill,
                                outline=CYAN if on else BORDER,
                                width=2 if on else 1)
            cv.create_text(x0 + 7, y0 + 11, anchor="w", text=str(s),
                           fill=TEXT if on else FAINT, font=FONTS["chip"])
            if s in special and special[s][0]:
                cv.create_text(x0 + cell / 2, y0 + cell / 2 + 4,
                               text=special[s][0], fill=special[s][1],
                               font=FONTS["mono_b"])
            elif self.policy_ref is not None and s != WALL:
                cv.create_text(x0 + cell / 2, y0 + cell / 2 + 5,
                               text=ACTION_ARROWS[int(self.policy_ref[s])],
                               fill=_mix(CARD, MUTED, 0.55), font=FONTS["body"])
        y = gy + GRID_NY * cell + 20
        bits = "".join(str(b) for b in state_bits(self.state))
        cv.create_text(18, y, anchor="w", fill=TEXT, font=FONTS["mono"],
                       text="s = %d  →  |%s>" % (self.state, bits))
        cv.create_text(18, y + 20, anchor="w", fill=FAINT, font=FONTS["small"],
                       text="grey arrows: Bellman policy · click to pick")
        cv.create_text(18, y + 38, anchor="w", fill=FAINT, font=FONTS["small"],
                       text="green dot: Bellman action for this state")

        y += 72
        cv.create_text(18, y, anchor="w", text="OUTPUT   Q(s,a) = %g · <Z_a>"
                       % self.q_scale, fill=AMBER, font=FONTS["card"])
        W, _G, _W0, _p = self._snapshot()
        if W is None:
            return
        q = q_values(W, self.state, self.q_scale)
        best = int(np.argmax(q))
        ref = None if self.policy_ref is None else int(self.policy_ref[self.state])
        bx0, bx1 = 70, w - 70
        mid = (bx0 + bx1) / 2
        y += 24
        for a in range(4):
            yy = y + a * 30
            is_best = a == best
            col = (GREEN if (ref is None or ref == a) else RED) if is_best else MUTED
            cv.create_text(20, yy, anchor="w", text=ACTION_ARROWS[a],
                           fill=TEXT, font=FONTS["mono_b"])
            if ref == a:   # the Bellman-optimal action: a green dot
                cv.create_oval(40, yy - 4, 48, yy + 4, fill=GREEN, outline="")
            cv.create_rectangle(bx0, yy - 7, bx1, yy + 7,
                                fill=_mix(CARD, BG, 0.5), outline="")
            xv = mid + (bx1 - bx0) / 2 * max(-1, min(1, q[a] / self.q_scale))
            cv.create_rectangle(min(mid, xv), yy - 7, max(mid, xv), yy + 7,
                                fill=col if is_best else _mix(CARD, AMBER, 0.55),
                                outline="")
            cv.create_line(mid, yy - 10, mid, yy + 10, fill=FAINT)
            cv.create_text(w - 14, yy, anchor="e", text="%+.3f" % q[a],
                           fill=col if is_best else TEXT, font=FONTS["mono"])
        y += 4 * 30 + 4
        if self.U_ref is not None and self.state not in (WALL, GOAL, DEATH):
            cv.create_text(18, y, anchor="w", fill=MUTED, font=FONTS["small"],
                           text="max Q = %+.3f    Bellman U* = %+.3f"
                                % (q.max(), float(self.U_ref[self.state])))
            y += 20
            if ref is not None:
                ok = best == ref
                cv.create_text(18, y, anchor="w", font=FONTS["small"],
                               fill=GREEN if ok else RED,
                               text=("greedy action matches Bellman" if ok else
                                     "greedy %s, Bellman says %s"
                                     % (ACTION_ARROWS[best], ACTION_ARROWS[ref])))
        elif self.state in (WALL, GOAL, DEATH):
            cv.create_text(18, y, anchor="w", fill=FAINT, font=FONTS["small"],
                           text="terminal / obstacle: never trained on")

    def _side_click(self, e):
        w, cell, gx, gy = self._side_geom()
        x = int((e.x - gx) // cell)
        yi = int((e.y - gy) // cell)
        if 0 <= x < GRID_NX and 0 <= yi < GRID_NY:
            s = x + GRID_NX * (GRID_NY - 1 - yi)
            if s != self.state:
                self.state = s
                self._draw_circuit()
                self._draw_side()

    # -- mini-map --------------------------------------------------------------
    def _mini_geom(self):
        L = max(self.hist.n_layers, 1)
        w = max(self.mini.winfo_width(), 100)
        h = max(self.mini.winfo_height(), 60)
        top, left = 30, 38
        ch = (h - top - 22) / 4.0
        blk = min((w - left - 10) / L, ch * 3 + 8)
        cw = (blk - 6) / 3.0
        return L, w, h, top, left, ch, blk, cw

    def _draw_minimap(self):
        cv = self.mini
        cv.delete("all")
        L, w, h, top, left, ch, blk, cw = self._mini_geom()
        title = "WEIGHTS" if self.mode == "angles" else "GRADIENT RMS"
        cv.create_text(12, 14, anchor="w", text=title + "  ·  all layers",
                       fill=MUTED, font=FONTS["chip"])
        W, G, _W0, _p = self._snapshot()
        if W is None:
            return
        for i in range(4):
            cv.create_text(left - 8, top + (i + 0.5) * ch, anchor="e",
                           text="q%d" % i, fill=FAINT, font=FONTS["chip"])
        dead = dead_mask(self.hist.n_layers)
        for l in range(self.hist.n_layers):
            xb = left + l * blk
            for i in range(4):
                for j in range(3):
                    v = W[l, i, j] if self.mode == "angles" else G[l, i, j]
                    col = angle_color(v) if self.mode == "angles" else grad_color(v)
                    x0 = xb + j * cw
                    y0 = top + i * ch
                    if dead[l, i, j]:
                        col = _mix(CARD, BG, 0.5)
                    cv.create_rectangle(x0, y0, x0 + cw - 1, y0 + ch - 1,
                                        fill=col, outline="")
                    if dead[l, i, j]:
                        cv.create_line(x0 + 2, y0 + ch - 3, x0 + cw - 3, y0 + 2,
                                       fill=BORDER_HI)
            if blk > 14:
                cv.create_text(xb + 1.5 * cw, top + 4 * ch + 9,
                               text=str(l + 1), fill=FAINT, font=FONTS["chip"])
            if self.selected and self.selected[0] == l:
                i = self.selected[1]
                cv.create_rectangle(xb - 1, top + i * ch - 1, xb + 3 * cw,
                                    top + (i + 1) * ch, outline=AMBER, width=2)
        # the window of layers currently drawn in the circuit
        x0 = left + self.layer0 * blk - 2
        x1 = left + (self.layer0 + self.n_fit) * blk - 4
        cv.create_rectangle(x0, top - 3, x1, top + 4 * ch + 1, outline=CYAN,
                            width=1)

    def _mini_click(self, e):
        L, w, h, top, left, ch, blk, cw = self._mini_geom()
        l = int((e.x - left) // blk)
        i = int((e.y - top) // ch)
        if 0 <= l < self.hist.n_layers and 0 <= i < 4:
            self._select((l, i))

    # -- detail strip: history of the selected gate ---------------------------
    def _draw_detail(self):
        cv = self.detail
        cv.delete("all")
        n = self.hist.n
        w = max(cv.winfo_width(), 200)
        h = max(cv.winfo_height(), 80)
        gate = self.selected or self.hover
        if gate is not None:
            l, i = gate
            head = "LAYER %d · q%d · Rot" % (l + 1, i) + (
                "   (pinned — click again to release)" if self.selected else
                "   (click to pin)")
        else:
            head = "WHOLE CIRCUIT   ·   click a gate to follow it"
        cv.create_text(12, 14, anchor="w", text=head, fill=MUTED,
                       font=FONTS["chip"])
        if n < 2:
            cv.create_text(w / 2, h / 2 + 6, text="history appears after a few "
                           "episodes", fill=FAINT, font=FONTS["small"])
            return
        games = self.hist.games
        half = (w - 24) / 2
        boxes = ((12, 28, 12 + half - 8, h - 10),
                 (12 + half + 8, 28, w - 12, h - 10))
        cur = self._cur()
        if gate is not None:
            Ws = self.hist.W[:, l, i, :]
            Gs = self.hist.G[:, l, i, :]
            series_a = [(Ws[:, j], PARAM_COLORS[j]) for j in range(3)]
            series_g = [(Gs[:, j], PARAM_COLORS[j]) for j in range(3)]
            title_a, title_g = "angles φ θ ω  (rad)", "gradient RMS φ θ ω  (log)"
        else:
            Wall = self.hist.W
            drift = np.linalg.norm((Wall - Wall[0]).reshape(n, -1), axis=1)
            Gall = self.hist.G[:, ~dead_mask(self.hist.n_layers)]
            with warnings.catch_warnings():   # row 0 (init) has no gradient
                warnings.simplefilter("ignore", RuntimeWarning)
                gn = np.sqrt(np.nanmean(Gall ** 2, axis=1))
            series_a = [(drift, CYAN)]
            series_g = [(gn, RED)]
            title_a, title_g = "‖W − W₀‖  distance from init", "gradient RMS  (log)"
        self._spark(cv, boxes[0], games, series_a, title_a, cur, log=False)
        self._spark(cv, boxes[1], games, series_g, title_g, cur, log=True)

    def _spark(self, cv, box, games, series, title, cur, log):
        x0, y0, x1, y1 = box
        cv.create_rectangle(x0, y0, x1, y1, fill=_mix(CARD, BG, 0.45),
                            outline="")
        cv.create_text(x0 + 6, y0 + 9, anchor="w", text=title, fill=FAINT,
                       font=FONTS["chip"])
        n = len(games)
        ys = []
        for arr, _c in series:
            a = np.asarray(arr, dtype=float)
            if log:
                # exact zeros (no signal that episode) are gaps, and the
                # scale stops at 1e-9 so one outlier cannot squash it
                a = np.where(a >= GRAD_ZERO, a, np.nan)
                a = np.log10(np.clip(a, 1e-9, None))
            ys.append(a)
        allv = np.concatenate([a[np.isfinite(a)] for a in ys]) if ys else []
        if len(allv) == 0:
            return
        lo, hi = float(np.min(allv)), float(np.max(allv))
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        pt, pb = y0 + 20, y1 - 6
        span = max(1, n - 1)
        k = max(1, n // int(max(x1 - x0, 50)))
        idx = np.arange(0, n, k)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        for a, (_arr, color) in zip(ys, series):
            pts = []
            for t in idx:
                v = a[t]
                if not np.isfinite(v):
                    continue
                pts += [x0 + 4 + (x1 - x0 - 8) * t / span,
                        pb - (pb - pt) * (v - lo) / (hi - lo)]
            if len(pts) >= 4:
                cv.create_line(*pts, fill=color, width=1.6)
        if log:
            fmt = lambda v: "1e%.1f" % v
        else:
            fmt = lambda v: "%+.2f" % v
        cv.create_text(x1 - 4, pt - 2, anchor="ne", text=fmt(hi), fill=FAINT,
                       font=FONTS["chip"])
        cv.create_text(x1 - 4, pb, anchor="se", text=fmt(lo), fill=FAINT,
                       font=FONTS["chip"])
        if cur >= 0:
            xc = x0 + 4 + (x1 - x0 - 8) * cur / span
            cv.create_line(xc, pt - 4, xc, pb, fill=TEXT, dash=(2, 3))

    # -- mouse -------------------------------------------------------------
    def _item_tag(self, e):
        items = self.cv.find_overlapping(e.x, e.y, e.x, e.y)
        for it in reversed(items):
            for t in self.cv.gettags(it):
                if t.count(":") >= 1 and t.split(":")[0] in (
                        "gate", "enc", "cnot", "meas"):
                    return t
        return None

    def _cv_motion(self, e):
        tag = self._item_tag(e)
        gate = None
        if tag and tag.startswith("gate:"):
            _g, l, i = tag.split(":")
            gate = (int(l), int(i))
        self._hover_tag = tag
        if gate != self.hover:
            self._set_hover(gate)
        else:
            self._update_info()

    def _cv_leave(self, _e=None):
        self._hover_tag = None
        self._set_hover(None)
        self._update_info()

    def _cv_click(self, e):
        tag = self._item_tag(e)
        if tag and tag.startswith("gate:"):
            _g, l, i = tag.split(":")
            self._select((int(l), int(i)))

    def _cv_wheel(self, e):
        L = self.hist.n_layers
        if self.n_fit >= L:
            return
        num = getattr(e, "num", None)
        up = num == 4 or getattr(e, "delta", 0) > 0
        self.layer0 = max(0, min(L - self.n_fit, self.layer0 + (-1 if up else 1)))
        self._draw_circuit()
        self._draw_minimap()

    def _update_info(self):
        tag = getattr(self, "_hover_tag", None)
        W, G, W0, _p = self._snapshot()
        if W is None:
            self.info.config(text="")
            return
        gate = self.hover or self.selected
        bits = state_bits(self.state)
        if tag and tag.startswith("enc:"):
            i = int(tag.split(":")[1])
            txt = ("re-uploading on q%d:  RX(π·x%d) RZ(π·x%d)  with x%d = %d%s"
                   % (i, i, i, i, bits[i],
                      "" if bits[i] else "   → identity for this state"))
        elif tag and tag.startswith("cnot:"):
            c = int(tag.split(":")[1])
            txt = "CNOT  q%d → q%d   (control → target; no q3 → q0 wrap-around)" % (c, c + 1)
        elif tag and tag.startswith("meas:"):
            a = int(tag.split(":")[1])
            txt = ("⟨Z%d⟩ × %g  =  Q(s=%d, %s)" % (a, self.q_scale, self.state,
                                                  ACTION_NAMES[a]))
        elif gate is not None:
            l, i = gate
            w, w0, gr = W[l, i], W0[l, i], G[l, i]
            dead = [PARAM_NAMES[j] + ": " + DEAD_WHY[j]
                    for j in range(3) if self._dead[l, i, j]]
            txt = ("layer %d · q%d    φ %+.4f  θ %+.4f  ω %+.4f  rad"
                   "    Δinit (%+.3f, %+.3f, %+.3f)    ∇rms (%s, %s, %s)"
                   % (l + 1, i, w[0], w[1], w[2], *(w - w0),
                      fmt_grad(gr[0]), fmt_grad(gr[1]), fmt_grad(gr[2])))
            if dead:
                txt += "\nno effect on the output → " + dead[0]
        else:
            txt = ("dials: needle = angle now, dashed = at init, wedge = distance "
                   "travelled   ·   hover a gate for exact numbers"
                   if self.mode == "angles" else
                   "gauges: gradient RMS over the episode, log scale %s … %s   "
                   "·   red = vanishing, green = healthy, grey 0 = no signal "
                   "this episode"
                   % (fmt_grad(10 ** GRAD_LO), fmt_grad(10 ** GRAD_HI)))
        self.info.config(text=txt)
