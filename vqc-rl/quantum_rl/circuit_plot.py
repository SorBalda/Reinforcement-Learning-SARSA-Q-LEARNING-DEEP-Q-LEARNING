"""Static matplotlib drawing of the VQC with its weights or gradients.

The live view (:mod:`quantum_rl.circuit_view`) is a Tk canvas: it only shows
the layers that fit on screen and exporting it needs ghostscript.  This module
draws the same picture -- same layout, symbols and colours -- as a matplotlib
figure with ALL the layers, so the end of a run can save the final circuit as
PNG and PDF anywhere.

Only the object-oriented API is used (no pyplot), so it is safe to call from
the Tk main thread.
"""

from __future__ import annotations

import math
import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Wedge

from .circuit_view import (ACTION_ARROWS, ACTION_NAMES, DEAD_WHY,
                           PARAM_COLORS, PARAM_NAMES, dead_mask, fmt_grad,
                           grad_color, grad_frac)
from .gui_widgets import (BG, BORDER_HI, CARD, CYAN, FAINT, MUTED, TEXT, VIOLET,
                          _mix)

# layout, in data units (1 unit ~ one gate slot)
_LANE = 1.7          # distance between wires
_ENC_W = 0.9
_CX_W = 0.45
_GAP = 0.3
_COL = 0.95          # width of one parameter column inside a Rot box
_ROT_W = 3 * _COL + 0.2
_PAD = 0.7
_LAYER_W = _ENC_W + _GAP + 3 * _CX_W + _GAP + _ROT_W + _PAD
_LEFT = 1.6
_RIGHT = 2.6


def _box(ax, x0, y0, w, h, fc, ec, lw=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def draw_circuit(W, *, W0=None, G=None, mode="angles", q_scale=3.0,
                 title="", figure=None) -> Figure:
    """Draw every layer of the circuit.

    Args:
        W: weights, shape ``(L, 4, 3)``.
        W0: initial weights; in ``angles`` mode the wedge of each dial spans
            from ``W0`` to ``W`` (distance travelled).  ``None`` = no wedge.
        G: gradient RMS, shape ``(L, 4, 3)``; required for ``mode="grads"``.
        mode: ``"angles"`` (dials) or ``"grads"`` (log gauges).
    """
    W = np.asarray(W, dtype=float)
    L = W.shape[0]
    dead = dead_mask(L)
    width = _LEFT + L * _LAYER_W + _RIGHT
    height = 4 * _LANE + 1.6
    fig = figure or Figure(figsize=(max(8.0, width * 0.62), height * 0.62 + 0.5),
                           facecolor=BG)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.9 if title else 0.98))
    ax.set_xlim(0, width)
    ax.set_ylim(-0.3, height)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=13, x=0.01, ha="left", y=0.98)

    ys = [height - 1.35 - _LANE * (i + 0.5) for i in range(4)]
    wire = _mix(BG, BORDER_HI, 0.9)
    x_end = width - _RIGHT + 0.4
    for i, y in enumerate(ys):
        ax.plot([_LEFT - 0.5, x_end], [y, y], color=wire, lw=1.6, zorder=1)
        ax.text(0.25, y + 0.15, "q%d" % i, color=TEXT, fontsize=12,
                fontweight="bold", va="center", family="monospace")
        ax.text(0.25, y - 0.32, "x%d" % i, color=FAINT, fontsize=8,
                va="center", family="monospace")

    top = ys[0] + _LANE / 2 + 0.05
    bot = ys[-1] - _LANE / 2 + 0.05
    bh = _LANE - 0.3
    r = 0.33
    for l in range(L):
        xl = _LEFT + l * _LAYER_W
        x_cx = xl + _ENC_W + _GAP
        x_rot = x_cx + 3 * _CX_W + _GAP
        _box(ax, xl - 0.15, bot, _LAYER_W - _PAD + 0.3, top - bot + 0.35,
             _mix(BG, "#ffffff", 0.035 if l % 2 else 0.055),
             _mix(BG, BORDER_HI, 0.5), z=0)
        ax.text(xl, top + 0.62, "LAYER %d" % (l + 1), color=MUTED, fontsize=9,
                fontweight="bold")
        for x, lab in ((xl + _ENC_W / 2, "encode"),
                       (x_cx + 1.5 * _CX_W, "entangle"),
                       (x_rot + _ROT_W / 2, "Rot(φ, θ, ω)")):
            ax.text(x, top + 0.18, lab, color=FAINT, fontsize=7, ha="center")

        # re-uploading of the input: RX(pi x) RZ(pi x)
        for y in ys:
            _box(ax, xl, y - 0.42, _ENC_W, 0.84, CARD, CYAN, lw=0.9, ls="--")
            ax.text(xl + _ENC_W / 2, y + 0.16, "RxRz", color=MUTED, fontsize=6.5,
                    ha="center", va="center", zorder=3)
            ax.text(xl + _ENC_W / 2, y - 0.17, "πx", color=TEXT, fontsize=9,
                    ha="center", va="center", zorder=3)

        # CNOT staircase
        for c in range(3):
            x = x_cx + (c + 0.5) * _CX_W
            y1, y2 = ys[c], ys[c + 1]
            ax.plot([x, x], [y1, y2 - 0.19], color=VIOLET, lw=1.6, zorder=2)
            ax.add_patch(Circle((x, y1), 0.08, fc=VIOLET, ec="none", zorder=3))
            ax.add_patch(Circle((x, y2), 0.19, fc=_mix(BG, "#ffffff", 0.05),
                                ec=VIOLET, lw=1.6, zorder=3))
            ax.plot([x - 0.19, x + 0.19], [y2, y2], color=VIOLET, lw=1.6, zorder=4)
            ax.plot([x, x], [y2 - 0.19, y2 + 0.19], color=VIOLET, lw=1.6, zorder=4)

        # Rot gates
        for i, y in enumerate(ys):
            if mode == "grads" and G is not None:
                live = G[l, i][~dead[l, i]]
                gm = np.nanmean(live) if np.isfinite(live).any() else np.nan
                ec = _mix(BORDER_HI, grad_color(gm), 0.6)
            else:
                ec = BORDER_HI
            _box(ax, x_rot, y - bh / 2, _ROT_W, bh, CARD, ec, lw=1.0)
            for j in range(3):
                xc = x_rot + 0.1 + _COL * (j + 0.5)
                yc = y + 0.05
                d = dead[l, i, j]
                ax.text(xc, y + bh / 2 - 0.2, PARAM_NAMES[j], ha="center",
                        va="center", fontsize=8,
                        color=FAINT if d else PARAM_COLORS[j], zorder=4)
                if d:
                    ax.add_patch(Circle((xc, yc), r, fc=_mix(CARD, BG, 0.5),
                                        ec=_mix(CARD, MUTED, 0.5), ls=":",
                                        lw=0.8, zorder=3))
                    ax.plot([xc - r / 2, xc + r / 2], [yc, yc], color=FAINT,
                            lw=1.5, zorder=4)
                    txt, fg = "n/a", FAINT
                elif mode == "grads":
                    g = G[l, i, j] if G is not None else np.nan
                    ax.add_patch(Arc((xc, yc), 2 * r, 2 * r, theta1=-45,
                                     theta2=225, color=_mix(CARD, BORDER_HI, 0.9),
                                     lw=3.5, zorder=3))
                    f = grad_frac(g)
                    if f > 0:
                        ax.add_patch(Arc((xc, yc), 2 * r, 2 * r,
                                         theta1=225 - 270 * f, theta2=225,
                                         color=grad_color(g), lw=3.5, zorder=4))
                    txt, fg = fmt_grad(g), grad_color(g)
                else:
                    v = W[l, i, j]
                    col = PARAM_COLORS[j]
                    ax.add_patch(Circle((xc, yc), r, fc=_mix(CARD, BG, 0.5),
                                        ec=_mix(CARD, col, 0.45), lw=0.9,
                                        zorder=3))
                    if W0 is not None:
                        v0 = W0[l, i, j]
                        dv = max(min(v - v0, 2 * math.pi), -2 * math.pi)
                        if abs(dv) > 1e-3:
                            a0 = 90 - math.degrees(v0)
                            a1 = 90 - math.degrees(v0 + dv)
                            ax.add_patch(Wedge((xc, yc), r * 0.92, min(a0, a1),
                                               max(a0, a1),
                                               fc=_mix(CARD, col, 0.32),
                                               ec="none", zorder=3))
                        ax.plot([xc, xc + 0.9 * r * math.sin(v0)],
                                [yc, yc + 0.9 * r * math.cos(v0)],
                                color=_mix(CARD, MUTED, 0.6), lw=0.8, ls=":",
                                zorder=4)
                    ax.plot([xc, xc + r * math.sin(v)], [yc, yc + r * math.cos(v)],
                            color=col, lw=2.0, solid_capstyle="round", zorder=5)
                    ax.add_patch(Circle((xc, yc), 0.05, fc=col, ec="none",
                                        zorder=6))
                    txt, fg = "%+.2f" % v, TEXT
                ax.text(xc, y - bh / 2 + 0.2, txt, ha="center", va="center",
                        fontsize=7.5, color=fg, family="monospace", zorder=4)

    # measurement: <Z_a> x q_scale is the Q-value of action a
    xm = width - _RIGHT + 0.5
    for a, y in enumerate(ys):
        _box(ax, xm, y - 0.3, 0.7, 0.6, CARD, BORDER_HI)
        ax.add_patch(Arc((xm + 0.35, y - 0.12), 0.46, 0.46, theta1=20, theta2=160,
                         color=MUTED, lw=1.2, zorder=3))
        ax.plot([xm + 0.35, xm + 0.52], [y - 0.12, y + 0.14], color=TEXT, lw=1.2,
                zorder=4)
        ax.text(xm + 0.9, y + 0.12, "%s %s" % (ACTION_ARROWS[a], ACTION_NAMES[a]),
                color=MUTED, fontsize=8.5, va="center")
        ax.text(xm + 0.9, y - 0.2, "Q = %g·<Z%d>" % (q_scale, a), color=FAINT,
                fontsize=7, va="center", family="monospace")

    # legend line
    if mode == "grads":
        note = ("gauges: gradient RMS of the episode, log scale %s … %s · red = "
                "vanishing, green = healthy · grey 0 = no signal (light cone) · "
                "n/a = no effect (%s; %s)"
                % (fmt_grad(10 ** -6), fmt_grad(10 ** -1), DEAD_WHY[0], DEAD_WHY[2]))
    else:
        note = ("dials: needle = angle (rad), dotted = initial value, wedge = "
                "distance travelled · 0 at twelve o'clock, positive clockwise · "
                "n/a = no effect on the output")
    ax.text(0.25, -0.1, note, color=FAINT, fontsize=7, va="bottom")
    return fig


def save_final_circuit(out_dir: str, history, q_scale: float) -> list[str]:
    """``circuit/circuit_angles`` and ``circuit/circuit_gradients`` (.png, .pdf)
    for the LAST snapshot of ``history``.  Returns the files written."""
    if history.n == 0:
        return []
    folder = os.path.join(out_dir, "circuit")
    os.makedirs(folder, exist_ok=True)
    W, W0 = history.W[-1], history.W[0]
    G = history.G[-1]
    game = int(history.games[-1])
    when = "initial weights" if game < 0 else "episode %d" % game
    written = []
    for mode, name, title in (
            ("angles", "circuit_angles",
             "VQC · final weights (%s) · %d layers" % (when, W.shape[0])),
            ("grads", "circuit_gradients",
             "VQC · gradient RMS of the last episode (%s)" % when)):
        fig = draw_circuit(W, W0=W0, G=G, mode=mode, q_scale=q_scale,
                           title=title)
        for ext in ("png", "pdf"):
            path = os.path.join(folder, "%s.%s" % (name, ext))
            fig.savefig(path, dpi=160, facecolor=BG)
            written.append(path)
    return written


__all__ = ["draw_circuit", "save_final_circuit"]
