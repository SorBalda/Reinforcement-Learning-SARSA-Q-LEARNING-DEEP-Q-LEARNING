"""Palette, fonts and hand-drawn widgets shared by the GUI modules.

Kept apart from :mod:`quantum_rl.gui` so that the circuit view
(:mod:`quantum_rl.circuit_view`) can use them without a circular import.
:data:`FONTS` is filled by ``App._init_fonts()`` once Tk is up.
"""

from __future__ import annotations

import tkinter as tk

# --------------------------------------------------------------------------- #
# "quantum" palette: deep navy, cyan and violet accents
# --------------------------------------------------------------------------- #

BG = "#070b16"        # window background, almost blue-black
PANEL = "#0b1222"     # bands (header, footer, bars)
CARD = "#101a2e"      # card body
CARD_HI = "#16233d"   # card under the mouse
FIELD = "#091121"     # entry background
BORDER = "#1d2d4f"    # normal border
BORDER_HI = "#2c4573"  # highlighted border
TEXT = "#e8eefb"      # primary text
MUTED = "#90a5c8"     # secondary text
FAINT = "#5e7199"     # tertiary text
CYAN = "#22d3ee"      # primary accent
VIOLET = "#a78bfa"    # secondary accent
AMBER = "#ffa94d"     # tertiary accent / "learned" series
RED = "#ff6b81"       # stop / error
GREEN = "#4ade80"     # positive outcome
PINK = "#f472b6"      # gradients

# --------------------------------------------------------------------------- #
# small drawing helpers
# --------------------------------------------------------------------------- #


def _mix(c1: str, c2: str, t: float) -> str:
    """Blend two hex colours (Tk has no alpha channel)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, int(round(x + (y - x) * t)))) for x, y in zip(a, b))


def _round_rect(cv: tk.Canvas, x1, y1, x2, y2, rad, **kw):
    """Rectangle with rounded corners: a smoothed polygon."""
    pts = [x1 + rad, y1, x2 - rad, y1, x2, y1, x2, y1 + rad,
           x2, y2 - rad, x2, y2, x2 - rad, y2, x1 + rad, y2,
           x1, y2, x1, y2 - rad, x1, y1 + rad, x1, y1]
    return cv.create_polygon(pts, smooth=True, **kw)


class Tooltip:
    """Dark help balloon shown on hover."""

    def __init__(self, widget, text: str, title: str = ""):
        self.widget, self.text, self.title = widget, text, title
        self.win = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def show(self, _event=None):
        if self.win is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        win.configure(bg=BORDER_HI)
        body = tk.Frame(win, bg=PANEL)
        body.pack(padx=1, pady=1)
        if self.title:
            tk.Label(body, text=self.title, bg=PANEL, fg=CYAN, anchor="w",
                     font=FONTS["tip_b"]).pack(fill="x", padx=10, pady=(7, 0))
        tk.Label(body, text=self.text, bg=PANEL, fg=MUTED, wraplength=380,
                 justify="left", anchor="w", font=FONTS["tip"]).pack(
                     fill="x", padx=10, pady=(2, 8))
        win.wm_geometry(f"+{x}+{y}")
        self.win = win

    def hide(self, _event=None):
        if self.win is not None:
            self.win.destroy()
            self.win = None


#: filled by App._init_fonts(); used by every widget in this module
FONTS: dict[str, tuple] = {}


class NeonButton(tk.Canvas):
    """Hand-drawn button: rounded corners, a glow, three style variants.

    Exposes ``config(state=...)`` and ``btn["state"]`` like a normal Tk widget,
    so the rest of the code can treat it as a ``ttk.Button``.
    """

    def __init__(self, master, text, command, *, accent=CYAN, kind="primary",
                 width=None, height=44, bg=None, font=None, pad=26):
        self._bg = bg or master.cget("bg")
        self._font = font or FONTS["btn"]
        # measured with Tcl's ``font measure``: passing the tuple to
        # ``tkfont.Font(font=...)`` converts it back to points and rescales it,
        # giving widths inflated by 60% on high-density screens.
        w = width or (master.tk.call("font", "measure", self._font, text)
                      + 2 * pad)
        super().__init__(master, width=w, height=height, bg=self._bg,
                         highlightthickness=0, bd=0, takefocus=1)
        self._text = text
        self._command = command
        self._accent = accent
        self._kind = kind
        self._state = "normal"
        self._hover = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", lambda _e: self._draw())
        self._draw()

    # -- Tk-style API ---------------------------------------------------
    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if isinstance(cnf, dict) and "state" in cnf:
            kw["state"] = cnf.pop("state")
        if "state" in kw:
            self._state = kw.pop("state")
            self._hover = False
            self._draw()
        if "text" in kw:
            self._text = kw.pop("text")
            self._draw()
        if cnf or kw:
            return super().configure(cnf, **kw)
        return None

    config = configure

    def __getitem__(self, key):
        if key == "state":
            return self._state
        return super().__getitem__(key)

    # -- events -------------------------------------------------------------
    def _on_enter(self, _e=None):
        if self._state == "normal":
            self._hover = True
            self.configure(cursor="hand2")
            self._draw()

    def _on_leave(self, _e=None):
        self._hover = False
        self._draw()

    def _on_click(self, _e=None):
        if self._state == "normal" and self._command is not None:
            self._command()

    # -- drawing ------------------------------------------------------------
    def _draw(self):
        self.delete("all")
        w = int(self.winfo_width() or self.cget("width"))
        h = int(self.winfo_height() or self.cget("height"))
        rad = min(14, h // 2)
        off = self._state == "disabled"
        acc = _mix(self._accent, PANEL, 0.62) if off else self._accent

        # glow: two frames fading into the background
        if not off:
            glow = 0.30 if self._hover else 0.16
            _round_rect(self, 1, 1, w - 1, h - 1, rad + 3,
                        fill=_mix(self._bg, acc, glow * 0.45), outline="")
            _round_rect(self, 3, 3, w - 3, h - 3, rad + 1,
                        fill=_mix(self._bg, acc, glow * 0.8), outline="")

        if self._kind == "primary":
            top = _mix(acc, "#ffffff", 0.22 if self._hover else 0.0)
            _round_rect(self, 5, 5, w - 5, h - 5, rad,
                        fill=top, outline=_mix(acc, "#ffffff", 0.45), width=1)
            fg = "#04121f" if not off else _mix("#04121f", PANEL, 0.4)
        else:
            body = CARD_HI if self._hover else CARD
            _round_rect(self, 5, 5, w - 5, h - 5, rad,
                        fill=body, outline=acc, width=1)
            fg = acc if not off else _mix(acc, PANEL, 0.3)

        self.create_text(w // 2, h // 2 + 1, text=self._text, fill=fg,
                         font=self._font)


class QToggle(tk.Frame):
    """Hand-drawn checkbox, matching the dark theme."""

    def __init__(self, master, text, variable, *, accent=AMBER, bg=CARD,
                 on_hover=None):
        super().__init__(master, bg=bg)
        self.var = variable
        self.accent = accent
        self._bg = bg
        side = 20
        self.box = tk.Canvas(self, width=side, height=side, bg=bg,
                             highlightthickness=0, bd=0)
        self.box.pack(side="left", padx=(0, 10))
        self.lbl = tk.Label(self, text=text, bg=bg, fg=TEXT, anchor="w",
                            justify="left", font=FONTS["body"], wraplength=600)
        self.lbl.pack(side="left", fill="x", expand=True)
        self.bind("<Configure>", self._wrap, add="+")
        for w in (self, self.box, self.lbl):
            w.bind("<Button-1>", self._toggle)
            w.configure(cursor="hand2")
            if on_hover is not None:
                w.bind("<Enter>", on_hover, add="+")
        variable.trace_add("write", lambda *_: self._draw())
        self._draw()

    def _wrap(self, event):
        """The label wraps instead of being clipped."""
        avail = max(80, event.width - 34)
        if abs(avail - int(self.lbl.cget("wraplength"))) > 8:
            self.lbl.configure(wraplength=avail)

    def _toggle(self, _e=None):
        self.var.set(not self.var.get())

    def _draw(self):
        cv, side = self.box, 20
        cv.delete("all")
        on = bool(self.var.get())
        if on:
            _round_rect(cv, 1, 1, side - 1, side - 1, 6,
                        fill=_mix(self._bg, self.accent, 0.30), outline="")
            _round_rect(cv, 3, 3, side - 3, side - 3, 5,
                        fill=self.accent, outline=_mix(self.accent, "#ffffff", .4))
            cv.create_line(6, 10, 9, 13.5, 14, 6, fill="#0b1222", width=2,
                           capstyle="round", joinstyle="round")
        else:
            _round_rect(cv, 3, 3, side - 3, side - 3, 5,
                        fill=FIELD, outline=BORDER_HI, width=1)


class QSlider(tk.Canvas):
    """Hand-drawn integer slider: thick track, filled part, round knob.

    Built for legibility rather than compactness: the knob is large, the hit
    area is the whole height of the widget, and it answers to click-to-jump,
    drag, mouse wheel and the arrow keys (Shift = 10 steps, Home/End).
    ``command(value)`` fires only on user changes, never from :meth:`set`.
    """

    def __init__(self, master, command=None, *, accent=CYAN, bg=None,
                 height=30, width=200):
        self._bg = bg or master.cget("bg")
        super().__init__(master, height=height, width=width, bg=self._bg,
                         highlightthickness=0, bd=0, takefocus=1,
                         cursor="hand2")
        self.command = command
        self.accent = accent
        self.lo, self.hi, self.value = 0, 0, 0
        self._hover = self._drag = self._focus = False
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, self._wheel)
        self.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.bind("<FocusOut>", lambda _e: self._set_focus(False))
        self.bind("<Left>", lambda e: self._step(-10 if e.state & 1 else -1))
        self.bind("<Right>", lambda e: self._step(10 if e.state & 1 else 1))
        self.bind("<Home>", lambda _e: self._user_set(self.lo))
        self.bind("<End>", lambda _e: self._user_set(self.hi))

    # -- API -----------------------------------------------------------------
    def set_range(self, lo: int, hi: int):
        self.lo, self.hi = int(lo), max(int(lo), int(hi))
        self.value = min(max(self.value, self.lo), self.hi)
        self._draw()

    def set(self, value: int):
        self.value = min(max(int(value), self.lo), self.hi)
        self._draw()

    def get(self) -> int:
        return self.value

    @property
    def dragging(self) -> bool:
        return self._drag

    # -- events ----------------------------------------------------------------
    def _set_hover(self, on):
        self._hover = on
        self._draw()

    def _set_focus(self, on):
        self._focus = on
        self._draw()

    def _geom(self):
        w = self.winfo_width() or int(self.cget("width"))
        h = self.winfo_height() or int(self.cget("height"))
        r = max(7, min(11, h // 3))
        return w, h, r, r + 3, w - r - 3

    def _value_at(self, x):
        _w, _h, _r, x0, x1 = self._geom()
        if self.hi == self.lo or x1 <= x0:
            return self.lo
        t = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
        return int(round(self.lo + t * (self.hi - self.lo)))

    def _user_set(self, v):
        v = min(max(int(v), self.lo), self.hi)
        if v != self.value:
            self.value = v
            self._draw()
            if self.command:
                self.command(v)

    def _press(self, e):
        self.focus_set()
        self._drag = True
        self._user_set(self._value_at(e.x))
        self._draw()

    def _motion(self, e):
        if self._drag:
            self._user_set(self._value_at(e.x))

    def _release(self, _e):
        self._drag = False
        self._draw()
        if self.command:
            self.command(self.value)

    def _step(self, k):
        self._user_set(self.value + k)

    def _wheel(self, e):
        num = getattr(e, "num", None)
        up = num == 4 or getattr(e, "delta", 0) > 0
        span = max(1, (self.hi - self.lo) // 50)
        self._step(span if up else -span)

    # -- drawing ---------------------------------------------------------------
    def _draw(self):
        self.delete("all")
        w, h, r, x0, x1 = self._geom()
        cy = h / 2
        off = self.hi == self.lo
        acc = _mix(self.accent, PANEL, 0.6) if off else self.accent
        t = 0.0 if off else (self.value - self.lo) / (self.hi - self.lo)
        xk = x0 + t * (x1 - x0)
        th = 3
        _round_rect(self, x0 - th, cy - th, x1 + th, cy + th, th,
                    fill=_mix(PANEL, BORDER_HI, 0.8), outline="")
        if xk > x0:
            _round_rect(self, x0 - th, cy - th, xk + th, cy + th, th,
                        fill=_mix(acc, PANEL, 0.25), outline="")
        # a few ticks when the range is short enough to count them
        n = self.hi - self.lo
        if 0 < n <= 24:
            for i in range(n + 1):
                x = x0 + i * (x1 - x0) / n
                self.create_line(x, cy + th + 3, x, cy + th + 6,
                                 fill=_mix(PANEL, MUTED, 0.6))
        big = self._hover or self._drag
        if big or self._focus:
            self.create_oval(xk - r - 4, cy - r - 4, xk + r + 4, cy + r + 4,
                             fill=_mix(self._bg, acc, 0.22), outline="")
        rr = r if big else r - 1
        self.create_oval(xk - rr, cy - rr, xk + rr, cy + rr,
                         fill=_mix(acc, "#ffffff", 0.25 if big else 0.0),
                         outline=_mix(acc, "#ffffff", 0.55), width=1)
        self.create_oval(xk - 2.5, cy - 2.5, xk + 2.5, cy + 2.5,
                         fill="#04121f", outline="")
