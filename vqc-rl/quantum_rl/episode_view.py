"""Animated playback of a monitored GridWorld episode.

The view is deliberately independent from the training loop: it accepts the
plain dictionary emitted by ``train(..., game_monitor=...)`` and can therefore
show the latest episode while training or after loading a run.
"""
from __future__ import annotations

import tkinter as tk
from typing import Any, Mapping, Optional

from .gui_widgets import (BG, CARD, BORDER, BORDER_HI, CYAN, GREEN, MUTED, RED, TEXT,
                           VIOLET, QSlider, FONTS)


class EpisodeView(tk.Toplevel):
    """A small 4x3 grid with Play/Pause, timeline slider and latest episode."""

    def __init__(self, master=None, episode: Optional[Mapping[str, Any]] = None,
                 *, nx: int = 4, ny: int = 3, obstacle_indexes=(),
                 alive_indexes=(), death_indexes=(), interval: int = 220):
        super().__init__(master)
        self.title("Episode playback")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.nx, self.ny = int(nx), int(ny)
        self.obstacles = set(int(x) for x in obstacle_indexes)
        self.alive = set(int(x) for x in alive_indexes)
        self.death = set(int(x) for x in death_indexes)
        self.interval = max(30, int(interval))
        self.episode = {}
        self.trajectory = []
        self.actions = []
        self.rewards = []
        self.frame = 0
        self.playing = False
        self._job = None
        self.cell = 92
        self.canvas = tk.Canvas(self, width=self.nx*self.cell, height=self.ny*self.cell,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(padx=18, pady=(18, 10))
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=18)
        self.play_btn = tk.Button(bar, text="Play", command=self.toggle,
                                  bg=CARD, fg=CYAN, activebackground=BORDER_HI,
                                  activeforeground=TEXT, relief="flat", bd=0, padx=14,
                                  font=FONTS.get("btn", ("TkDefaultFont", 11, "bold")))
        self.play_btn.pack(side="left")
        self.status = tk.Label(bar, text="Nessun episodio", bg=BG, fg=MUTED, anchor="w",
                               font=FONTS.get("body", ("TkDefaultFont", 11)))
        self.status.pack(side="left", padx=12, fill="x", expand=True)
        self.slider = QSlider(self, command=self._seek, accent=VIOLET, bg=BG,
                              width=self.nx*self.cell-36, height=32)
        self.slider.pack(fill="x", padx=18, pady=(4, 16))
        self.protocol("WM_DELETE_WINDOW", self.close)
        if episode is not None:
            self.set_episode(episode)

    def set_episode(self, episode: Mapping[str, Any]) -> None:
        """Replace playback data and show its last frame."""
        self.episode = dict(episode)
        self.trajectory = [int(s) for s in episode.get("trajectory", ())]
        self.actions = [int(a) for a in episode.get("actions", ())]
        self.rewards = [float(r) for r in episode.get("rewards", ())]
        n = max(0, len(self.trajectory) - 1)
        self.slider.set_range(0, n)
        self.frame = n
        self.slider.set(n)
        self._draw()

    def replay(self) -> None:
        """Restart the latest episode from its first state and play it."""
        if not self.trajectory:
            return
        self.playing = False
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.frame = 0
        self.slider.set(0)
        self._draw()
        self.playing = True
        self.play_btn.configure(text="Pause")
        # Keep the initial state visible for one interval before advancing.
        self._job = self.after(self.interval, self._tick)

    def toggle(self) -> None:
        if not self.trajectory:
            return
        if not self.playing and self.frame >= len(self.trajectory) - 1:
            self.replay()
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()
        elif self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.frame >= len(self.trajectory) - 1:
            self.playing = False
            self.play_btn.configure(text="Play")
            self._job = None
            return
        self.frame += 1
        self.slider.set(self.frame)
        self._draw()
        self._job = self.after(self.interval, self._tick)

    def _seek(self, value: int) -> None:
        self.frame = int(value)
        self._draw()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        for y in range(self.ny):
            for x in range(self.nx):
                state = x + self.nx*y
                x0, y0 = x*self.cell+3, (self.ny-1-y)*self.cell+3
                fill = BORDER
                if state in self.obstacles: fill = "#283451"
                elif state in self.alive: fill = "#16452f"
                elif state in self.death: fill = "#4b2331"
                c.create_rectangle(x0, y0, x0+self.cell-6, y0+self.cell-6, fill=fill, outline=BORDER_HI)
                c.create_text(x0+10, y0+10, text=str(state), fill=MUTED, anchor="nw",
                              font=FONTS.get("small", ("TkDefaultFont", 10)))
        shown = self.trajectory[:self.frame+1]
        points = []
        for state in shown:
            x, y = int(state) % self.nx, int(state) // self.nx
            points.extend((x*self.cell+self.cell/2, (self.ny-1-y)*self.cell+self.cell/2))
        if len(points) >= 4:
            c.create_line(*points, fill=VIOLET, width=4, capstyle="round", joinstyle="round")
        if shown:
            state = shown[-1]
            x, y = int(state) % self.nx, int(state) // self.nx
            cx, cy = x*self.cell+self.cell/2, (self.ny-1-y)*self.cell+self.cell/2
            c.create_oval(cx-17, cy-17, cx+17, cy+17, fill=CYAN, outline=TEXT, width=2)
        reward = self.rewards[self.frame-1] if self.frame > 0 and self.frame <= len(self.rewards) else "—"
        self.status.configure(text=f"Frame {self.frame}/{max(0, len(self.trajectory)-1)}   •   reward: {reward}")

    def close(self) -> None:
        self.playing = False
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.destroy()


# Friendly alias for callers that prefer a descriptive name.
EpisodePlayback = EpisodeView

__all__ = ["EpisodeView", "EpisodePlayback"]
