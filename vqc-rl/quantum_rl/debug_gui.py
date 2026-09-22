"""Debug GUI: the normal GUI plus switches that bring the notebook bugs back.

Launch::

    ./debug_gui.py        (or: .venv/bin/python -m quantum_rl.debug_gui)

Same window, same automatic saving as :mod:`quantum_rl.gui`, with one extra
card in the third column: ticking a switch sets the matching
:class:`~quantum_rl.config.FixFlags` field back to ``True``, i.e. restores the
original, broken behaviour of the notebook, to A/B it against the fix.  The
flags used are recorded in ``summary.json`` under ``config.fixes``.

The plain GUI never offers these switches: it always trains ``FixFlags()``.
"""

from __future__ import annotations

import tkinter as tk

from .config import FixFlags
from .gui import App
from .gui_widgets import AMBER, CARD, FAINT, FONTS, QToggle, Tooltip

#: the four notebook bugs.  Ticking one restores the original behaviour.
BUG_HELP = {
    "td_pred_uses_next_state":
        "THE root cause of the failure: the prediction was indexed on the arrival "
        "state instead of the one the action was taken from, so every action "
        "converged to the same value.",
    "epsilon_greedy_inverted":
        "The epsilon-greedy test was inverted: the agent explored "
        "5% of the time at the start and 95% at the end.",
    "env_deterministic_transitions":
        "Training ran on a deterministic grid while the comparison against "
        "Bellman was on the stochastic one.",
    "greedy_policy_ignores_reward":
        "The Bellman policy maximised P*U ignoring the immediate reward, "
        "so it did not 'see' the -1.",
}

#: readable captions for the bug checkboxes (the key stays the FixFlags one)
BUG_LABELS = {
    "td_pred_uses_next_state": "TD prediction on the arrival state",
    "epsilon_greedy_inverted": "Inverted epsilon-greedy",
    "env_deterministic_transitions": "Deterministic grid while training",
    "greedy_policy_ignores_reward": "Greedy policy without reward",
}


class DebugApp(App):
    WINDOW_TITLE = "Quantum RL  ·  DEBUG  ·  notebook bugs can be re-enabled"

    def __init__(self):
        self.bug_vars: dict[str, tk.BooleanVar] = {}
        super().__init__()

    def _build_side_cards(self, col):
        super()._build_side_cards(col)
        inner = self._card(col, "Notebook bugs", AMBER,
                           note="tick = restore the original behaviour")
        for name, help_text in BUG_HELP.items():
            v = tk.BooleanVar(value=False)
            self.bug_vars[name] = v
            hover = (lambda _e, t=help_text: self.hint.set(t))
            tog = QToggle(inner, BUG_LABELS.get(name, name), v, accent=AMBER,
                          bg=CARD, on_hover=hover)
            tog.pack(fill="x", pady=(5, 2))
            Tooltip(tog.lbl, help_text, title=name)
            lab = tk.Label(inner, text=help_text, bg=CARD, fg=FAINT,
                           font=FONTS["small"], justify="left", anchor="w",
                           wraplength=380)
            lab.pack(fill="x", padx=(30, 0), pady=(0, 6))
            lab.bind("<Enter>", hover, add="+")
            self._auto_wrap(lab, inner, pad=46)

    def _extra_setup_state(self):
        return {"debug": {"bug_flags": {
            name: bool(var.get()) for name, var in self.bug_vars.items()}}}

    def _apply_extra_setup_state(self, data):
        flags = data.get("debug", {}).get("bug_flags", {})
        if not isinstance(flags, dict):
            return
        for name, value in flags.items():
            if name in self.bug_vars:
                self.bug_vars[name].set(bool(value))

    def _fixes(self) -> FixFlags:
        return FixFlags(**{n: v.get() for n, v in self.bug_vars.items()})


def main():
    DebugApp().mainloop()


if __name__ == "__main__":
    main()
