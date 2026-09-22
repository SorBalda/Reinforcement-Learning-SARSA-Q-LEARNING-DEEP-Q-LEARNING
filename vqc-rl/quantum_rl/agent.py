"""Action selection and replay memory for the VQC agent.

Faithful port of ``StochFrozenLake_QUANTUM.ipynb``:

    cell 45  -> ``epsilon_greedy``
    cell 47  -> ``sample_random_batch``  (-> :meth:`ReplayMemory.sample`)
    cell 48  -> ``loss_fnc = nn.MSELoss()``
    cell 50  -> the replay-memory list ``D`` and its ``append`` / ``pop(0)`` policy

Two notebook behaviours that look like bugs are ported AS THEY ARE:

    * ``epsilon_greedy`` takes the GREEDY branch when ``np.random.rand() < epsilon``
      -- inverted w.r.t. the usual convention, so the agent explores LESS while
      epsilon is still large and MORE as epsilon decays.  Controlled by
      ``cfg.fixes.epsilon_greedy_inverted`` (``True`` = notebook).
    * the replay memory evicts with ``if len(D) >= N: D.pop(0)`` (``>=``, not
      ``>``) right after appending -- see :class:`ReplayMemory` for what that
      does to the steady-state length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .circuit import decimal_to_binary_fix_length, variational_classifier

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import order free
    from .config import Config

__all__ = [
    "loss_fnc",
    "ReplayMemory",
    "sample_random_batch",
    "epsilon_greedy",
]


# notebook cell 48
loss_fnc = nn.MSELoss()


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------
# `rng=None` reproduces the notebook EXACTLY: it uses the numpy legacy global
# RNG (`np.random.rand` / `np.random.randint`), so `np.random.seed(...)` is what
# makes a run reproducible.  Passing an explicit `np.random.Generator` (or a
# `RandomState`) gives an isolated stream instead -- same distributions, a
# different sequence of draws, so runs seeded either way are NOT comparable.
def _rand(rng: Optional[Any] = None) -> float:
    """Uniform float in [0, 1)."""
    if rng is None:
        return float(np.random.rand())
    if hasattr(rng, "random"):
        return float(rng.random())
    return float(rng.random_sample())


def _randint(n: int, rng: Optional[Any] = None) -> int:
    """Uniform integer in [0, n)."""
    if rng is None:
        return int(np.random.randint(0, n))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, n))
    return int(rng.randint(0, n))


# ---------------------------------------------------------------------------
# notebook cell 47 (+ cell 50 for the D / pop(0) policy)
# ---------------------------------------------------------------------------
def sample_random_batch(D: Sequence, batch_len: int, rng: Optional[Any] = None) -> List:
    """Notebook cell 47 (``sample_random_batch``).

    Sample ``batch_len`` transitions WITH REPLACEMENT, using the notebook's
    ``int(np.random.rand() * len(D))`` index draw (kept verbatim -- it consumes
    exactly one uniform per sampled item, which matters for reproducibility).
    """
    output = []
    for _ in range(batch_len):
        random_index = int(_rand(rng) * len(D))
        output.append(D[random_index])
    return output


class ReplayMemory:
    """The notebook's replay memory ``D`` (cell 50), as an object.

    A transition is the notebook's 4-element list ``[s, a, reward, s_next]``.

    EVICTION (faithful, and subtle): the notebook's training loop does::

        D.append([s_1, a, reward, s_2])
        if len(D) >= N:
            D.pop(0)

    i.e. it appends first and then pops whenever the length REACHES ``N``
    (``>=``, not ``>``).  Consequences:

      * if the memory is filled from empty through :meth:`append`, it saturates
        at ``maxlen - 1`` transitions, never ``maxlen``;
      * the notebook does NOT hit that case, because its initial random fill
        (cell 50, the ``while True`` pre-fill loop) appends to the bare list with
        no eviction check and stops at exactly ``N`` entries -- so during
        training the steady state is ``N`` (append -> N+1 -> pop -> N).

    To reproduce the notebook the pre-fill MUST therefore bypass eviction: use
    :meth:`prefill_append` (or ``append(t, evict=False)``) for the initial random
    fill and plain :meth:`append` inside the training loop.  Filling with
    :meth:`append` instead is off by one transition, which shifts every
    subsequent ``int(rand() * len(D))`` index and hence the whole trajectory.
    """

    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self.memory: List = []

    # -- notebook cell 50 -------------------------------------------------
    def append(self, t, evict: bool = True) -> None:
        """Append a transition, then apply the notebook's ``pop(0)`` eviction.

        ``evict=False`` reproduces the bare ``D.append(...)`` of the cell-50
        pre-fill loop (no eviction at all).
        """
        self.memory.append(t)
        if evict and len(self.memory) >= self.maxlen:
            self.memory.pop(0)

    def prefill_append(self, t) -> None:
        """Notebook cell 50 pre-fill: append with NO eviction check."""
        self.append(t, evict=False)

    # -- notebook cell 47 -------------------------------------------------
    def sample(self, batch_len: int, rng: Optional[Any] = None) -> List:
        """Notebook cell 47: ``batch_len`` transitions sampled WITH replacement."""
        return sample_random_batch(self.memory, batch_len, rng=rng)

    # -- plumbing ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]

    def __iter__(self):
        return iter(self.memory)

    def clear(self) -> None:
        self.memory = []

    @property
    def D(self) -> List:
        """The underlying list, under the notebook's name."""
        return self.memory

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ReplayMemory(maxlen={self.maxlen}, len={len(self.memory)})"


# ---------------------------------------------------------------------------
# notebook cell 45
# ---------------------------------------------------------------------------
def epsilon_greedy(
    var_Q_circuit,
    epsilon: float,
    n_action: int,
    s: int,
    qnode=None,
    cfg: "Optional[Config]" = None,
    rng: Optional[Any] = None,
    train: bool = False,
) -> int:
    """Notebook cell 45 (``epsilon_greedy``).

    The notebook body is::

        if train or np.random.rand() < epsilon:
            action = torch.argmax(torch.tensor(variational_classifier(
                var_Q_circuit=var_Q_circuit.clone().detach(),
                angles=decimalToBinaryFixLength(4, s))))
            action = action.item()
        else:
            action = np.random.randint(0, 4)
        return action

    so the GREEDY branch is taken when ``rand() < epsilon``.  That is inverted
    w.r.t. the standard convention: with ``epsilon_0 = 0.95`` decaying to
    ``0.05`` the agent starts out 95% greedy and ends up 95% random, i.e. it
    explores more and more as training proceeds.  Ported verbatim under
    ``cfg.fixes.epsilon_greedy_inverted = True`` (the default); with the flag set
    to ``False`` the conventional rule (random when ``rand() < epsilon``) is used
    instead.

    Both branches consume exactly ONE uniform draw, and -- exactly like the
    notebook's ``or`` short-circuit -- none at all when ``train=True``, so the
    two flag settings can be A/B'd on the same RNG stream.

    ``var_Q_circuit.clone().detach()`` is kept: action selection must never build
    an autograd graph.

    Returns:
        int in ``[0, n_action)`` -- 0=up, 1=right, 2=down, 3=left.
    """
    inverted = True if cfg is None else bool(cfg.fixes.epsilon_greedy_inverted)
    scale = 3.0 if cfg is None else float(cfg.q_scale)
    n_wires = 4 if cfg is None else int(cfg.n_wires)

    if train:
        # notebook: `train or ...` short-circuits, no random number is drawn
        greedy = True
    else:
        u = _rand(rng)
        greedy = (u < epsilon) if inverted else (u >= epsilon)

    if greedy:
        q_values = variational_classifier(
            var_Q_circuit=var_Q_circuit.clone().detach(),
            angles=decimal_to_binary_fix_length(n_wires, s),
            qnode=qnode,
            scale=scale,
        )
        # notebook wraps this in `torch.tensor(...)` purely to detach; the
        # weights are already detached above, so `.detach()` is equivalent and
        # avoids torch's copy-construct warning.
        action = torch.argmax(q_values.detach())
        action = int(action.item())
    else:
        # notebook hardcodes `np.random.randint(0, 4)`; n_action is always 4 here
        action = _randint(n_action, rng=rng)
    return action
