"""Plain-numpy simulator of the VQC, for the GUI's live circuit view.

Same circuit as :func:`quantum_rl.circuit.make_qnode`: for every layer the
state is re-uploaded (``RX(pi*b)``, ``RZ(pi*b)`` on each wire), then the CNOT
chain 0->1->2->3 and ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)``
on each wire; the outputs are the four ``<Z_i>``.

Why not the QNode: the GUI evaluates the circuit on the Tk thread, for any
state and any past snapshot of the weights, while the training thread is busy
with PennyLane/torch.  Sixteen amplitudes are cheaper to push around by hand
than to share a device across threads.  Wire 0 is the most significant qubit,
as in PennyLane; ``tests/`` checks the two against each other.

No torch, no pennylane: importing this module is free.
"""

from __future__ import annotations

import numpy as np

N_WIRES = 4


def _rx(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(a):
    return np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]])


def _rot(phi, theta, omega):
    return _rz(omega) @ _ry(theta) @ _rz(phi)


def _apply_1q(psi, u, wire):
    """``psi`` has shape ``(2,)*N_WIRES``; apply ``u`` on ``wire``."""
    psi = np.tensordot(u, psi, axes=([1], [wire]))
    return np.moveaxis(psi, 0, wire)


def _apply_cnot(psi, control, target):
    psi = psi.copy()
    idx = [slice(None)] * N_WIRES
    idx[control] = 1
    sub = psi[tuple(idx)]
    # after fixing ``control`` the target axis shifts down by one if it came after
    t = target - 1 if target > control else target
    psi[tuple(idx)] = np.flip(sub, axis=t)
    return psi


def state_bits(s: int) -> np.ndarray:
    """4-bit big-endian encoding of the state index (bit 0 goes on wire 0)."""
    return np.array([(int(s) >> (N_WIRES - 1 - i)) & 1 for i in range(N_WIRES)])


def expvals(weights, bits) -> np.ndarray:
    """``<Z_0..3>`` of the circuit for ``weights`` of shape ``(L, 4, 3)``."""
    weights = np.asarray(weights, dtype=float)
    psi = np.zeros((2,) * N_WIRES, dtype=complex)
    psi[(0,) * N_WIRES] = 1.0
    enc = [_rz(np.pi * b) @ _rx(np.pi * b) for b in bits]
    for W in weights:
        for w in range(N_WIRES):
            psi = _apply_1q(psi, enc[w], w)
        for w in range(N_WIRES - 1):
            psi = _apply_cnot(psi, w, w + 1)
        for w in range(N_WIRES):
            psi = _apply_1q(psi, _rot(*W[w]), w)
    prob = np.abs(psi) ** 2
    out = np.empty(N_WIRES)
    for w in range(N_WIRES):
        p = np.moveaxis(prob, w, 0).reshape(2, -1).sum(axis=1)
        out[w] = p[0] - p[1]
    return out


def q_values(weights, s: int, scale: float = 3.0) -> np.ndarray:
    """The four action Q-values of state ``s``: ``scale * <Z_a>``."""
    return scale * expvals(weights, state_bits(s))
