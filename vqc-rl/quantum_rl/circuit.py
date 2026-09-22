"""Variational quantum circuit (VQC) used as the Q-value function approximator.

Faithful port of ``StochFrozenLake_QUANTUM.ipynb``:

    cell 40  -> ``decimal_to_binary_fix_length``  (notebook: ``decimalToBinaryFixLength``)
    cell 41  -> ``DTYPE`` (``torch.DoubleTensor``), re-stated by cell 49
    cell 42  -> ``qml.device('default.qubit', wires=4)``, re-stated by cell 49
    cell 43  -> the ``circuit`` QNode + ``variational_classifier``
    cell 44  -> ``state_preparation`` (notebook: ``statepreparation``) and ``layer``

Architecture, verbatim from the notebook:

    * 4 qubits, ``default.qubit``, torch interface, float64 parameters.
    * The state index ``s`` is turned into a 4-bit vector by
      ``decimal_to_binary_fix_length(4, s)`` and angle-encoded with RX/RZ
      (angle ``pi * bit``) on the matching wire.
    * ``weights`` has shape ``(deep_layers, 4, 3)``.  For EVERY layer the input
      state is RE-PREPARED before the layer is applied (data re-uploading) --
      this is the literal notebook loop ``for W in weights: statepreparation(angles); layer(W)``
      and it is kept on purpose.
    * Each layer is the CNOT chain 0->1->2->3 followed by one ``qml.Rot`` per wire.
    * The 4 ``PauliZ`` expectation values, multiplied by ``q_scale`` (3.0 in the
      notebook), ARE the four action Q-values (0=up, 1=right, 2=down, 3=left).

Deliberate deviations from the notebook (all numerically neutral, see the
function docstrings):

    1. ``decimal_to_binary_fix_length`` uses plain ``numpy`` instead of
       ``pennylane.numpy``.  The notebook's ``pennylane.numpy`` arrays carry
       ``requires_grad=True`` by default, which would mark the *input angles* as
       trainable; the returned values are bit-for-bit the same.
    2. The globals ``dev`` / ``circuit`` become ``make_qnode()`` so the device is
       not shared implicitly between experiments.  A lazily built module-level
       default QNode (``default_qnode()``) reproduces the notebook's single
       global device for callers that do not pass one.
    3. ``variational_classifier`` returns a ``torch.Tensor`` of shape ``(4,)``
       (``torch.stack`` of the QNode outputs) instead of the raw python list that
       PennyLane 0.45 hands back.  Every
       downstream use in the notebook (``max(...)``, ``[action_j]``,
       ``nn.MSELoss``) behaves identically on a stacked tensor, and stacking
       keeps the autograd graph intact (``loss.backward()`` still reaches the
       weights -- verified by the ``__main__`` self-check at the bottom).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

__all__ = [
    "DTYPE",
    "TORCH_DTYPE",
    "N_WIRES",
    "decimal_to_binary_fix_length",
    "state_preparation",
    "layer",
    "make_qnode",
    "default_qnode",
    "variational_classifier",
]


# notebook cells 41 / 49: `dtype = torch.DoubleTensor` (everything runs in float64)
DTYPE = torch.DoubleTensor
TORCH_DTYPE = torch.float64

# notebook cells 42 / 49: `dev = qml.device('default.qubit', wires=4)`
N_WIRES = 4


# ---------------------------------------------------------------------------
# notebook cell 40
# ---------------------------------------------------------------------------
def decimal_to_binary_fix_length(length, decimal):
    """Notebook cell 40 (``decimalToBinaryFixLength``).

    Encode ``decimal`` as a fixed-length big-endian bit vector, left padded with
    zeros.  ``decimal_to_binary_fix_length(4, 9) -> array([1, 0, 0, 1])``.

    Faithful quirks kept from the notebook:
      * ``decimal`` is truncated with ``int()``;
      * if ``bin(decimal)`` is LONGER than ``length`` the value is NOT truncated,
        the oversized array is returned as-is (never happens here: state indices
        are 0..11 and ``length`` is 4);
      * the dtype is not normalised -- the padded branch yields float64, the
        exact-length branch yields int64, exactly like the original.

    Deviation: plain ``numpy`` instead of ``pennylane.numpy`` (see module docstring).
    """
    bin_num = bin(int(decimal))[2:]
    output_num = [int(item) for item in bin_num]
    if len(output_num) < length:
        output_num = np.concatenate(
            (np.zeros((length - len(output_num),)), np.array(output_num))
        )
    else:
        output_num = np.array(output_num)
    return output_num


# ---------------------------------------------------------------------------
# notebook cell 44
# ---------------------------------------------------------------------------
def state_preparation(a):
    """Notebook cell 44 (``statepreparation``).

    Angle-encode the feature vector ``a`` (the 4 bits of the state index):
    ``RX(pi * a[i])`` then ``RZ(pi * a[i])`` on wire ``i``.

    Note that RZ right after RX on |0>-initialised wires only adds a phase on the
    FIRST layer, but since the state is re-prepared on top of an already entangled
    register at every later layer it is not a no-op there.  Kept verbatim.
    """
    for ind in range(len(a)):
        qml.RX(np.pi * a[ind], wires=ind)
        qml.RZ(np.pi * a[ind], wires=ind)


def layer(W):
    """Notebook cell 44 (``layer``).

    One variational layer: CNOT chain 0->1->2->3, then a general rotation per
    wire parameterised by ``W`` (shape ``(4, 3)``).

    Hardcoded for 4 wires, exactly as in the notebook (there is no wrap-around
    CNOT 3->0, so wire 0 is never the target of an entangling gate).
    """
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)


# ---------------------------------------------------------------------------
# notebook cells 42 + 43 + 49
# ---------------------------------------------------------------------------
def make_qnode(n_wires=N_WIRES, interface="torch"):
    """Notebook cells 42/49 (device) + 43 (``circuit``).

    Build a fresh ``default.qubit`` device and the variational QNode on top of it.

    The returned QNode has the notebook signature ``circuit(weights, angles=None)``
    and returns a list of ``n_wires`` PauliZ expectation values -- one Q-value per
    action.

    ``n_wires`` must be 4: :func:`layer` and the (4, 3) weight blocks are
    hardcoded for a 4-qubit register, just like the notebook.

    .. warning::
       ``qml.device('default.qubit', ...)`` defaults to ``seed="global"``, and
       PennyLane implements that as ``np.random.randint(0, 10000000)`` -- so
       **constructing a device consumes one draw from the legacy global numpy
       RNG**.  Anything that wants a reproducible, notebook-identical RNG stream
       must therefore build its QNode *before* calling
       :func:`quantum_rl.config.apply_seed` (the notebook builds ``dev`` in cell
       42, long before the training loop of cell 50).  :func:`quantum_rl.train.train`
       does exactly that; building a QNode lazily mid-run silently shifts every
       subsequent action, transition and minibatch index.
    """
    if n_wires != 4:
        raise ValueError(
            "make_qnode only supports n_wires=4: layer() and the (4, 3) weight "
            "blocks are hardcoded for 4 qubits in the notebook (cell 44)."
        )

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface=interface)
    def circuit(weights, angles=None):
        """The circuit of the variational classifier (notebook cell 43).

        NOTE (faithful): the state is re-prepared before EVERY layer -- this is
        data re-uploading and it is what the notebook does.
        """
        for W in weights:
            state_preparation(angles)
            layer(W)
        return [qml.expval(qml.PauliZ(ind)) for ind in range(n_wires)]

    return circuit


_DEFAULT_QNODE = None


def default_qnode():
    """Lazily built process-wide QNode, standing in for the notebook's global
    ``circuit`` (cell 43) on the global ``dev`` (cells 42/49).

    Used only when a caller does not pass an explicit ``qnode``.
    """
    global _DEFAULT_QNODE
    if _DEFAULT_QNODE is None:
        _DEFAULT_QNODE = make_qnode()
    return _DEFAULT_QNODE


def _stack_qnode_output(raw_output):
    """Normalise a multi-measurement QNode return into a 1-D torch tensor.

    PennyLane 0.45 returns a python ``list`` of 0-dim torch tensors here (older
    versions returned a torch tensor, some return a tuple).  ``torch.stack`` is
    differentiable, so the autograd graph from the weights to the 4 Q-values
    survives -- this is the one place where a careless port silently kills
    ``loss.backward()``.
    """
    if isinstance(raw_output, torch.Tensor):
        return raw_output

    elements = []
    for element in raw_output:
        if not isinstance(element, torch.Tensor):
            # non-torch interfaces (e.g. autograd/numpy): no graph to preserve
            element = torch.as_tensor(np.asarray(element))
        elements.append(element)
    return torch.stack(tuple(elements))


def variational_classifier(var_Q_circuit, angles=None, qnode=None, scale=3.0):
    """Notebook cell 43 (``variational_classifier``).

    Evaluate the VQC and return the 4 action Q-values as a differentiable
    ``torch.Tensor`` of shape ``(4,)``.

    The notebook scales every expectation value by 3.0 with an in-place python
    loop (``for i in range(len(raw_output)): raw_output[i] = raw_output[i]*3.0``).
    The loop is kept here (on a clone of the stacked tensor, which autograd
    handles through ``CopySlices``); ``scale`` is the ``cfg.q_scale`` knob and
    defaults to the notebook's 3.0.

    Args:
        var_Q_circuit: weights, shape ``(deep_layers, 4, 3)``, torch float64.
        angles: 4-bit encoded state, e.g. ``decimal_to_binary_fix_length(4, s)``.
        qnode: QNode from :func:`make_qnode`; ``None`` uses :func:`default_qnode`.
        scale: multiplicative output scaling (notebook: 3.0).
    """
    if qnode is None:
        qnode = default_qnode()

    weights = var_Q_circuit
    raw_output = _stack_qnode_output(qnode(weights, angles=angles))

    # notebook cell 43: `for i in range(len(raw_output)): raw_output[i] = raw_output[i]*3.0`
    output = raw_output.clone()
    for i in range(len(output)):
        output[i] = output[i] * scale
    return output


# ---------------------------------------------------------------------------
# self-check: forward pass + backward pass through variational_classifier
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    deep_layers =3
    qnode = make_qnode()

    # notebook cell 50 initialisation of var_Q_circuit
    var_Q_circuit = torch.tensor(
        0.3 * np.random.randn(deep_layers, 4, 3), device="cpu", requires_grad=True
    ).type(DTYPE)

    angles = decimal_to_binary_fix_length(4, 9)
    print("angles(4, 9) =", angles)

    q_values = variational_classifier(
        var_Q_circuit=var_Q_circuit, angles=angles, qnode=qnode, scale=3.0
    )
    print("q_values      =", q_values)
    print("is tensor     =", isinstance(q_values, torch.Tensor), q_values.shape, q_values.dtype)
    print("requires_grad =", q_values.requires_grad)

    total = q_values.sum()
    total.backward()

    ok_grad = var_Q_circuit.grad is not None
    print("weight grad is not None =", ok_grad)
    if ok_grad:
        print("grad norm =", float(var_Q_circuit.grad.norm()))

    # cross-check: the in-place scaling loop must give the same gradient as a
    # plain vectorised `raw * scale` (i.e. the loop does not corrupt autograd).
    w2 = var_Q_circuit.clone().detach().requires_grad_(True)
    raw2 = _stack_qnode_output(qnode(w2, angles=angles))
    (raw2 * 3.0).sum().backward()
    same = torch.allclose(var_Q_circuit.grad, w2.grad)
    print("loop-scaled grad == vectorised grad =", same)

    # the un-scaled gradient must be exactly 1/3 of the scaled one
    w3 = var_Q_circuit.clone().detach().requires_grad_(True)
    raw3 = _stack_qnode_output(qnode(w3, angles=angles))
    raw3.sum().backward()
    scaling_ok = torch.allclose(var_Q_circuit.grad, 3.0 * w3.grad)
    print("scaled grad == 3 x unscaled grad   =", scaling_ok)

    # action selection path must NOT build a graph
    detached = variational_classifier(
        var_Q_circuit=var_Q_circuit.clone().detach(), angles=angles, qnode=qnode, scale=3.0
    )
    print("detached path requires_grad =", detached.requires_grad)

    print()
    print(qml.draw(qnode)(var_Q_circuit, angles=decimal_to_binary_fix_length(4, 9)))
    print()
    if ok_grad and same and scaling_ok and not detached.requires_grad:
        print("SELF-CHECK PASSED: gradients flow through variational_classifier.")
    else:
        raise SystemExit("SELF-CHECK FAILED")
