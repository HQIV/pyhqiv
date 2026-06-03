"""
Lean-backed quantum simulator primitives for HQIV workflows.

This module is intentionally lightweight: it provides an executable Python
simulation layer while preserving clear hooks to Lean-backed formal artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


ComplexArray = np.ndarray
GateSpec = Tuple[str, Tuple[int, ...], Tuple[float, ...]]


@dataclass(frozen=True)
class ValueWithError:
    value: float
    sigma: float


def _basis_size(n_qubits: int) -> int:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    return 1 << n_qubits


def _assert_valid_target(n_qubits: int, q: int) -> None:
    if q < 0 or q >= n_qubits:
        raise ValueError(f"qubit index {q} out of range for n_qubits={n_qubits}")


def _assert_valid_control_target(n_qubits: int, control: int, target: int) -> None:
    _assert_valid_target(n_qubits, control)
    _assert_valid_target(n_qubits, target)
    if control == target:
        raise ValueError("control and target must be different qubits")


@dataclass
class NoiseModel:
    bit_flip_p: float = 0.0
    phase_flip_p: float = 0.0
    depolarizing_p: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("bit_flip_p", self.bit_flip_p),
            ("phase_flip_p", self.phase_flip_p),
            ("depolarizing_p", self.depolarizing_p),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


class QuantumState:
    """Dense statevector in computational basis order."""

    def __init__(self, n_qubits: int, amplitudes: ComplexArray | None = None):
        self.n_qubits = n_qubits
        dim = _basis_size(n_qubits)
        if amplitudes is None:
            self.amplitudes = np.zeros(dim, dtype=np.complex128)
            self.amplitudes[0] = 1.0 + 0.0j
        else:
            arr = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
            if arr.size != dim:
                raise ValueError(f"Expected {dim} amplitudes, got {arr.size}")
            self.amplitudes = arr.copy()
            self.normalize()

    def copy(self) -> "QuantumState":
        return QuantumState(self.n_qubits, self.amplitudes.copy())

    def norm_sq(self) -> float:
        return float(np.vdot(self.amplitudes, self.amplitudes).real)

    def normalize(self) -> None:
        norm = math.sqrt(self.norm_sq())
        if norm == 0.0:
            raise ValueError("Cannot normalize zero state")
        self.amplitudes = self.amplitudes / norm

    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def apply_single_qubit_gate(self, gate: np.ndarray, target: int) -> None:
        _assert_valid_target(self.n_qubits, target)
        if gate.shape != (2, 2):
            raise ValueError("single-qubit gate must have shape (2, 2)")
        stride = 1 << target
        dim = self.amplitudes.size
        block = stride << 1
        out = self.amplitudes.copy()
        for base in range(0, dim, block):
            for offset in range(stride):
                i0 = base + offset
                i1 = i0 + stride
                a0 = self.amplitudes[i0]
                a1 = self.amplitudes[i1]
                out[i0] = gate[0, 0] * a0 + gate[0, 1] * a1
                out[i1] = gate[1, 0] * a0 + gate[1, 1] * a1
        self.amplitudes = out

    def apply_cnot(self, control: int, target: int) -> None:
        _assert_valid_control_target(self.n_qubits, control, target)
        if control == target:
            raise ValueError("control and target must differ")
        dim = self.amplitudes.size
        out = self.amplitudes.copy()
        control_mask = 1 << control
        target_mask = 1 << target
        for i in range(dim):
            if (i & control_mask) != 0 and (i & target_mask) == 0:
                j = i | target_mask
                out[i] = self.amplitudes[j]
                out[j] = self.amplitudes[i]
        self.amplitudes = out

    def apply_swap(self, q0: int, q1: int) -> None:
        _assert_valid_control_target(self.n_qubits, q0, q1)
        if q0 == q1:
            return
        dim = self.amplitudes.size
        out = self.amplitudes.copy()
        m0 = 1 << q0
        m1 = 1 << q1
        for i in range(dim):
            b0 = (i & m0) >> q0
            b1 = (i & m1) >> q1
            if b0 != b1:
                j = i ^ m0 ^ m1
                if i < j:
                    out[i], out[j] = self.amplitudes[j], self.amplitudes[i]
        self.amplitudes = out

    def apply_controlled_phase(self, control: int, target: int, theta: float) -> None:
        _assert_valid_control_target(self.n_qubits, control, target)
        phase = np.exp(1j * theta)
        cm = 1 << control
        tm = 1 << target
        for i in range(self.amplitudes.size):
            if (i & cm) and (i & tm):
                self.amplitudes[i] *= phase

    def apply_qft(self, qubits: Sequence[int] | None = None, inverse: bool = False) -> None:
        q = list(range(self.n_qubits)) if qubits is None else list(qubits)
        if not q:
            return
        for qb in q:
            _assert_valid_target(self.n_qubits, qb)
        sign = -1.0 if inverse else 1.0
        m = len(q)
        for i in range(m):
            qi = q[i]
            for j in range(i):
                qj = q[j]
                theta = sign * math.pi / (1 << (i - j))
                self.apply_controlled_phase(control=qi, target=qj, theta=theta)
            self.apply_single_qubit_gate(Gates.H, qi)
        for i in range(m // 2):
            self.apply_swap(q[i], q[m - 1 - i])

    def apply_qft_exact(self, qubits: Sequence[int] | None = None, inverse: bool = False) -> None:
        """
        Exact QFT / inverse-QFT on the selected qubits by block-subspace Fourier matrix.

        This is slower than the gate-decomposition implementation in `apply_qft`, but
        is reliable for small simulations and ensures correct interference for
        OSHoracle-style period finding.
        """
        q = list(range(self.n_qubits)) if qubits is None else list(qubits)
        if not q:
            return
        q_sorted = sorted(q)
        m = len(q_sorted)
        for qb in q_sorted:
            _assert_valid_target(self.n_qubits, qb)
        if m == 0:
            return

        # Remaining qubits form independent "blocks".
        other_positions = [p for p in range(self.n_qubits) if p not in set(q_sorted)]
        k = len(other_positions)
        other_dim = 1 << k
        sub_dim = 1 << m

        # Fourier matrix in little-endian convention relative to q_sorted ordering.
        # F[sub_out, sub_in] = exp(±2π i sub_out * sub_in / sub_dim) / sqrt(sub_dim)
        omega_sign = -1.0 if inverse else 1.0
        F = np.empty((sub_dim, sub_dim), dtype=np.complex128)
        norm = 1.0 / math.sqrt(float(sub_dim))
        for out_i in range(sub_dim):
            for in_j in range(sub_dim):
                angle = omega_sign * 2.0 * math.pi * (out_i * in_j) / float(sub_dim)
                F[out_i, in_j] = norm * np.exp(1j * angle)

        old = self.amplitudes.copy()
        new = np.zeros_like(old)

        # Helper: build global index from (other, sub) bit patterns.
        def global_index(other_val: int, sub_val: int) -> int:
            idx = 0
            # Place sub bits into q positions.
            for j, pos in enumerate(q_sorted):
                if (sub_val >> j) & 1:
                    idx |= 1 << pos
            # Place other bits into the remaining positions.
            for j, pos in enumerate(other_positions):
                if (other_val >> j) & 1:
                    idx |= 1 << pos
            return idx

        for other_val in range(other_dim):
            vec = np.zeros((sub_dim,), dtype=np.complex128)
            for sub_val in range(sub_dim):
                gi = global_index(other_val, sub_val)
                vec[sub_val] = old[gi]
            vec_out = F @ vec
            for sub_val in range(sub_dim):
                gi = global_index(other_val, sub_val)
                new[gi] = vec_out[sub_val]

        self.amplitudes = new

    def measure_shots(self, shots: int, seed: int | None = None) -> Dict[str, int]:
        if shots <= 0:
            raise ValueError("shots must be > 0")
        rng = random.Random(seed)
        probs = self.probabilities()
        cdf = np.cumsum(probs)
        counts: Dict[str, int] = {}
        for _ in range(shots):
            r = rng.random()
            idx = int(np.searchsorted(cdf, r, side="right"))
            bitstring = format(idx, f"0{self.n_qubits}b")[::-1]
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def apply_noise(self, model: NoiseModel, seed: int | None = None) -> None:
        model.validate()
        rng = random.Random(seed)
        for q in range(self.n_qubits):
            if rng.random() < model.bit_flip_p:
                self.apply_single_qubit_gate(Gates.X, q)
            if rng.random() < model.phase_flip_p:
                self.apply_single_qubit_gate(Gates.Z, q)
            if rng.random() < model.depolarizing_p:
                pauli = rng.choice((Gates.X, Gates.Y, Gates.Z))
                self.apply_single_qubit_gate(pauli, q)


class Gates:
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H = (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    S = np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
    T = np.array([[1.0, 0.0], [0.0, np.exp(1j * math.pi / 4.0)]], dtype=np.complex128)

    @staticmethod
    def phase(theta: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)


@dataclass
class Circuit:
    n_qubits: int
    ops: List[GateSpec]

    def __init__(self, n_qubits: int):
        _basis_size(n_qubits)
        self.n_qubits = n_qubits
        self.ops = []

    def h(self, q: int) -> "Circuit":
        self.ops.append(("single", (q,), ()))
        return self

    def x(self, q: int) -> "Circuit":
        self.ops.append(("x", (q,), ()))
        return self

    def y(self, q: int) -> "Circuit":
        self.ops.append(("y", (q,), ()))
        return self

    def z(self, q: int) -> "Circuit":
        self.ops.append(("z", (q,), ()))
        return self

    def s(self, q: int) -> "Circuit":
        self.ops.append(("s", (q,), ()))
        return self

    def t(self, q: int) -> "Circuit":
        self.ops.append(("t", (q,), ()))
        return self

    def phase(self, q: int, theta: float) -> "Circuit":
        self.ops.append(("phase", (q,), (theta,)))
        return self

    def cnot(self, control: int, target: int) -> "Circuit":
        self.ops.append(("cnot", (control, target), ()))
        return self

    def swap(self, q0: int, q1: int) -> "Circuit":
        self.ops.append(("swap", (q0, q1), ()))
        return self

    def qft(self, qubits: Sequence[int] | None = None, inverse: bool = False) -> "Circuit":
        q = tuple(range(self.n_qubits)) if qubits is None else tuple(qubits)
        self.ops.append(("qft", q, (1.0 if inverse else 0.0,)))
        return self


@dataclass
class RunResult:
    state: QuantumState
    counts: Dict[str, int] | None

    def probability_with_error(self, index: int) -> ValueWithError:
        probs = self.state.probabilities()
        if index < 0 or index >= probs.size:
            raise ValueError("index out of range")
        p = float(probs[index])
        if self.counts is None:
            return ValueWithError(value=p, sigma=0.0)

        shots = sum(self.counts.values())
        if shots <= 0:
            return ValueWithError(value=p, sigma=0.0)
        sigma = math.sqrt(max(0.0, p * (1.0 - p)) / float(shots))
        return ValueWithError(value=p, sigma=sigma)


def run_circuit(
    circuit: Circuit,
    *,
    initial_state: QuantumState | None = None,
    shots: int | None = None,
    noise: NoiseModel | None = None,
    seed: int | None = None,
) -> RunResult:
    state = QuantumState(circuit.n_qubits) if initial_state is None else initial_state.copy()
    if state.n_qubits != circuit.n_qubits:
        raise ValueError("initial_state.n_qubits must match circuit.n_qubits")

    for kind, qubits, params in circuit.ops:
        if kind == "single":
            state.apply_single_qubit_gate(Gates.H, qubits[0])
        elif kind == "x":
            state.apply_single_qubit_gate(Gates.X, qubits[0])
        elif kind == "y":
            state.apply_single_qubit_gate(Gates.Y, qubits[0])
        elif kind == "z":
            state.apply_single_qubit_gate(Gates.Z, qubits[0])
        elif kind == "s":
            state.apply_single_qubit_gate(Gates.S, qubits[0])
        elif kind == "t":
            state.apply_single_qubit_gate(Gates.T, qubits[0])
        elif kind == "phase":
            state.apply_single_qubit_gate(Gates.phase(params[0]), qubits[0])
        elif kind == "cnot":
            state.apply_cnot(qubits[0], qubits[1])
        elif kind == "swap":
            state.apply_swap(qubits[0], qubits[1])
        elif kind == "qft":
            state.apply_qft(qubits=qubits, inverse=bool(params[0]))
        else:
            raise ValueError(f"Unsupported gate kind: {kind}")

        if noise is not None:
            state.apply_noise(noise, seed=seed)

    counts = None if shots is None else state.measure_shots(shots, seed=seed)
    return RunResult(state=state, counts=counts)


__all__ = [
    "Circuit",
    "ComplexArray",
    "Gates",
    "NoiseModel",
    "QuantumState",
    "RunResult",
    "ValueWithError",
    "run_circuit",
]

