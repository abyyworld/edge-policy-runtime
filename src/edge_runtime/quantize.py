"""Weight-only int8 quantization, and the check that it did not break anything.

Scope, stated plainly so the numbers are not read as more than they are:

**This is weight-only quantization.** Weights are stored as int8 with a float
scale; activations stay float32. That is a real deployment mode — it is what
memory-bound models ship as, and the arithmetic here is exactly what a
weight-only int8 kernel computes, so the fidelity numbers transfer. Activation
quantization is a different job (it needs calibrated ranges per tensor and it
interacts with the accumulator width), and pretending to do it in numpy would
produce numbers that mean nothing.

**The speedup is not measured here, because numpy cannot show it.** numpy
upcasts int8 operands, so an int8 matmul in this process is *slower*, not
faster. What this module measures is the footprint reduction and — the part
that actually decides whether you can ship it — how far the quantized policy's
actions move from the float policy's. Latency is measured separately, against
whatever kernel the device really runs, by :mod:`edge_runtime.runtime`.

**Per-channel, not per-tensor.** A single scale for a whole weight matrix is set
by its largest-magnitude channel, and every other channel loses resolution in
proportion. That matters most in the action head, whose output channels carry
different physical quantities: seven joint deltas capped at 0.15 rad, and one
normalised gripper command. The gripper channel sets the scale, and the joint
channels — the ones measured in radians that the robot actually tracks — pay for
it. Measured on the reference policy, per-tensor is **6.4x** worse on the joint
deltas (2.5e-3 vs 3.9e-4 rad worst case) for a footprint difference of 5%.

That is enough to fail a tolerance of 1% of the rate cap while per-channel
passes, so it is not an academic difference; per-tensor is also what you get by
default from most toolchains. :func:`quantization_report` measures both on the
same data so the choice is visible rather than assumed, and
`test_per_tensor_loses_the_joint_channels` holds the result in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import Policy, PolicyError

INT8_MAX = 127.0


# The actuator's per-step rate cap. Tolerances are expressed as a fraction of it
# rather than as a bare number, because "5e-4 is small" is only true relative to
# something and the something is how far the joint can move in one control step.
RATE_CAP_RAD = 0.15
JOINT_TOLERANCE_RAD = 0.01 * RATE_CAP_RAD
GRIPPER_TOLERANCE = 0.02  # normalised [0, 1] command


@dataclass(frozen=True)
class FidelityReport:
    """How far the quantized policy's actions moved, per unit.

    Joint deltas and the gripper command are reported separately because they
    are not the same quantity. Collapsing them into one number means the
    dimensionless gripper channel sets a tolerance that is then applied to
    radians, and the resulting figure is not in any unit at all.
    """

    scheme: str
    float_bytes: int
    quantized_bytes: int
    n_calibration: int

    joint_mae_rad: float
    joint_max_abs_rad: float
    gripper_max_abs: float
    worst_column: str

    @property
    def compression(self) -> float:
        return self.float_bytes / max(self.quantized_bytes, 1)

    def qualifies(
        self,
        joint_tolerance_rad: float = JOINT_TOLERANCE_RAD,
        gripper_tolerance: float = GRIPPER_TOLERANCE,
    ) -> bool:
        """Fidelity is judged on the worst case, not the average.

        A mean error inside tolerance with a tail outside it is a policy that is
        fine almost always, which on a robot is a different sentence from fine.
        """
        return (
            self.joint_max_abs_rad <= joint_tolerance_rad
            and self.gripper_max_abs <= gripper_tolerance
        )

    def summary(self) -> str:
        return (
            f"{self.scheme}: {self.compression:.2f}x smaller, "
            f"joint MAE {self.joint_mae_rad:.2e} rad, "
            f"worst {self.joint_max_abs_rad:.2e} rad on {self.worst_column}"
        )


class QuantizedPolicy(Policy):
    """A policy whose weights are stored as int8 codes plus float scales.

    Subclasses :class:`Policy` so it is a drop-in for the runtime, the bundler
    and the health gate — none of which should have to know how the weights are
    stored.
    """

    def __init__(
        self,
        codes: list[np.ndarray],
        scales: list[np.ndarray],
        biases: list[np.ndarray],
        obs_columns: list[str],
        action_columns: list[str] | None = None,
        name: str = "policy-int8",
        scheme: str = "per_channel",
    ) -> None:
        self.codes = [np.asarray(c, dtype=np.int8) for c in codes]
        self.scales = [np.asarray(s, dtype=np.float32) for s in scales]
        self.scheme = scheme
        weights = [c.astype(np.float32) * s for c, s in zip(self.codes, self.scales, strict=True)]
        super().__init__(weights, biases, obs_columns, action_columns, name)

    @property
    def stored_bytes(self) -> int:
        return sum(
            c.nbytes + s.nbytes + b.nbytes
            for c, s, b in zip(self.codes, self.scales, self.biases, strict=True)
        )


def quantize(policy: Policy, *, scheme: str = "per_channel") -> QuantizedPolicy:
    """Symmetric int8 quantization of every weight matrix.

    Symmetric (zero maps to zero, no zero-point) because these weights are
    roughly zero-centred and an asymmetric scheme buys nothing for the extra
    field to get wrong. Biases stay float32: they are a rounding error of the
    footprint and a real source of error if quantized.
    """
    if scheme not in {"per_channel", "per_tensor"}:
        raise PolicyError(f"unknown quantization scheme {scheme!r}")

    codes, scales = [], []
    for w in policy.weights:
        if scheme == "per_channel":
            # One scale per output column, so a wide channel cannot set the
            # resolution of a narrow one.
            amax = np.abs(w).max(axis=0, keepdims=True)
        else:
            amax = np.abs(w).max(keepdims=True)
        # An all-zero channel has no scale to speak of; 1.0 keeps it at zero
        # instead of producing NaN.
        scale = np.where(amax > 0, amax / INT8_MAX, 1.0).astype(np.float32)
        code = np.clip(np.rint(w / scale), -INT8_MAX, INT8_MAX).astype(np.int8)
        codes.append(code)
        scales.append(scale)

    return QuantizedPolicy(
        codes,
        scales,
        policy.biases,
        policy.obs_columns,
        policy.action_columns,
        name=f"{policy.name}-int8",
        scheme=scheme,
    )


def float_bytes(policy: Policy) -> int:
    return sum(w.nbytes + b.nbytes for w, b in zip(policy.weights, policy.biases, strict=True))


def measure_fidelity(
    policy: Policy,
    quantized: QuantizedPolicy,
    calibration: np.ndarray,
) -> FidelityReport:
    """Compare actions on a calibration set of real observations.

    Calibration observations must come from the distribution the device will
    actually see. Quantization error is input-dependent, and a report built on
    Gaussian noise will happily approve a policy that falls apart on the robot.
    """
    calibration = np.atleast_2d(np.asarray(calibration, dtype=np.float32))
    if calibration.shape[1] != policy.obs_dim:
        raise PolicyError(
            f"calibration set has {calibration.shape[1]} columns, policy expects {policy.obs_dim}"
        )
    if len(calibration) < 32:
        raise PolicyError(
            f"{len(calibration)} calibration observations is not enough to see the "
            "tail of the error distribution; use at least 32"
        )

    reference = policy.forward(calibration)
    candidate = quantized.forward(calibration)
    error = np.abs(candidate - reference)

    joint_cols = [i for i, c in enumerate(policy.action_columns) if c.endswith("_delta")]
    other_cols = [i for i in range(error.shape[1]) if i not in set(joint_cols)]
    joint_error = error[:, joint_cols] if joint_cols else error
    worst_col = joint_cols[int(joint_error.max(axis=0).argmax())] if joint_cols else 0

    return FidelityReport(
        scheme=quantized.scheme,
        float_bytes=float_bytes(policy),
        quantized_bytes=quantized.stored_bytes,
        n_calibration=len(calibration),
        joint_mae_rad=float(joint_error.mean()),
        joint_max_abs_rad=float(joint_error.max()),
        gripper_max_abs=float(error[:, other_cols].max()) if other_cols else 0.0,
        worst_column=policy.action_columns[worst_col],
    )


def quantization_report(policy: Policy, calibration: np.ndarray) -> dict[str, FidelityReport]:
    """Both schemes, measured on the same data, so the difference is visible."""
    return {
        scheme: measure_fidelity(policy, quantize(policy, scheme=scheme), calibration)
        for scheme in ("per_tensor", "per_channel")
    }
