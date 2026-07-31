"""The inference loop that runs on the device, and what it refuses to do.

Two decisions carry most of the weight here.

**Latency is reported as p95 and p99, and the mean is not a headline number.**
A control loop with a 20 ms budget, a 4 ms mean and a 60 ms p99 misses its
deadline several times a minute, and on a robot each miss is a discontinuity in
the commanded trajectory. Averages hide exactly the events that matter, so this
module counts deadline misses directly instead of inferring them.

**An inference failure produces a safe action, not an exception.** A policy that
raises — a malformed observation, a NaN, an unloaded model mid-swap — must not
take the control loop down with it. The runtime returns the hold action (zero
joint deltas, gripper unchanged), counts it, and flags it upstream. The robot
stops moving, which is a bad outcome; the alternative is an unhandled exception
in a loop that owns actuators, which is a worse one.

The runtime never decides to *stop*. Deciding a device is unhealthy enough to
be pulled from service is a fleet-level decision made with fleet-level context,
and it belongs in the hub. The device's job is to keep telling the truth about
itself.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .policy import Policy


@dataclass
class StepRecord:
    """One inference. The unit the telemetry up-link aggregates."""

    latency_ms: float
    deadline_missed: bool
    failed: bool
    action: np.ndarray
    error: str | None = None


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    deadline_misses: int
    failures: int
    budget_ms: float

    @property
    def deadline_miss_rate(self) -> float:
        return self.deadline_misses / self.n if self.n else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "mean_ms": round(self.mean_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": round(self.deadline_miss_rate, 6),
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 6),
            "budget_ms": self.budget_ms,
        }

    def summary(self) -> str:
        return (
            f"n={self.n} p50={self.p50_ms:.2f}ms p95={self.p95_ms:.2f}ms "
            f"p99={self.p99_ms:.2f}ms max={self.max_ms:.2f}ms  "
            f"misses={self.deadline_misses} ({self.deadline_miss_rate:.2%}) "
            f"failures={self.failures}"
        )


@dataclass
class InferenceRuntime:
    """Runs a policy under a deadline, and keeps a bounded record of how it went.

    The latency window is a ring buffer, not a growing list. A device runs for
    weeks; an unbounded metrics buffer is a memory leak with a longer fuse than
    most, and it will take the process down at 3am on the one deployment nobody
    is watching.
    """

    policy: Policy
    budget_ms: float = 20.0
    window: int = 4096

    _latencies: deque[float] = field(default_factory=deque, init=False)
    _n: int = field(default=0, init=False)
    _misses: int = field(default=0, init=False)
    _failures: int = field(default=0, init=False)
    _last_gripper: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._latencies = deque(maxlen=self.window)
        try:
            self._gripper_idx = self.policy.action_columns.index("gripper_target")
        except ValueError:
            self._gripper_idx = -1

    # -- the loop ------------------------------------------------------------

    def warmup(self, n: int = 20) -> None:
        """Run the policy on zeros before the first real step.

        The first inference pays for lazily-allocated buffers and, on a real
        accelerator, for kernel selection and clock ramp-up. Left unwarmed, that
        cost lands on the first control step after a policy swap — which is
        precisely the step a health gate is watching.
        """
        zeros = np.zeros(self.policy.obs_dim, dtype=np.float32)
        for _ in range(n):
            self.policy.forward(zeros)

    def hold_action(self) -> np.ndarray:
        """Zero joint deltas, gripper held where it was."""
        action = np.zeros(self.policy.action_dim, dtype=np.float32)
        if self._gripper_idx >= 0:
            action[self._gripper_idx] = self._last_gripper
        return action

    def step(self, observation: dict[str, float] | np.ndarray) -> StepRecord:
        start = time.perf_counter_ns()
        failed = False
        error: str | None = None
        try:
            vector = (
                self.policy.bind(observation)
                if isinstance(observation, dict)
                else np.asarray(observation, dtype=np.float32)
            )
            action = np.asarray(self.policy.forward(vector), dtype=np.float32)
            if not np.all(np.isfinite(action)):
                raise ValueError("policy produced a non-finite action")
        except Exception as exc:  # noqa: BLE001 - the whole point is not to propagate
            failed = True
            error = f"{type(exc).__name__}: {exc}"
            action = self.hold_action()
        else:
            if self._gripper_idx >= 0:
                self._last_gripper = float(action[self._gripper_idx])

        latency_ms = (time.perf_counter_ns() - start) / 1e6
        missed = latency_ms > self.budget_ms

        self._latencies.append(latency_ms)
        self._n += 1
        self._misses += int(missed)
        self._failures += int(failed)

        return StepRecord(
            latency_ms=latency_ms,
            deadline_missed=missed,
            failed=failed,
            action=action,
            error=error,
        )

    # -- reporting -----------------------------------------------------------

    def stats(self) -> LatencyStats:
        """Percentiles over the retained window; counters over all time.

        The two have different horizons on purpose. Latency drifts and the
        recent window is what describes the device now; failures are rare events
        and the lifetime count is what an operator wants to see.
        """
        samples = np.asarray(self._latencies, dtype=np.float64)
        if samples.size == 0:
            return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, self.budget_ms)
        p50, p95, p99 = np.percentile(samples, [50, 95, 99])
        return LatencyStats(
            n=self._n,
            mean_ms=float(samples.mean()),
            p50_ms=float(p50),
            p95_ms=float(p95),
            p99_ms=float(p99),
            max_ms=float(samples.max()),
            deadline_misses=self._misses,
            failures=self._failures,
            budget_ms=self.budget_ms,
        )

    def reset_counters(self) -> None:
        self._latencies.clear()
        self._n = self._misses = self._failures = 0


def benchmark(
    policy: Policy,
    observations: np.ndarray,
    *,
    budget_ms: float = 20.0,
    warmup: int = 20,
) -> LatencyStats:
    """Time a policy on real observations, one at a time.

    One at a time, not batched. Batched throughput is the wrong measurement for
    a control loop: the robot needs the action for *this* observation before the
    next tick, and a batch of 64 does not exist. Batched numbers look far better
    and describe a workload the device does not have.
    """
    runtime = InferenceRuntime(policy, budget_ms=budget_ms)
    runtime.warmup(warmup)
    for obs in np.atleast_2d(observations):
        runtime.step(obs)
    return runtime.stats()
