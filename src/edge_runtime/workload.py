"""Something for the device to do, so the runtime has real inputs to run on.

A kinematic reach task: seven joints integrate the commanded deltas subject to
limits and a rate cap, and the observation carries the error to a goal. There
are no contacts, no dynamics and no friction — it is the same stand-in the
evaluation harness upstream uses, and success rates from it are harness results,
not robot results.

It exists here for one reason: latency, flagging and OTA gating all need a
stream of observations with realistic structure — correlated across time,
occasionally out of distribution, occasionally saturating the actuator. Feeding
the runtime Gaussian noise would exercise the code and measure nothing.

Replacing it with a real robot or a simulator means implementing `reset` and
`step`. Nothing in the runtime, the OTA client or the telemetry spool knows what
is on the other side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .policy import default_obs_columns

JOINT_LIMIT = 2.6  # rad
RATE_CAP = 0.15  # rad per control step


@dataclass
class ReachWorkload:
    """Episodic reach task producing the default observation contract."""

    seed: int = 0
    episode_length: int = 200
    # Fraction of episodes started from a pose well outside the usual range, so
    # the out-of-distribution detector has something true to find.
    anomaly_rate: float = 0.05

    obs_columns: list[str] = field(default_factory=default_obs_columns)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        anomalous = self._rng.random() < self.anomaly_rate
        spread = 2.2 if anomalous else 0.5
        self.joints = self._rng.normal(0, spread, 7).astype(np.float32)
        self.goal = self._rng.uniform(-0.8, 0.8, 7).astype(np.float32)
        self.gripper = 0.0
        self.t = 0
        self.last_action_norm = 0.0
        self.anomalous = anomalous
        return self.observe()

    def observe(self) -> np.ndarray:
        error = self.goal - self.joints
        return np.array(
            [
                *self.joints,
                *error,
                self.gripper,
                self.t / self.episode_length,
                float(np.linalg.norm(error)),
                self.last_action_norm,
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply an action. Returns (next observation, episode_done)."""
        action = np.asarray(action, dtype=np.float32)
        delta = np.clip(action[:7], -RATE_CAP, RATE_CAP)
        self.joints = np.clip(self.joints + delta, -JOINT_LIMIT, JOINT_LIMIT)
        self.gripper = float(np.clip(action[7], 0.0, 1.0)) if action.size > 7 else self.gripper
        self.last_action_norm = float(np.linalg.norm(delta))
        self.t += 1

        reached = float(np.linalg.norm(self.goal - self.joints)) < 0.05
        done = reached or self.t >= self.episode_length
        obs = self.observe()
        if done:
            self.reset()
        return obs, done

    def sample_observations(self, n: int, policy_forward=None) -> np.ndarray:
        """Collect `n` observations, for calibration and probe sets.

        Rolled out under a policy when one is given, because the states a policy
        visits are not the states a random walk visits — and calibrating an
        out-of-distribution detector on the wrong distribution is how you get a
        detector that fires on everything or on nothing.
        """
        out = np.zeros((n, len(self.obs_columns)), dtype=np.float32)
        obs = self.observe()
        for i in range(n):
            out[i] = obs
            action = (
                policy_forward(obs)
                if policy_forward is not None
                else self._rng.normal(0, 0.05, 8).astype(np.float32)
            )
            obs, _ = self.step(action)
        return out
