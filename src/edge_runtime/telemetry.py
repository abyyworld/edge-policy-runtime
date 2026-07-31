"""The up-link: what the device says about itself, and how it survives silence.

A fleet is only as observable as its worst-connected node, so the hard part of
telemetry is not what to measure — it is what happens during the six hours the
device cannot reach the hub.

**Events are spooled locally first, and only deleted once acknowledged.**
Delivery is at-least-once: a device that uploads a batch and loses the network
before reading the response will send it again. Every event carries a UUID and
the hub deduplicates on it. Exactly-once across an unreliable link is not
available at any price, and pretending otherwise means silently losing events
instead of occasionally duplicating them.

**The spool is bounded, and it evicts by priority before it evicts by age.**
When a device has been offline long enough to fill its budget, the events worth
keeping are the failures and the flagged edge cases — not the routine heartbeats
that say everything is fine. A plain ring buffer drops precisely backwards:
it keeps the last hour of "all normal" and discards the crash that started it.
Here, heartbeats are shed first, and a fresh heartbeat will never displace an
old failure.

**Flagging happens on the device.** Uploading everything and triaging centrally
requires bandwidth nobody has; uploading only aggregates means the interesting
episodes are gone by the time anyone asks. So the detectors run locally and the
device sends the tail, not the mean. What counts as interesting is in
:class:`EdgeCaseDetector`.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .bundle import write_atomically
from .runtime import LatencyStats, StepRecord

EventKind = Literal["heartbeat", "failure", "flag", "ota"]

# Lower number, kept longer. Failures and update lifecycle events are the record
# of what went wrong; heartbeats are re-derivable from the next one.
PRIORITY = {"failure": 0, "ota": 0, "flag": 1, "heartbeat": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TelemetryEvent:
    kind: EventKind
    device_id: str
    policy_version: int
    payload: dict
    ts: str = field(default_factory=_now)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.kind, 1)

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_id": self.event_id,
                "kind": self.kind,
                "device_id": self.device_id,
                "policy_version": self.policy_version,
                "ts": self.ts,
                "payload": self.payload,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, d: dict) -> TelemetryEvent:
        return cls(
            kind=d["kind"],
            device_id=d["device_id"],
            policy_version=d["policy_version"],
            payload=d.get("payload", {}),
            ts=d.get("ts", _now()),
            event_id=d.get("event_id", uuid.uuid4().hex),
        )


# -- flagging ----------------------------------------------------------------


@dataclass
class EdgeCaseDetector:
    """Decides which control steps are worth a human's attention.

    Four signals, chosen because each catches a failure the others miss:

    ``inference_failed``
        The policy raised or produced a non-finite action. Always interesting.
    ``deadline_burst``
        Consecutive deadline misses. One slow step is jitter; five in a row is a
        thermal throttle, a memory-pressure stall, or another process on the
        device — all of which look like a policy problem in aggregate latency
        and are not one.
    ``action_saturated``
        The policy is commanding at or beyond the actuator limit, so the robot
        is no longer doing what the policy asked. Invisible in action-error
        metrics because the error is measured before the clamp.
    ``observation_ood``
        The observation is far outside the range seen during calibration, in
        units of that column's own spread. This is the one that finds the
        situations the training set never contained — the reason for collecting
        from the fleet at all.

    Deliberately not included: a learned uncertainty estimate. It would be the
    strongest signal here and it needs a policy that produces one; adding a
    threshold on softmax entropy for a deterministic regression head would be
    decoration.

    **Each reason has a cooldown, and suppressed occurrences are counted rather
    than discarded.** An out-of-distribution episode is out of distribution for
    all two hundred of its steps. Emitting two hundred near-identical events
    spends the spool and the operator's attention on one incident, and a device
    that flags a tenth of its steps has not flagged anything. So the first
    occurrence carries the full context and the rest increment a counter that
    rides along on the next heartbeat — the incident is still visible, and its
    duration is still measurable, at one event instead of two hundred.
    """

    action_limit: float = 0.15  # rad per step, matching the actuator rate cap
    ood_sigma: float = 4.0
    burst_length: int = 5
    cooldown_steps: int = 100

    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None

    _consecutive_misses: int = field(default=0, init=False)
    _step: int = field(default=0, init=False)
    _last_emitted: dict[str, int] = field(default_factory=dict, init=False)
    _suppressed: dict[str, int] = field(default_factory=dict, init=False)

    @classmethod
    def calibrated_on(cls, observations: np.ndarray, **kwargs) -> EdgeCaseDetector:
        """Fit the in-distribution range from the data the policy was trained on."""
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        std = obs.std(axis=0)
        # A column that never varied has no scale; treating its std as zero makes
        # every future value infinitely out of distribution.
        std = np.where(std > 1e-6, std, 1.0)
        return cls(obs_mean=obs.mean(axis=0), obs_std=std, **kwargs)

    def inspect(self, observation: np.ndarray, record: StepRecord) -> list[str]:
        """Return the reasons this step is interesting. Empty means routine."""
        self._step += 1
        reasons: list[str] = []

        if record.failed:
            reasons.append(f"inference_failed:{record.error}")

        self._consecutive_misses = self._consecutive_misses + 1 if record.deadline_missed else 0
        if self._consecutive_misses == self.burst_length:
            reasons.append(f"deadline_burst:{self.burst_length}")

        action = np.asarray(record.action, dtype=np.float64)
        if action.size and np.max(np.abs(action[:7])) >= self.action_limit:
            reasons.append(f"action_saturated:{np.max(np.abs(action[:7])):.3f}")

        if self.obs_mean is not None and self.obs_std is not None:
            obs = np.asarray(observation, dtype=np.float64).ravel()
            if obs.shape == self.obs_mean.shape:
                z = np.abs(obs - self.obs_mean) / self.obs_std
                if z.max() >= self.ood_sigma:
                    reasons.append(f"observation_ood:{z.max():.1f}sigma")

        return self._apply_cooldown(reasons)

    def _apply_cooldown(self, reasons: list[str]) -> list[str]:
        """Emit the first occurrence of each kind; count the rest."""
        emitted: list[str] = []
        for reason in reasons:
            kind = reason.split(":", 1)[0]
            last = self._last_emitted.get(kind)
            if last is not None and self._step - last < self.cooldown_steps:
                self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
                continue
            self._last_emitted[kind] = self._step
            emitted.append(reason)
        return emitted

    def drain_suppressed(self) -> dict[str, int]:
        """Counts of occurrences held back since the last call, then reset.

        Reported on the heartbeat. Without this the cooldown would be lossy and
        "one flag" would be indistinguishable from "one flag and four hundred
        more just like it", which is the difference between a blip and an outage.
        """
        counts, self._suppressed = self._suppressed, {}
        return counts


# -- the spool ---------------------------------------------------------------


class Spool:
    """A bounded, restart-surviving event queue with priority-aware eviction."""

    def __init__(self, path: Path | str, capacity: int = 2000) -> None:
        self.path = Path(path)
        self.capacity = capacity
        self.events: deque[TelemetryEvent] = deque()
        self.dropped: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                self.events.append(TelemetryEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                # A partial final line is what a power cut during append looks
                # like. Losing one event beats refusing to start.
                continue

    def _flush(self) -> None:
        write_atomically(self.path, "\n".join(e.to_json() for e in self.events) + "\n")

    def __len__(self) -> int:
        return len(self.events)

    def append(self, event: TelemetryEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.capacity:
            self._evict()
        self._flush()

    def _evict(self) -> None:
        """Drop the least valuable event: lowest priority class, then oldest.

        Note what this cannot do: if the spool is full of failures, a new failure
        still displaces the oldest failure. The alternative — refusing new
        events — means a device that broke once stops reporting that it is still
        broken.
        """
        while len(self.events) > self.capacity:
            worst_priority = max(e.priority for e in self.events)
            for i, event in enumerate(self.events):
                if event.priority == worst_priority:
                    del self.events[i]
                    self.dropped[event.kind] = self.dropped.get(event.kind, 0) + 1
                    break

    def batch(self, limit: int = 128) -> list[TelemetryEvent]:
        """The next events to send, most important first.

        Importance before recency: on a link that may drop halfway through, the
        first bytes should be the ones worth the most.
        """
        ordered = sorted(self.events, key=lambda e: (e.priority, e.ts))
        return ordered[:limit]

    def acknowledge(self, event_ids: Iterable[str]) -> int:
        """Remove events the hub confirmed it stored. Nothing else removes them."""
        ids = set(event_ids)
        before = len(self.events)
        self.events = deque(e for e in self.events if e.event_id not in ids)
        self._flush()
        return before - len(self.events)

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for event in self.events:
            by_kind[event.kind] = by_kind.get(event.kind, 0) + 1
        return {
            "pending": len(self.events),
            "capacity": self.capacity,
            "by_kind": by_kind,
            "dropped": dict(self.dropped),
        }


# -- the client --------------------------------------------------------------


class Uploader(Protocol):
    """Whatever moves events off the device.

    A protocol rather than a class because the transport is the part that
    changes per deployment — HTTP here, MQTT or a USB stick elsewhere — and none
    of the buffering, prioritisation or acknowledgement logic above should have
    to change with it.
    """

    def send(self, events: list[TelemetryEvent]) -> list[str]:
        """Send events; return the ids the receiver has durably stored."""
        ...


@dataclass
class TelemetryClient:
    """Records events locally and drains them upstream when it can."""

    device_id: str
    spool: Spool
    uploader: Uploader | None = None
    batch_size: int = 128

    policy_version: int = 0
    consecutive_failures: int = field(default=0, init=False)

    def record(self, kind: EventKind, payload: dict) -> TelemetryEvent:
        event = TelemetryEvent(
            kind=kind,
            device_id=self.device_id,
            policy_version=self.policy_version,
            payload=payload,
        )
        self.spool.append(event)
        return event

    def heartbeat(self, stats: LatencyStats, extra: dict | None = None) -> TelemetryEvent:
        return self.record("heartbeat", {"latency": stats.as_dict(), **(extra or {})})

    def flag(
        self, reasons: list[str], observation: np.ndarray, record: StepRecord
    ) -> TelemetryEvent:
        """Record a flagged step, with enough context to act on and no more.

        The observation is included because a flag without the input that caused
        it is an alert nobody can investigate. Full trajectories are the fleet
        loop's job, and they cost far more bandwidth than this.
        """
        return self.record(
            "flag",
            {
                "reasons": reasons,
                "observation": [round(float(v), 5) for v in np.asarray(observation).ravel()],
                "action": [round(float(v), 5) for v in np.asarray(record.action).ravel()],
                "latency_ms": round(record.latency_ms, 3),
            },
        )

    def failure(self, record: StepRecord) -> TelemetryEvent:
        return self.record(
            "failure",
            {"error": record.error, "latency_ms": round(record.latency_ms, 3)},
        )

    def flush(self) -> dict:
        """Attempt one upload. Never raises: the control loop outranks the link."""
        if self.uploader is None or not self.spool.events:
            return {"sent": 0, "pending": len(self.spool), "ok": True}

        batch = self.spool.batch(self.batch_size)
        try:
            acknowledged = self.uploader.send(batch)
        except Exception as exc:  # noqa: BLE001 - a dead hub must not stop the robot
            self.consecutive_failures += 1
            return {
                "sent": 0,
                "pending": len(self.spool),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "consecutive_failures": self.consecutive_failures,
            }

        self.consecutive_failures = 0
        removed = self.spool.acknowledge(acknowledged)
        return {"sent": removed, "pending": len(self.spool), "ok": True}
