"""Tests for the up-link.

The interesting cases are all failure cases: the hub is down, the device has
been offline for a day, the spool is full, the same incident is happening for
the two-hundredth consecutive step.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge_runtime.runtime import InferenceRuntime, StepRecord
from edge_runtime.telemetry import (
    EdgeCaseDetector,
    Spool,
    TelemetryClient,
    TelemetryEvent,
)


def make_event(kind: str, i: int = 0) -> TelemetryEvent:
    return TelemetryEvent(kind=kind, device_id="dev-a", policy_version=1, payload={"i": i})


class RecordingUploader:
    def __init__(self, fail: bool = False, accept: float = 1.0) -> None:
        self.fail = fail
        self.accept = accept
        self.batches: list[list[TelemetryEvent]] = []

    def send(self, events):
        if self.fail:
            raise ConnectionError("hub unreachable")
        self.batches.append(list(events))
        keep = int(len(events) * self.accept)
        return [e.event_id for e in events[:keep]]


# -- the spool ---------------------------------------------------------------


def test_the_spool_survives_a_restart(tmp_path):
    """A device that loses its buffer on reboot loses exactly the events that
    explain the reboot."""
    spool = Spool(tmp_path / "spool.jsonl", capacity=100)
    spool.append(make_event("failure"))
    spool.append(make_event("heartbeat"))

    reopened = Spool(tmp_path / "spool.jsonl", capacity=100)
    assert len(reopened) == 2
    assert {e.kind for e in reopened.events} == {"failure", "heartbeat"}


def test_a_truncated_final_line_does_not_stop_the_device(tmp_path):
    path = tmp_path / "spool.jsonl"
    spool = Spool(path, capacity=100)
    spool.append(make_event("failure"))
    with path.open("a") as fh:
        fh.write('{"kind": "heartbe')  # what a power cut mid-append looks like

    reopened = Spool(path, capacity=100)
    assert len(reopened) == 1


def test_eviction_sheds_heartbeats_before_failures(tmp_path):
    """A plain ring buffer drops precisely backwards.

    Fill a device's spool during an outage and a FIFO queue keeps the last hour
    of "all normal" and discards the failure that started the incident.
    """
    spool = Spool(tmp_path / "spool.jsonl", capacity=10)
    spool.append(make_event("failure"))
    for i in range(50):
        spool.append(make_event("heartbeat", i))

    kinds = [e.kind for e in spool.events]
    assert len(spool) == 10
    assert "failure" in kinds, "the failure was evicted before the heartbeats"
    assert spool.dropped["heartbeat"] == 41


def test_a_fresh_heartbeat_never_displaces_an_old_flag(tmp_path):
    spool = Spool(tmp_path / "spool.jsonl", capacity=5)
    for i in range(5):
        spool.append(make_event("flag", i))
    spool.append(make_event("heartbeat"))

    assert [e.kind for e in spool.events] == ["flag"] * 5
    assert spool.dropped == {"heartbeat": 1}


def test_a_full_spool_of_failures_still_accepts_new_failures(tmp_path):
    """The alternative — refusing new events — means a device that broke once
    stops reporting that it is still broken."""
    spool = Spool(tmp_path / "spool.jsonl", capacity=3)
    for i in range(6):
        spool.append(make_event("failure", i))

    assert len(spool) == 3
    assert [e.payload["i"] for e in spool.events] == [3, 4, 5]


def test_batches_are_ordered_by_importance_then_age(tmp_path):
    """On a link that may drop halfway through, the first bytes should be the
    ones worth the most."""
    spool = Spool(tmp_path / "spool.jsonl", capacity=100)
    spool.append(make_event("heartbeat", 0))
    spool.append(make_event("flag", 1))
    spool.append(make_event("failure", 2))

    assert [e.kind for e in spool.batch()] == ["failure", "flag", "heartbeat"]


# -- delivery ----------------------------------------------------------------


def test_events_are_kept_until_the_hub_acknowledges_them(tmp_path):
    """At-least-once. Anything else silently loses events."""
    spool = Spool(tmp_path / "spool.jsonl", capacity=100)
    uploader = RecordingUploader(fail=True)
    client = TelemetryClient("dev-a", spool, uploader)
    client.record("failure", {})

    outcome = client.flush()
    assert outcome["ok"] is False
    assert outcome["pending"] == 1
    assert client.consecutive_failures == 1

    uploader.fail = False
    assert client.flush()["sent"] == 1
    assert len(spool) == 0
    assert client.consecutive_failures == 0


def test_a_partial_acknowledgement_keeps_the_rest(tmp_path):
    """A hub that stores nine of ten events and returns nine ids loses nothing."""
    spool = Spool(tmp_path / "spool.jsonl", capacity=100)
    client = TelemetryClient("dev-a", spool, RecordingUploader(accept=0.5))
    for i in range(10):
        client.record("flag", {"i": i})

    outcome = client.flush()
    assert outcome["sent"] == 5
    assert len(spool) == 5


def test_flush_never_raises_when_the_hub_is_down(tmp_path):
    """The control loop outranks the link."""
    client = TelemetryClient("dev-a", Spool(tmp_path / "s.jsonl"), RecordingUploader(fail=True))
    client.record("heartbeat", {})
    assert client.flush()["ok"] is False  # and did not raise


def test_flush_with_no_uploader_is_a_no_op(tmp_path):
    client = TelemetryClient("dev-a", Spool(tmp_path / "s.jsonl"))
    client.record("heartbeat", {})
    assert client.flush() == {"sent": 0, "pending": 1, "ok": True}


# -- flagging ----------------------------------------------------------------


def failing_step() -> StepRecord:
    return StepRecord(
        latency_ms=1.0, deadline_missed=False, failed=True, action=np.zeros(8), error="boom"
    )


def clean_step(action=None) -> StepRecord:
    return StepRecord(
        latency_ms=1.0,
        deadline_missed=False,
        failed=False,
        action=np.zeros(8) if action is None else action,
    )


def test_a_routine_step_is_not_flagged():
    detector = EdgeCaseDetector.calibrated_on(np.random.default_rng(0).normal(0, 1, (500, 18)))
    assert detector.inspect(np.zeros(18), clean_step()) == []


def test_an_inference_failure_is_always_flagged():
    detector = EdgeCaseDetector()
    assert detector.inspect(np.zeros(18), failing_step())[0].startswith("inference_failed")


def test_an_out_of_distribution_observation_is_flagged():
    calibration = np.random.default_rng(0).normal(0, 1, (1000, 18))
    detector = EdgeCaseDetector.calibrated_on(calibration, ood_sigma=4.0)

    far = np.zeros(18)
    far[3] = 12.0
    reasons = detector.inspect(far, clean_step())
    assert any(r.startswith("observation_ood") for r in reasons)


def test_a_column_that_never_varied_does_not_make_everything_ood():
    """Zero variance is not infinite sensitivity."""
    calibration = np.random.default_rng(0).normal(0, 1, (500, 18))
    calibration[:, 5] = 0.7  # a constant channel, as a wired-off sensor would be
    detector = EdgeCaseDetector.calibrated_on(calibration)
    assert detector.inspect(calibration[0], clean_step()) == []


def test_saturated_actions_are_flagged():
    detector = EdgeCaseDetector(action_limit=0.15)
    action = np.zeros(8)
    action[2] = 0.2
    assert any(
        r.startswith("action_saturated") for r in detector.inspect(np.zeros(18), clean_step(action))
    )


def test_one_slow_step_is_jitter_but_a_burst_is_an_incident():
    detector = EdgeCaseDetector(burst_length=5)
    slow = StepRecord(latency_ms=99.0, deadline_missed=True, failed=False, action=np.zeros(8))

    for _ in range(4):
        assert detector.inspect(np.zeros(18), slow) == []
    assert any(r.startswith("deadline_burst") for r in detector.inspect(np.zeros(18), slow))


def test_the_burst_counter_resets_on_a_healthy_step():
    detector = EdgeCaseDetector(burst_length=3)
    slow = StepRecord(latency_ms=99.0, deadline_missed=True, failed=False, action=np.zeros(8))
    detector.inspect(np.zeros(18), slow)
    detector.inspect(np.zeros(18), slow)
    detector.inspect(np.zeros(18), clean_step())
    assert detector.inspect(np.zeros(18), slow) == []


def test_a_sustained_incident_produces_one_event_and_a_count():
    """A device that flags a tenth of its steps has not flagged anything.

    The incident is still visible, and its duration is still measurable, at one
    event instead of two hundred.
    """
    detector = EdgeCaseDetector(cooldown_steps=100)
    emitted = sum(bool(detector.inspect(np.zeros(18), failing_step())) for _ in range(200))

    assert emitted == 2  # step 1, then step 101
    assert detector.drain_suppressed() == {"inference_failed": 198}
    assert detector.drain_suppressed() == {}, "draining should reset the counters"


def test_the_cooldown_is_per_reason_not_global():
    """A new kind of problem during an ongoing one is news."""
    detector = EdgeCaseDetector(cooldown_steps=1000, action_limit=0.15)
    detector.inspect(np.zeros(18), failing_step())

    action = np.zeros(8)
    action[0] = 0.9
    reasons = detector.inspect(np.zeros(18), clean_step(action))
    assert any(r.startswith("action_saturated") for r in reasons)


# -- what a flag carries -----------------------------------------------------


def test_a_flag_carries_the_observation_that_caused_it(tmp_path):
    """A flag without its input is an alert nobody can investigate."""
    client = TelemetryClient("dev-a", Spool(tmp_path / "s.jsonl"))
    obs = np.arange(18, dtype=np.float32)
    event = client.flag(["observation_ood:9.1sigma"], obs, clean_step())

    assert event.payload["observation"][:3] == [0.0, 1.0, 2.0]
    assert event.payload["reasons"] == ["observation_ood:9.1sigma"]


def test_events_carry_the_policy_version_that_produced_them(tmp_path):
    """Without this a dashboard can show which devices are unhealthy but not
    whether the release is why."""
    client = TelemetryClient("dev-a", Spool(tmp_path / "s.jsonl"), policy_version=7)
    assert client.record("heartbeat", {}).policy_version == 7


def test_heartbeats_report_the_latency_the_runtime_measured(tmp_path):
    runtime = InferenceRuntime(_tiny_policy(), budget_ms=1e-9)
    for _ in range(20):
        runtime.step(np.zeros(runtime.policy.obs_dim))

    client = TelemetryClient("dev-a", Spool(tmp_path / "s.jsonl"))
    payload = client.heartbeat(runtime.stats()).payload
    assert payload["latency"]["n"] == 20
    assert payload["latency"]["deadline_misses"] == 20


def _tiny_policy():
    from edge_runtime.policy import reference_policy

    return reference_policy(hidden=(8,))


@pytest.mark.parametrize("kind", ["failure", "ota", "flag", "heartbeat"])
def test_every_event_kind_round_trips_through_json(kind, tmp_path):
    spool = Spool(tmp_path / "s.jsonl", capacity=10)
    spool.append(make_event(kind))
    assert Spool(tmp_path / "s.jsonl", capacity=10).events[0].kind == kind
