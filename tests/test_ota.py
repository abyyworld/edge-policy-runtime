"""Tests for the down-link.

The properties asserted here are the ones that decide whether an update is
something you can do to a robot in the field or something you have to drive to
a robot to do.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from edge_runtime import bundle as bundle_mod
from edge_runtime.bundle import BundleError, HealthSpec
from edge_runtime.ota import (
    DirectoryReleaseSource,
    OTAClient,
    OTAError,
    health_check,
    in_rollout,
    publish,
)
from edge_runtime.policy import default_obs_columns, reference_policy

KEY = None  # the demo key; these tests are about the protocol, not the secret


def make_bundle(tmp_path, version: int, *, policy=None, health=None, columns=None, **kwargs):
    policy = policy or reference_policy(obs_columns=columns, seed=version)
    probe = np.random.default_rng(version).normal(0, 0.4, (64, policy.obs_dim)).astype(np.float32)
    return bundle_mod.build(
        policy,
        version=version,
        out_dir=tmp_path / "build",
        probe=probe,
        health=health or HealthSpec(max_action_deviation=1e-4),
        **kwargs,
    )


@pytest.fixture
def fleet(tmp_path):
    """A release root with v1 published, and a device booted on it."""
    releases = tmp_path / "releases"
    publish(make_bundle(tmp_path, 1), releases)

    client = OTAClient(
        tmp_path / "device",
        "dev-a",
        DirectoryReleaseSource(releases),
        device_obs_columns=default_obs_columns(),
    )
    # Boot the device onto v1 the way a factory image would.
    pointer = client.check()
    client.stage(pointer)
    client.activate(1)
    return client, releases, tmp_path


# -- the happy path ----------------------------------------------------------


def test_a_device_adopts_a_newer_release(fleet):
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)

    outcome = client.poll()
    assert outcome["updated"] is True
    assert client.state.current_version == 2
    assert client.state.previous_version == 1


def test_current_is_a_symlink_that_always_points_somewhere(fleet):
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)
    client.poll()

    assert client.current_link.is_symlink()
    assert (client.current_link / "manifest.json").exists()
    assert json.loads((client.current_link / "manifest.json").read_text())["version"] == 2


def test_the_previous_bundle_is_kept_not_deleted(fleet):
    """Rollback is only cheap if the bytes are still there."""
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)
    client.poll()
    assert client.bundle_dir(1).exists()


def test_polling_when_up_to_date_does_nothing_and_says_why(fleet):
    client, _, _ = fleet
    outcome = client.poll()
    assert outcome["updated"] is False
    assert "already on v1" in outcome["reason"]


# -- refusing bad releases ---------------------------------------------------


def test_a_tampered_archive_is_refused_and_never_activated(fleet):
    """A bad download must not be able to reach the running policy."""
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)

    archive = releases / "2.tar.gz"
    archive.write_bytes(archive.read_bytes()[:-40])  # truncated transfer

    outcome = client.poll()
    assert outcome["updated"] is False
    assert client.state.current_version == 1
    assert "2" in client.state.quarantined


def test_a_release_whose_manifest_disagrees_with_the_pointer_is_refused(fleet):
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)

    pointer = json.loads((releases / "stable.json").read_text())
    pointer["version"] = 3  # advertise a version the signed manifest does not claim
    (releases / "stable.json").write_text(json.dumps(pointer))
    (releases / "3.tar.gz").write_bytes((releases / "2.tar.gz").read_bytes())

    with pytest.raises((OTAError, BundleError)):
        client.stage(client.check())
    assert client.state.current_version == 1


def test_a_release_needing_a_sensor_this_device_lacks_is_rejected(fleet):
    """The failure that otherwise presents as a mysteriously bad policy.

    A model retrained with a wrist force-torque channel is a perfectly good
    model. Activated on a device that does not have that sensor, it receives
    whatever is in that slot — and every symptom points at the model.
    """
    client, releases, tmp_path = fleet
    publish(
        make_bundle(tmp_path, 2, columns=[*default_obs_columns(), "wrist_force_z"]),
        releases,
    )

    outcome = client.poll()
    assert outcome["updated"] is False
    assert "wrist_force_z" in client.state.quarantined["2"]
    assert client.state.current_version == 1


def test_a_release_that_cannot_meet_its_own_latency_budget_is_rejected(fleet):
    """A heterogeneous fleet: the release was validated on faster hardware.

    The budget travels with the release, so the same bundle can be healthy on an
    Orin and rejected on a Nano. The device is the only party that can answer
    the question, and it answers it before the policy touches the control loop.
    """
    client, releases, tmp_path = fleet
    publish(
        make_bundle(
            tmp_path,
            2,
            health=HealthSpec(latency_budget_ms=1e-6, max_deadline_miss_rate=0.0),
        ),
        releases,
    )

    outcome = client.poll()
    assert outcome["updated"] is False
    assert "budget" in client.state.quarantined["2"]
    assert client.state.current_version == 1


# -- quarantine --------------------------------------------------------------


def test_a_rejected_version_is_not_downloaded_again(fleet):
    """The update loop this prevents is a fleet-wide bandwidth event.

    Without quarantine every device re-downloads the same bad release on every
    poll — and they all got it at the same time, so they all do it at the same
    time.
    """
    client, releases, tmp_path = fleet
    publish(
        make_bundle(tmp_path, 2, columns=[*default_obs_columns(), "wrist_force_z"]),
        releases,
    )
    client.poll()

    fetched = []
    original_fetch = client.source.fetch

    def counting_fetch(version, dest):
        fetched.append(version)
        return original_fetch(version, dest)

    client.source.fetch = counting_fetch
    for _ in range(5):
        client.poll()

    assert fetched == [], "a quarantined version was downloaded again"
    assert client.check() is None


def test_quarantine_does_not_block_a_later_good_release(fleet):
    client, releases, tmp_path = fleet
    publish(
        make_bundle(tmp_path, 2, columns=[*default_obs_columns(), "wrist_force_z"]),
        releases,
    )
    client.poll()
    publish(make_bundle(tmp_path, 3), releases)

    assert client.poll()["updated"] is True
    assert client.state.current_version == 3


# -- rollback ----------------------------------------------------------------


def test_rollback_returns_to_the_previous_version_and_quarantines_this_one(fleet):
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)
    client.poll()

    assert client.rollback("success rate collapsed in the field") == 1
    assert client.state.current_version == 1
    assert "collapsed" in client.state.quarantined["2"]
    assert json.loads((client.current_link / "manifest.json").read_text())["version"] == 1


def test_rollback_with_nothing_to_roll_back_to_is_an_error_not_a_brick(fleet):
    client, _, _ = fleet
    with pytest.raises(OTAError, match="only ever run one version"):
        client.rollback()
    assert client.state.current_version == 1


def test_repeated_rollbacks_freeze_updates_for_an_operator(fleet):
    """Cycling is worse than sitting still.

    Three rollbacks in short order is a statement about this device, not about
    the releases. Continuing to auto-update it means a robot that keeps changing
    behaviour while nobody understands why.
    """
    client, releases, tmp_path = fleet
    for version in (2, 3, 4):
        publish(make_bundle(tmp_path, version), releases)
        assert client.poll()["updated"] is True
        client.rollback(f"regression after v{version}")

    assert client.state.updates_frozen is True
    publish(make_bundle(tmp_path, 5), releases)
    assert client.poll()["updated"] is False
    assert "frozen" in client.poll()["reason"]

    client.unfreeze()
    assert client.poll()["updated"] is True


def test_state_survives_a_restart(fleet):
    """Everything above is worthless if it lives only in memory."""
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases)
    client.poll()
    client.rollback("regression")

    restarted = OTAClient(
        tmp_path / "device",
        "dev-a",
        DirectoryReleaseSource(releases),
        device_obs_columns=default_obs_columns(),
    )
    assert restarted.state.current_version == 1
    assert "2" in restarted.state.quarantined
    assert restarted.poll()["updated"] is False


# -- staged rollout ----------------------------------------------------------


def test_rollout_percentage_selects_a_stable_subset_of_the_fleet():
    devices = [f"dev-{i:04d}" for i in range(2000)]
    canary = [d for d in devices if in_rollout(d, version=7, rollout_percent=10)]

    assert 0.08 < len(canary) / len(devices) < 0.12
    # Stable: a device does not drift in and out between polls.
    assert canary == [d for d in devices if in_rollout(d, 7, 10)]
    # Wider rollouts are supersets, so a canary device is never taken backwards.
    wider = {d for d in devices if in_rollout(d, 7, 40)}
    assert set(canary).issubset(wider)


def test_the_canary_group_changes_between_versions():
    """Otherwise the same unlucky devices are the canary for every release."""
    devices = [f"dev-{i:04d}" for i in range(2000)]
    seven = {d for d in devices if in_rollout(d, 7, 10)}
    eight = {d for d in devices if in_rollout(d, 8, 10)}
    overlap = len(seven & eight) / max(len(seven), 1)
    assert overlap < 0.3


def test_a_device_outside_the_rollout_stays_where_it_is(fleet):
    client, releases, tmp_path = fleet
    publish(make_bundle(tmp_path, 2), releases, rollout_percent=1)

    # dev-a is deliberately not in the 1% cohort for v2.
    assert not in_rollout("dev-a", 2, 1)

    outcome = client.poll()
    assert outcome["updated"] is False
    assert "not in the cohort" in outcome["reason"]
    assert client.state.current_version == 1
    assert "2" not in client.state.quarantined, "declining a rollout is not a failure"


# -- the health gate on its own ---------------------------------------------


def test_health_check_passes_a_bundle_built_for_this_device(tmp_path):
    path = make_bundle(tmp_path, 1)
    manifest = bundle_mod.verify(path)
    result = health_check(path, manifest, device_obs_columns=default_obs_columns())
    assert result.passed, result.reasons
    assert result.measurements["latency"]["n"] == 64


def test_health_check_catches_a_bundle_that_does_not_compute_what_it_did(tmp_path):
    """The probe's recorded actions are the build-time ground truth.

    A bundle whose policy no longer reproduces them on this device has been
    corrupted, or is being run by a runtime that is not the one it was built
    for. Either way it is not the thing that was evaluated.
    """
    path = make_bundle(tmp_path, 1)
    manifest = bundle_mod.verify(path)

    with np.load(path / "probe.npz") as data:
        observations, expected = data["observations"], data["expected_actions"]
    np.savez(path / "probe.npz", observations=observations, expected_actions=expected + 0.5)

    result = health_check(path, manifest, device_obs_columns=default_obs_columns())
    assert not result.passed
    assert "does not compute here what it computed at build time" in result.summary()


def test_poll_never_raises_even_when_the_release_source_is_broken(fleet):
    """This runs on a timer next to a control loop."""
    client, _, _ = fleet

    def explode(channel):
        raise OSError("network is down")

    client.source.latest = explode
    outcome = client.poll()
    assert outcome["updated"] is False
    assert "network is down" in outcome["reason"]
    assert client.state.current_version == 1
