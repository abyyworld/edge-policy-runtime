"""Both links, over real HTTP, against a real device agent.

Everything else in this suite tests a component. This tests the claim: a policy
can be replaced on a running device without touching the device, and the device
reports what happened.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from edge_runtime import bundle as bundle_mod
from edge_runtime.agent import AgentConfig, DeviceAgent, build_calibration
from edge_runtime.bundle import HealthSpec
from edge_runtime.ota import publish
from edge_runtime.policy import default_obs_columns, reference_policy
from edge_runtime.server import serve
from edge_runtime.transport import HttpReleaseSource, HttpTelemetryUploader


def make_bundle(tmp_path, version: int, *, skill: float = 1.0, columns=None):
    policy = reference_policy(obs_columns=columns, seed=version, skill=skill)
    probe = build_calibration(reference_policy(seed=version, skill=skill), n=64)
    if probe.shape[1] != policy.obs_dim:
        probe = np.hstack(
            [probe, np.zeros((len(probe), policy.obs_dim - probe.shape[1]), np.float32)]
        )
    return bundle_mod.build(
        policy,
        version=version,
        out_dir=tmp_path / "build",
        probe=probe,
        health=HealthSpec(max_action_deviation=1e-4),
        notes=f"v{version}",
    )


@pytest.fixture
def hub(tmp_path):
    releases = tmp_path / "releases"
    releases.mkdir()
    server, store = serve(releases, port=0)
    yield f"http://127.0.0.1:{server.server_address[1]}", releases, store, tmp_path
    server.shutdown()


@pytest.fixture
def device(hub, tmp_path):
    url, releases, store, _ = hub
    publish(make_bundle(tmp_path, 1, skill=0.6), releases)

    agent = DeviceAgent(
        AgentConfig(device_id="jetson-a1", root=tmp_path / "dev", ota_every=200),
        source=HttpReleaseSource(url),
        uploader=HttpTelemetryUploader(url),
    )
    agent.install_local(tmp_path / "build" / "1")
    return agent


def test_the_device_boots_on_the_factory_bundle(device):
    assert device.ota.state.current_version == 1
    assert device.runtime is not None


def test_a_policy_is_replaced_mid_run_without_touching_the_device(device, hub, tmp_path):
    url, releases, store, _ = hub
    result = device.run(400)
    assert result["policy_version"] == 1

    publish(make_bundle(tmp_path, 2), releases)
    result = device.run(400)

    assert result["updates_applied"] == 1
    assert result["policy_version"] == 2
    assert (
        device.runtime.policy.content_hash()
        == bundle_mod.load_policy(device.ota.bundle_dir(2)).content_hash()
    ), "the running policy is the one that was verified and activated"


def test_the_hub_learns_which_version_each_device_is_on(device, hub, tmp_path):
    url, releases, store, _ = hub
    device.run(400)
    publish(make_bundle(tmp_path, 2), releases)
    device.run(400)
    device.drain_telemetry()

    events = store.events()
    assert {e["policy_version"] for e in events} == {1, 2}

    ota_events = [e for e in events if e["kind"] == "ota"]
    assert ota_events and ota_events[-1]["payload"]["to_version"] == 2
    assert "latency" in ota_events[-1]["payload"]["health"]


def test_a_rejected_release_is_reported_upstream(device, hub, tmp_path):
    """The hub has to be able to tell "not rolled out yet" from "refused"."""
    url, releases, store, _ = hub
    publish(make_bundle(tmp_path, 2, columns=[*default_obs_columns(), "wrist_force_z"]), releases)

    device.poll_updates()
    device.drain_telemetry()

    rejections = [
        e for e in store.events() if e["kind"] == "ota" and "rejected_version" in e["payload"]
    ]
    assert rejections
    assert "wrist_force_z" in rejections[-1]["payload"]["reason"]
    assert device.ota.state.current_version == 1


def test_telemetry_survives_the_hub_being_down(device, hub, tmp_path):
    """The outage the up-link exists to survive."""
    url, releases, store, _ = hub
    device.telemetry.uploader = HttpTelemetryUploader("http://127.0.0.1:1")  # nothing listening

    device.run(600)
    pending = len(device.telemetry.spool)
    assert pending > 0
    assert store.events() == []

    device.telemetry.uploader = HttpTelemetryUploader(url)
    drained = device.drain_telemetry()
    assert drained["pending"] == 0
    assert len(store.events()) == pending


def test_the_hub_deduplicates_a_replayed_batch(device, hub):
    """At-least-once delivery means the hub will see duplicates."""
    url, releases, store, _ = hub
    device.run(600)
    device.drain_telemetry()
    stored = store.events()
    assert stored, "nothing to replay"

    # A device that lost the response and resent: same ids, same payloads.
    accepted = store.accept([dict(e) for e in stored])
    assert len(accepted) == len(stored), "the device must be told it can drop them"
    assert len(store.events()) == len(stored), "but nothing was stored twice"


def test_a_malformed_batch_is_not_acknowledged(hub):
    """Acknowledging what you did not store turns at-least-once into at-most-once."""
    url, _, store, _ = hub
    request = urllib.request.Request(
        f"{url}/telemetry",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 400
    assert store.events() == []


def test_the_release_endpoint_refuses_a_traversal_attempt(hub):
    url, _, _, _ = hub
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{url}/bundles/..%2f..%2fetc%2fpasswd.tar.gz", timeout=5)
    assert exc.value.code == 404


def test_an_unknown_channel_is_a_404_not_a_crash(hub):
    url, _, _, _ = hub
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{url}/releases/nightly.json", timeout=5)
    assert exc.value.code == 404


def test_the_device_snapshot_is_what_a_dashboard_would_render(device, hub, tmp_path):
    url, releases, store, _ = hub
    publish(make_bundle(tmp_path, 2), releases)
    device.run(400)

    snapshot = device.snapshot()
    assert snapshot["current_version"] == 2
    assert snapshot["latency"]["n"] > 0
    assert 0 <= snapshot["cohort"] < 100
    assert json.dumps(snapshot)  # a dashboard has to be able to serialise it
