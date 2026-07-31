"""Tests for the thing that decides whether a device will run a set of bytes."""

from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from edge_runtime import bundle as bundle_mod
from edge_runtime.bundle import BundleError, HealthSpec, Manifest
from edge_runtime.policy import reference_policy

KEY = b"test-fleet-key"
OTHER_KEY = b"someone-elses-key"


@pytest.fixture
def built(tmp_path):
    policy = reference_policy(seed=3)
    observations = np.random.default_rng(0).normal(0, 0.4, (64, policy.obs_dim)).astype(np.float32)
    return bundle_mod.build(
        policy, version=1, out_dir=tmp_path / "build", probe=observations, key=KEY
    )


def test_a_clean_bundle_verifies(built):
    manifest = bundle_mod.verify(built, key=KEY)
    assert manifest.version == 1
    assert set(manifest.files) == {"policy.npz", "probe.npz"}


def test_modifying_a_file_invalidates_the_bundle(built):
    """The signature covers the manifest; the manifest covers the file hashes.

    So swapping the policy for a different one is caught even though the
    signature itself was never touched — which is the whole reason the manifest
    carries hashes rather than just a version number.
    """
    reference_policy(seed=99).save(built / "policy.npz")
    with pytest.raises(BundleError, match="does not match the manifest"):
        bundle_mod.verify(built, key=KEY)


def test_modifying_the_manifest_invalidates_the_signature(built):
    manifest = json.loads((built / "manifest.json").read_text())
    manifest["health"]["latency_budget_ms"] = 10_000.0
    (built / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="signature verification"):
        bundle_mod.verify(built, key=KEY)


def test_a_bundle_signed_with_another_key_is_refused(built):
    with pytest.raises(BundleError, match="signature verification"):
        bundle_mod.verify(built, key=OTHER_KEY)


def test_a_missing_file_is_caught(built):
    (built / "probe.npz").unlink()
    with pytest.raises(BundleError, match="missing probe.npz"):
        bundle_mod.verify(built, key=KEY)


def test_canonical_bytes_are_stable_across_reserialisation(built):
    """Signatures must survive a round-trip through the model.

    If they did not, a device that parsed and re-serialised a manifest — which
    it does, every time it reads one — could invalidate a release it had already
    accepted.
    """
    original = (built / "manifest.json").read_bytes()
    manifest = Manifest.model_validate_json(original)
    reparsed = Manifest.model_validate_json(json.dumps(manifest.model_dump(mode="json")))
    assert manifest.canonical_bytes() == reparsed.canonical_bytes()


def test_versions_are_immutable_once_built(tmp_path):
    policy = reference_policy()
    bundle_mod.build(policy, version=4, out_dir=tmp_path / "b", key=KEY)
    with pytest.raises(BundleError, match="immutable"):
        bundle_mod.build(policy, version=4, out_dir=tmp_path / "b", key=KEY)


def test_a_bundle_needing_a_newer_runtime_is_refused(built):
    manifest = Manifest.model_validate_json((built / "manifest.json").read_text())
    manifest.runtime_min_version = 99
    (built / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json")))
    (built / "manifest.sig").write_text(bundle_mod._sign(manifest.canonical_bytes(), KEY))
    with pytest.raises(BundleError, match="needs runtime >= v99"):
        bundle_mod.verify(built, key=KEY)


def test_provenance_gaps_are_named(built):
    manifest = bundle_mod.verify(built, key=KEY)
    assert "no dataset_hash" in manifest.provenance_gaps()
    assert "never evaluated closed-loop" in manifest.provenance_gaps()


def test_health_spec_travels_with_the_release(tmp_path):
    """The envelope is a property of the release, not of the device.

    A device that hard-codes its latency budget cannot be sent a policy with a
    different one without being reflashed — which is the thing this repo exists
    to avoid.
    """
    path = bundle_mod.build(
        reference_policy(),
        version=1,
        out_dir=tmp_path / "b",
        health=HealthSpec(latency_budget_ms=3.5, max_deadline_miss_rate=0.0),
        key=KEY,
    )
    assert bundle_mod.verify(path, key=KEY).health.latency_budget_ms == 3.5


# -- transport ---------------------------------------------------------------


def test_pack_unpack_round_trips(built, tmp_path):
    archive = bundle_mod.pack(built, tmp_path / "v1.tar.gz")
    out = bundle_mod.unpack(archive, tmp_path / "unpacked")
    assert bundle_mod.verify(out, key=KEY).version == 1


def test_unpack_refuses_a_member_that_escapes_the_directory(tmp_path):
    """The archive arrives from the network, and `..` is a valid tar member name.

    Default extraction honours it. A release server that has been compromised,
    or a proxy that has been, should not be able to write to the device's
    filesystem outside the staging directory.
    """
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("pwned")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../../escaped.txt")

    with pytest.raises(BundleError, match="escapes the bundle directory"):
        bundle_mod.unpack(evil, tmp_path / "dest")
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_unpack_refuses_a_symlink_member(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo("policy.npz")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(BundleError, match="non-regular member"):
        bundle_mod.unpack(evil, tmp_path / "dest")


def test_write_atomically_leaves_no_partial_file(tmp_path):
    target = tmp_path / "state.json"
    bundle_mod.write_atomically(target, '{"a": 1}')
    bundle_mod.write_atomically(target, '{"a": 2}')
    assert json.loads(target.read_text()) == {"a": 2}
    # No temporary files left behind to be mistaken for state later.
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
