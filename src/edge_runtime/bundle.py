"""The deployable unit: a signed, versioned, self-describing policy bundle.

A bundle is the only thing that crosses the wire to a device. It contains the
policy, the probe set the device uses to decide whether the policy works, and a
manifest that describes both.

**The chain of custody is: signature over manifest, manifest over file hashes.**
One signature therefore covers every byte in the bundle. Verifying a bundle
means checking the signature, then checking that each file still hashes to what
the manifest says. Swapping `policy.npz` for a different one invalidates the
bundle even though the signature itself was never touched.

**Versions are monotonic integers, not semver strings.** The device's only
question is "is this newer than what I am running", and integers answer it
without a parser, a comparison operator, or an argument about pre-release
ordering. What the version *means* lives in `notes` and in the hub's release
log, where humans read it.

**The signature is HMAC-SHA256 over a shared secret, and that is a real
limitation.** Any party who can verify a release can also forge one, so a
compromised device compromises the fleet. Production wants asymmetric signing:
the hub holds a private key, devices hold only the public half. That is a
change to :func:`_sign` and :func:`_verify_signature` and nothing else — the
chain of custody above does not move — and it is left undone deliberately
rather than half-done, so the boundary is honest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .policy import Policy

SCHEMA_VERSION = 1
RUNTIME_VERSION = 1

# A default only so the demos run out of the box. A device that ships with the
# default key is a device with no signature check at all.
DEFAULT_KEY_ENV = "EDGE_SIGNING_KEY"
_DEMO_KEY = b"edge-runtime-demo-key-not-for-production"


class BundleError(RuntimeError):
    pass


class FileEntry(BaseModel):
    sha256: str
    bytes: int


class HealthSpec(BaseModel):
    """What the device must confirm before it will run this policy.

    These live in the bundle rather than on the device because the acceptable
    latency and the acceptable action deviation are properties of *the release*,
    and a device that hard-codes them cannot be sent a policy with a different
    envelope without being reflashed — which is the thing this repo exists to
    avoid.
    """

    latency_budget_ms: float = 20.0
    # Fraction of probe steps allowed to miss the budget during the health run.
    max_deadline_miss_rate: float = 0.02
    # Max mean absolute deviation from the probe's recorded actions, in radians.
    # `inf` means the probe checks liveness and latency only, not behaviour.
    max_action_deviation: float = float("inf")
    probe_file: str | None = "probe.npz"


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    version: int
    policy_id: str
    policy_name: str
    created_at: str
    channel: str = "stable"

    runtime_min_version: int = RUNTIME_VERSION
    obs_columns: list[str]
    action_columns: list[str]
    quantization: str = "none"

    files: dict[str, FileEntry] = Field(default_factory=dict)
    health: HealthSpec = Field(default_factory=HealthSpec)

    # Provenance, carried down from the training pipeline and the eval harness.
    # A device should be able to say which dataset produced the policy it is
    # currently running, without anyone having to reconstruct it from timestamps.
    source_checkpoint_id: str | None = None
    dataset_hash: str | None = None
    eval_success_rate: float | None = None
    notes: str = ""

    def canonical_bytes(self) -> bytes:
        """The exact bytes that get signed.

        Sorted keys and no whitespace, so that re-serialising a manifest on a
        different Python version cannot invalidate a signature.
        """
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    def provenance_gaps(self) -> list[str]:
        gaps = []
        if not self.source_checkpoint_id:
            gaps.append("no source checkpoint")
        if not self.dataset_hash:
            gaps.append("no dataset_hash")
        if self.eval_success_rate is None:
            gaps.append("never evaluated closed-loop")
        return gaps


# -- signing -----------------------------------------------------------------


def signing_key() -> bytes:
    key = os.environ.get(DEFAULT_KEY_ENV)
    return key.encode() if key else _DEMO_KEY


def _sign(payload: bytes, key: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _verify_signature(payload: bytes, signature: str, key: bytes) -> bool:
    # compare_digest, not `==`: signature comparison is the one place a timing
    # side-channel is worth the two extra words.
    return hmac.compare_digest(_sign(payload, key), signature)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# -- building ----------------------------------------------------------------


def build(
    policy: Policy,
    *,
    version: int,
    out_dir: Path | str,
    probe: np.ndarray | None = None,
    health: HealthSpec | None = None,
    channel: str = "stable",
    quantization: str = "none",
    source_checkpoint_id: str | None = None,
    dataset_hash: str | None = None,
    eval_success_rate: float | None = None,
    notes: str = "",
    key: bytes | None = None,
) -> Path:
    """Write and sign a bundle directory. Returns its path.

    `probe` is the observation set the device replays to decide whether the
    policy is healthy. The *expected* actions are recorded from this policy at
    build time, so the health gate on the device is comparing the policy against
    what the hub saw it do — which catches a corrupted download, a quantization
    step that went wrong, and a runtime that is subtly not the one the bundle was
    built for.
    """
    if version < 1:
        raise BundleError("versions start at 1")
    out_dir = Path(out_dir) / str(version)
    if out_dir.exists():
        raise BundleError(f"{out_dir} already exists; versions are immutable once published")
    out_dir.mkdir(parents=True)

    policy.save(out_dir / "policy.npz")

    if probe is not None:
        probe = np.atleast_2d(np.asarray(probe, dtype=np.float32))
        np.savez(
            out_dir / "probe.npz",
            observations=probe,
            expected_actions=policy.forward(probe).astype(np.float32),
        )

    spec = health or HealthSpec()
    if probe is None:
        spec = spec.model_copy(update={"probe_file": None})

    manifest = Manifest(
        version=version,
        policy_id=policy.content_hash(),
        policy_name=policy.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        channel=channel,
        obs_columns=policy.obs_columns,
        action_columns=policy.action_columns,
        quantization=quantization,
        health=spec,
        source_checkpoint_id=source_checkpoint_id,
        dataset_hash=dataset_hash,
        eval_success_rate=eval_success_rate,
        notes=notes,
    )
    manifest.files = {
        p.name: FileEntry(sha256=sha256_file(p), bytes=p.stat().st_size)
        for p in sorted(out_dir.iterdir())
        if p.name not in {"manifest.json", "manifest.sig"}
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (out_dir / "manifest.sig").write_text(
        _sign(manifest.canonical_bytes(), key or signing_key()), encoding="utf-8"
    )
    return out_dir


def verify(bundle_dir: Path | str, *, key: bytes | None = None) -> Manifest:
    """Check signature, then file hashes, then runtime compatibility.

    Order matters. Hashing an unsigned bundle's files tells you only that it is
    internally consistent, which a forger gets for free.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    sig_path = bundle_dir / "manifest.sig"
    if not manifest_path.exists() or not sig_path.exists():
        raise BundleError(f"{bundle_dir} is not a bundle: manifest or signature missing")

    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if not _verify_signature(
        manifest.canonical_bytes(),
        sig_path.read_text(encoding="utf-8").strip(),
        key or signing_key(),
    ):
        raise BundleError(
            f"bundle v{manifest.version} failed signature verification — "
            "it was not produced by the holder of this fleet's signing key, or it "
            "was modified after signing"
        )

    if manifest.schema_version > SCHEMA_VERSION:
        raise BundleError(
            f"bundle uses manifest schema v{manifest.schema_version}; this runtime "
            f"understands v{SCHEMA_VERSION}. Update the device before publishing it."
        )
    if manifest.runtime_min_version > RUNTIME_VERSION:
        raise BundleError(
            f"bundle v{manifest.version} needs runtime >= v{manifest.runtime_min_version}, "
            f"this device has v{RUNTIME_VERSION}"
        )

    for name, entry in manifest.files.items():
        path = bundle_dir / name
        if not path.exists():
            raise BundleError(f"bundle v{manifest.version} is missing {name}")
        actual = sha256_file(path)
        if actual != entry.sha256:
            raise BundleError(
                f"{name} does not match the manifest "
                f"(expected {entry.sha256[:12]}, got {actual[:12]})"
            )

    return manifest


def load_policy(bundle_dir: Path | str) -> Policy:
    return Policy.load(Path(bundle_dir) / "policy.npz")


def load_probe(bundle_dir: Path | str) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(bundle_dir) / "probe.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return data["observations"], data["expected_actions"]


# -- transport ---------------------------------------------------------------


def pack(bundle_dir: Path | str, out_path: Path | str) -> Path:
    """Tar+gzip a bundle for transport, with a flat archive layout."""
    bundle_dir = Path(bundle_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for member in sorted(bundle_dir.iterdir()):
            if member.is_file():
                tar.add(member, arcname=member.name)
    return out_path


def unpack(archive: Path | str, dest_dir: Path | str) -> Path:
    """Extract a bundle archive, refusing anything that escapes `dest_dir`.

    The archive arrives from the network. Python's default extraction will
    happily honour `../../etc/`, absolute paths, symlinks and device nodes, and
    "the release server is trusted" stops being true the moment it is not. Only
    flat regular files are accepted; anything else fails the whole extraction
    rather than being skipped, because a bundle that is not what it claims to be
    should not be half-installed.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                raise BundleError(f"archive contains a non-regular member: {member.name!r}")
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
                raise BundleError(f"archive member escapes the bundle directory: {member.name!r}")
        # The explicit checks above are the guarantee; `data` filter is a second
        # layer where the interpreter offers one (3.12+, and backported).
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest_dir, **extra)  # noqa: S202 - every member was checked above
    return dest_dir


def write_atomically(path: Path, data: str) -> None:
    """Write a file such that a power cut leaves either the old or the new one.

    The device state file is written through this. A truncated JSON state file
    on a robot that lost power mid-update is a device that will not boot into
    anything, and it is entirely avoidable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
