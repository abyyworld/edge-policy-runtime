"""The down-link: replacing the policy on a running device, without reflashing.

The shape of the problem is that the device is the party that cannot be
supervised. Whatever it does with a new release, it does alone, possibly on a
bad network, possibly moments before losing power, and if it gets it wrong
somebody drives to where the robot is.

So the update is staged, gated, atomic, and reversible:

1. **Check.** Is there a version newer than mine, on my channel, that my rollout
   cohort has been reached by, that I have not already quarantined?
2. **Stage.** Download to a scratch directory, verify the signature and every
   file hash. The running policy has not been touched, and a failure here costs
   nothing but bandwidth.
3. **Gate.** Load the staged policy and prove it runs on *this* device: the
   observation contract resolves against the columns this device actually
   publishes, latency fits the budget the release itself declares, and the probe
   actions match what the hub recorded when it built the bundle.
4. **Activate.** Repoint a symlink with :func:`os.replace`, which is atomic. A
   power cut leaves the device on exactly one of the two versions, never on
   half of each.
5. **Roll back.** The previous bundle is kept, not deleted. Reverting is the
   same atomic repoint in the other direction.

Two things that are easy to leave out and expensive to leave out:

**Quarantine.** A version that failed the gate is recorded as failed. Without
that, the device checks again in sixty seconds, sees the same newest version,
downloads it again, fails again, and does that until someone notices — burning
the bandwidth of every device in the fleet simultaneously, because they all got
the bad release at the same time.

**A rollback budget.** A device that has rolled back repeatedly in a short
window stops updating itself and says so. Something is wrong that is specific to
this device — thermal, storage, a wedged accelerator — and it is a worse outcome
for it to keep cycling than to sit still on a known version and wait for a human.

**What this gate cannot do is tell you a policy is *worse*.** It catches broken,
incompatible and slow. Catching "runs fine, succeeds less often" needs
closed-loop evaluation against the incumbent, which needs episodes, which the
device does not have at activation time. That decision belongs to the fleet — it
is the canary stage in `robot-fleet-loop` — and conflating the two produces a
health check that quietly approves regressions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from . import bundle as bundle_mod
from .bundle import BundleError, Manifest, write_atomically
from .policy import Policy
from .runtime import InferenceRuntime


class OTAError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- rollout cohorts ---------------------------------------------------------


def cohort_of(device_id: str) -> int:
    """A stable 0–99 bucket for a device, from its id alone.

    Staged rollout without the hub tracking per-device state: the release says
    "version 7, 20% rolled out", and each device works out for itself whether it
    is in the first 20%. Deterministic, so a device does not drift in and out of
    the canary group between polls, and salted per version so the same unlucky
    devices are not the canary for every single release.
    """
    digest = hashlib.sha256(device_id.encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100


def in_rollout(device_id: str, version: int, rollout_percent: int) -> bool:
    if rollout_percent >= 100:
        return True
    if rollout_percent <= 0:
        return False
    salted = hashlib.sha256(f"{device_id}:{version}".encode()).digest()
    return int.from_bytes(salted[:4], "big") % 100 < rollout_percent


# -- device state ------------------------------------------------------------


class UpdateRecord(BaseModel):
    ts: str
    version: int
    action: str  # staged | activated | rejected | rolled_back
    detail: str = ""


class DeviceState(BaseModel):
    """Everything the device needs to know after a reboot."""

    device_id: str
    channel: str = "stable"
    current_version: int = 0
    previous_version: int | None = None
    quarantined: dict[str, str] = Field(default_factory=dict)
    history: list[UpdateRecord] = Field(default_factory=list)
    updates_frozen: bool = False

    def recent_rollbacks(self, window: int = 10) -> int:
        return sum(1 for r in self.history[-window:] if r.action == "rolled_back")


# -- release transport -------------------------------------------------------


class ReleaseSource(Protocol):
    """Where releases come from. HTTP, a mounted share, or a USB stick."""

    def latest(self, channel: str) -> dict | None:
        """Release pointer: {version, rollout_percent, sha256, ...} or None."""
        ...

    def fetch(self, version: int, dest: Path) -> Path:
        """Download the bundle archive for `version` to `dest`."""
        ...


@dataclass
class DirectoryReleaseSource:
    """Releases published as `<root>/<channel>.json` plus `<root>/<version>.tar.gz`.

    Used by the tests and the local demo. It is also not a toy: a device that
    updates from a mounted share or a technician's USB stick uses exactly this
    path, and those deployments are common in places with no reliable uplink.
    """

    root: Path

    def latest(self, channel: str) -> dict | None:
        pointer = Path(self.root) / f"{channel}.json"
        if not pointer.exists():
            return None
        return json.loads(pointer.read_text(encoding="utf-8"))

    def fetch(self, version: int, dest: Path) -> Path:
        src = Path(self.root) / f"{version}.tar.gz"
        if not src.exists():
            raise OTAError(f"release {version} is advertised but {src} does not exist")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest


def publish(
    bundle_dir: Path | str,
    release_root: Path | str,
    *,
    rollout_percent: int = 100,
    channel: str | None = None,
) -> dict:
    """Put a built bundle where devices can find it, and point the channel at it.

    The archive is written before the pointer is updated. A device polling in
    the middle of a publish sees the old version, never a pointer to an archive
    that does not exist yet.
    """
    bundle_dir = Path(bundle_dir)
    manifest = bundle_mod.verify(bundle_dir)
    release_root = Path(release_root)
    release_root.mkdir(parents=True, exist_ok=True)

    archive = bundle_mod.pack(bundle_dir, release_root / f"{manifest.version}.tar.gz")
    pointer = {
        "version": manifest.version,
        "channel": channel or manifest.channel,
        "rollout_percent": int(rollout_percent),
        "sha256": bundle_mod.sha256_file(archive),
        "bytes": archive.stat().st_size,
        "policy_id": manifest.policy_id,
        "published_at": _now(),
        "notes": manifest.notes,
    }
    write_atomically(release_root / f"{pointer['channel']}.json", json.dumps(pointer, indent=2))
    return pointer


# -- the health gate ---------------------------------------------------------


@dataclass
class HealthResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    def summary(self) -> str:
        return "healthy" if self.passed else "; ".join(self.reasons)


def health_check(
    bundle_dir: Path | str,
    manifest: Manifest,
    *,
    device_obs_columns: list[str] | None = None,
) -> HealthResult:
    """Prove the staged policy runs acceptably on this device, before it flies."""
    bundle_dir = Path(bundle_dir)
    reasons: list[str] = []
    measurements: dict = {}

    try:
        policy: Policy = bundle_mod.load_policy(bundle_dir)
    except OSError as exc:
        # The host could not read the staged bundle. That is a fault here, not
        # evidence the release is bad, and returning it as a health verdict
        # would quarantine a good release for a local problem and leave the
        # device on its old policy with no indication why. Raise instead.
        raise OTAError(f"could not read the staged bundle for v{manifest.version}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        # A bundle that is present but unreadable as a policy really is a bad
        # release, so this one is a health verdict.
        return HealthResult(False, [f"policy failed to load: {type(exc).__name__}: {exc}"])

    # The contract check, first, because it is the failure that would otherwise
    # look like a mysteriously bad policy. A release that needs a sensor channel
    # this device does not publish must never be activated on it.
    if device_obs_columns is not None:
        missing = [c for c in policy.obs_columns if c not in set(device_obs_columns)]
        if missing:
            reasons.append(
                f"observation contract mismatch: this device does not publish "
                f"{', '.join(missing[:4])}" + (" …" if len(missing) > 4 else "")
            )
            return HealthResult(False, reasons)

    probe = bundle_mod.load_probe(bundle_dir)
    if probe is None:
        observations = np.zeros((64, policy.obs_dim), dtype=np.float32)
        expected = None
    else:
        observations, expected = probe

    runtime = InferenceRuntime(policy, budget_ms=manifest.health.latency_budget_ms)
    runtime.warmup()
    actions = []
    for obs in observations:
        actions.append(runtime.step(obs).action)
    stats = runtime.stats()
    measurements["latency"] = stats.as_dict()

    if stats.failures:
        reasons.append(f"{stats.failures}/{stats.n} probe steps raised")
    if stats.deadline_miss_rate > manifest.health.max_deadline_miss_rate:
        reasons.append(
            f"missed the {manifest.health.latency_budget_ms:g} ms budget on "
            f"{stats.deadline_miss_rate:.1%} of probe steps "
            f"(allowed {manifest.health.max_deadline_miss_rate:.1%}, p99 {stats.p99_ms:.2f} ms)"
        )

    if expected is not None:
        deviation = float(np.abs(np.asarray(actions) - expected).mean())
        measurements["action_deviation"] = deviation
        if deviation > manifest.health.max_action_deviation:
            reasons.append(
                f"probe actions differ from the recorded ones by {deviation:.2e} rad "
                f"(allowed {manifest.health.max_action_deviation:.2e}) — the bundle "
                "does not compute here what it computed at build time"
            )

    return HealthResult(not reasons, reasons, measurements)


# -- the client --------------------------------------------------------------


class OTAClient:
    """The device side of the down-link.

    Layout under `root`:

        state.json          what version is live, what failed, what to avoid
        bundles/<version>/  verified bundles, current and previous
        current -> bundles/<version>
    """

    MAX_ROLLBACKS = 3

    def __init__(
        self,
        root: Path | str,
        device_id: str,
        source: ReleaseSource | None = None,
        *,
        channel: str = "stable",
        device_obs_columns: list[str] | None = None,
        key: bytes | None = None,
    ) -> None:
        self.root = Path(root)
        self.device_id = device_id
        self.source = source
        self.device_obs_columns = device_obs_columns
        self.key = key
        self.bundles = self.root / "bundles"
        self.state_path = self.root / "state.json"
        self.current_link = self.root / "current"

        self.root.mkdir(parents=True, exist_ok=True)
        self.bundles.mkdir(exist_ok=True)
        self.state = self._load_state(channel)

    # -- state ---------------------------------------------------------------

    def _load_state(self, channel: str) -> DeviceState:
        if self.state_path.exists():
            return DeviceState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        return DeviceState(device_id=self.device_id, channel=channel)

    def _save_state(self) -> None:
        write_atomically(self.state_path, self.state.model_dump_json(indent=2))

    def _record(self, version: int, action: str, detail: str = "") -> None:
        self.state.history.append(
            UpdateRecord(ts=_now(), version=version, action=action, detail=detail)
        )
        self.state.history = self.state.history[-100:]
        self._save_state()

    def bundle_dir(self, version: int) -> Path:
        return self.bundles / str(version)

    def current_policy(self) -> Policy | None:
        if self.state.current_version == 0:
            return None
        return bundle_mod.load_policy(self.bundle_dir(self.state.current_version))

    def current_manifest(self) -> Manifest | None:
        if self.state.current_version == 0:
            return None
        path = self.bundle_dir(self.state.current_version) / "manifest.json"
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))

    def status(self) -> dict:
        return {
            "device_id": self.device_id,
            "channel": self.state.channel,
            "cohort": cohort_of(self.device_id),
            "current_version": self.state.current_version,
            "previous_version": self.state.previous_version,
            "quarantined": self.state.quarantined,
            "updates_frozen": self.state.updates_frozen,
            "recent_rollbacks": self.state.recent_rollbacks(),
        }

    # -- the update path -----------------------------------------------------

    def check(self) -> dict | None:
        """Return the release this device should move to, or None.

        Every reason for declining is a reason a fleet operator will eventually
        have to ask about, so each one is named rather than folded into a bare
        `None`.
        """
        if self.source is None:
            return None
        if self.state.updates_frozen:
            return None

        pointer = self.source.latest(self.state.channel)
        if not pointer:
            return None

        version = int(pointer["version"])
        if version <= self.state.current_version:
            return None
        if str(version) in self.state.quarantined:
            return None
        if not in_rollout(self.device_id, version, int(pointer.get("rollout_percent", 100))):
            return None
        return pointer

    def stage(self, pointer: dict) -> Path:
        """Download and verify into `bundles/<version>`. Does not activate.

        Verification happens in a scratch directory. A bundle that fails is
        removed rather than left where a later run might mistake it for a
        verified one.
        """
        if self.source is None:
            raise OTAError("no release source configured")
        version = int(pointer["version"])
        scratch = Path(tempfile.mkdtemp(dir=self.root, prefix=f".staging-{version}-"))
        try:
            archive = self.source.fetch(version, scratch / "bundle.tar.gz")

            expected_sha = pointer.get("sha256")
            if expected_sha and bundle_mod.sha256_file(archive) != expected_sha:
                raise OTAError(
                    f"downloaded archive for v{version} does not match the "
                    "published hash — truncated transfer, or a substituted file"
                )

            unpacked = bundle_mod.unpack(archive, scratch / "bundle")
            manifest = bundle_mod.verify(unpacked, key=self.key)
            if manifest.version != version:
                raise OTAError(
                    f"release pointer advertises v{version} but the signed manifest "
                    f"says v{manifest.version}"
                )

            dest = self.bundle_dir(version)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(unpacked), str(dest))
        except (OTAError, BundleError) as exc:
            self.quarantine(version, f"staging failed: {exc}")
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        self._record(version, "staged")
        return dest

    def activate(self, version: int) -> HealthResult:
        """Gate, then atomically swap. Rejects quarantine on failure."""
        bundle_dir = self.bundle_dir(version)
        if not bundle_dir.exists():
            raise OTAError(f"v{version} is not staged")

        manifest = bundle_mod.verify(bundle_dir, key=self.key)
        result = health_check(bundle_dir, manifest, device_obs_columns=self.device_obs_columns)
        if not result.passed:
            self.quarantine(version, result.summary())
            self._record(version, "rejected", result.summary())
            return result

        self._point_current_at(version)
        self.state.previous_version = (
            self.state.current_version if self.state.current_version else None
        )
        self.state.current_version = version
        self._record(version, "activated", f"from v{self.state.previous_version or 0}")
        return result

    def _point_current_at(self, version: int) -> None:
        """Repoint `current` atomically.

        `os.symlink` then `os.replace`, rather than unlink-then-link: the second
        form has a window in which `current` does not exist, and a device that
        reboots inside that window comes up with no policy at all.
        """
        target = Path("bundles") / str(version)
        tmp = self.root / f".current.{os.getpid()}.{version}"
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(target, tmp)
        os.replace(tmp, self.current_link)

    def rollback(self, reason: str = "") -> int:
        """Return to the previous version and quarantine the current one."""
        if self.state.previous_version is None:
            raise OTAError("nothing to roll back to — this device has only ever run one version")

        failed = self.state.current_version
        target = self.state.previous_version
        self.quarantine(failed, reason or "rolled back")
        self._point_current_at(target)
        self.state.current_version = target
        self.state.previous_version = None
        self._record(failed, "rolled_back", reason)

        if self.state.recent_rollbacks() >= self.MAX_ROLLBACKS:
            # Repeated rollbacks are a property of this device, not of the
            # releases. Stop guessing and hand it to a person.
            self.state.updates_frozen = True
            self._save_state()
        return target

    def quarantine(self, version: int, reason: str) -> None:
        self.state.quarantined[str(version)] = reason
        self._save_state()

    def unfreeze(self, clear_quarantine: bool = False) -> None:
        """Operator action after investigating. Not something the device does."""
        self.state.updates_frozen = False
        if clear_quarantine:
            self.state.quarantined.clear()
        self._save_state()

    # -- the whole thing -----------------------------------------------------

    def poll(self) -> dict:
        """One full check → stage → gate → activate cycle. Never raises.

        This runs on a timer next to a control loop. Anything it lets escape
        takes the robot's supervisor process with it.
        """
        outcome: dict = {"checked": True, "updated": False, "version": self.state.current_version}
        try:
            pointer = self.check()
            if pointer is None:
                outcome["reason"] = self._why_not_updating()
                return outcome

            version = int(pointer["version"])
            outcome["candidate"] = version
            self.stage(pointer)
            result = self.activate(version)
            outcome["updated"] = result.passed
            outcome["version"] = self.state.current_version
            outcome["health"] = result.measurements
            if not result.passed:
                outcome["reason"] = f"v{version} rejected by health gate: {result.summary()}"
        except (OTAError, BundleError) as exc:
            outcome["reason"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            outcome["reason"] = f"unexpected: {type(exc).__name__}: {exc}"
        return outcome

    def _why_not_updating(self) -> str:
        if self.state.updates_frozen:
            return "updates frozen after repeated rollbacks; needs an operator"
        if self.source is None:
            return "no release source configured"
        pointer = self.source.latest(self.state.channel)
        if not pointer:
            return f"no release published on channel {self.state.channel!r}"
        version = int(pointer["version"])
        if version <= self.state.current_version:
            return f"already on v{self.state.current_version}"
        if str(version) in self.state.quarantined:
            return f"v{version} is quarantined: {self.state.quarantined[str(version)]}"
        percent = int(pointer.get("rollout_percent", 100))
        if not in_rollout(self.device_id, version, percent):
            return f"v{version} is at {percent}% rollout; this device is not in the cohort yet"
        return "up to date"
