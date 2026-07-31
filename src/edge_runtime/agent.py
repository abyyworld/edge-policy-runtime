"""The device agent: one process holding the control loop and both links.

The ordering here is the design. The control step runs first and unconditionally;
telemetry, and the OTA poll, are things that happen *between* control steps and
are allowed to fail. Nothing on the network path can extend a control step, and
nothing on the network path can raise into one.

**Telemetry upload backs off exponentially and the control loop does not care.**
A hub that has been down for an hour should be costing the device one failed
connection every few minutes, not one per tick. The spool is what makes that
safe: backing off loses nothing, it just delays.

**An activated update is itself a telemetry event.** The version a device is
running, when it changed, whether the health gate had anything to say — that is
the join between the down-link and the up-link, and without it a fleet
dashboard can show you which devices are unhealthy but not whether the release
is why.

**The policy is reloaded from the activated bundle, not from the staged one.**
They should be the same bytes. If they ever are not, the device should run what
it verified and pointed `current` at, not what it happened to have in memory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bundle as bundle_mod
from .ota import OTAClient, ReleaseSource
from .policy import Policy, default_obs_columns
from .runtime import InferenceRuntime
from .telemetry import EdgeCaseDetector, Spool, TelemetryClient, Uploader
from .workload import ReachWorkload


def stable_seed(identifier: str) -> int:
    """A seed derived from a string that is the same in every process."""
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big")


@dataclass
class AgentConfig:
    device_id: str
    root: Path
    channel: str = "stable"
    budget_ms: float = 20.0

    heartbeat_every: int = 200
    ota_every: int = 500
    upload_every: int = 100
    spool_capacity: int = 2000
    max_backoff_multiplier: int = 32


@dataclass
class DeviceAgent:
    config: AgentConfig
    source: ReleaseSource | None = None
    uploader: Uploader | None = None
    workload: ReachWorkload | None = None

    ota: OTAClient = field(init=False)
    telemetry: TelemetryClient = field(init=False)
    runtime: InferenceRuntime | None = field(default=None, init=False)
    detector: EdgeCaseDetector = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.config.root)
        # sha256, not the builtin hash(): string hashing is salted per process,
        # so a device would draw a different workload on every restart and the
        # demo would print different numbers every run. Anything that seeds
        # behaviour from an identifier needs a hash that is stable across
        # processes, and this is the cheapest place to get that wrong.
        self.workload = self.workload or ReachWorkload(seed=stable_seed(self.config.device_id))
        self.obs_columns = list(self.workload.obs_columns or default_obs_columns())

        self.ota = OTAClient(
            root / "ota",
            self.config.device_id,
            self.source,
            channel=self.config.channel,
            device_obs_columns=self.obs_columns,
        )
        self.telemetry = TelemetryClient(
            device_id=self.config.device_id,
            spool=Spool(root / "spool.jsonl", capacity=self.config.spool_capacity),
            uploader=self.uploader,
            policy_version=self.ota.state.current_version,
        )
        self.detector = EdgeCaseDetector()
        self._steps_since_upload = 0
        self._load_current()

    # -- policy lifecycle ----------------------------------------------------

    def _load_current(self) -> None:
        policy = self.ota.current_policy()
        if policy is None:
            self.runtime = None
            return
        self.runtime = InferenceRuntime(policy, budget_ms=self.config.budget_ms)
        self.runtime.warmup()
        self.telemetry.policy_version = self.ota.state.current_version
        self._calibrate_detector(policy)

    def _calibrate_detector(self, policy: Policy) -> None:
        """Refit the out-of-distribution detector to the new policy's own states.

        A new policy visits different states, so a detector calibrated on the
        old one's rollouts will fire constantly for the first hour after every
        update — which trains operators to ignore it.
        """
        # On a private copy of the workload: calibration must not advance the
        # episode the control loop is in the middle of.
        assert self.workload is not None
        scratch = ReachWorkload(seed=self.workload.seed + 991)
        probe = scratch.sample_observations(512, policy_forward=policy.forward)
        self.detector = EdgeCaseDetector.calibrated_on(probe)

    def install_local(self, bundle_dir: Path | str, version: int | None = None) -> int:
        """Install a bundle from local storage — the factory image.

        A device has to boot into *something*. It arrives with a bundle already
        on disk, verified and activated by this path, and only then does the OTA
        client have a version to compare an advertised release against.
        """
        bundle_dir = Path(bundle_dir)
        manifest = bundle_mod.verify(bundle_dir)
        version = version or manifest.version
        dest = self.ota.bundle_dir(version)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.mkdir()
            for item in bundle_dir.iterdir():
                if item.is_file():
                    dest.joinpath(item.name).write_bytes(item.read_bytes())
        result = self.ota.activate(version)
        if not result.passed:
            raise RuntimeError(
                f"factory bundle v{version} failed the health gate: {result.summary()}"
            )
        self._load_current()
        return version

    # -- the loop ------------------------------------------------------------

    def poll_updates(self) -> dict:
        before = self.ota.state.current_version
        outcome = self.ota.poll()
        if outcome.get("updated"):
            self._load_current()
            self.telemetry.record(
                "ota",
                {
                    "from_version": before,
                    "to_version": self.ota.state.current_version,
                    "health": outcome.get("health", {}),
                },
            )
        elif outcome.get("candidate") and not outcome.get("updated"):
            self.telemetry.record(
                "ota",
                {
                    "from_version": before,
                    "rejected_version": outcome.get("candidate"),
                    "reason": outcome.get("reason", ""),
                },
            )
        return outcome

    def maybe_upload(self) -> dict | None:
        """Upload on schedule, backing off after consecutive failures."""
        self._steps_since_upload += 1
        multiplier = min(2**self.telemetry.consecutive_failures, self.config.max_backoff_multiplier)
        if self._steps_since_upload < self.config.upload_every * multiplier:
            return None
        self._steps_since_upload = 0
        return self.telemetry.flush()

    def drain_telemetry(self, max_batches: int = 64) -> dict:
        """Keep uploading until the spool is empty, the link fails, or we stop
        making progress.

        One batch per opportunity is right inside the control loop, where the
        budget is a few milliseconds. At the end of a run — or when a device
        comes back after an outage with thousands of events queued — draining is
        what actually clears the backlog. Bounded, because "until empty" against
        a hub that acknowledges nothing is a loop that never returns.
        """
        total = 0
        for _ in range(max_batches):
            outcome = self.telemetry.flush()
            if not outcome.get("ok") or not outcome.get("sent"):
                break
            total += outcome["sent"]
        return {"sent": total, "pending": len(self.telemetry.spool)}

    def run(self, steps: int) -> dict:
        """Run `steps` control steps with both links live."""
        if self.runtime is None:
            raise RuntimeError(
                f"device {self.config.device_id} has no active policy — "
                "install a factory bundle before running"
            )
        assert self.workload is not None

        flagged = 0
        updates = 0
        obs = self.workload.observe()

        for i in range(steps):
            record = self.runtime.step(obs)

            reasons = self.detector.inspect(obs, record)
            if record.failed:
                self.telemetry.failure(record)
                flagged += 1
            elif reasons:
                self.telemetry.flag(reasons, obs, record)
                flagged += 1

            obs, _ = self.workload.step(record.action)

            if (i + 1) % self.config.heartbeat_every == 0:
                self.telemetry.heartbeat(
                    self.runtime.stats(),
                    extra={
                        "spool": self.telemetry.spool.stats(),
                        "step": i + 1,
                        "suppressed_flags": self.detector.drain_suppressed(),
                    },
                )
            if self.config.ota_every and (i + 1) % self.config.ota_every == 0:
                outcome = self.poll_updates()
                updates += int(bool(outcome.get("updated")))
                if outcome.get("updated"):
                    obs = self.workload.reset()
            self.maybe_upload()

        stats = self.runtime.stats()
        self.telemetry.heartbeat(
            stats, extra={"final": True, "suppressed_flags": self.detector.drain_suppressed()}
        )
        self.drain_telemetry()

        return {
            "device_id": self.config.device_id,
            "steps": steps,
            "policy_version": self.ota.state.current_version,
            "flagged": flagged,
            "updates_applied": updates,
            "latency": stats.as_dict(),
            "spool": self.telemetry.spool.stats(),
        }

    def snapshot(self) -> dict:
        """What a fleet dashboard would ask this device for."""
        stats = self.runtime.stats() if self.runtime else None
        return {
            **self.ota.status(),
            "latency": stats.as_dict() if stats else None,
            "spool": self.telemetry.spool.stats(),
            "telemetry_link_failures": self.telemetry.consecutive_failures,
        }


def build_calibration(policy: Policy, n: int = 512, seed: int = 0) -> np.ndarray:
    """Observations under `policy`, for quantization calibration and probe sets."""
    return ReachWorkload(seed=seed).sample_observations(n, policy_forward=policy.forward)
