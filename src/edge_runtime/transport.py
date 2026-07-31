"""HTTP transport for both links, over the standard library.

Every network call carries a timeout. A device whose telemetry upload blocks
forever on a half-open socket is a device that has silently stopped updating and
stopped reporting, and it will look healthy from every angle except the one that
matters. The default is deliberately short: telemetry is not worth waiting on.

Retries are not here. Whether to retry, and how long to back off, depends on
what else the device is doing — and both callers already hold that context:
:class:`~edge_runtime.telemetry.TelemetryClient` keeps the events until they are
acknowledged, and :meth:`~edge_runtime.ota.OTAClient.poll` is driven by a timer that
is itself the retry interval.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .ota import OTAError
from .telemetry import TelemetryEvent

DEFAULT_TIMEOUT = 10.0


@dataclass
class HttpReleaseSource:
    """Reads release pointers and bundle archives from a release server."""

    base_url: str
    timeout: float = DEFAULT_TIMEOUT

    def latest(self, channel: str) -> dict | None:
        url = f"{self.base_url.rstrip('/')}/releases/{channel}.json"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:  # noqa: S310
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise OTAError(f"release server returned {exc.code} for {url}") from exc
        except OSError as exc:
            # Unreachable hub is normal, not exceptional. The device stays on the
            # version it has and tries again on the next tick.
            raise OTAError(f"cannot reach the release server: {exc}") from exc

    def fetch(self, version: int, dest: Path) -> Path:
        url = f"{self.base_url.rstrip('/')}/bundles/{version}.tar.gz"
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                urllib.request.urlopen(url, timeout=self.timeout) as response,  # noqa: S310
                dest.open("wb") as fh,
            ):
                # Streamed, not read into memory: a bundle is as large as the
                # model in it, and the device has less RAM than the hub.
                shutil.copyfileobj(response, fh)
        except OSError as exc:
            dest.unlink(missing_ok=True)
            raise OTAError(f"download of v{version} failed: {exc}") from exc
        return dest


@dataclass
class HttpTelemetryUploader:
    """Posts a batch of events and returns the ids the hub says it stored.

    The return value is the whole contract: the device deletes exactly what came
    back and keeps everything else. A hub that stores nine of ten events and
    returns nine ids loses nothing — the tenth is sent again.
    """

    base_url: str
    timeout: float = DEFAULT_TIMEOUT

    def send(self, events: list[TelemetryEvent]) -> list[str]:
        if not events:
            return []
        url = f"{self.base_url.rstrip('/')}/telemetry"
        body = json.dumps({"events": [json.loads(e.to_json()) for e in events]}).encode()
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            return list(json.loads(response.read().decode()).get("accepted", []))
