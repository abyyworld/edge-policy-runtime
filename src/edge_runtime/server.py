"""A release and telemetry endpoint, in the standard library.

This is the smallest thing that makes both links real rather than simulated: the
device polls actual HTTP, downloads actual bytes, and posts actual events. It is
a development server — single process, no auth, no durability guarantees beyond
an fsync-less append — and the fleet hub in `robot-fleet-loop` is what a real
deployment points at.

It is included because "the OTA client works" is a claim that should be
demonstrable against a server rather than against a mock, and because the
acknowledgement contract is easy to get subtly wrong on the receiving side:
returning ids for events you have not durably stored converts at-least-once
delivery into at-most-once, silently, and the loss only shows up as gaps nobody
is looking for.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 8 << 20


class ReleaseStore:
    """Serves published releases and accepts telemetry, deduplicating by id."""

    def __init__(self, release_root: Path | str, telemetry_path: Path | str | None = None) -> None:
        self.release_root = Path(release_root)
        self.telemetry_path = Path(telemetry_path or self.release_root / "telemetry.jsonl")
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if self.telemetry_path.exists():
            for line in self.telemetry_path.read_text().splitlines():
                try:
                    self._seen.add(json.loads(line)["event_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    def pointer(self, channel: str) -> bytes | None:
        path = self.release_root / f"{channel}.json"
        return path.read_bytes() if path.exists() else None

    def archive(self, version: str) -> bytes | None:
        # `version` arrives from the network; only digits are ever a valid
        # version, and checking that is cheaper than reasoning about what a
        # path separator would do here.
        if not version.isdigit():
            return None
        path = self.release_root / f"{version}.tar.gz"
        return path.read_bytes() if path.exists() else None

    def accept(self, events: list[dict]) -> list[str]:
        """Append events, returning the ids now durably stored.

        Ids already present are returned too. The device is telling us about
        something it still holds; confirming it lets the device drop it, and
        re-confirming a duplicate is exactly what at-least-once delivery needs.
        """
        accepted: list[str] = []
        with self._lock, self.telemetry_path.open("a") as fh:
            for event in events:
                event_id = event.get("event_id")
                if not event_id:
                    continue
                if event_id not in self._seen:
                    fh.write(json.dumps(event, separators=(",", ":")) + "\n")
                    self._seen.add(event_id)
                accepted.append(event_id)
            fh.flush()
        return accepted

    def events(self) -> list[dict]:
        if not self.telemetry_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.telemetry_path.read_text().splitlines()
            if line.strip()
        ]


def make_handler(store: ReleaseStore, quiet: bool = True):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: A002
            if not quiet:
                super().log_message(*args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send(200, b'{"ok":true}', "application/json")
            elif path.startswith("/releases/") and path.endswith(".json"):
                body = store.pointer(path[len("/releases/") : -len(".json")])
                if body is None:
                    self._send(404, b'{"error":"no such channel"}', "application/json")
                else:
                    self._send(200, body, "application/json")
            elif path.startswith("/bundles/") and path.endswith(".tar.gz"):
                body = store.archive(path[len("/bundles/") : -len(".tar.gz")])
                if body is None:
                    self._send(404, b'{"error":"no such version"}', "application/json")
                else:
                    self._send(200, body, "application/gzip")
            else:
                self._send(404, b'{"error":"not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/telemetry":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                self._send(413, b'{"error":"batch too large"}', "application/json")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode())
                accepted = store.accept(payload.get("events", []))
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                # A malformed batch must not be acknowledged: the device keeps
                # it and someone gets to see the bad event rather than a gap.
                self._send(400, b'{"error":"malformed batch"}', "application/json")
                return
            self._send(200, json.dumps({"accepted": accepted}).encode(), "application/json")

    return Handler


def serve(
    release_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8720,
    *,
    quiet: bool = True,
) -> tuple[ThreadingHTTPServer, ReleaseStore]:
    """Start the server on a background thread. Returns (server, store).

    Port 0 asks the OS for a free one, which is how the tests avoid the flake
    that comes from assuming a fixed port is available.
    """
    store = ReleaseStore(release_root)
    server = ThreadingHTTPServer((host, port), make_handler(store, quiet=quiet))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, store
