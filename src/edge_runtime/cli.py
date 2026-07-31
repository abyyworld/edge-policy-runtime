"""Command-line interface."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from . import bundle as bundle_mod
from .agent import AgentConfig, DeviceAgent, build_calibration
from .bundle import BundleError, HealthSpec
from .ota import DirectoryReleaseSource, OTAError, cohort_of, publish
from .policy import Policy, default_obs_columns, reference_policy
from .quantize import JOINT_TOLERANCE_RAD, RATE_CAP_RAD, quantization_report, quantize
from .runtime import benchmark
from .server import ReleaseStore, serve

app = typer.Typer(
    add_completion=False,
    help="Edge policy runtime: signed over-the-air updates and a telemetry up-link.",
    no_args_is_help=True,
)
bundle_app = typer.Typer(help="Build, verify and publish policy bundles.", no_args_is_help=True)
device_app = typer.Typer(help="Operate a device.", no_args_is_help=True)
app.add_typer(bundle_app, name="bundle")
app.add_typer(device_app, name="device")

console = Console()

ReleasesOpt = typer.Option("releases", "--releases", help="Release root directory.")
RootOpt = typer.Option("var/device", "--root", help="Device state directory.")


@app.command()
def version() -> None:
    console.print(f"edge-policy-runtime {__version__}")


def _load_or_build(source: str | None, skill: float, seed: int, hidden: str) -> Policy:
    if source:
        return Policy.load(source)
    dims = tuple(int(x) for x in hidden.split(",") if x)
    return reference_policy(hidden=dims, seed=seed, skill=skill)


# -- optimisation ------------------------------------------------------------


@app.command()
def quantise(
    policy_path: str = typer.Option(
        None, "--policy", help="Policy .npz (reference policy if omitted)."
    ),
    out: str = typer.Option(None, "--out", help="Write the int8 policy here."),
    tolerance: float = typer.Option(
        JOINT_TOLERANCE_RAD, help="Max acceptable joint-delta deviation, in radians."
    ),
    calibration: int = typer.Option(512, help="Calibration observations."),
) -> None:
    """Quantize to int8 and report what it cost in action fidelity.

    Both schemes are measured on the same data, because per-tensor is what you
    get by default and the difference is not small.
    """
    policy = _load_or_build(policy_path, 1.0, 0, "128,128")
    obs = build_calibration(policy, n=calibration)
    reports = quantization_report(policy, obs)

    table = Table(title=f"int8 quantization — {policy.name} ({policy.n_parameters:,} params)")
    table.add_column("scheme")
    table.add_column("size", justify="right")
    table.add_column("joint MAE (rad)", justify="right")
    table.add_column("joint worst (rad)", justify="right")
    table.add_column("gripper worst", justify="right")
    table.add_column("ships?", justify="center")
    for scheme, report in reports.items():
        table.add_row(
            scheme,
            f"{report.compression:.2f}x",
            f"{report.joint_mae_rad:.2e}",
            f"{report.joint_max_abs_rad:.2e}",
            f"{report.gripper_max_abs:.2e}",
            "[green]yes[/]" if report.qualifies(tolerance) else "[red]no[/]",
        )
    console.print(table)
    console.print(
        f"tolerance {tolerance:.2e} rad — 1% of the {RATE_CAP_RAD:g} rad/step rate cap, "
        "judged on the worst case rather than the mean"
    )

    if out:
        path = quantize(policy, scheme="per_channel").save(out)
        console.print(f"wrote {path}")


@app.command()
def bench(
    policy_path: str = typer.Option(None, "--policy", help="Policy .npz."),
    steps: int = typer.Option(2000, help="Control steps to time."),
    budget: float = typer.Option(20.0, "--budget-ms", help="Control-loop deadline, in ms."),
    int8: bool = typer.Option(False, "--int8", help="Benchmark the quantized policy instead."),
) -> None:
    """Time single-observation inference against a deadline."""
    policy = _load_or_build(policy_path, 1.0, 0, "128,128")
    if int8:
        policy = quantize(policy)
    obs = build_calibration(policy, n=min(steps, 1024))
    reps = (steps // len(obs)) + 1
    stats = benchmark(policy, obs.repeat(reps, axis=0)[:steps], budget_ms=budget)
    console.print(f"[bold]{policy.name}[/]  {stats.summary()}")
    if stats.deadline_misses:
        console.print(
            f"[yellow]{stats.deadline_miss_rate:.2%} of steps missed the "
            f"{budget:g} ms budget[/] — the mean would not have told you"
        )


# -- bundles -----------------------------------------------------------------


@bundle_app.command("build")
def bundle_build(
    version_number: int = typer.Argument(..., help="Monotonic release version."),
    out: str = typer.Option("build", "--out", help="Where to write the bundle."),
    policy_path: str = typer.Option(None, "--policy", help="Policy .npz."),
    skill: float = typer.Option(1.0, help="Reference-policy skill (demo releases)."),
    seed: int = typer.Option(0, help="Reference-policy seed."),
    hidden: str = typer.Option("128,128", help="Reference-policy hidden widths."),
    channel: str = typer.Option("stable", help="Release channel."),
    int8: bool = typer.Option(False, "--int8", help="Ship the quantized policy."),
    latency_budget: float = typer.Option(20.0, help="Deadline this release declares, in ms."),
    requires_column: str = typer.Option(
        None, help="Add an observation column the policy needs (to demo a contract mismatch)."
    ),
    notes: str = typer.Option("", help="What changed."),
) -> None:
    """Build and sign a bundle."""
    columns = default_obs_columns()
    if requires_column:
        columns.append(requires_column)
    policy = (
        Policy.load(policy_path)
        if policy_path
        else reference_policy(
            obs_columns=columns,
            hidden=tuple(int(x) for x in hidden.split(",") if x),
            seed=seed,
            skill=skill,
        )
    )
    if int8:
        policy = quantize(policy)

    probe = build_calibration(policy, n=128, seed=seed + 1)
    if probe.shape[1] != policy.obs_dim:
        # A policy needing a column the workload does not produce cannot have a
        # probe recorded from it. Pad, and let the device's contract check be
        # the thing that catches it — which is the point of the demo.
        import numpy as np

        probe = np.hstack(
            [probe, np.zeros((len(probe), policy.obs_dim - probe.shape[1]), np.float32)]
        )

    path = bundle_mod.build(
        policy,
        version=version_number,
        out_dir=out,
        probe=probe,
        health=HealthSpec(latency_budget_ms=latency_budget, max_action_deviation=1e-4),
        channel=channel,
        quantization="int8-per-channel" if int8 else "none",
        notes=notes,
    )
    manifest = bundle_mod.verify(path)
    console.print(f"[green]built[/] v{manifest.version}  policy {manifest.policy_id[:12]}  {path}")
    gaps = manifest.provenance_gaps()
    if gaps:
        console.print(f"[yellow]provenance gaps:[/] {', '.join(gaps)}")


@bundle_app.command("verify")
def bundle_verify(bundle_dir: str = typer.Argument(...)) -> None:
    """Check a bundle's signature and file hashes."""
    try:
        manifest = bundle_mod.verify(bundle_dir)
    except BundleError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]verified[/] v{manifest.version}  policy {manifest.policy_id[:12]}")
    console.print(
        f"  {len(manifest.files)} files, {sum(f.bytes for f in manifest.files.values()):,} bytes"
    )
    console.print(f"  budget {manifest.health.latency_budget_ms:g} ms, channel {manifest.channel}")


@bundle_app.command("publish")
def bundle_publish(
    bundle_dir: str = typer.Argument(...),
    releases: str = ReleasesOpt,
    rollout: int = typer.Option(100, "--rollout", help="Percent of the fleet eligible."),
) -> None:
    """Publish a bundle and point its channel at it."""
    pointer = publish(bundle_dir, releases, rollout_percent=rollout)
    console.print(
        f"[green]published[/] v{pointer['version']} on {pointer['channel']} "
        f"at {pointer['rollout_percent']}% rollout ({pointer['bytes']:,} bytes)"
    )


@app.command()
def serve_releases(
    releases: str = ReleasesOpt,
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8720),
) -> None:
    """Run the development release and telemetry endpoint."""
    server, _ = serve(releases, host, port, quiet=False)
    console.print(f"serving {releases} on http://{host}:{port}  (ctrl-c to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


# -- device ------------------------------------------------------------------


def _agent(root: str, device_id: str, releases: str | None, channel: str = "stable") -> DeviceAgent:
    source = DirectoryReleaseSource(Path(releases)) if releases else None
    return DeviceAgent(
        AgentConfig(device_id=device_id, root=Path(root), channel=channel), source=source
    )


@device_app.command("install")
def device_install(
    bundle_dir: str = typer.Argument(..., help="Factory bundle to boot into."),
    root: str = RootOpt,
    device_id: str = typer.Option("dev-001", "--device-id"),
) -> None:
    """Install the factory bundle a device boots into."""
    agent = _agent(root, device_id, None)
    installed = agent.install_local(bundle_dir)
    console.print(f"[green]installed[/] v{installed} on {device_id}")


@device_app.command("run")
def device_run(
    root: str = RootOpt,
    device_id: str = typer.Option("dev-001", "--device-id"),
    releases: str = ReleasesOpt,
    steps: int = typer.Option(2000),
    ota_every: int = typer.Option(500),
) -> None:
    """Run the control loop with both links live."""
    agent = _agent(root, device_id, releases)
    agent.config.ota_every = ota_every
    result = agent.run(steps)
    console.print_json(json.dumps(result))


@device_app.command("status")
def device_status(
    root: str = RootOpt,
    device_id: str = typer.Option("dev-001", "--device-id"),
    releases: str = ReleasesOpt,
) -> None:
    """What version this device is on, and why it is not on a newer one."""
    agent = _agent(root, device_id, releases)
    status = agent.ota.status()
    table = Table(show_header=False)
    table.add_row("device", f"{device_id}  (cohort {cohort_of(device_id)}/100)")
    table.add_row(
        "version", f"v{status['current_version']}  (previous v{status['previous_version'] or 0})"
    )
    table.add_row("update", agent.ota._why_not_updating())
    table.add_row("quarantined", ", ".join(f"v{k}" for k in status["quarantined"]) or "—")
    table.add_row("frozen", "[red]yes[/]" if status["updates_frozen"] else "no")
    table.add_row("spool", json.dumps(agent.telemetry.spool.stats()))
    console.print(table)


@device_app.command("rollback")
def device_rollback(
    root: str = RootOpt,
    device_id: str = typer.Option("dev-001", "--device-id"),
    reason: str = typer.Option("operator rollback", "--reason"),
) -> None:
    """Return to the previous version and quarantine the current one."""
    agent = _agent(root, device_id, None)
    try:
        target = agent.ota.rollback(reason)
    except OTAError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    console.print(f"[yellow]rolled back[/] to v{target}")


@app.command()
def telemetry(
    releases: str = ReleasesOpt,
    kind: str = typer.Option(None, help="Filter by event kind."),
    limit: int = typer.Option(20),
) -> None:
    """Show what devices have reported."""
    events = ReleaseStore(releases).events()
    if kind:
        events = [e for e in events if e["kind"] == kind]

    counts: dict[str, int] = {}
    for event in events:
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    console.print(
        f"{len(events)} events received: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    table = Table()
    table.add_column("device")
    table.add_column("v", justify="right")
    table.add_column("kind")
    table.add_column("detail", overflow="fold")
    for event in events[-limit:]:
        payload = event.get("payload", {})
        if event["kind"] == "flag":
            detail = ", ".join(payload.get("reasons", []))
        elif event["kind"] == "heartbeat":
            latency = payload.get("latency", {})
            detail = (
                f"p99 {latency.get('p99_ms', 0):.2f}ms, {latency.get('deadline_misses', 0)} misses"
            )
        else:
            detail = json.dumps({k: v for k, v in payload.items() if k != "health"})
        table.add_row(event["device_id"], str(event["policy_version"]), event["kind"], detail[:110])
    console.print(table)


# -- the demo ----------------------------------------------------------------


@app.command()
def demo(
    workdir: str = typer.Option("var/demo", "--workdir"),
    steps: int = typer.Option(1200, help="Control steps per phase."),
) -> None:
    """The whole down-link, end to end, including a release that gets rejected.

    v1 ships. v2 ships and is adopted. v3 needs a sensor this device does not
    have, is rejected by the health gate before it ever runs, and is quarantined
    so the device does not re-download it on every poll.
    """
    work = Path(workdir)
    shutil.rmtree(work, ignore_errors=True)
    releases, build_dir, device_root = work / "releases", work / "build", work / "device"
    device_id = "jetson-a1"

    def line(text: str) -> None:
        console.print(f"[dim]│[/] {text}")

    console.rule("[bold]v1 — the factory image")
    v1 = bundle_mod.build(
        reference_policy(skill=0.6, seed=0),
        version=1,
        out_dir=build_dir,
        probe=build_calibration(reference_policy(skill=0.6, seed=0), n=128),
        health=HealthSpec(latency_budget_ms=20.0, max_action_deviation=1e-4),
        notes="factory image",
    )
    publish(v1, releases)
    agent = DeviceAgent(
        AgentConfig(device_id=device_id, root=device_root, ota_every=400),
        source=DirectoryReleaseSource(releases),
    )
    agent.install_local(v1)
    line(
        f"device {device_id} booted on v{agent.ota.state.current_version}, cohort {cohort_of(device_id)}/100"
    )
    result = agent.run(steps)
    line(
        f"ran {result['steps']} steps, p99 {result['latency']['p99_ms']:.2f} ms, {result['flagged']} flagged"
    )

    console.rule("[bold]v2 — a real update, adopted over the air")
    v2 = bundle_mod.build(
        reference_policy(skill=1.0, seed=1),
        version=2,
        out_dir=build_dir,
        probe=build_calibration(reference_policy(skill=1.0, seed=1), n=128),
        health=HealthSpec(latency_budget_ms=20.0, max_action_deviation=1e-4),
        notes="retrained on fleet data",
    )
    publish(v2, releases, rollout_percent=100)
    line("published v2 at 100% rollout")
    result = agent.run(steps)
    line(
        f"device is now on v{result['policy_version']} — {result['updates_applied']} update applied in-flight"
    )

    console.rule("[bold]v3 — a release this device must refuse")
    bad_columns = [*default_obs_columns(), "wrist_force_z"]
    bad = reference_policy(obs_columns=bad_columns, skill=1.0, seed=2)
    import numpy as np

    probe = build_calibration(reference_policy(skill=1.0, seed=2), n=128)
    probe = np.hstack([probe, np.zeros((len(probe), 1), np.float32)])
    v3 = bundle_mod.build(
        bad,
        version=3,
        out_dir=build_dir,
        probe=probe,
        health=HealthSpec(latency_budget_ms=20.0, max_action_deviation=1e-4),
        notes="adds a wrist force-torque channel",
    )
    publish(v3, releases, rollout_percent=100)
    line("published v3 — it needs a wrist_force_z channel this device does not publish")

    outcome = agent.poll_updates()
    line(f"[yellow]{outcome.get('reason', '')}[/]")
    line(f"still serving v{agent.ota.state.current_version}, uninterrupted")

    outcome = agent.poll_updates()
    line(f"second poll: [green]{outcome.get('reason', '')}[/] — no re-download, no update loop")

    console.rule("[bold]what the fleet was told")
    store = ReleaseStore(releases)
    server, store = serve(releases, port=0)
    try:
        from .transport import HttpTelemetryUploader

        port = server.server_address[1]
        agent.telemetry.uploader = HttpTelemetryUploader(f"http://127.0.0.1:{port}")
        sent = agent.drain_telemetry()
        line(f"uploaded {sent['sent']} events over HTTP, {sent['pending']} still spooled")
    finally:
        server.shutdown()

    counts: dict[str, int] = {}
    for event in store.events():
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    line(", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    console.print()
    console.print(f"device state: [bold]{json.dumps(agent.ota.status())}[/]")
