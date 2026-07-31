# edge-policy-runtime

**A policy runtime for edge devices, with a signed down-link and a telemetry
up-link.**

Optimising a model for a device is a day's work. Being able to *replace* that
model on a hundred devices in the field, prove each one still works before it
runs, and find out what happened afterwards — that is the part that decides
whether a robot fleet can improve after it ships.

```bash
git clone https://github.com/abyyworld/edge-policy-runtime
cd edge-policy-runtime
make install
make demo        # v1 ships, v2 is adopted over the air, v3 is refused
make quantize    # int8 fidelity, per-tensor vs per-channel, measured
make test        # 92 tests
```

Standalone: it takes a trained policy and everything after that is this
repo's job. The fleet-scale version of both links —  many devices, selective
data collection, retraining and canaried releases — is
[`robot-fleet-loop`](https://github.com/abyyworld/robot-fleet-loop).

---

## What `make demo` does

```
──────────────── v1 — the factory image ────────────────
│ device jetson-a1 booted on v1, cohort 49/100
│ ran 1200 steps, p99 0.09 ms, 5 flagged
──────── v2 — a real update, adopted over the air ──────
│ published v2 at 100% rollout
│ device is now on v2 — 1 update applied in-flight
──────── v3 — a release this device must refuse ────────
│ published v3 — it needs a wrist_force_z channel this device does not publish
│ v3 rejected by health gate: observation contract mismatch: this device does
  not publish wrist_force_z
│ still serving v2, uninterrupted
│ second poll: v3 is quarantined — no re-download, no update loop
──────────────── what the fleet was told ───────────────
│ uploaded 29 events over HTTP, 0 still spooled
│ flag=12, heartbeat=14, ota=2
```

The third phase is the one worth looking at. A model retrained with a wrist
force-torque channel is a perfectly good model. Activated on a device that does
not have that sensor, it receives whatever happens to be in that slot, and every
symptom points at the model rather than at the deployment. The device catches it
before the policy touches the control loop, refuses it, and — crucially —
remembers that it refused it.

## The down-link

```
check → stage → gate → activate → (roll back)
```

1. **Check.** Is there a version newer than mine, on my channel, that my rollout
   cohort has been reached by, that I have not already quarantined?
2. **Stage.** Download to scratch. Verify the signature and every file hash. The
   running policy has not been touched and a failure here costs only bandwidth.
3. **Gate.** Prove the staged policy runs *on this device*: the observation
   contract resolves against the columns this device actually publishes, latency
   fits the budget the release itself declares, and the probe actions match what
   the hub recorded when it built the bundle.
4. **Activate.** Repoint a symlink with `os.replace`. A power cut leaves the
   device on exactly one of the two versions, never on half of each.
5. **Roll back.** The previous bundle is kept, not deleted. Reverting is the same
   atomic repoint in the other direction.

### The chain of custody

**Signature over manifest, manifest over file hashes.** One signature covers
every byte in the bundle, so swapping `policy.npz` for a different one is caught
even though the signature itself was never touched.

```
manifest.json   version, policy id, obs/action contract, health envelope,
                sha256 of every other file, provenance from the training run
manifest.sig    HMAC-SHA256 over the canonical manifest bytes
policy.npz      weights and the named contract they were trained with
probe.npz       observations, and the actions the hub saw this policy produce
```

### Three things that are easy to leave out

**Quarantine.** A version that failed the gate is recorded as failed. Without it
the device checks again in sixty seconds, sees the same newest version,
downloads it again, and fails again — and every device in the fleet got the bad
release at the same time, so they all do that simultaneously.

**A rollback budget.** After three rollbacks in a short window the device stops
updating itself and says so. Something is wrong that is specific to this device,
and it is better for it to sit still on a known version than to keep cycling
while nobody understands why.

**Staged rollout without hub-side state.** The release says "v7, 20% rolled
out"; each device hashes its own id with the version and works out whether it is
in the first 20%. Deterministic, so a device does not drift in and out between
polls, and salted per version, so the same unlucky devices are not the canary for
every release.

### What the gate cannot do

It catches **broken**, **incompatible** and **too slow**. It cannot catch *runs
fine, succeeds less often* — that needs closed-loop evaluation against the
incumbent, which needs episodes the device does not have at activation time.
That decision belongs to the fleet, and conflating the two produces a health
check that quietly approves regressions. The canary stage in `robot-fleet-loop` is
where it lives.

## The up-link

The hard part of telemetry is not what to measure. It is the six hours the
device cannot reach the hub.

**Spool first, delete only on acknowledgement.** Delivery is at-least-once: a
device that uploads a batch and loses the network before reading the response
sends it again, and the hub deduplicates on event id. Exactly-once across an
unreliable link is not available at any price, and pretending otherwise means
losing events instead of occasionally duplicating them.

**The spool evicts by priority before it evicts by age.** A plain ring buffer
drops precisely backwards — it keeps the last hour of "all normal" and discards
the failure that started the incident. Here heartbeats are shed first, and a
fresh heartbeat never displaces an old failure.

| priority | kinds | evicted |
| ---: | --- | --- |
| 0 | `failure`, `ota` | last |
| 1 | `flag` | second |
| 2 | `heartbeat` | first |

**Flagging happens on the device.** Uploading everything needs bandwidth nobody
has; uploading only aggregates means the interesting episodes are gone by the
time anyone asks. Four detectors, each catching something the others miss:

| detector | catches |
| --- | --- |
| `inference_failed` | the policy raised, or produced a non-finite action |
| `deadline_burst` | consecutive deadline misses — thermal throttle, memory stall, another process |
| `action_saturated` | commanding beyond the actuator limit, so the robot is no longer doing what the policy asked |
| `observation_ood` | the situation the training set never contained — the reason to collect from the fleet at all |

**Each reason has a cooldown, and suppressed occurrences are counted rather than
discarded.** An out-of-distribution episode is out of distribution for all two
hundred of its steps. The first occurrence carries the full context; the rest
increment a counter that rides on the next heartbeat. The incident stays visible
and its duration stays measurable, at one event instead of two hundred. Before
this was added the demo device flagged **785 of 1200 steps**; it now flags 5, and
no information was lost.

**An activated update is itself a telemetry event.** Without the join between
which version a device is on and how that device is doing, a dashboard can show
you which devices are unhealthy but not whether the release is why.

## Inference

**Latency is reported as p95 and p99, and the mean is not a headline number.** A
control loop with a 20 ms budget, a 4 ms mean and a 60 ms p99 misses its deadline
several times a minute, and each miss is a discontinuity in the commanded
trajectory. Deadline misses are counted directly rather than inferred.

**An inference failure produces a safe action, not an exception.** A malformed
observation, a NaN from a dead sensor, a model mid-swap — the runtime returns
the hold action (zero joint deltas, gripper unchanged), counts it, and flags it.
The robot stopping is a bad outcome; an unhandled exception in a loop that owns
actuators is a worse one.

**Benchmarked one observation at a time, not batched.** The robot needs the
action for *this* observation before the next tick, and a batch of 64 does not
exist. Batched numbers look far better and describe a workload the device does
not have.

### int8, and the scheme most toolchains give you by default

The action head's output channels carry different physical quantities: seven
joint deltas capped at 0.15 rad, and one normalised gripper command. A single
scale for the whole matrix is set by the gripper channel, and the joint deltas —
the ones the robot actually tracks, in radians — pay for it.

| scheme | size | joint MAE | joint worst | gripper worst | ships? |
| --- | ---: | ---: | ---: | ---: | :---: |
| per-tensor | 3.85x | 6.26e-04 | **2.53e-03** | 3.86e-03 | no |
| per-channel | 3.66x | 1.67e-04 | **3.95e-04** | 1.92e-03 | yes |

Tolerance is 1% of the rate cap (1.5e-3 rad), judged on the worst case rather
than the mean — a mean inside tolerance with a tail outside it is a policy that
is fine almost always, which on a robot is a different sentence from fine.
Per-tensor is **6.4x** worse on the joint channels for a footprint difference of
5%, and it fails a tolerance per-channel passes.

Joint and gripper errors are reported separately because they are not the same
quantity. Collapsing them means the dimensionless channel sets a tolerance that
is then applied to radians, and the resulting figure is not in any unit at all.

## What this is not

Stated plainly, because each of these would otherwise be read as more than it is.

- **The signature is HMAC-SHA256 over a shared secret.** Any party who can verify
  a release can also forge one, so a compromised device compromises the fleet.
  Production wants asymmetric signing — hub holds the private key, devices hold
  only the public half. That is a change to two functions and nothing else; the
  chain of custody does not move. It is left undone deliberately rather than
  half-done, so the boundary is honest.
- **The int8 speedup is not measured, because numpy cannot show it.** numpy
  upcasts int8 operands, so an int8 matmul in this process is slower, not faster.
  What is measured is footprint and fidelity — the numbers that decide whether
  you can ship it. Latency comes from whatever kernel the device really runs.
- **Weight-only quantization.** Activations stay float32. That is a real
  deployment mode and the arithmetic here is exactly what a weight-only int8
  kernel computes, so the fidelity numbers transfer. Activation quantization
  needs calibrated per-tensor ranges and interacts with accumulator width;
  faking it in numpy would produce numbers that mean nothing.
- **The workload is kinematic.** Joints integrate commanded deltas subject to
  limits and a rate cap. No contacts, no dynamics, no friction. It exists so the
  runtime has inputs with realistic structure; replacing it with a simulator or a
  real robot means implementing `reset` and `step`, and nothing in the runtime,
  the OTA client or the spool knows what is on the other side.
- **`server.py` is a development endpoint.** Single process, no auth. It is here
  so both links can be demonstrated over real HTTP rather than against a mock;
  `robot-fleet-loop` is what a deployment points at.

## Layout

```
src/edge_runtime/
  policy.py      the artifact and its named observation contract
  quantize.py    weight-only int8, per-channel, and the fidelity check
  runtime.py     deadline-aware inference, safe fallback, p95/p99
  bundle.py      signed, versioned, self-describing releases
  ota.py         check → stage → gate → activate → roll back; quarantine, cohorts
  telemetry.py   detectors, the priority spool, at-least-once delivery
  transport.py   HTTP for both links, with timeouts
  server.py      development release + telemetry endpoint
  agent.py       the device process holding all of it
  workload.py    a kinematic task, so the runtime has something to run on
tests/           92 tests, mostly about what happens when things fail
```

## Commands

```bash
make demo                                   # the whole down-link, end to end
make quantize                               # int8 fidelity, both schemes
make bench                                  # latency against a control deadline

edge-runtime bundle build 2 --notes "retrained" # build and sign a release
edge-runtime bundle verify build/2              # signature + every file hash
edge-runtime bundle publish build/2 --rollout 20
edge-runtime serve-releases                     # dev endpoint on :8720

edge-runtime device install build/1             # the factory image
edge-runtime device run --steps 5000            # control loop, both links live
edge-runtime device status                      # version, and why not a newer one
edge-runtime device rollback --reason "..."
edge-runtime telemetry --kind flag              # what devices reported
```

## Status

Working end to end. A device boots on a factory bundle, adopts a published
release over HTTP mid-run, refuses one it cannot run, quarantines it, and reports
all of it upstream — with the whole path covered by tests that assert the failure
behaviour rather than the happy path. The signing scheme and the workload are
placeholders and say so; the update state machine, the spool and the health gate
are not.

## Licence

MIT.
