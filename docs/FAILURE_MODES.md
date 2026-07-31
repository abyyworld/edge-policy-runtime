# Failure modes of an over-the-air update

The device is the party that cannot be supervised. Whatever it does with a new
release it does alone, possibly on a bad network, possibly moments before losing
power, and if it gets it wrong somebody drives to where the robot is.

This is the list the update state machine was written against. Each row names
the failure, what happens instead, and the test that holds it.

## Transport and integrity

| Failure | What this does | Test |
| --- | --- | --- |
| Truncated download | Archive hash checked against the published pointer before unpacking; version quarantined | `test_a_tampered_archive_is_refused_and_never_activated` |
| Substituted `policy.npz` inside a correctly-signed bundle | Manifest carries a hash per file; the signature covers the manifest | `test_modifying_a_file_invalidates_the_bundle` |
| Manifest edited after signing | Signature is over canonical manifest bytes | `test_modifying_the_manifest_invalidates_the_signature` |
| Release from someone else's key | `hmac.compare_digest` against the fleet key | `test_a_bundle_signed_with_another_key_is_refused` |
| Archive member named `../../etc/...` | Every member checked for absolute paths, `..`, and nesting; whole extraction fails | `test_unpack_refuses_a_member_that_escapes_the_directory` |
| Archive member is a symlink or device node | Only flat regular files accepted | `test_unpack_refuses_a_symlink_member` |
| Pointer advertises a version the signed manifest does not claim | Compared explicitly after verification | `test_a_release_whose_manifest_disagrees_with_the_pointer_is_refused` |

A bundle that fails any of these is removed from scratch space rather than left
where a later run might mistake it for a verified one.

## Compatibility

| Failure | What this does | Test |
| --- | --- | --- |
| Policy needs a sensor channel this device does not publish | Contract resolved by name against the device's columns, before the policy runs | `test_a_release_needing_a_sensor_this_device_lacks_is_rejected` |
| Release validated on faster hardware than this device | The latency budget travels *in the release*; the device measures itself against it | `test_a_release_that_cannot_meet_its_own_latency_budget_is_rejected` |
| Bundle uses a manifest schema this runtime does not understand | `schema_version` and `runtime_min_version` checked during verification | `test_a_bundle_needing_a_newer_runtime_is_refused` |
| Bundle does not reproduce the actions the hub recorded from it | Probe replay compared to `expected_actions` from build time | `test_health_check_catches_a_bundle_that_does_not_compute_what_it_did` |

The first row is the one that costs the most when it is missing, because it does
not present as a deployment problem. The model receives whatever is in that slot,
behaves oddly, and every symptom points at the model.

## Interruption

| Failure | What this does | Test |
| --- | --- | --- |
| Power cut during activation | `os.symlink` then `os.replace` — never unlink-then-link, which has a window where `current` does not exist | `test_current_is_a_symlink_that_always_points_somewhere` |
| Power cut while writing device state | State written to a temp file, fsynced, then `os.replace`d | `test_write_atomically_leaves_no_partial_file` |
| Power cut mid-append to the telemetry spool | Partial final line skipped on load; losing one event beats refusing to start | `test_a_truncated_final_line_does_not_stop_the_device` |
| Device restarts after a rollback | Current version, quarantine list and history are on disk | `test_state_survives_a_restart` |
| Release server disappears mid-poll | `poll()` catches everything; the control loop is not on the network path | `test_poll_never_raises_even_when_the_release_source_is_broken` |

## Loops and cascades

| Failure | What this does | Test |
| --- | --- | --- |
| Bad release re-downloaded on every poll, fleet-wide | Rejected versions are quarantined by version number | `test_a_rejected_version_is_not_downloaded_again` |
| Quarantine blocks the fix | Quarantine is per version, so a later good release is adopted normally | `test_quarantine_does_not_block_a_later_good_release` |
| One device cycles between versions indefinitely | Three rollbacks in a short window freezes updates and asks for an operator | `test_repeated_rollbacks_freeze_updates_for_an_operator` |
| Whole fleet takes a bad release simultaneously | Rollout percentage, computed device-side from a salted hash of the device id | `test_rollout_percentage_selects_a_stable_subset_of_the_fleet` |
| The same devices are always the canary | The rollout hash is salted with the version | `test_the_canary_group_changes_between_versions` |
| Hub down, device retries every tick | Exponential backoff on the upload path; the spool makes backing off free | `DeviceAgent.maybe_upload` |

## Observability

| Failure | What this does | Test |
| --- | --- | --- |
| Device offline long enough to fill its buffer | Priority eviction: heartbeats shed before flags, flags before failures | `test_eviction_sheds_heartbeats_before_failures` |
| Upload succeeds but the response is lost | At-least-once; the hub deduplicates on event id | `test_the_hub_deduplicates_a_replayed_batch` |
| Hub stores some events and not others | Device deletes exactly the ids returned | `test_a_partial_acknowledgement_keeps_the_rest` |
| Hub acknowledges a batch it could not parse | Malformed batches return 400 and store nothing | `test_a_malformed_batch_is_not_acknowledged` |
| One incident produces hundreds of identical events | Per-reason cooldown; suppressed occurrences counted on the heartbeat | `test_a_sustained_incident_produces_one_event_and_a_count` |
| A device is unhealthy but nobody can tell whether the release is why | Every event carries its policy version; activations are themselves events | `test_the_hub_learns_which_version_each_device_is_on` |

## Runtime

| Failure | What this does | Test |
| --- | --- | --- |
| Dead sensor reports NaN | Non-finite actions are treated as failures; the hold action is issued instead | `test_a_nan_observation_does_not_reach_the_actuators` |
| Policy raises mid-loop | Caught; hold action, counted, flagged | `test_a_policy_that_raises_produces_a_hold_action` |
| Latency mean looks fine, tail does not | p95/p99 reported, deadline misses counted directly | `test_deadline_misses_are_counted_not_inferred` |
| Metrics buffer grows for weeks | Ring buffer for percentiles; lifetime counters kept separately | `test_the_latency_window_is_bounded` |

## What is deliberately not handled here

**"The new policy runs fine but succeeds less often."** The device cannot answer
this at activation time — it needs episodes, and a comparison against the
incumbent on matched conditions. Answering it on the device would mean a health
gate that either blocks every update for an hour or approves regressions
silently. It belongs to the fleet: canary a release to a subset, compare the
outcomes those devices report against the incumbent's, and promote or roll back
on the result. That is `robot-fleet-loop`.

**Key compromise.** HMAC with a shared secret means any device that can verify a
release can forge one. Asymmetric signing fixes it and is two functions of work;
see the note in `bundle.py` for why it is left explicitly undone rather than
partially done.
