"""Tests for the control-loop runtime and the int8 conversion in front of it."""

from __future__ import annotations

import numpy as np
import pytest

from edge_runtime.agent import build_calibration
from edge_runtime.policy import Policy, PolicyError, default_obs_columns, reference_policy
from edge_runtime.quantize import (
    GRIPPER_TOLERANCE,
    JOINT_TOLERANCE_RAD,
    measure_fidelity,
    quantization_report,
    quantize,
)
from edge_runtime.runtime import InferenceRuntime, benchmark


@pytest.fixture
def policy():
    return reference_policy(seed=0)


@pytest.fixture
def calibration(policy):
    return build_calibration(policy, n=512)


# -- the observation contract ------------------------------------------------


def test_binding_names_the_missing_column(policy):
    """The failure that would otherwise be a silently mis-slotted sensor."""
    observation = {c: 0.0 for c in policy.obs_columns}
    del observation["gripper_state"]

    with pytest.raises(PolicyError, match="gripper_state"):
        policy.bind(observation)


def test_binding_respects_recorded_order_not_dict_order(policy):
    """Two policies with the same columns in a different order are not the same
    policy, and a dictionary does not carry the order that matters."""
    observation = {c: float(i) for i, c in enumerate(policy.obs_columns)}
    shuffled = dict(reversed(list(observation.items())))
    assert np.array_equal(policy.bind(observation), policy.bind(shuffled))


def test_a_contract_that_disagrees_with_the_weights_is_refused():
    with pytest.raises(PolicyError, match="observation columns"):
        Policy([np.zeros((4, 8), np.float32)], [np.zeros(8, np.float32)], ["a", "b"])


def test_content_hash_identifies_the_bytes_not_the_filename(tmp_path, policy):
    policy.save(tmp_path / "policy_final_v2_REAL.npz")
    reloaded = Policy.load(tmp_path / "policy_final_v2_REAL.npz")
    assert reloaded.content_hash() == policy.content_hash()
    assert reference_policy(seed=1).content_hash() != policy.content_hash()


def test_the_contract_survives_a_save_load_round_trip(tmp_path, policy):
    policy.save(tmp_path / "p.npz")
    assert Policy.load(tmp_path / "p.npz").obs_columns == default_obs_columns()


# -- the deadline ------------------------------------------------------------


def test_deadline_misses_are_counted_not_inferred(policy):
    """Averages hide exactly the events that matter."""
    runtime = InferenceRuntime(policy, budget_ms=1e-9)  # nothing can meet this
    for _ in range(50):
        runtime.step(np.zeros(policy.obs_dim, np.float32))

    stats = runtime.stats()
    assert stats.deadline_misses == 50
    assert stats.deadline_miss_rate == 1.0


def test_a_generous_budget_is_met(policy, calibration):
    stats = benchmark(policy, calibration[:200], budget_ms=1000.0)
    assert stats.deadline_misses == 0
    assert stats.p99_ms >= stats.p50_ms


def test_the_latency_window_is_bounded(policy):
    """A device runs for weeks; an unbounded metrics buffer is a memory leak
    with a longer fuse than most."""
    runtime = InferenceRuntime(policy, window=64)
    for _ in range(5000):
        runtime.step(np.zeros(policy.obs_dim, np.float32))

    assert len(runtime._latencies) == 64
    assert runtime.stats().n == 5000, "the lifetime count is not windowed"


# -- failing safely ----------------------------------------------------------


def test_a_policy_that_raises_produces_a_hold_action(policy):
    """An unhandled exception in a loop that owns actuators is the worse
    outcome."""
    runtime = InferenceRuntime(policy)
    record = runtime.step(np.zeros(3, np.float32))  # wrong shape

    assert record.failed
    assert "PolicyError" in record.error
    assert np.allclose(record.action[:7], 0.0)
    assert runtime.stats().failures == 1


def test_a_nan_observation_does_not_reach_the_actuators(policy):
    """A dead sensor reporting NaN is a shape-correct observation.

    Nothing in the forward pass raises on it — it propagates silently to the
    action, and NaN reaching the actuators is worse than not moving.
    """
    runtime = InferenceRuntime(policy)
    obs = np.ones(policy.obs_dim, np.float32)
    obs[4] = np.nan

    record = runtime.step(obs)
    assert record.failed
    assert "non-finite" in record.error
    assert np.all(np.isfinite(record.action))


def test_the_hold_action_keeps_the_gripper_where_it_was(policy):
    runtime = InferenceRuntime(policy)
    runtime.step(np.ones(policy.obs_dim, np.float32) * 0.3)
    commanded = runtime._last_gripper

    record = runtime.step(np.zeros(3, np.float32))  # forced failure
    assert record.action[7] == pytest.approx(commanded)


def test_a_failure_does_not_stop_the_loop(policy):
    runtime = InferenceRuntime(policy)
    for i in range(20):
        runtime.step(np.zeros(3 if i % 2 else policy.obs_dim, np.float32))
    assert runtime.stats().n == 20
    assert runtime.stats().failures == 10


# -- quantization ------------------------------------------------------------


def test_quantized_policy_is_a_drop_in(policy, calibration):
    quantized = quantize(policy)
    assert quantized.obs_columns == policy.obs_columns
    assert quantized.forward(calibration[:4]).shape == policy.forward(calibration[:4]).shape


def test_int8_is_roughly_four_times_smaller(policy):
    quantized = quantize(policy)
    from edge_runtime.quantize import float_bytes

    assert 3.5 < float_bytes(policy) / quantized.stored_bytes < 4.0


def test_per_tensor_loses_the_joint_channels(policy, calibration):
    """The measured cost of the scheme most toolchains give you by default.

    The action head's output channels carry different physical quantities. One
    scale for the matrix is set by the largest — the normalised gripper command
    — and the joint deltas, which are what the robot tracks in radians, pay for
    it.
    """
    reports = quantization_report(policy, calibration)
    per_tensor, per_channel = reports["per_tensor"], reports["per_channel"]

    assert per_tensor.joint_max_abs_rad > 5 * per_channel.joint_max_abs_rad
    # And the footprint difference that buys is negligible.
    assert per_channel.compression > 0.9 * per_tensor.compression


def test_the_default_scheme_fails_a_tolerance_the_good_one_passes(policy, calibration):
    reports = quantization_report(policy, calibration)
    assert not reports["per_tensor"].qualifies(JOINT_TOLERANCE_RAD, GRIPPER_TOLERANCE)
    assert reports["per_channel"].qualifies(JOINT_TOLERANCE_RAD, GRIPPER_TOLERANCE)


def test_fidelity_is_judged_on_the_worst_case(policy, calibration):
    report = measure_fidelity(policy, quantize(policy), calibration)
    assert report.joint_max_abs_rad > report.joint_mae_rad
    assert not report.qualifies(joint_tolerance_rad=report.joint_mae_rad)


def test_joint_and_gripper_errors_are_reported_separately(policy, calibration):
    """Collapsing them means the dimensionless channel sets a tolerance that is
    then applied to radians."""
    report = measure_fidelity(policy, quantize(policy), calibration)
    assert report.gripper_max_abs > report.joint_max_abs_rad
    assert report.worst_column.endswith("_delta")


def test_a_calibration_set_too_small_to_see_the_tail_is_refused(policy):
    with pytest.raises(PolicyError, match="not enough"):
        measure_fidelity(policy, quantize(policy), np.zeros((8, policy.obs_dim), np.float32))


def test_calibration_shape_is_checked(policy):
    with pytest.raises(PolicyError, match="calibration set has"):
        measure_fidelity(policy, quantize(policy), np.zeros((64, 3), np.float32))


def test_an_all_zero_channel_does_not_produce_nan(policy, calibration):
    """A dead output channel has no scale to speak of."""
    policy.weights[-1][:, 3] = 0.0
    quantized = quantize(policy)
    assert np.all(np.isfinite(quantized.forward(calibration[:16])))


def test_an_unknown_scheme_is_refused(policy):
    with pytest.raises(PolicyError, match="unknown quantization scheme"):
        quantize(policy, scheme="int4-magic")
