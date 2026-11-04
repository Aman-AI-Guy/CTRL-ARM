"""Read-only xArm telemetry. These tests never open a controller and never enable motion."""

import socket

import pytest

from trova.arm.telemetry import (
    GuardedSDK, TelemetryWriteRefused, XArmTelemetry, observation_record, open_reader,
)


class _Sdk:
    def __init__(self, pose=None, joints=None, state=2, err=(0, 0), moving=False, ik=None):
        self.pose = [1.0, 2.0, 3.0, 180.0, 0.0, 12.0] if pose is None else pose
        self.joints = [0.0, 10.0, -20.0, 30.0, 40.0] if joints is None else joints
        self.state = state
        self.err = err
        self.moving = moving
        self.ik = [0.0, 10.0, -20.0, 30.0, 40.0] if ik is None else ik
        self.closed = False
        self.writes = []

    def get_position(self):
        return 0, list(self.pose)

    def get_servo_angle(self):
        return 0, list(self.joints)

    def get_state(self):
        return 0, self.state

    def get_err_warn_code(self, show=False):
        return 0, list(self.err)

    def get_is_moving(self):
        return self.moving

    def get_inverse_kinematics(self, pose, input_is_radian=False, return_is_radian=False, ref_angles=None):
        return 0, list(self.ik)

    def motion_enable(self, *args, **kwargs):
        self.writes.append("motion_enable")

    def set_state(self, *args, **kwargs):
        self.writes.append("set_state")

    def set_position(self, *args, **kwargs):
        self.writes.append("set_position")

    def set_tcp_offset(self, *args, **kwargs):
        self.writes.append("set_tcp_offset")

    def clean_error(self, *args, **kwargs):
        self.writes.append("clean_error")

    def disconnect(self):
        self.closed = True


def test_read_never_calls_an_actuator():
    raw = _Sdk()
    arm = XArmTelemetry(raw, "192.0.2.1")
    reading = arm.read()
    assert reading["source"] == "telemetry"
    assert reading["simulated"] is False
    assert reading["pose6"] == [1.0, 2.0, 3.0, 180.0, 0.0, 12.0]
    assert reading["joints"] == [0.0, 10.0, -20.0, 30.0, 40.0]
    assert reading["state_code"] == 2
    assert reading["error_code"] == 0
    assert reading["motion_sent"] is False
    assert reading["movements_enabled"] is False
    assert reading["motors_enabled_by_trova"] is False
    assert reading["tcp_changed"] is False
    assert reading["sync_precision_claimed"] is False
    assert reading["read_finished_at"] >= reading["read_started_at"]
    for name in ("motion_enable", "set_state", "set_position", "set_tcp_offset", "clean_error"):
        with pytest.raises(TelemetryWriteRefused):
            getattr(arm.sdk, name)(True)
    assert raw.writes == []
    assert "motion_enable" not in arm.sdk.calls
    assert set(arm.sdk.calls) <= {
        "get_position", "get_servo_angle", "get_state", "get_err_warn_code", "get_is_moving",
    }
    solved = arm.inverse(reading["pose6"], reading["joints"])
    assert solved["ik_verified"] is True and solved["motion_sent"] is False
    assert solved["joints"] == [0.0, 10.0, -20.0, 30.0, 40.0]
    raw.ik = [0.0, 0.0, 50.0, 0.0, 0.0]
    refused = arm.inverse(reading["pose6"])
    assert refused["ik_verified"] is False and "joint 3" in refused["reason"]
    assert refused["motion_sent"] is False
    arm.close()
    assert raw.closed is True


def test_a_failed_read_is_not_replaced_with_a_simulated_pose():
    class Dead:
        def get_position(self):
            raise OSError("controller down")

    reading = XArmTelemetry(Dead()).read()
    assert reading["source"] == "unavailable"
    assert reading["pose6"] is None and reading["xyz_mm"] is None
    assert reading["motion_sent"] is False and reading["sync_precision_claimed"] is False
    assert "controller down" in reading["error"]


def test_open_reader_does_not_connect_until_a_read(monkeypatch):
    calls = []

    def refused(address, timeout=None):
        calls.append(address)
        raise OSError("no robot")

    monkeypatch.setattr("trova.arm.telemetry.socket.create_connection", refused)
    assert open_reader(None) is None
    assert open_reader("") is None
    assert open_reader("off") is None
    assert open_reader("sim") is None
    assert calls == []
    reader = open_reader("xarm:192.0.2.9")
    assert calls == []
    reading = reader.read()
    assert calls == [("192.0.2.9", 18333)]
    assert reading["source"] == "unavailable" and reading["motion_sent"] is False
    assert "192.0.2.9" in reading["error"]
    inverse = reader.inverse([0, 0, 0, 180, 0, 0])
    assert inverse["ik_verified"] is False and inverse["motion_sent"] is False


def test_guard_refuses_unknown_names_before_the_sdk():
    class Only:
        looked = False

        def __getattr__(self, name):
            self.looked = True
            raise AssertionError(name)

    guard = GuardedSDK(Only())
    with pytest.raises(TelemetryWriteRefused):
        guard.move_gohome()
    assert guard._raw.looked is False


def test_observation_record_reports_a_wall_clock_gap_and_does_not_claim_sync():
    record = observation_record(
        {"requested_at": 10.0, "received_at": 10.4, "request_s": 0.4, "width": 1640, "height": 1232,
         "rotate_reported": 180, "extra_rotate": 0, "rotated_again": False,
         "provenance": "raspberry_pi", "kind": "pi"},
        {"source": "telemetry", "pose6": [1, 2, 3, 180, 0, 0], "joints": [0, 0, 0, 0, 0],
         "state_code": 2, "error_code": 0, "moving": False, "read_started_at": 10.1,
         "read_finished_at": 10.2},
        {"found": True, "corners": 54},
    )
    assert record["skew_s"] == 0.2
    assert record["sync_precision_claimed"] is False
    assert "not a measured trigger" in record["sync_note"]
    assert record["robot"]["motion_sent"] is False
    assert record["image"]["rotate_reported"] == 180
    assert record["image"]["rotated_again"] is False
    assert record["detection"]["found"] is True
    # socket is imported so a future edit that connects at import time is visible in this module
    assert hasattr(socket, "create_connection")
