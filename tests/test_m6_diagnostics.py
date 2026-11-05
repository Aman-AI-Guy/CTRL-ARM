"""Whole-chain diagnostics use guarded readers and a driverless gateway, never a motion driver."""

from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import pytest
import numpy as np

from trova.diagnose import diagnose, STATUSES
from trova.localization import CalibrationRouter
from trova.geom import VisionStore


class Camera:
    host = "test-camera"

    def status(self):
        return {"camera": "streaming", "width": 1600, "height": 900, "rotate": 180}

    def capture(self):
        raw = (
            Path(__file__).parent / "fixtures/selection/red_mouse_close.jpg"
        ).read_bytes()
        return SimpleNamespace(
            jpeg=raw,
            meta={
                "kind": "pi",
                "rotate_reported": 180,
                "extra_rotate": 0,
                "requested_at": time.time(),
                "received_at": time.time(),
            },
        )


def test_live_observation_reaches_metric_stage_without_initializing_driver(
    tmp_path, monkeypatch
):
    from test_localization_providers import qualified
    from test_selection_fallback import RecordedVision
    import trova.arm

    monkeypatch.setattr(
        trova.arm,
        "make_arm",
        lambda *a, **k: pytest.fail("A diagnostic must never initialize a driver"),
    )
    provider = qualified(tmp_path)
    reader = SimpleNamespace(
        read=lambda: provider.robot | {"state_code": 2, "warn_code": 0},
        configuration=lambda: {"mode": 0},
    )
    result = diagnose(
        camera=Camera(),
        reader=reader,
        grok=RecordedVision(),
        data=tmp_path,
        vision=CalibrationRouter(VisionStore(), provider),
        pixel=[800, 450],
    )
    stages = {row["stage"]: row for row in result["stages"]}
    assert stages["fresh_image"]["status"] == "PASS"
    assert stages["semantic_recognition"]["status"] == "PASS"
    assert stages["pixel_to_base"]["status"] == "PASS"
    assert result["location"]["object_id"] > 0
    assert result["location"]["robot_joints_deg"] == [0] * 5
    assert (
        stages["five_axis_ik"]["status"] == "BLOCKED"
    )  # measured object geometry intentionally missing
    assert stages["hardware_authorization"]["status"] == "BLOCKED"
    assert all(
        s["status"] in STATUSES and s["module"] and "actual_input" in s
        for s in result["stages"]
    )
    assert not result["motion_sent"] and not result["ready_for_physical_execution"]


def test_camera_failure_does_not_hide_other_independent_checks(tmp_path):
    class Broken(Camera):
        def capture(self):
            raise RuntimeError("network disconnected")

    result = diagnose(camera=Broken(), data=tmp_path)
    stages = {row["stage"]: row for row in result["stages"]}
    assert stages["camera_connectivity"]["status"] == "PASS"
    assert stages["fresh_image"]["status"] == "FAIL"
    assert stages["calibration_loading"]["status"] == "PASS"
    assert stages["robot_telemetry"]["status"] == "NOT_CONFIGURED"
    assert "reconstruction_backend" in stages and "hardware_authorization" in stages
    assert not result["motion_sent"]


def test_negative_live_response_is_not_object_recognition(tmp_path):
    from trova.grok import GrokResult

    class Missing:
        def json(self, **kwargs):
            return GrokResult(
                {
                    "box_id": None,
                    "candidate_ids": [],
                    "visible_but_unboxed": False,
                    "clarifying_question": None,
                    "label": "nessun mouse rosso",
                    "reason": "No red mouse is visible.",
                    "confidence": 0.99,
                },
                0,
                "recorded",
                False,
            )

    report = diagnose(camera=Camera(), grok=Missing(), data=tmp_path)
    stages = {row["stage"]: row for row in report["stages"]}
    assert stages["grok_vision_connectivity"]["status"] == "PASS"
    assert stages["semantic_recognition"]["status"] == "BLOCKED"
    assert stages["image_localization"]["status"] == "BLOCKED"
    assert report["selected"] is None


def test_guarded_fk_and_configuration_never_call_writes():
    from trova.arm.telemetry import GuardedSDK, XArmTelemetry, TelemetryWriteRefused
    from trova.scan3d.flange import flange_pose

    class SDK:
        mode = 0
        tcp_offset = [0] * 6
        world_offset = [0] * 6

        def get_forward_kinematics(self, q, **kwargs):

            matrix = flange_pose(q)
            return 0, [*matrix[:3, 3], *cv2.RQDecomp3x3(matrix[:3, :3])[0]]

    sdk = GuardedSDK(SDK())
    reader = XArmTelemetry(sdk)
    assert reader.configuration()["mode"] == 0
    result = reader.forward([0] * 5)
    assert result["ok"] and not result["motion_sent"]
    assert set(sdk.calls) <= {
        "get_forward_kinematics",
        "read_property:mode",
        "read_property:tcp_offset",
        "read_property:world_offset",
    }
    with pytest.raises(TelemetryWriteRefused):
        sdk.read_property("motion_enable")


def test_measured_base_goal_is_corroborated_with_image_and_provider(tmp_path):
    from test_localization_providers import qualified

    p = qualified(tmp_path)
    router = CalibrationRouter(VisionStore(), p)
    selected = {
        "track_id": 5,
        "box": [700, 350, 900, 550],
        "geometric_reference": {
            "kind": "measured_base_point",
            "pixel": [800, 450],
            "point_mm": [207, 0, 35],
            "point_uncertainty_mm": 0.5,
            "measurement_note": "synthetic measured point",
            "snapshot_id": "independent-reference",
        },
    }
    result = router.localize_region(p.robot, selected, [1600, 900])
    assert result["ok"] and result["point_mm"] == [207, 0, 35]
    assert result["reference_height_mm"] == pytest.approx(35)
    assert result["uncertainty_mm"] == 0.5 and result["frame"] == "robot_base"
    selected["geometric_reference"]["point_mm"] = [1000, 0, 35]
    assert not router.localize_region(p.robot, selected, [1600, 900])["ok"]


def test_metric_localization_rejects_expired_live_image():
    from test_agent import _Host, _arm, _task
    from test_geom import _fixture_store
    from trova.agent import skill_locate

    host = _Host(_arm(), _fixture_store())
    host.requires_fresh_observation = True
    task = _task()
    task["snapshot_pose"] = host.robot()
    task["scene_received_at"] = time.time() - 31
    result = skill_locate(host, task, "")
    assert not result["ok"] and "30 seconds" in result["reason"]


def test_scene_metadata_belongs_to_captured_frame_not_display_poll(
    tmp_path, monkeypatch
):
    from test_pipeline import _client

    app, _ = _client(tmp_path, monkeypatch)
    st = app.state.trova
    frame = np.zeros((100, 120, 3), np.uint8)
    st.source = SimpleNamespace(
        read_capture=lambda: {
            "frame": frame,
            "meta": {"requested_at": 123, "received_at": 124},
        },
        info=lambda: {"kind": "pi", "requested_at": 999},
    )
    result, meta = st.capture_frame()
    assert result.shape == (100, 120, 3)
    assert meta["requested_at"] == 123 and meta["received_at"] == 124
