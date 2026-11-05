"""Retained physical hover inputs through imported IK, geometry and Trova's gateway. No SDK."""

import json

import numpy as np
import pytest

from trova.scan3d.approach import propose
from trova.scan3d.evidence import workspace
from trova.scan3d.flange import FlangeCalibration
from trova.scan3d.orbit_planner import JointGeometry, recorded_context
from trova.scan3d.trajectory import IntervalValidator


def hover():
    root = workspace()
    retained, old = recorded_context(root)
    cal = FlangeCalibration.load(root / "calib/camera_robot.json")
    record = json.loads(
        (root / "arm_pc/runs/20261105-031058_hover/hover_target.json").read_text()
    )
    old.settings.object_height_mm = record["height_mm"]
    old.settings.object_radius_mm = 45
    foot = np.asarray(record["target_mm"]) - [0, 0, record["height_mm"]]
    context = JointGeometry(
        cal.T_fc,
        {"normal": [0, 0, 1], "point_mm": [0, 0, cal.table_z]},
        foot,
        old.settings,
        old.gateway,
        qualified=False,
    )
    return record, context


def test_actual_hover_geometry_generates_reachable_camera_pose():
    record, context = hover()
    result = propose(context, record["q0"], record["above_mm"])
    chosen = result["next"]
    assert chosen, result
    assert not chosen["executable"] and chosen["gateway"]["status"] == "dry_run"
    assert chosen["elevation_deg"] < 90  # strict existing joint-jump limit retained
    camera, direction = context.cam.ray(chosen["joints_deg"])
    top = np.asarray(record["target_mm"])
    assert camera[2] - top[2] == pytest.approx(120, abs=1)
    np.testing.assert_allclose(
        direction, (top - camera) / np.linalg.norm(top - camera), atol=0.01
    )
    assert max(abs(a - b) for a, b in zip(record["q0"], chosen["joints_deg"])) <= 45
    certificate = context.swept_check(record["q0"], chosen["joints_deg"])
    assert certificate["ok"], certificate
    assert len(certificate["joint_waypoints"]) > 2
    assert (
        certificate["joint_linear_model_bounded"]
        and not certificate["collision_avoidance"]
    )
    assert certificate["assembly_contact_exclusion"] == ["link4", "head"]
    validator = IntervalValidator(context)
    for left, right in zip(
        certificate["joint_waypoints"], certificate["joint_waypoints"][1:]
    ):
        for u in np.linspace(0, 1, 7):
            assert all(
                gap >= 0
                for _, gap, _ in validator.constraints(
                    np.asarray(left) * (1 - u) + np.asarray(right) * u
                )
            )


def test_motion_envelope_missing_or_wrong_binding_is_refused():
    record, context = hover()
    reasons = context.hardware_blockers(record["q0"], record["q0"])
    assert any("operator" in r for r in reasons)
    assert any("interpolation" in r for r in reasons)
    context.settings.motion_qualification = {
        "operator": "test",
        "measurement_note": "synthetic envelope",
        "model_sha256": "wrong",
        "mount_sha256": "wrong",
    }
    decision = context.gateway.authorize_joints(
        record["q0"],
        0.1,
        source="vision_joint_approach",
        geometry=context,
        start_q=record["q0"],
    )
    assert not decision.ok and not decision.allow_motion
    assert "bind" in decision.reason


def test_swept_certificate_refuses_midpath_body_obstacle():
    from trova.motion import KeepOut

    record, context = hover()
    candidate = propose(context, record["q0"], 120)["next"]
    mid = (np.asarray(record["q0"]) + candidate["joints_deg"]) / 2
    from trova.scan3d.orbit_math import fk_frames

    world = context.geometry.world(fk_frames(mid), context.turn)
    point = world["link3"][0][0]
    context.gateway.keepouts = [
        KeepOut(
            (point[0] - 2, point[0] + 2),
            (point[1] - 2, point[1] + 2),
            (point[2] - 2, point[2] + 2),
        )
    ]
    certificate = context.swept_check(record["q0"], candidate["joints_deg"])
    assert not certificate["ok"] and "keepout" in certificate["reason"]


def test_agent_on_arm_approach_uses_joint_executor_and_not_scan_continuation(
    tmp_path, monkeypatch
):
    from test_m5_agent import orbit_scene

    st, client, backend = orbit_scene(tmp_path, monkeypatch)
    st.camera_on_arm = True
    st.observe_clearance_mm = 200
    # The recording is the fixture's known setup, never a live qualification.
    result = client.post(
        "/api/agent/task", json={"text": "Ora posizionati sopra."}
    ).json()
    assert result["approach_plan"] is not None, result
    assert result["joint_operation"] == "approach", result
    assert result["confirmation"] and not backend.calls, result
    moved = client.post(
        "/api/agent/confirm", json={"proposal_id": result["confirmation"]["id"]}
    ).json()
    assert backend.calls, moved
    assert not moved["hardware_moved"] and moved["simulation_moved"]
    skills = [row["skill"] for row in moved["steps"]]
    assert "plan_orbit" not in skills and "start_scan" not in skills
    assert not st.scans.active


def test_controller_mode_changed_is_an_explicit_blocker(tmp_path, monkeypatch):
    from test_voice_workflow import setup_scene, select_and_plan

    st, client = setup_scene(tmp_path, monkeypatch, True)
    previous = st.robot_telemetry
    st.robot_telemetry = lambda: previous() | {"state_code": 5}
    result = select_and_plan(st, client)
    assert not result["confirmation"] and not result["hardware_moved"]
    assert "MODE_CHANGED" in result["reason"]
