"""Provider qualification uses synthetic measurements; the real saved model remains invalid."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trova.geom import VisionStore, tool_pose_matrix
from trova.localization import CalibrationRouter, HistoricalProvider, make_router
from trova.scan3d.evidence import sha256
from trova.scan3d.flange import flange_pose

RETAINED = Path(__file__).parent / "fixtures/calibration/historical_camera_robot.json"


def robot(q):
    matrix = flange_pose(q)
    angles = cv2.RQDecomp3x3(matrix[:3, :3])[0]
    pose = [*matrix[:3, 3].tolist(), *angles]
    np.testing.assert_allclose(tool_pose_matrix(pose), matrix, atol=1e-6)
    return dict(
        joints=q,
        pose6=pose,
        xyz_mm=pose[:3],
        moving=False,
        error_code=0,
        source="telemetry",
    )


def qualified(tmp_path):
    # Known synthetic geometry, independent of the archived physical model and never used on hardware.
    candidate = dict(
        image_size=[1600, 900],
        K=[[1000, 0, 800], [0, 1000, 450], [0, 0, 1]],
        dist=[0] * 5,
        T_flange_camera=np.eye(4).tolist(),
        table_z_mm=0,
        quality={"reprojection_rms_px": 0.1, "provenance": "synthetic_test_only"},
    )
    path = tmp_path / "synthetic_candidate.json"
    path.write_text(json.dumps(candidate))
    station = dict(
        provider="historical_flange",
        calibration_sha256=sha256(path),
        camera_identity="synthetic camera",
        optical_profile="synthetic 1600x900",
        mount_id="synthetic mount",
        robot_base_id="synthetic base",
        table_id="synthetic table",
        measurement_note="Synthetic test ground truth",
        tcp_offset_mm=[0] * 6,
        units="mm",
        frame="robot_base",
        camera_host="test-camera",
        table_plane=dict(a=0, b=0, c=0, residual_mm=0),
    )
    (tmp_path / "calibration_station.json").write_text(json.dumps(station))
    provider = HistoricalProvider(path, tmp_path)
    camera = dict(kind="pi", host="test-camera", rotate_reported=180, extra_rotate=0)
    provider.camera, provider.robot = camera, robot([0] * 5)
    refs = []
    for index, q in enumerate([[0] * 5, [0, 0, 0, 0, 20]]):
        for n, point in enumerate([[177, -30, 0], [237, -30, 0], [207, 30, 0]]):
            image = f"validation_{index}_{n}.png"
            cv2.imwrite(str(tmp_path / image), np.zeros((900, 1600, 3), np.uint8))
            refs.append(
                dict(
                    calibration_sha256=provider.digest,
                    binding_sha256=provider.binding_hash(station),
                    image=image,
                    image_sha256=sha256(tmp_path / image),
                    image_size=[1600, 900],
                    robot=robot(q),
                    measured_mm=point,
                    pixel=provider.cal.robot_to_pixel(point, q),
                    camera_profile=camera,
                    provenance="operator_independent_reference",
                    measurement_note="synthetic",
                )
            )
    (tmp_path / "calibration_references.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in refs)
    )
    return provider


def test_real_saved_calibration_is_not_activated_by_replay(tmp_path):
    router = make_router(VisionStore(), tmp_path, RETAINED)
    router.set_context(
        dict(kind="pi", host="test", rotate_reported=180, extra_rotate=0),
        robot([0] * 5),
    )
    report = router.assess([1600, 900])
    assert not report["valid"] and report["active_provider"] == "image_space"
    assert len(report["providers"]) == 3
    assert any("6.148" in reason for reason in report["reasons"])
    assert any("independent" in reason for reason in report["reasons"])
    assert router.current.hand_eye is None
    result = router.localize_observation(
        robot([0] * 5), dict(center=[800, 450], box=[750, 400, 850, 500]), [1600, 900]
    )
    assert not result["ok"] and result["point_mm"] is None


def test_qualified_joint_provider_loads_without_checkerboard_artifact_shape(tmp_path):
    provider = qualified(tmp_path)
    report = provider.assess([1600, 900])
    assert report["valid"], report["reasons"]
    assert report["validation"]["independent_samples"] == 6
    assert report["validation"]["reference_p95_mm"] < 1e-6
    router = CalibrationRouter(VisionStore(), provider)
    assert router.assess([1600, 900])["active_provider"] == "historical_flange"
    assert router.current.hand_eye is None  # no fabricated checkerboard observations
    selected = dict(
        box=[700, 350, 900, 550],
        center=[800, 450],
        geometric_reference=dict(kind="support_plane_point", pixel=[800, 450]),
    )
    result = router.localize_observation(provider.robot, selected, [1600, 900])
    assert result["ok"] and result["frame"] == "robot_base" and result["units"] == "mm"
    np.testing.assert_allclose(result["point_mm"], [207, 0, 0], atol=1e-6)
    assert result["uncertainty_mm"] > 0
    goal = router.observation_target(result["point_mm"], 150, yaw_deg=0)
    assert goal["ok"] and goal["target_mm"] == [207, 0, 150]
    assert not goal["ik_verified"]


@pytest.mark.parametrize(
    "case",
    [
        "resolution",
        "rotation",
        "extra_rotation",
        "tcp",
        "units",
        "references",
        "changed_image",
        "pose_range",
        "artifact",
    ],
)
def test_qualification_failures_are_closed(tmp_path, case):
    p = qualified(tmp_path)
    size = [1600, 900]
    if case == "resolution":
        size = [1640, 1232]
    elif case == "rotation":
        p.camera["rotate_reported"] = 0
    elif case == "extra_rotation":
        p.camera["extra_rotate"] = 180
    elif case == "tcp":
        p.robot["pose6"][0] += 10
    elif case == "units":
        path = tmp_path / "calibration_station.json"
        d = json.loads(path.read_text())
        d["units"] = "metres"
        path.write_text(json.dumps(d))
    elif case == "references":
        (tmp_path / "calibration_references.jsonl").write_text("")
    elif case == "changed_image":
        (tmp_path / "validation_0_0.png").write_bytes(b"changed")
    elif case == "pose_range":
        p.robot = robot([0, 0, 0, 0, 30])
    elif case == "artifact":
        p.path.write_text(p.path.read_text() + "\n")
    report = p.assess(size)
    assert not report["valid"], case
    assert not p.localize(p.robot, [800, 450], size)["ok"]


def test_cup_center_is_not_a_support_point(tmp_path):
    provider = qualified(tmp_path)
    router = CalibrationRouter(VisionStore(), provider)
    selected = dict(name="green paper cup", center=[800, 450], box=[700, 350, 900, 550])
    result = router.localize_observation(provider.robot, selected, [1600, 900])
    assert (
        not result["ok"]
        and result["point_mm"] is None
        and result["object_center_mm"] is None
    )
    assert "height" in result["reason"] and "bounding-box center" in result["reason"]


def test_provider_target_still_passes_existing_motion_gateway(tmp_path):
    from test_agent import _Host, _arm
    from trova.agent import skill_plan

    provider = qualified(tmp_path)
    router = CalibrationRouter(VisionStore(), provider)
    router.assess([1600, 900])
    host = _Host(_arm("0,40,-10,10,0,10"), router)
    host.image_size = [1600, 900]
    host.robot = lambda: provider.robot
    task = dict(
        goal="hover",
        phase="located",
        selected=dict(track_id=1, name="test", label_authoritative=True),
        location=dict(ok=True, point_mm=[207, 0, 0], uncertainty_mm=1),
        motion_lock=None,
        authoritative=True,
        steps=[],
        timings=[],
        calibration_valid=True,
    )
    before = list(host.arm.tcp_mm())
    result = skill_plan(host, task, "")
    assert not result["ok"] and not result["executed"]
    assert result["motion"]["status"] == "rejected"
    assert list(host.arm.tcp_mm()) == before


def test_missing_or_malformed_import_does_not_crash_live_server(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{}")
    router = make_router(VisionStore(), tmp_path, path)
    assert not router.assess([1600, 900])["valid"]
    assert any(
        "cannot be loaded" in reason for reason in router.assess([1600, 900])["reasons"]
    )


def test_followup_without_selection_never_selects_a_different_object(
    tmp_path, monkeypatch
):
    from test_pipeline import _client

    app, client = _client(tmp_path, monkeypatch)
    app.state.trova.grok.json = lambda **kwargs: pytest.fail(
        "No model should choose a new target for an unbound pronoun"
    )
    result = client.post("/api/agent/task", json={"text": "Now hover above it."}).json()
    assert result["status"] == "refused" and not result["hardware_moved"]
    assert "retained target" in result["reason"]


def test_agent_joint_provider_localizes_and_proposes_without_execution(tmp_path):
    from test_agent import _Host, _arm
    from trova.agent import skill_locate, skill_plan

    provider = qualified(tmp_path)
    router = CalibrationRouter(VisionStore(), provider)
    host = _Host(_arm(), router)
    host.image_size = [1600, 900]
    host.robot = lambda: provider.robot
    host.execute = lambda *args: pytest.fail(
        "A planning test must never execute a move"
    )
    selected = {
        "track_id": 41,
        "name": "synthetic target",
        "label_authoritative": True,
        "box": [700, 350, 900, 550],
        "center": [800, 450],
        "confidence": 0.9,
        "geometric_reference": {"kind": "support_plane_point", "pixel": [800, 450]},
    }
    task = {
        "goal": "move above it",
        "phase": "selected",
        "selected": selected,
        "snapshot_pose": provider.robot,
        "authoritative": True,
        "motion_lock": None,
        "steps": [],
        "timings": [],
    }
    located = skill_locate(host, task, "")
    assert located["ok"] and task["selected"]["track_id"] == 41
    assert task["location"]["provider"] == "historical_flange"
    plan = skill_plan(host, task, "")
    assert plan["ok"], plan["reason"]
    assert task["planned_target"] == [207, 0, 150]
    assert not plan["executed"] and not plan["motion"]["execution_authorized"]


def test_malformed_binding_is_reported_instead_of_activated(tmp_path):
    provider = qualified(tmp_path)
    (tmp_path / "calibration_station.json").write_text("[]")
    report = provider.assess([1600, 900])
    assert not report["valid"]
    assert any("JSON object" in reason for reason in report["reasons"])


def test_uploaded_scene_never_becomes_live_reference_evidence(tmp_path, monkeypatch):
    from test_pipeline import _client
    from trova.server import AgentHost

    app, client = _client(tmp_path, monkeypatch)
    st = app.state.trova
    st.override = np.zeros((900, 1600, 3), np.uint8)
    snap = client.post("/api/scene").json()["snapshot"]
    assert snap["camera_kind"] == "upload"
    assert snap["provenance"] == "uploaded_image"
    assert AgentHost(st).camera_kind == "upload"
    result = client.post(
        "/api/vision/provider/reference",
        json={
            "snapshot_id": snap["id"],
            "pixel": [800, 450],
            "measured_mm": [200, 0, 0],
            "measurement_note": "test must be rejected",
        },
    )
    assert result.status_code == 400
    assert not (tmp_path / "calibration_references.jsonl").exists()
