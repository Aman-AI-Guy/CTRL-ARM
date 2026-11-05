"""Operator calibration, localization, and a closed loop that does not move hardware."""

import json

import cv2
import numpy as np
import pytest

from trova.acquire import ScanBook
from trova.agent import run, skill_angle, skill_center, skill_evaluate, skill_inspect, skill_move_above, skill_plan
from trova.arm import Limits
from trova.arm.telemetry import GuardedSDK, TelemetryWriteRefused, open_reader
from trova.boxes import Box
from trova.geom import DOF_NOTE, VisionStore, camera_pose_matrix, reference_error, tool_pose_matrix
from trova.inventory import describe
from trova.motion import apply_operator_safety
from trova.scene import build_objects


def test_read_only_reader_refuses_actuator_names_without_connecting():
    assert open_reader("") is None
    assert open_reader("off") is None
    assert open_reader("none") is None
    assert open_reader("sim") is None
    reader = open_reader("xarm:192.0.2.1")
    assert reader is not None and reader._inner is None
    guard = GuardedSDK(object())
    for name in ("motion_enable", "set_state", "set_position", "set_tcp_offset", "clean_error"):
        with pytest.raises(TelemetryWriteRefused):
            getattr(guard, name)()
    assert guard.calls == []


def test_rejected_sample_is_kept_out_and_a_drop_clears_matrices(tmp_path):
    store = VisionStore(tmp_path / "vision_calib.json")
    store.intrinsic_views = [{"width": 8, "height": 8, "square_mm": 20.0}] * 3
    store.intrinsics = {"k": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                        "width": 8, "height": 8, "reprojection_rms_px": 2.4}
    store.hand_eye = {"matrix": np.eye(4).tolist(), "consistency_mm": 0.0,
                      "max_rotation_deg": 20.0, "max_translation_mm": 40.0, "method": "fixture"}
    store.table = {"point_mm": [0, 0, 0], "normal": [0, 0, 1], "residual_mm": 9.0}
    noted = store.note_rejection("handeye", "no checkerboard in this image")
    assert noted["ok"] is False and noted["rejected"] == 1
    assert store.handeye_views == [] and store.assess(None)["valid"] is False
    assert store.intrinsics is not None and store.hand_eye is not None
    assert "not relaxed" in store.next_instruction()
    store.intrinsics["reprojection_rms_px"] = 0.2
    spread = {"one_axis": True, "reason": "the rotations stay near one axis", "ready_to_solve": False}
    instruction = store.next_instruction(spread)
    assert "near one axis" in instruction and DOF_NOTE in instruction
    assert "not a hand-eye matrix" in instruction
    dropped = store.drop_sample("intrinsics", 0)
    assert dropped["ok"] is True and dropped["persisted_matrix"] is False
    assert store.intrinsics is None and store.hand_eye is None and store.table is None
    missing = store.drop_sample("handeye", 4)
    assert missing["ok"] is False
    store.save()
    loaded = VisionStore.load(store.path)
    assert loaded.intrinsics is None and loaded.rejected and loaded.rejected[0]["accepted"] is False


def test_camera_pose_is_the_tool_pose_times_the_mount():
    pose = (120.0, -40.0, 310.0, 180.0, 0.0, 25.0)
    mount = tool_pose_matrix((15.0, -8.0, 40.0, 0.0, 0.0, 10.0))
    composed = tool_pose_matrix(pose) @ mount
    assert np.allclose(camera_pose_matrix(pose, mount), composed)
    assert np.allclose(camera_pose_matrix(pose, np.eye(4)), tool_pose_matrix(pose))


def test_reference_error_needs_a_valid_calibration_and_never_authorizes_motion():
    blocked = reference_error(None, [10, 20, 0], calibration_valid=False, pose_source="telemetry")
    assert blocked["error_mm"] is None and blocked["accepts_motion"] is False
    assert blocked["estimate_from_calibration"] is False and blocked["moved"] is False
    assert blocked["centered_image_is_not_localization"] is True
    from test_geom import _fixture_store
    store = _fixture_store()
    hit = store.localize((0, 0, 400, 180, 0, 0), 840, 300, [1280, 800])
    assert hit["ok"] and hit["verified_tool_position"] is False
    compared = reference_error(
        hit["point_mm"], [110.0, 60.0, 0.0], uncertainty_mm=hit["uncertainty_mm"],
        calibration_version=3, image_time="t", pose_source="telemetry", calibration_valid=True)
    assert compared["accepts_motion"] is False and compared["estimate_from_calibration"] is True
    assert compared["error_mm"] == pytest.approx(14.14, abs=0.5)
    assert compared["predicted_mm"][2] == pytest.approx(0.0, abs=1.0)
    assert compared["calibration_version"] == 3 and compared["pose_source"] == "telemetry"


def test_unknown_names_are_not_authoritative_and_stay_distinguishable():
    frame = np.zeros((40, 40, 3), np.uint8)
    boxes = [Box(0, 0, 10, 10, id=1), Box(12, 0, 30, 16, id=2)]

    class _Result:
        data = {"objects": [{"box_id": 1, "name": "", "note": "pink"},
                            {"box_id": 99, "name": "ghost"}], "summary": "s"}
        latency_s = 0.2
        offline = False
        mock = False

    class _Grok:
        def json(self, **_kwargs):
            return _Result()

    inv = describe(_Grok(), frame, boxes)
    assert inv.grok_mode == "live"
    assert inv.items[1]["name"] == "unknown" and inv.items[1]["label_authoritative"] is False
    assert inv.items[1]["note"] == "pink" and inv.items[1]["name_from_model"] is False
    assert inv.items[2]["name"] == "unknown" and 99 not in inv.items
    for box, track in zip(boxes, (7, 8), strict=True):
        box.extra["track_id"] = track
    objects = build_objects(boxes, frame, inv.items)
    assert [obj["name"] for obj in objects] == ["unknown", "unknown"]
    assert all(obj["geometric_detection"] and obj["semantic"] == obj["note"] for obj in objects)
    assert all(obj["ambiguous"] and not obj["label_authoritative"] for obj in objects)
    assert objects[0]["semantic"] == "pink" and objects[0]["name"] != "pink"

    _Result.data = {"objects": [{"box_id": 1, "name": "pink highlighter", "note": "cap on the left"}],
                    "summary": "s"}
    _Result.mock = True
    named = describe(_Grok(), frame, boxes[:1])
    assert named.items[1]["name"] == "pink highlighter"
    assert named.items[1]["label_authoritative"] is False and named.grok_mode == "mock"


def test_operator_safety_rejects_unconfirmed_and_over_limit_values():
    from test_agent import _arm
    arm = _arm()
    gateway = arm.gateway
    mode = gateway.mode
    limits = gateway.limits
    refused = apply_operator_safety(gateway, {"confirm": False, "limits": "0,500,-200,200,0,400"})
    assert refused["ok"] is False and refused["mode_unchanged"] is True
    assert gateway.mode == mode and gateway.limits == limits and gateway.keepout_assessed is False
    stored = apply_operator_safety(gateway, {"confirm": True, "keep_out": "none", "speed_mm_s": 40,
                                             "observe_clearance_mm": 80})
    assert stored["ok"] is True and stored["collision_avoidance"] is False
    assert gateway.mode == mode and gateway.keepout_assessed is True and gateway.keepouts == []
    assert gateway.max_speed_mm_s == 40 and arm.speed == 40
    assert stored["clearance_mm"] == 80
    too_fast = apply_operator_safety(gateway, {"confirm": True, "speed_mm_s": 9999})
    assert too_fast["ok"] is False and gateway.max_speed_mm_s == 40 and gateway.mode == mode
    unknown = apply_operator_safety(gateway, {"confirm": True, "mode": "hardware"})
    assert unknown["ok"] is False and "mode" in unknown["reason"] and gateway.mode == mode
    joints_before = gateway.joint_limits
    blank_joints = apply_operator_safety(gateway, {"confirm": True, "joint_limits": " "})
    assert blank_joints["ok"] is False and gateway.joint_limits == joints_before
    assert "empty" in blank_joints["reason"]
    assert "links" in " ".join(stored["unchecked"]).lower() or any("not in the model" in line for line in stored["unchecked"])


def test_lost_target_stops_on_the_second_miss_and_centering_is_not_a_move():
    from test_agent import _Host, _arm, _task
    from test_geom import _fixture_store
    arm = _arm()
    host = _Host(arm, _fixture_store())
    host.redetect = lambda frame: []
    task = _task()
    task["planned_target"] = [100, 50, 150]
    before = list(arm.tcp_mm())
    first = skill_evaluate(host, task, "")
    assert not first.get("terminal") and task["phase"] == "lost"
    assert task["selected"]["track_id"] == 1 and task["replans"] == 1
    assert first["goal_achieved"] is False
    second = skill_evaluate(host, task, "")
    assert second["terminal"] is True and task["phase"] == "blocked"
    assert "lost again" in task["motion_lock"]
    assert list(arm.tcp_mm()) == before

    centered = _task()
    centered["selected"]["center"] = [640, 400]
    near = skill_center(host, centered, "")
    assert near["executed"] is False and near.get("motion_allowed") is False
    assert "not a verified 3D position" in near["reason"]
    assert centered.get("planned_target") is None
    assert list(arm.tcp_mm()) == before

    host.robot = lambda: {"xyz_mm": [300, 0, 200], "pose6": [300, 0, 200], "joints": [0, 0, 0, 0, 0],
                          "moving": False, "source": "telemetry", "error_code": 0}
    angled = skill_angle(host, _task(), "")
    assert angled["executed"] is False and "yaw was not read" in angled["reason"]
    assert "20" not in angled["reason"]
    assert list(arm.tcp_mm()) == before

    host.robot = lambda: {"xyz_mm": [400.0, 10.0, 250.0], "pose6": [400.0, 10.0, 250.0, 180.0, 0.0, 25.0],
                          "joints": [1, 2, 3, 4, 5], "moving": False, "source": "telemetry", "error_code": 0}
    host.inverse = lambda pose, joints=None: {
        "ok": False, "ik_verified": False, "joints": None, "motion_sent": False, "reason": "not reached"}
    unverified = skill_angle(host, _task(), "")
    assert unverified["executed"] is False and unverified.get("ik_verified") is False
    assert "No move was sent" in unverified["reason"]
    assert list(arm.tcp_mm()) == before


def test_uncertainty_outside_the_workspace_blocks_without_widening_limits():
    from test_agent import _Host, _arm, _task
    from test_geom import _fixture_store
    limits = "50,350,-80,80,100,250"
    arm = _arm(limits)
    before_limits = (arm.gateway.limits.x, arm.gateway.limits.y, arm.gateway.limits.z)
    host = _Host(arm, _fixture_store())
    task = _task()
    task["location"]["uncertainty_mm"] = 80.0
    blocked = skill_plan(host, task, "")
    assert blocked["ok"] is False and blocked["executed"] is False
    assert blocked["motion"]["execution_authorized"] is False
    assert blocked["motion"]["object_id"] == 1
    assert blocked["motion"]["uncertainty_mm"] == 80.0
    assert blocked["motion"]["uncertainty_does_not_expand_limits"] is True
    assert "uncertainty" in blocked["reason"]
    assert list(arm.tcp_mm())[0] == pytest.approx(300, abs=1)
    assert (arm.gateway.limits.x, arm.gateway.limits.y, arm.gateway.limits.z) == before_limits
    host2 = _Host(_arm(limits), _fixture_store())
    wide = _task()
    wide["location"]["uncertainty_mm"] = 10.0
    planned = skill_plan(host2, wide, "")
    assert planned["ok"] is True and planned["executed"] is False
    assert planned["motion"]["uncertainty_mm"] == 10.0
    assert planned["motion"]["localization_mm"] == [100.0, 50.0, 0.0]
    assert list(host2.arm.tcp_mm())[0] == pytest.approx(300, abs=1)


def test_a_rejected_approach_is_not_retried_after_the_limits_are_widened():
    from test_agent import _Host, _arm, _task
    from test_geom import _fixture_store
    arm = _arm("0,40,-10,10,0,10")
    host = _Host(arm, _fixture_store())
    task = _task()
    refused = skill_plan(host, task, "")
    assert refused["ok"] is False and refused["executed"] is False
    assert refused["motion"]["execution_authorized"] is False
    assert task["motion_lock"]
    calls = {"n": 0}
    real = host.gateway.authorize_cartesian

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    host.gateway.authorize_cartesian = wrapped
    host.gateway.limits = Limits.parse("0,600,-400,400,0,500")
    before = list(arm.tcp_mm())
    again = skill_move_above(host, task, "forward 20")
    assert again["executed"] is False and calls["n"] == 0
    assert list(arm.tcp_mm()) == before
    long = host.gateway.authorize_cartesian((900, 0, 200), 50, start=(300, 0, 200))
    assert long.status == "rejected" and long.allow_motion is False
    assert list(arm.tcp_mm()) == before


def test_step_mode_is_bounded_and_a_fresh_snapshot_is_not_reused_after_a_move():
    from test_agent import _Host, _arm
    from trova.geom import VisionStore

    class _Grok:
        def __init__(self):
            self.skills = ["inspect_scene", "list_objects", "list_objects"]
            self.index = 0

        def json(self, **_kwargs):
            skill = self.skills[min(self.index, len(self.skills) - 1)]
            self.index += 1
            return type("A", (), {"data": {"skill": skill, "argument": "", "stop": False},
                                  "offline": False, "mock": True})()

    host = _Host(_arm(), VisionStore())
    host.grok = _Grok()
    shots = {"n": 0}

    def snapshot():
        shots["n"] += 1
        return {"objects": [{"track_id": 1, "name": "unknown", "label_authoritative": False}],
                "grok_mode": "mock", "summary": "one", "width": 8, "height": 8,
                "camera_kind": "pi", "id": "snap", "provenance": "pi"}

    host.snapshot = snapshot
    host.note_stage = lambda stage, seconds: None
    first = run(host, "trova l'evidenziatore rosa e posizionati sopra", step=True)
    assert first["status"] == "step" and first["loop"] == "step"
    assert len(first["steps"]) == 1 and first["steps"][0]["skill"] == "inspect_scene"
    assert first["hardware_moved"] is False and first["realtime_claimed"] is False
    assert first["autonomous_execution_allowed"] is False and shots["n"] == 1
    second = run(host, first["goal"], step=True, continued=first["resume"])
    assert second["status"] == "step" and len(second["steps"]) == 2
    assert second["hardware_moved"] is False and shots["n"] == 1
    third = run(host, first["goal"], step=True, continued=second["resume"])
    assert third["status"] == "blocked" and third["hardware_moved"] is False
    assert len(third["steps"]) == 3

    pose = host.robot()
    host.last_snapshot = {"monotonic": __import__("time").monotonic(), "objects": [], "grok_mode": "live",
                          "summary": "cached", "width": 8, "height": 8, "id": "old",
                          "pose": pose, "camera_kind": "pi", "provenance": "pi"}
    cached = skill_inspect(host, {"phase": "start", "goal": "again"}, "")
    assert cached["snapshot_reused"] is True and shots["n"] == 1
    moved = skill_inspect(host, {"phase": "moved", "goal": "again"}, "")
    assert moved["snapshot_reused"] is False and shots["n"] == 2


def test_scan_keeps_pose_relation_without_claiming_sync(tmp_path):
    book = ScanBook(tmp_path)
    book.start("highlighter")
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    jpeg = buf.tobytes()
    mount = np.eye(4).tolist()
    good = book.capture(jpeg, {
        "pose6": [10, 20, 200, 180, 0, 10], "robot_tcp_mm": [10, 20, 200],
        "joints_deg": [1, 2, 3, 4, 5], "pose_source": "telemetry",
        "requested_at": 20.0, "stopped_at": 19.0, "received_at": 20.1, "read_finished_at": 20.12,
        "width": 8, "height": 8, "tool_camera": mount, "calibration_version": 0,
        "moving": False, "error_code": 0,
    }, {"track_id": 4, "name": "unknown", "label_authoritative": False, "association": "same_view"})
    assert good["accepted"] is True and good["sync_precision_claimed"] is False
    assert good["pose_image_relation"] == "T_base_camera = T_base_tool @ T_tool_camera"
    assert good["sync_quality"] == "wall_clock_gap_only" and good["skew_s"] == pytest.approx(0.02, abs=1e-4)
    assert good["camera_pose"] is not None and good["viewpoint"]["id"] == good["viewpoint_id"]
    assert good["detection"]["label_authoritative"] is False
    bad = book.capture(b"not-a-jpeg", {"pose6": [10, 20, 200, 180, 0, 10], "pose_source": "telemetry",
                                       "width": 8, "height": 8, "tool_camera": mount})
    assert bad["accepted"] is False and bad["camera_pose"] is None
    assert bad["pose_image_relation"] is None and bad["sync_precision_claimed"] is False


def test_qualify_cli_checks_a_file_and_does_not_connect(tmp_path):
    from trova.qualify import main
    path = tmp_path / "safety.json"
    path.write_text(json.dumps({"confirm": True, "keep_out": "none", "speed_mm_s": 20}))
    assert main([str(path)]) == 0
    path.write_text(json.dumps({"confirm": False, "speed_mm_s": 20}))
    assert main([str(path)]) == 1
    path.write_text(json.dumps({"confirm": True, "speed_mm_s": 9999}))
    assert main([str(path)]) == 1


def test_api_reference_safety_and_page_do_not_move(tmp_path, monkeypatch, capsys):
    from test_pipeline import _client
    app, client = _client(tmp_path, monkeypatch)
    arm = app.state.trova.arm
    before = list(arm.tip)
    blocked = client.post("/api/vision/reference", json={"x_mm": 10, "y_mm": 20, "z_mm": 0, "u": 4, "v": 4}).json()
    assert blocked["error_mm"] is None and blocked["accepts_motion"] is False and blocked["moved"] is False
    assert blocked["estimate_from_calibration"] is False and blocked["missing"]
    missing_track = client.post("/api/vision/reference", json={"x_mm": 1, "y_mm": 2, "z_mm": 3, "track_id": 1})
    assert missing_track.status_code == 400
    st = app.state.trova
    robot = st.robot_telemetry()
    st.inventory = {"snapshot": {
        "time": "t", "width": 8, "height": 8,
        "pose": {"pose6": [0, 0, 0, 180, 0, 0], "moving": False},
        "objects": [{"track_id": 7, "center": [3, 3]}],
    }}
    stale = client.post("/api/vision/reference", json={"x_mm": 1, "y_mm": 2, "z_mm": 3, "u": 3, "v": 3}).json()
    assert stale["error_mm"] is None and stale["accepts_motion"] is False and stale["moved"] is False
    assert "picture" in stale["reason"]
    assert robot["source"] == "simulated"
    st.inventory["snapshot"]["pose"] = {"pose6": robot.get("pose6"), "moving": True}
    moving = client.post("/api/vision/reference", json={"x_mm": 1, "y_mm": 2, "z_mm": 3, "track_id": 7}).json()
    assert moving["moved"] is False and moving["error_mm"] is None and "moving" in moving["reason"]

    mode = arm.gateway.mode
    quiet = client.post("/api/safety", json={"confirm": False, "speed_mm_s": 30}).json()
    assert quiet["ok"] is False and quiet["moved"] is False and quiet["grok_can_edit"] is False
    assert arm.gateway.mode == mode
    assert client.post("/api/safety", json={"confirm": True, "mode": "hardware"}).status_code == 422
    stored = client.post("/api/safety", json={"confirm": True, "keep_out": "none", "speed_mm_s": 25}).json()
    assert stored["ok"] is True and stored["moved"] is False and stored["collision_avoidance"] is False
    assert arm.gateway.mode == mode and arm.gateway.keepout_assessed is True
    fast = client.post("/api/safety", json={"confirm": True, "speed_mm_s": 900}).json()
    assert fast["ok"] is False and arm.gateway.max_speed_mm_s == 25
    assert client.post("/api/agent/task", json={"text": "go", "limits": "0,1,0,1,0,1"}).status_code == 422
    dropped = client.post("/api/vision/sample/drop", json={"kind": "handeye", "index": 3}).json()
    assert dropped["ok"] is False and dropped["moved"] is False
    st.vision.intrinsic_views.append({"width": 4, "height": 4})
    st.vision.intrinsics = {"k": [[1]]}
    cleared = client.post("/api/vision/sample/drop", json={"kind": "intrinsics", "index": 0}).json()
    assert cleared["ok"] is True and cleared["moved"] is False and st.vision.intrinsics is None
    page = client.get("/").text
    for needle in ("runAgent(true)", "previewBoard()", "captureBoard('handeye')", "Salva i valori misurati",
                   'id="calibnext"', 'id="locinfo"', "arresto di emergenza fisico", "anticollisione: no",
                   "calibrazione non valida", "movimento eseguito", 'id="mode"'):
        assert needle in page, needle
    tools = client.get("/api/voice/config").json()["session"]["tools"]
    task_tool = next(item for item in tools if item["name"] == "run_vision_task")
    assert set(task_tool["parameters"]["properties"]) == {"text"}
    state = client.get("/api/state").json()
    assert state["safety"]["collision_avoidance"] is False and state["safety"]["grok_can_edit"] is False
    assert state["latency"]["realtime_claimed"] is False
    assert state["robot"]["movements_enabled"] is False
    latency = client.get("/api/latency").json()
    assert latency["realtime_claimed"] is False
    assert list(arm.tip) == before
    from trova.qualify import main
    assert main([]) == 2
    capsys.readouterr()
