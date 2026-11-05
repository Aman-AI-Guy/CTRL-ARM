"""Skill loop, gateway enforcement, and scan sessions. The arm in these tests is simulated."""

import time

import cv2
import numpy as np

from trova.acquire import ScanBook
from trova.agent import skill_evaluate, skill_move_above, skill_plan
from trova.arm import CartesianArm, FakeCartesian, Limits
from trova.motion import CameraBody


def _arm(limits="0,600,-400,400,0,500"):
    arm = CartesianArm(FakeCartesian(start=(300.0, 0.0, 200.0)), z_hover_mm=200,
                       home_mm=(300.0, 0.0, 200.0), limits=Limits.parse(limits), speed_mm_s=800)
    arm.gateway.camera = CameraBody((0.0, 0.0, 30.0), 20.0)
    return arm


class _Host:
    def __init__(self, arm, vision):
        self.arm = arm
        self.vision = vision
        self.gateway = arm.gateway
        self.grok = None
        self.settle_s = 0
        self.cancel = type("E", (), {"is_set": staticmethod(lambda: False)})()
        self.camera_kind = "pi"
        self.image_size = [1280, 800]
        self.observe_clearance_mm = 150.0
        self.frame = np.zeros((800, 1280, 3), np.uint8)
        self.boxes = []

    def robot(self):
        pose = [float(v) for v in self.arm.backend.tcp_pose()]
        return {"xyz_mm": pose[:3], "pose6": pose, "joints": self.arm.backend.joints(),
                "moving": self.arm.backend.is_moving(), "source": "telemetry"}

    def execute(self, target):
        ok = bool(self.gateway._execute_cartesian(tuple(target)))
        return {"ok": ok, "reason": self.arm.last_error or ("moved" if ok else "not moved"),
                "final_mm": [float(v) for v in self.arm.tcp_mm()]}

    def read_frame(self):
        return self.frame, {"kind": "pi", "requested_at": time.time()}

    def redetect(self, frame):
        return []


def _task(point=(100.0, 50.0, 0.0)):
    return {
        "goal": "move above the pen", "phase": "located",
        "selected": {"track_id": 1, "name": "pen", "center": [840, 300], "box": [800, 260, 880, 340],
                     "confidence": 0.8},
        "location": {"ok": True, "point_mm": list(point)},
        "planned_target": None, "proposal": None, "motion_lock": None, "calibration_valid": True,
        "authoritative": True, "evaluation": None,
    }


def test_rejected_move_is_not_retried_and_coordinates_are_ignored():
    from test_geom import _fixture_store
    arm = _arm("0,40,-10,10,0,10")
    host = _Host(arm, _fixture_store())
    calls = {"n": 0}
    real = host.gateway.authorize_cartesian

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    host.gateway.authorize_cartesian = wrapped
    task = _task()
    refused = skill_plan(host, task, "x=100 y=50 z=10")
    assert refused["ok"] is False and calls["n"] == 0
    assert "coordinates" in task["motion_lock"]
    task["motion_lock"] = None
    rejected = skill_plan(host, task, "")
    assert rejected["ok"] is False and calls["n"] == 1
    assert task["motion_lock"]
    before = list(arm.tcp_mm())
    again = skill_move_above(host, task, "forward 400")
    assert again["executed"] is False and calls["n"] == 1
    assert list(arm.tcp_mm()) == before


def test_qualified_simulation_moves_only_through_the_gateway():
    from test_geom import _fixture_store
    arm = _arm()
    host = _Host(arm, _fixture_store())
    task = _task()
    planned = skill_plan(host, task, "")
    assert planned["ok"] and planned["executed"] is False
    assert planned["motion"]["collision_avoidance"] is False
    assert task["planned_target"] == [100.0, 50.0, 150.0] or abs(task["planned_target"][0] - 100) < 1
    # Drop the qualification: a synthetic camera must not be sent.
    host.camera_kind = "demo"
    held = skill_move_above(host, task, "")
    assert held["executed"] is False and "synthetic" in held["reason"]
    assert abs(arm.tcp_mm()[0] - 300) < 1
    task["motion_lock"] = None
    host.camera_kind = "pi"
    moved = skill_move_above(host, task, "")
    assert moved["executed"] is True
    assert np.allclose(arm.tcp_mm(), [100, 50, 150], atol=2.0)


def test_image_center_is_not_a_reached_goal():
    from test_geom import _fixture_store
    host = _Host(_arm(), _fixture_store())
    host.redetect = lambda frame: [{"track_id": 1, "center": [640, 400], "association": "same_view",
                                    "identity_uncertain": False}]
    task = _task()
    task["planned_target"] = [100, 50, 150]
    task["selected"] = {"track_id": 1}
    result = skill_evaluate(host, task, "")
    checks = result["evaluation"]
    assert checks["near_intended_image_location"] is True
    assert checks["goal_achieved"] is False
    assert checks["image_centering_is_not_3d_proximity"] is True
    assert checks["robot_reached_planned_position"] is False


def test_scan_session_keeps_metadata_and_does_not_reconstruct(tmp_path):
    book = ScanBook(tmp_path)
    book.start("pen")
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    jpeg = buf.tobytes()
    stale = book.capture(jpeg, {"moving": False, "pose6": [0, 0, 200, 180, 0, 0], "robot_tcp_mm": [0, 0, 200],
                                "requested_at": 1.0, "stopped_at": 5.0})
    assert stale["accepted"] is False and stale["sync_precision_claimed"] is False
    good = book.capture(jpeg, {"pose6": [10, 20, 200, 180, 0, 0], "robot_tcp_mm": [10, 20, 200],
                               "requested_at": 20.0, "stopped_at": 19.0, "width": 8, "height": 8,
                               "tool_camera": np.eye(4).tolist(), "calibration_version": 2,
                               "sync": "snapshot_after_request"})
    assert good["accepted"] is True and good["camera_pose"] is not None
    assert good["calibration_version"] == 2
    assert good["object_id"] == "pen" and good["recorded_at"]
    simulated = book.capture(jpeg, {"pose6": [10, 20, 200, 180, 0, 0], "robot_tcp_mm": [10, 20, 200],
                                    "pose_source": "simulated", "requested_at": 30.0, "stopped_at": 29.0,
                                    "width": 8, "height": 8, "tool_camera": np.eye(4).tolist(),
                                    "joints_deg": [1, 2, 3, 4, 5], "calibration_version": 2})
    assert simulated["accepted"] is False and simulated["camera_pose"] is None
    assert "telemetry" in simulated["reject_reason"]
    faulted = book.capture(jpeg, {"pose6": [10, 20, 200, 180, 0, 0], "robot_tcp_mm": [10, 20, 200],
                                  "pose_source": "telemetry", "requested_at": 40.0, "stopped_at": 39.0,
                                  "width": 8, "height": 8, "tool_camera": np.eye(4).tolist(),
                                  "error_code": 22, "calibration_version": 2})
    assert faulted["accepted"] is False and faulted["camera_pose"] is None
    assert "error code 22" in faulted["reject_reason"]
    assert faulted["sync_precision_claimed"] is False
    assert simulated["object_id"] == "pen" and simulated["joints_deg"] == [1, 2, 3, 4, 5]
    book.sessions[book.active]["intrinsics"] = {"k": [[800, 0, 4], [0, 800, 4], [0, 0, 1]], "width": 8, "height": 8}
    ranked = book.candidates([10, 20, 0], 100)
    assert ranked["reconstructed"] is False and ranked["coverage_is_not_a_reconstruction"] is True
    assert any(item["requires_joint_move"] and not item["executable_cartesian"] for item in ranked["candidates"])
    exported = book.export_colmap()
    assert exported["reconstructed"] is False and exported["cad"] is False and exported["textured_mesh"] is False
    folder = __import__("pathlib").Path(exported["folder"])
    assert "not been run" in (folder / "points3D.txt").read_text()
    assert "PINHOLE" in (folder / "cameras.txt").read_text()
    assert "v0003" not in (folder / "images.txt").read_text()
    assert not (folder / "images" / "v0003.jpg").exists()


def test_api_task_stays_blocked_on_the_demo_and_rejects_a_mode_field(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, client = _client(tmp_path, monkeypatch)
    arm = app.state.trova.arm
    before = list(arm.tip)
    body = client.post("/api/agent/task", json={"text": "trova la penna blu e posizionati sopra"}).json()
    assert body["hardware_moved"] is False
    assert body["verified_tool_position"] is False
    assert body["collision_avoidance"] is False
    assert body["status"] in ("held", "blocked", "clarify")
    assert list(arm.tip) == before
    assert client.post("/api/agent/task", json={"text": "go", "mode": "hardware"}).status_code == 422
    calib = client.get("/api/vision/calibration").json()
    assert calib["intrinsics"] is None and calib["status"]["valid"] is False and calib["moved"] is False
    names = [item["name"] for item in client.get("/api/voice/config").json()["session"]["tools"]]
    assert "run_vision_task" in names
    shot = client.post("/api/vision/handeye", json={"cols": 5, "rows": 4, "square_mm": 25, "on_table": True}).json()
    assert shot["moved"] is False and list(arm.tip) == before
    assert client.post("/api/vision/intrinsics", json={"cols": 5, "rows": 4, "square_mm": 25, "unlock": "MOVE"}).status_code == 422
    started = client.post("/api/scan/start", json={"object_id": "pen"}).json()
    assert started["reconstructed"] is False
    captured = client.post("/api/scan/capture").json()
    assert captured["sync_precision_claimed"] is False
    assert captured["accepted"] is False
    assert captured["pose_source"] == "simulated"
    assert "telemetry" in (captured.get("reject_reason") or "")
    page = client.get("/").text
    assert 'id="cambadge"' in page and 'id="motionbadge"' in page and 'id="movedbadge"' in page
    assert ("calibration not valid" in page or "calibrazione non valida" in page)
    assert ("movement not executed" in page or "movimento non autorizzato" in page)
    assert ("Grok online" in page or "S.grok" in page)
    exported = client.post("/api/scan/export").json()
    assert exported["reconstructed"] is False
    assert client.post("/api/scan/finish").json()["status"] == "finished"


def test_uncalibrated_plan_names_measurements_and_does_not_invent_a_target():
    from trova.agent import skill_plan
    from trova.geom import VisionStore
    arm = _arm()
    host = _Host(arm, VisionStore())
    host.observe_clearance_mm = None
    host.gateway.limits = None
    host.gateway.camera = None
    host.robot = lambda: {"xyz_mm": [1, 2, 3], "pose6": None, "source": "simulated", "moving": False}
    task = _task()
    task["authoritative"] = False
    task["grok_mode"] = "mock"
    task["calibration_valid"] = False
    before = list(arm.tcp_mm())
    planned = skill_plan(host, task, "")
    assert planned["executed"] is False and task.get("planned_target") is None
    assert list(arm.tcp_mm()) == before
    for needle in ("intrinsics", "hand-eye", "table plane", "TROVA_OBSERVE_CLEARANCE_MM",
                   "TROVA_ARM_LIMITS", "TROVA_CAMERA_BODY_MM", "mock", "simulated"):
        assert needle in planned["reason"], needle


def test_calibrated_proposal_is_not_executed_from_a_simulated_pose():
    from test_geom import _fixture_store
    arm = _arm()
    host = _Host(arm, _fixture_store())
    host.robot = lambda: {"xyz_mm": [300.0, 0.0, 200.0], "pose6": [300.0, 0.0, 200.0, 180, 0, 0],
                          "joints": [0, 0, 0, 0, 0], "moving": False, "source": "simulated"}
    task = _task()
    before = list(arm.tcp_mm())
    planned = skill_plan(host, task, "")
    assert planned["ok"] and planned["executed"] is False
    assert np.allclose(task["planned_target"], [100, 50, 150], atol=1.0)
    moved = skill_move_above(host, task, "")
    assert moved["executed"] is False and "simulated" in moved["reason"]
    assert np.allclose(arm.tcp_mm(), before)


def test_a_repeated_skill_stops_and_names_the_blockers():
    from trova.agent import run
    from trova.geom import VisionStore

    class _Answer:
        data = {"skill": "list_objects", "argument": "", "stop": False}
        offline = False
        mock = False

    class _Grok:
        def json(self, **_kwargs):
            return _Answer()

    host = _Host(_arm(), VisionStore())
    host.grok = _Grok()
    host.observe_clearance_mm = None
    host.snapshot = lambda: {"objects": [], "grok_mode": "live", "summary": "nothing", "width": 8, "height": 8}
    result = run(host, "list the table")
    assert result["hardware_moved"] is False and result["motion_proposed"] is False
    assert result["status"] == "blocked"
    assert len(result["steps"]) == 2
    assert "TROVA_OBSERVE_CLEARANCE_MM" in result["reason"]
    assert "intrinsics" in result["reason"]


def test_cancel_stops_before_a_skill_and_a_stale_pose_blocks_localization():
    import threading

    from trova.agent import run, skill_locate
    from trova.geom import VisionStore

    host = _Host(_arm(), VisionStore())
    host.cancel = threading.Event()
    host.cancel.set()
    host.grok = None
    stopped = run(host, "Grok, trova l'evidenziatore rosa e posizionati sopra.")
    assert stopped["status"] == "stopped" and stopped["steps"] == []
    assert stopped["hardware_moved"] is False and stopped["verified_tool_position"] is False

    host.cancel = type("E", (), {"is_set": staticmethod(lambda: False)})()
    task = _task()
    task["location"] = None
    task["snapshot_pose"] = {"pose6": [0, 0, 200, 180, 0, 0], "moving": False}
    host.robot = lambda: {"xyz_mm": [30, 0, 200], "pose6": [30, 0, 200, 180, 0, 0],
                          "moving": False, "source": "telemetry"}
    stale = skill_locate(host, task, "")
    assert stale["ok"] is False and "changed" in stale["reason"]
    assert task["location"] is None

    task["motion_lock"] = None
    task["snapshot_pose"] = {"pose6": [0, 0, 200, 180, 0, 0], "moving": True}
    host.robot = lambda: {"xyz_mm": [0, 0, 200], "pose6": [0, 0, 200, 180, 0, 0],
                          "moving": False, "source": "telemetry"}
    moving = skill_locate(host, task, "")
    assert "stationary" in moving["reason"]

    task["motion_lock"] = None
    task["snapshot_pose"] = {"pose6": [0, 0, 200, 180, 0, 0], "moving": False}
    missed = skill_locate(host, task, "")
    assert missed["ok"] is False and "intrinsics" in missed["reason"]
    assert (missed.get("location") or {}).get("point_mm") is None


def test_approach_uses_telemetry_and_refuses_unestablished_clearance():
    from test_geom import _fixture_store
    arm = _arm()
    host = _Host(arm, _fixture_store())
    host.robot = lambda: {"xyz_mm": [400.0, 10.0, 250.0], "pose6": [400.0, 10.0, 250.0, 180.0, 0.0, 25.0],
                          "joints": [1, 2, 3, 4, 5], "moving": False, "source": "telemetry", "error_code": 0}
    before = list(arm.tcp_mm())
    task = _task()
    planned = skill_plan(host, task, "")
    assert planned["ok"] and planned["executed"] is False
    assert np.allclose(planned["motion"]["proposal"]["start_mm"], [400, 10, 250])
    assert np.allclose(planned["motion"]["proposal"]["target_mm"], [100, 50, 150], atol=1)
    assert planned["motion"]["trajectory_mm"][0] == planned["motion"]["proposal"]["start_mm"]
    assert planned["motion"]["pose6"][3:] == [180.0, 0.0, 25.0]
    assert planned["motion"]["collision_avoidance"] is False
    assert planned["motion"]["ik_verified"] is False
    assert planned["target"]["yaw_from_telemetry"] is True
    assert np.allclose(arm.tcp_mm(), before)

    host.inverse = lambda pose, joints=None: {
        "ok": True, "ik_verified": True, "joints": [0, 10, -20, 30, 40], "motion_sent": False, "reason": "ok"}
    confirmed = skill_plan(host, _task(), "")
    assert confirmed["motion"]["ik_verified"] is True
    assert confirmed["motion"]["ik_joints_deg"] == [0, 10, -20, 30, 40]
    assert np.allclose(arm.tcp_mm(), before)

    bare_arm = _arm()
    bare = _Host(bare_arm, _fixture_store())
    bare.robot = host.robot
    bare.gateway.camera = None
    refused = skill_plan(bare, _task(), "")
    assert refused["executed"] is False and refused["ok"] is False
    assert "clearance cannot be established" in refused["reason"]
    assert np.allclose(refused["motion"]["proposal"]["target_mm"], [100, 50, 150], atol=1)
    assert np.allclose(bare_arm.tcp_mm(), before)

    locked_arm = _arm()
    locked = _Host(locked_arm, _fixture_store())
    locked.robot = host.robot
    locked.inverse = host.inverse
    locked.read_only_robot = True
    proposal = skill_plan(locked, _task(), "")
    assert proposal["ok"] and proposal["executed"] is False
    locked.gateway.mode = "hardware"
    held = skill_move_above(locked, locked_task(proposal), "")
    assert held["executed"] is False
    assert "keep-out" in held["reason"]
    assert "read-only" in held["reason"]
    assert np.allclose(locked_arm.tcp_mm(), before)


def locked_task(planned):
    task = _task()
    task["planned_target"] = planned["motion"]["proposal"]["target_mm"]
    task["proposal"] = planned["motion"]
    task["ik_verified"] = True
    task["location"] = {"ok": True, "point_mm": [100.0, 50.0, 0.0]}
    task["_robot"] = {"xyz_mm": [400.0, 10.0, 250.0], "pose6": [400.0, 10.0, 250.0, 180.0, 0.0, 25.0],
                      "source": "telemetry", "moving": False, "error_code": 0}
    return task


def test_state_does_not_snapshot_the_pi(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, client = _client(tmp_path, monkeypatch, TROVA_CAMERA="pi", TROVA_CAM_HOST="127.0.0.1", TROVA_CAM_PORT="9")
    source = app.state.trova.source
    source.cam.timeout = 0.2
    source.cam.retries = 0
    source.cam.status_timeout = 0.2

    def boom():
        raise AssertionError("the status poll took a snapshot")

    source.read = boom
    source.read_capture = boom
    body = client.get("/api/state").json()
    assert body["source"]["kind"] == "pi"
    assert body["source"]["online"] is False
    assert body["vision"]["valid"] is False
    assert body["robot"]["source"] == "simulated"
