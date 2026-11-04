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
    book.sessions[book.active]["intrinsics"] = {"k": [[800, 0, 4], [0, 800, 4], [0, 0, 1]], "width": 8, "height": 8}
    ranked = book.candidates([10, 20, 0], 100)
    assert ranked["reconstructed"] is False and ranked["coverage_is_not_a_reconstruction"] is True
    assert any(item["requires_joint_move"] and not item["executable_cartesian"] for item in ranked["candidates"])
    exported = book.export_colmap()
    assert exported["reconstructed"] is False and exported["cad"] is False and exported["textured_mesh"] is False
    folder = __import__("pathlib").Path(exported["folder"])
    assert "not been run" in (folder / "points3D.txt").read_text()
    assert "PINHOLE" in (folder / "cameras.txt").read_text()


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
    exported = client.post("/api/scan/export").json()
    assert exported["reconstructed"] is False
    assert client.post("/api/scan/finish").json()["status"] == "finished"


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
