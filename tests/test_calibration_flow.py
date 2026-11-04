"""Checkerboard workflow and a Pi JPEG, with the motion driver left simulated."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from trova.pi_cam import PiCam

from test_geom import _checker


def _pose(values, *, moving=False, error=0, source="telemetry"):
    return {
        "xyz_mm": list(values[:3]), "pose6": list(values), "joints": [1, 2, 3, 4, 5],
        "moving": moving, "source": source, "simulated": False, "error": None,
        "state_code": 2, "error_code": error, "warn_code": 0,
        "read_started_at": 10.0, "read_finished_at": 10.05,
        "motion_sent": False, "movements_enabled": False,
    }


class _Reader:
    def __init__(self, poses, **flags):
        self.poses = [list(pose) for pose in poses]
        self.flags = flags
        self.reads = 0

    def read(self):
        self.reads += 1
        pose = self.poses.pop(0) if len(self.poses) > 1 else self.poses[0]
        return _pose(pose, **self.flags)

    def inverse(self, pose, ref_joints=None):
        return {"ok": False, "ik_verified": False, "joints": None, "motion_sent": False,
                "reason": "not asked to move"}


def test_pi_jpeg_is_decoded_with_the_reported_rotation():
    image = np.zeros((40, 60, 3), np.uint8)
    image[5:20, 10:30] = (0, 0, 255)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/status.json"):
                body = b'{"camera":"streaming","protection":"normal","width":60,"height":40,"rotate":180}'
                self._send(body, "application/json")
            elif self.path.startswith("/snapshot.jpg"):
                self._send(jpeg, "image/jpeg")
            else:
                self.send_response(404)
                self.end_headers()

        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cam = PiCam("127.0.0.1", server.server_address[1], timeout=2, retries=0, status_timeout=2)
        captured = cam.capture()
        frame = cv2.imdecode(np.frombuffer(captured.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert frame.shape == (40, 60, 3)
        assert captured.meta["rotate_reported"] == 180
        assert captured.meta["rotated_again"] is False
        assert captured.meta["sync_precision_claimed"] is False
        assert captured.meta["width_reported"] == 60 and captured.meta["height_reported"] == 40
        assert captured.meta["provenance"] == "raspberry_pi"
    finally:
        server.shutdown()


def test_preview_and_handeye_capture_do_not_move_or_invent_a_calibration(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, client = _client(tmp_path, monkeypatch)
    state = app.state.trova
    tip = list(state.arm.tip)
    state.reader = _Reader([[10, 20, 300, 180, 0, 5]])
    preview = client.post("/api/vision/preview", json={"cols": 5, "rows": 4}).json()
    assert preview["moved"] is False and preview["persisted"] is False
    assert preview["found"] is False
    assert preview["observation"]["sync_precision_claimed"] is False
    assert preview["observation"]["robot"]["source"] == "telemetry"
    assert preview["observation"]["robot"]["motion_sent"] is False
    calib = client.get("/api/vision/calibration").json()
    assert calib["n_intrinsics"] == 0 and calib["n_handeye"] == 0
    assert calib["status"]["valid"] is False
    assert "never moves" in calib["status"]["procedure"][0]
    assert any("MotionGateway" in step for step in calib["status"]["first_move"])
    assert list(state.arm.tip) == tip

    state.override = _checker()
    state.reader = _Reader([[10, 20, 300, 180, 0, 5], [10, 20, 300, 180, 0, 5]])
    stored = client.post("/api/vision/handeye",
                         json={"cols": 5, "rows": 4, "square_mm": 25, "on_table": True}).json()
    assert stored["ok"] is True and stored["moved"] is False
    assert stored["found"] is True and stored["count"] == 1
    assert stored["pose_source"] == "telemetry"
    assert stored["progress"]["valid"] is False
    assert stored["progress"]["persisted_matrix"] is False
    assert stored["observation"]["sync_precision_claimed"] is False
    assert list(state.arm.tip) == tip

    state.reader = _Reader([[10, 20, 300, 180, 0, 0], [80, 20, 300, 180, 0, 0]])
    drifted = client.post("/api/vision/handeye",
                          json={"cols": 5, "rows": 4, "square_mm": 25, "on_table": True}).json()
    assert drifted["ok"] is False and drifted["moved"] is False
    assert "moved while the picture" in drifted["reason"]
    assert drifted["count"] == 1

    state.reader = _Reader([[10, 20, 300, 180, 0, 5], [10, 20, 300, 180, 0, 5]], moving=True)
    moving = client.post("/api/vision/handeye",
                        json={"cols": 5, "rows": 4, "square_mm": 25}).json()
    assert moving["ok"] is False and "moving" in moving["reason"] and moving["count"] == 1

    state.reader = _Reader([[10, 20, 300, 180, 0, 5], [10, 20, 300, 180, 0, 5]], error=22)
    fault = client.post("/api/vision/handeye",
                       json={"cols": 5, "rows": 4, "square_mm": 25}).json()
    assert fault["ok"] is False and "22" in fault["reason"] and fault["count"] == 1

    state.reader = _Reader([[10, 20, 300, 180, 0, 5]], source="simulated")
    simulated = client.post("/api/vision/handeye",
                           json={"cols": 5, "rows": 4, "square_mm": 25}).json()
    assert simulated["ok"] is False and "not telemetry" in simulated["reason"]
    assert simulated["count"] == 1 and list(state.arm.tip) == tip

    page = client.get("/").text
    assert "previewBoard()" in page and "captureBoard('handeye')" in page
    assert "collision avoidance: no" in page and "physical e-stop" in page
    state.reader = _Reader([[10, 20, 300, 180, 0, 5]])
    body = client.get("/api/state").json()
    assert body["robot"]["source"] == "telemetry"
    assert body["robot"]["movements_enabled"] is False
    assert body["robot"]["process_can_move"] is False
    assert body["vision_progress"]["n_handeye"] == 1
    assert body["vision_progress"]["valid"] is False
    assert body["motion"]["collision_avoidance"] is False
    assert body["motion"]["keep_out"] == "not_assessed"


def test_a_size_mismatch_and_a_live_driver_are_not_labeled_as_a_solved_calibration(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, client = _client(tmp_path, monkeypatch)
    state = app.state.trova
    frame = _checker()

    class _Pi:
        def info(self):
            return {"kind": "pi", "provenance": "raspberry_pi"}

        def read_capture(self):
            ok, encoded = cv2.imencode(".jpg", frame)
            assert ok
            return {"frame": frame, "jpeg": encoded.tobytes(), "meta": {
                "kind": "pi", "requested_at": 1.0, "received_at": 1.2, "width_reported": 10,
                "height_reported": 10, "rotate_reported": 180, "extra_rotate": 0,
                "rotated_again": False, "provenance": "raspberry_pi",
            }}

    state.source = _Pi()
    state.reader = _Reader([[10, 20, 300, 180, 0, 5], [10, 20, 300, 180, 0, 5]])
    mismatch = client.post("/api/vision/handeye",
                          json={"cols": 5, "rows": 4, "square_mm": 25}).json()
    assert mismatch["ok"] is False and mismatch["moved"] is False
    assert "10x10" in mismatch["reason"]
    assert mismatch["count"] == 0
    shot = client.post("/api/vision/intrinsics",
                      json={"cols": 5, "rows": 4, "square_mm": 25}).json()
    assert shot["ok"] is False and shot["count"] == 0
    solved = client.post("/api/vision/intrinsics/solve").json()
    assert solved["status"]["valid"] is False and solved["moved"] is False
    payload = json.loads((tmp_path / "vision_calib.json").read_text())
    assert payload["intrinsics"] is None and payload["hand_eye"] is None

    class _Backend:
        def tcp_pose(self):
            return (4.0, 5.0, 6.0, 180.0, 0.0, 1.0)

        def joints(self):
            return (0, 1, 2, 3, 4)

        def is_moving(self):
            return False

    class _Arm:
        def __init__(self):
            self.backend = _Backend()
            self.gateway = type("G", (), {"mode": "dry_run", "simulated": False})()

        def status(self):
            return {"state": "idle", "simulated": False, "tcp_mm": [4, 5, 6], "tip_mm": [4, 5, 6]}

    state.reader = None
    state.arm = _Arm()
    live = state.robot_telemetry()
    assert live["source"] == "telemetry" and live["movements_enabled"] is True
    assert live["process_can_move"] is False
    state.arm.gateway.mode = "hardware"
    armed = state.robot_telemetry()
    assert armed["process_can_move"] is True and armed["read_only"] is False
    state.reader = _Reader([[1, 2, 3, 180, 0, 0]])
    watched = state.robot_telemetry()
    assert watched["movements_enabled"] is False and watched["arm_state"] == "read-only"
    assert watched["process_can_move"] is True and watched["motion_sent"] is False
