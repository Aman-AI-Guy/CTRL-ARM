"""Pi camera client against a local stand-in of the camera server. No robot, no network camera."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from trova.camera import PiSource, open_source
from trova.pi_cam import CameraError, PiCam

_OK, _BUF = cv2.imencode(".jpg", np.full((40, 60, 3), 180, np.uint8))
assert _OK
JPEG = _BUF.tobytes()
assert JPEG.startswith(b"\xff\xd8") and JPEG.endswith(b"\xff\xd9")


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.startswith("/status.json"):
            if _Handler.mode == "paused":
                body = (b'{"camera":"paused","width":640,"height":480,"rotate":180,"temp":85,'
                        b'"protection":"paused","message":"cooling","warnings":["hot"]}')
            else:
                body = (b'{"camera":"idle","width":60,"height":40,"rotate":180,"temp":41.2,'
                        b'"protection":"normal","warnings":[]}')
            self._send(200, body, "application/json")
            return
        if self.path.startswith("/snapshot.jpg"):
            if _Handler.mode == "503":
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"cooling")
                return
            if _Handler.mode == "corrupt":
                self._send(200, b"\xff\xd8not-a-complete-jpeg", "image/jpeg")
                return
            if _Handler.mode == "hang":
                time.sleep(2)
                return
            self._send(200, JPEG, "image/jpeg")
            return
        self.send_error(404)

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def camera_server():
    _Handler.mode = "ok"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    yield host, port
    httpd.shutdown()


def _cam(host, port, **kwargs):
    return PiCam(host, port, timeout=kwargs.get("timeout", 1), retries=0, status_timeout=0.5)


def test_snapshot_is_a_fresh_jpeg_and_reports_status(camera_server):
    host, port = camera_server
    cam = _cam(host, port)
    status = cam.status()
    assert status["rotate"] == 180 and status["protection"] == "normal"
    cap = cam.capture()
    assert cap.jpeg == JPEG
    assert cap.meta["sync"] == "snapshot_after_request"
    assert cap.meta["sync_precision_claimed"] is False
    assert cap.meta["extra_rotate"] == 0
    assert cap.meta["requested_at"] <= cap.meta["received_at"]


def test_failure_drops_the_previous_frame(camera_server):
    host, port = camera_server
    src = open_source(f"pi:{host}:{port}")
    assert isinstance(src, PiSource)
    src.cam.timeout = 1
    src.cam.retries = 0
    src.cam.status_timeout = 0.5
    frame = src.read()
    assert frame.shape == (40, 60, 3)
    assert src.info()["online"] is True
    assert src.info()["image_held"] is False
    _Handler.mode = "503"
    with pytest.raises(RuntimeError):
        src.read()
    assert src._frame is None and src.last_capture is None
    assert src.info()["online"] is False
    _Handler.mode = "corrupt"
    with pytest.raises(CameraError):
        src.cam.get_jpeg()


def test_paused_camera_and_timeout_do_not_return_an_old_jpeg(camera_server):
    host, port = camera_server
    cam = _cam(host, port)
    assert cam.get_jpeg() == JPEG
    _Handler.mode = "paused"
    with pytest.raises(CameraError, match="cooling"):
        cam.capture()
    _Handler.mode = "hang"
    impatient = _cam(host, port, timeout=0.2)
    with pytest.raises(CameraError, match="can't reach"):
        impatient.get_jpeg()


def test_pi_spec_is_not_opened_as_a_video_file():
    src = open_source("pi:127.0.0.1:9")
    assert src.kind == "pi"
    assert src.info()["online"] is False
    src.close()
