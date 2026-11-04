"""Read-only check of the Raspberry Pi camera. Skips when the host does not answer.

This test does not enable the robot and does not send a motion command.
Set TROVA_CAM_HOST to the current DHCP address when trova-cam.local does not resolve.
"""

import os

import cv2
import numpy as np
import pytest

from trova.pi_cam import CameraError, PiCam


def test_live_camera_status_and_snapshot():
    host = os.environ.get("TROVA_CAM_HOST") or "trova-cam.local"
    port = int(os.environ.get("TROVA_CAM_PORT") or 8080)
    cam = PiCam(host, port, timeout=4, retries=0, status_timeout=2)
    try:
        status = cam.status()
        jpeg = cam.get_jpeg()
    except CameraError as exc:
        pytest.skip(f"camera unreachable at {host}:{port}: {exc}")
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.ndim == 3
    assert status.get("camera") in ("idle", "streaming", "paused")
    if status.get("protection") == "normal" and status.get("width") and status.get("height"):
        assert (image.shape[1], image.shape[0]) == (int(status["width"]), int(status["height"]))
    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
