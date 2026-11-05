import argparse
import socket

import pytest

from trova.station import configure


def args(**values):
    return argparse.Namespace(**({"profile": "preview", "camera_host": "trova-cam.local",
                                 "camera_fallback": None, "port": 8000, "robot_ip": "192.168.1.111",
                                 "camera_on_arm": False} | values))


def test_preview_overrides_existing_hardware_settings(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    monkeypatch.setenv("TROVA_ARM", "xarm:192.168.1.111")
    monkeypatch.setenv("TROVA_MOTION_MODE", "hardware")
    monkeypatch.setenv("TROVA_HARDWARE_UNLOCK", "MOVE")
    monkeypatch.setenv("TROVA_ARM_READ", "xarm:192.168.1.111")
    import os
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        configure(args())
        assert os.environ["TROVA_ARM"] == "sim"
        assert os.environ["TROVA_ARM_READ"] == "off"
        assert os.environ["TROVA_MOTION_MODE"] == "dry_run"
        assert os.environ["TROVA_HARDWARE_UNLOCK"] == ""
        assert os.environ["TROVA_ARM_LIMITS"] == "-500,500,-500,500,200,1000"
        assert os.environ["TROVA_ARM_MAX_SPEED_MM_S"] == "20"
        assert os.environ["TROVA_JOG_MAX_MM"] == "300"


def test_camera_fallback_requires_operator_address(monkeypatch):
    def unavailable(*a, **k):
        raise socket.gaierror("unknown hostname")
    monkeypatch.setattr(socket, "getaddrinfo", unavailable)
    with pytest.raises(RuntimeError, match="--camera-fallback"):
        configure(args())
    import os
    monkeypatch.setattr(os, "environ", dict(os.environ))
    configure(args(camera_fallback="192.168.5.160"))
    assert os.environ["TROVA_CAMERA"] == "pi:192.168.5.160"
