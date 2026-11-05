"""Launch the Pi/xArm workstation. Only the operator runs qualify or operate."""

import argparse
import importlib.util
import os
import socket
import json
from pathlib import Path


def configure(args):
    workspace_env = {}
    workspace_file = Path(__file__).resolve().parent.parent / "station_workspace.json"
    if workspace_file.exists():
        from .arm.cartesian import Limits
        data = json.loads(workspace_file.read_text())
        if data.get("confirm") is not True:
            raise RuntimeError("station_workspace.json must contain the operator's confirm: true")
        limits = Limits.parse(data["limits"])
        if limits is None:
            raise RuntimeError("station_workspace.json has no workspace limits")
        workspace_env = {"TROVA_ARM_LIMITS": limits.describe(),
                         "TROVA_ARM_SPEED_MM_S": str(data["speed_mm_s"]),
                         "TROVA_ARM_MAX_SPEED_MM_S": str(data["speed_mm_s"]),
                         "TROVA_JOG_MAX_MM": str(data["jog_max_mm"])}
    host = args.camera_host
    try:
        socket.getaddrinfo(host, 8080, type=socket.SOCK_STREAM)
    except socket.gaierror:
        if not args.camera_fallback:
            raise RuntimeError("Camera hostname did not resolve. Pass --camera-fallback with its current DHCP address.") from None
        host = args.camera_fallback
        print(f"Camera hostname unavailable; using operator-supplied address {host}")
    if args.profile != "preview" and importlib.util.find_spec("xarm") is None:
        raise RuntimeError('Install the robot SDK first: .venv/bin/pip install -e ".[robots]"')
    os.environ.update(workspace_env)
    if workspace_env:
        print("Operator workspace: " + workspace_env["TROVA_ARM_LIMITS"] + " (robot-base mm)")
    os.environ.update(TROVA_CAMERA=f"pi:{host}", TROVA_HOST="127.0.0.1", TROVA_PORT=str(args.port),
                      TROVA_ARM="sim" if args.profile != "operate" else f"xarm:{args.robot_ip}",
                      TROVA_ARM_READ=f"xarm:{args.robot_ip}" if args.profile == "qualify" else "off",
                      TROVA_CAMERA_ON_ARM="1" if args.camera_on_arm else "0")
    if args.profile != "operate":
        os.environ.update(TROVA_MOTION_MODE="dry_run", TROVA_HARDWARE_UNLOCK="")
    elif os.environ.get("TROVA_MOTION_MODE") != "hardware":
        os.environ["TROVA_MOTION_MODE"] = "dry_run"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("preview", "qualify", "operate"), default="preview",
                        help="preview: no robot connection; qualify: telemetry only; operate: real driver, dry-run unless operator unlocked")
    parser.add_argument("--camera-host", default="trova-cam.local")
    parser.add_argument("--camera-fallback", help="Current DHCP address, used only if hostname resolution fails")
    parser.add_argument("--robot-ip", default="192.168.1.111")
    parser.add_argument("--camera-on-arm", action="store_true", help="Use only when the Pi camera is mounted on the robot")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from .config import load_settings  # Load .env silently, then apply the selected profile.
    try:
        configure(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    import uvicorn
    from .server import create_app
    settings = load_settings()
    print(f"Trova {args.profile}: http://localhost:{args.port} (open on this laptop for microphone access)")
    print("Motion mode: " + os.environ["TROVA_MOTION_MODE"])
    if os.environ.get("TROVA_TOKEN"):
        print("API access token is configured; use your existing authenticated browser session or token link.")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
