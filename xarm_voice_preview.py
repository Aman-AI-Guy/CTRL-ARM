#!/usr/bin/env python3
"""Read-only integration test: microphone/text -> Grok -> JSON -> xArm 5 telemetry.

Run from Grok_Hackathon project root after copying this file into it:
    python3 xarm_voice_preview.py --text "Move left by 2 centimeters"
    python3 xarm_voice_preview.py --mic

Never enables motion, changes robot state/mode, or sends a movement instruction.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "speech_to_text"))

from dotenv import load_dotenv
from grok_to_xarm_json import RobotCommand, grok_to_command, record_wav, transcribe_wav


def read_result(arm, name: str, getter):
    code, value = getter()
    if code != 0:
        raise RuntimeError(f"xArm {name} failed (SDK code {code})")
    return value


def main():
    parser = argparse.ArgumentParser(description="Grok voice commands and xArm telemetry: READ ONLY")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mic", action="store_true", help="Record a spoken command")
    group.add_argument("--text", help="Typed command, instead of microphone")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--ip", default=os.getenv("XARM_IP", "192.168.1.111"))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        parser.error("XAI_API_KEY is missing from the environment or repo .env")
    if args.mic:
        if not 0.5 <= args.seconds <= 15:
            parser.error("--seconds must be between 0.5 and 15")
        text = transcribe_wav(record_wav(args.seconds), api_key)
        print("Transcript:", text)
    else:
        text = args.text

    cmd = grok_to_command(text, api_key) if text.strip() else RobotCommand(
        action="ignore", direction="none", distance_mm=0
    )
    print("\nValidated Grok command:")
    print(json.dumps(cmd.model_dump(), indent=2))

    # Connect to the robot only to read its current telemetry.
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip, is_radian=False)
    try:
        if not arm.connected:
            raise ConnectionError(f"Cannot reach xArm at {args.ip}")
        state = read_result(arm, "state", arm.get_state)
        err_warn = read_result(arm, "errors", arm.get_err_warn_code)
        pose = read_result(arm, "position", arm.get_position)
        joints = read_result(arm, "joint angles", arm.get_servo_angle)
        print(f"\nRobot at {args.ip} (READ ONLY):")
        print(f"State: {state} | error/warning: {err_warn}")
        print(f"TCP [x,y,z,roll,pitch,yaw]: {[round(v, 3) for v in pose]}")
        print(f"Joint angles: {[round(v, 3) for v in joints]}")
        if state == 5:
            print("State 5 = MODE_CHANGED; do not automatically reset/enable the arm.")
        elif state not in (1, 2):
            print("Robot is not in a normal ready/moving feedback state.")
    finally:
        arm.disconnect()

    print("\nNO MOTION SENT. This script does not enable motors or change the controller mode.")
    print("A voice STOP is not a safety-rated emergency stop. Use the physical e-stop.")
    if cmd.action == "move":
        from trova.motion import MotionGateway
        decision = MotionGateway.standalone().plan_relative(
            cmd.direction, cmd.distance_mm, pose=tuple(pose[:3]), execute=False)
        print(f"Requested: {cmd.distance_mm} mm {cmd.direction} (robot base frame).")
        print("Gateway proposal (dry-run, nothing was sent):")
        print(json.dumps(decision.as_dict(), indent=2))
    elif cmd.action == "stop":
        print("Voice STOP was parsed. This read-only preview does not stop the arm.")


if __name__ == "__main__":
    main()
