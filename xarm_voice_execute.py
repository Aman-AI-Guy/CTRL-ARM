#!/usr/bin/env python3
"""Spoken or typed relative move, checked by Trova's motion gateway.

Default is a dry-run. This process does not import the xArm SDK and does not connect.

    python3 xarm_voice_execute.py --text "Move backward by 3 millimeters"
    python3 xarm_voice_execute.py --text "Sposta indietro di 40 centimetri"

--execute moves a real arm only when the operator has already set
TROVA_MOTION_MODE=hardware and TROVA_HARDWARE_UNLOCK=MOVE, and then types MOVE.
Pytest cannot reach that path. The requested distance is never rewritten: a distance
outside the jog limit comes back as a rejection with the heard millimetres.

Directions are the robot base frame (backward = -X). The first physical jog stays
backward-only unless --allow-all-base-directions is passed after each direction is checked.

A voice STOP is not a safety-rated emergency stop. Keep the physical e-stop in reach.
This script does not claim collision avoidance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_command(text: str, api_key: str | None):
    speech = str(ROOT / "speech_to_text")
    if speech not in sys.path:
        sys.path.insert(0, speech)
    from grok_to_xarm_json import RobotCommand, grok_to_command, offline_robot_command

    if not (text or "").strip():
        return RobotCommand(action="ignore", direction="none", distance_mm=0)
    if api_key:
        return grok_to_command(text, api_key)
    return offline_robot_command(text)


def _execute_on_hardware(command, env: dict):
    """Operator path. Builds a controller and lets the gateway move it. Tests never call this."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("refusing hardware execution")
    ip = (env.get("XARM_IP") or "").strip()
    if not ip:
        raise RuntimeError("XARM_IP is not set")
    from trova.arm.cartesian import CartesianArm, Limits
    from trova.arm.xarm import XArmBackend

    limits = Limits.parse(env.get("TROVA_ARM_LIMITS", ""))
    if limits is None:
        raise RuntimeError("TROVA_ARM_LIMITS is required before hardware execution")
    backend = XArmBackend(ip)   # read-only connect: motors stay as they are until a ticketed move
    speed = float(env.get("TROVA_ARM_SPEED_MM_S") or 30)
    arm = CartesianArm(backend, float(env.get("TROVA_ARM_Z_MM") or 150), None, limits, speed_mm_s=speed)
    if arm.gateway.mode != "hardware":
        raise RuntimeError(arm.gateway.note or "the gateway is not in hardware mode")
    return arm.gateway.plan_relative(command.direction, command.distance_mm, execute=True)


def run_text(text: str, *, execute: bool = False, confirm: str = "", allow_all_directions: bool = False,
             environ: dict | None = None, api_key: str | None = None) -> dict:
    """Parse one sentence and return the gateway proposal. Does not connect to a robot."""
    from trova.motion import MotionGateway, resolve_mode

    env = os.environ if environ is None else environ
    command = _load_command(text, api_key)
    mode, note = resolve_mode(False, environ=env)
    payload = {"command": command.model_dump(), "sdk_called": False}
    if command.action != "move":
        reason = ("Voice STOP was parsed. It holds nothing here and it is not the physical emergency stop."
                  if command.action == "stop" else "Not a movement command.")
        payload["motion"] = {
            "ok": False, "status": "stopped" if command.action == "stop" else "rejected", "mode": mode,
            "reason": reason, "requested": command.model_dump(), "proposal": {},
            "safety": {"software_stop_is_not_safety_rated": True, "collision_avoidance": False},
            "allow_motion": False, "software_stop_is_not_safety_rated": True, "collision_avoidance": False,
            "final_mm": None,
        }
        return payload
    allowed = None if allow_all_directions else ("backward",)
    decision = MotionGateway.standalone(environ=env).plan_relative(
        command.direction, command.distance_mm, allowed_directions=allowed, execute=False)
    motion = decision.as_dict()
    if execute:
        if mode != "hardware" or confirm != "MOVE":
            motion["ok"] = False
            motion["status"] = "rejected"
            motion["allow_motion"] = False
            motion["mode"] = mode
            if mode != "hardware":
                why = note or "hardware mode is not enabled by the operator"
            else:
                why = "hardware execution needs the operator to type MOVE"
            motion["reason"] = why + ". The requested distance was not changed."
        else:
            moved = _execute_on_hardware(command, env)
            motion = moved.as_dict()
            payload["sdk_called"] = True
    payload["motion"] = motion
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice or text to a gateway-checked xArm jog (dry-run by default)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mic", action="store_true")
    source.add_argument("--text")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--execute", action="store_true", help="Ask to move only when hardware mode is already unlocked")
    parser.add_argument("--allow-all-base-directions", action="store_true",
                        help="Allow all six base-frame directions after each one has been checked on the station")
    args = parser.parse_args()
    if not 0.5 <= args.seconds <= 15:
        parser.error("--seconds must be between 0.5 and 15")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.environ.get("XAI_API_KEY") or None
    if args.mic:
        if not api_key:
            parser.error("Missing XAI_API_KEY for the microphone")
        speech = str(ROOT / "speech_to_text")
        if speech not in sys.path:
            sys.path.insert(0, speech)
        from grok_to_xarm_json import record_wav, transcribe_wav
        text = transcribe_wav(record_wav(args.seconds), api_key)
        print("Transcript:", text)
    else:
        text = args.text

    from trova.motion import resolve_mode
    mode, note = resolve_mode(False)
    confirm = ""
    if args.execute and mode == "hardware":
        print("Keep a hand on the physical e-stop. A voice STOP is not a safety-rated emergency stop.")
        confirm = input("Type MOVE to execute this step: ").strip()
    elif args.execute and note:
        print(note)
    # A typed sentence uses Grok when a key is present, and the offline parser otherwise.
    out = run_text(text, execute=args.execute, confirm=confirm,
                   allow_all_directions=args.allow_all_base_directions, api_key=None if not api_key else api_key)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
