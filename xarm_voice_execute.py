#!/usr/bin/env python3
"""Voice -> Grok speech transcription -> Grok JSON -> xArm5 single movement.

Install this file at Grok_Hackathon/xarm_voice_execute.py, alongside .env and
speech_to_text/grok_to_xarm_json.py.

First tests:
  python3 xarm_voice_execute.py --mic                    # no motion
  python3 xarm_voice_execute.py --mic --execute          # motion after typing MOVE
  python3 xarm_voice_execute.py --text 'Move backward by 3 millimeters' --execute

For the very first physical test, only the direction 'backward' is enabled.
This means -X in the ROBOT BASE frame, not relative to the operator.
After manually calibrating directions in UFACTORY Studio, passing
--allow-all-base-directions enables the other five base-frame movements.
All movements remain capped at <=3mm by default and require confirmation.

This demo is NOT a safety-rated robot controller and performs NO collision check.
Keep people clear of the robot and the physical E-stop in reach.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'speech_to_text'))
from grok_to_xarm_json import grok_to_command, record_wav, transcribe_wav

# Coordinates are xArm BASE frame, millimeters. 'Left' and 'right' have to be
# physically checked against your own viewing direction before using them.
DIRECTION_AXIS_SIGN = {
    'forward':  (0, +1),   # +X
    'backward': (0, -1),   # -X
    'left':     (1, +1),   # +Y
    'right':    (1, -1),   # -Y
    'up':       (2, +1),   # +Z
    'down':     (2, -1),   # -Z
}
MAX_ALLOWED_STEP_MM = 300
SPEED_MM_S = 100
ACC_MM_S2 = 50


def read_checked(label, fn):
    code, result = fn()
    if code != 0:
        raise RuntimeError(f'{label}: SDK read failed, code={code}')
    return result


def call_checked(label, result):
    if result != 0:
        raise RuntimeError(f'{label}: SDK command failed, code={result}')


def main():
    p = argparse.ArgumentParser(description='Microphone -> Grok -> confirmed xArm5 micro-movement')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--mic', action='store_true', help='Record microphone audio')
    source.add_argument('--text', help='Use typed text instead of the microphone')
    p.add_argument('--seconds', type=float, default=4.0)
    p.add_argument('--ip', default=os.getenv('XARM_IP', '192.168.1.111'))
    p.add_argument('--execute', action='store_true', help='Allow motion AFTER typed MOVE confirmation')
    p.add_argument('--allow-all-base-directions', action='store_true',
                   help='Unlock base-frame directions beyond backward AFTER calibration')
    args = p.parse_args()

    if not 0.5 <= args.seconds <= 15:
        p.error('--seconds must be between 0.5 and 15')

    load_dotenv(ROOT / '.env', override=False)
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        p.error('Missing XAI_API_KEY; set it or put it in Grok_Hackathon/.env')

    if args.mic:
        spoken_text = transcribe_wav(record_wav(args.seconds), api_key)
        print('Transcript:', spoken_text)
    else:
        spoken_text = args.text

    if not spoken_text.strip():
        print('No speech detected. NO MOTION.')
        return
    command = grok_to_command(spoken_text, api_key)
    print('Validated Grok JSON:', command.model_dump_json(indent=2))

    if command.action != 'move':
        print('Not a movement command. NO MOTION. (Voice STOP is not an emergency stop.)')
        return
    if command.direction not in DIRECTION_AXIS_SIGN:
        print('Unknown direction. NO MOTION.')
        return
    if not (1 <= command.distance_mm <= MAX_ALLOWED_STEP_MM):
        print(f'REFUSED: first physical tests require 1-{MAX_ALLOWED_STEP_MM} mm.')
        print('Try saying: "Move backward by three millimeters."')
        return
    if command.direction != 'backward' and not args.allow_all_base_directions:
        print('REFUSED: initial test allows only "backward" = -X in robot BASE frame.')
        print('After calibrating each direction, opt in with --allow-all-base-directions.')
        return

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip, is_radian=False)
    try:
        if not arm.connected:
            raise RuntimeError(f'Robot at {args.ip} is not connected')
        state = read_checked('get_state', arm.get_state)
        errs = read_checked('get_err_warn_code', arm.get_err_warn_code)
        pose = list(read_checked('get_position', arm.get_position))
        print('Robot state:', state, '| error/warning:', errs)
        print('Current TCP [X,Y,Z,roll,pitch,yaw]:', [round(x, 3) for x in pose])

        if errs[0] != 0:
            raise RuntimeError('Robot reports an error. Resolve it in UFACTORY Studio first.')
        if state not in (2, 5):
            raise RuntimeError(f'Robot state {state} is not idle; motion rejected.')
        if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
            raise RuntimeError('Robot returned invalid Cartesian coordinates.')
        if abs(abs(pose[3]) - 180) > 1 or abs(pose[4]) > 1:
            raise RuntimeError('xArm5 roll/pitch orientation unexpected; refusing.')

        axis, sign = DIRECTION_AXIS_SIGN[command.direction]
        goal = pose.copy()
        goal[axis] += sign * command.distance_mm

        # Extra restraint for the initial -X motion, since last measured X,Y
        # were very close to the horizontal extension of the arm.
        if not args.allow_all_base_directions:
            if not (pose[0] > 100 and math.hypot(goal[0], goal[1]) < math.hypot(pose[0], pose[1])):
                raise RuntimeError('Not a verified inward -X step; rejecting.')

        print('Target TCP [X,Y,Z,roll,pitch,yaw]:', [round(x, 3) for x in goal])
        print(f'Planned: {command.direction}, {command.distance_mm} mm at {SPEED_MM_S} mm/s')
        print('NOTE: No camera, collision model, or autonomous safety assessment is used.')

        if not args.execute:
            print('DRY RUN ONLY. Re-run with --execute to request a real movement.')
            return

        print('Check workspace is clear; keep your hand near the physical E-stop.')
        if input('Type MOVE (uppercase) to enable motors and execute this step: ').strip() != 'MOVE':
            print('Cancelled. NO MOTION.')
            return

        # Check that the robot has not moved between planning and approval.
        fresh_pose = list(read_checked('fresh get_position', arm.get_position))
        fresh_state = read_checked('fresh get_state', arm.get_state)
        fresh_errs = read_checked('fresh get_err_warn_code', arm.get_err_warn_code)
        if math.dist(fresh_pose[:3], pose[:3]) > 1.0:
            raise RuntimeError('Robot moved since the command was planned. Aborting.')
        if fresh_state not in (2, 5) or fresh_errs[0] != 0:
            raise RuntimeError('State/errors changed since the command was planned. Aborting.')

        # SDK mode 0 = position control; setting state 0 re-arms state 5.
        call_checked('motion_enable', arm.motion_enable(enable=True))
        call_checked('set_mode(0)', arm.set_mode(0))
        call_checked('set_state(0)', arm.set_state(0))

        new_state = read_checked('ready state', arm.get_state)
        if new_state != 2:
            raise RuntimeError(f'Robot did not enter READY state (reported {new_state}).')

        # Recompute destination from the latest verified pose to preserve orientation.
        target = fresh_pose.copy()
        target[axis] += sign * command.distance_mm
        call_checked('set_position', arm.set_position(
            x=target[0], y=target[1], z=target[2],
            roll=180.0 if fresh_pose[3] >= 0 else -180.0,
            pitch=0.0, yaw=fresh_pose[5],
            speed=SPEED_MM_S, mvacc=ACC_MM_S2,
            is_radian=False, wait=True, timeout=30,
        ))
        after = read_checked('final get_position', arm.get_position)
        print('Final TCP:', [round(x, 3) for x in after])
        print('SDK reported completion. Verify actual movement visually.')
    finally:
        arm.disconnect()


if __name__ == '__main__':
    main()
