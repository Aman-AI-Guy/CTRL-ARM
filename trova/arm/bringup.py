"""Guided first connection to the UFactory xArm 5, before Trova drives it:

    .venv/bin/python -m trova.arm.bringup 192.168.1.xxx        # the IP on the control box label

With museum staff next to the e-stop. Every motion asks for "y" first, runs at <= 30 mm/s, and starts from a
pose you set BY HAND (joint teaching mode), never from the folded transport pose. Steps:

  1. connect: firmware, state, mode, error / warning codes, joints, tool pose, TCP offset, controller safety box
  2. clear errors and enable (if needed)
  3. by hand: bring the flange above the table, pointing down (the pose Trova works in)
  4. by hand: park pose out of the camera's view          -> TROVA_ARM_HOME_MM
  5. by hand: the 4 corners of the work area on the table  -> TROVA_ARM_LIMITS, TROVA_ARM_Z_MM
  6. test move: up 30 mm and back (straight line, orientation unchanged)
  7. STOP test: a slow 100 mm move, stopped after ~1 s: must halt early and hold
  8. prints the .env lines and saves data/xarm_bringup.json

Nothing here is called by the server. Ctrl+C at any prompt ends it; the arm keeps holding its position.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SPEED = 30.0        # mm/s for every bring-up move
ACC = 300.0         # mm/s^2
MARGIN_XY = 40.0    # safety box = work-area corners +- this
Z_FLOOR = 15.0      # lowest allowed tool height above the table
HOVER = 100.0       # pointing height above the table


class Abort(RuntimeError):
    pass


class Bringup:
    def __init__(self, api, ask=input, say=print, sleep=time.sleep, camera_on_arm: bool = False):
        self.api, self.ask, self.say, self.sleep = api, ask, say, sleep
        self.camera_on_arm = camera_on_arm      # the event's sphere camera: home is the SURVEY pose, not a park pose
        self.result: dict = {}

    # ---------- small helpers
    def yes(self, q: str) -> bool:
        return self.ask(f"{q} [y/N] ").strip().lower().startswith("y")

    def wait_enter(self, q: str) -> None:
        self.ask(f"{q}  (Enter when done) ")

    def pose(self) -> list[float]:
        code, p = self.api.get_position()
        if code != 0 or not p:
            raise Abort(f"could not read the tool pose (code {code})")
        return [round(float(v), 1) for v in p[:6]]

    def errors(self) -> tuple[int, int]:
        code, ew = self.api.get_err_warn_code()
        return (ew[0], ew[1]) if code == 0 and ew else (-1, -1)

    def enable(self) -> None:
        self.api.clean_error()
        self.api.clean_warn()
        self.api.motion_enable(True)
        self.api.set_mode(0)
        self.api.set_state(0)
        self.sleep(0.5)

    def hand_mode(self, on: bool) -> None:
        """Joint teaching mode: the arm is gravity compensated and can be moved by hand."""
        self.api.set_mode(2 if on else 0)
        self.api.set_state(0)
        self.sleep(0.3)

    def move(self, p: list[float], speed: float = SPEED, wait: bool = True) -> int:
        x, y, z, r, pi, yaw = p
        return self.api.set_position(x=x, y=y, z=z, roll=r, pitch=pi, yaw=yaw, speed=speed, mvacc=ACC,
                                     wait=wait, timeout=30)

    def stop(self) -> None:          # the same sequence the Trova driver uses (trova/arm/xarm.py)
        code = self.api.set_state(4)
        code = code[0] if isinstance(code, (list, tuple)) and code else code
        if type(code) is not int or code != 0:
            raise Abort(f"STOP failed (set_state(4) returned {code}); use the physical E-stop")

    # ---------- the steps
    def step_connect(self) -> None:
        a = self.api
        err, warn = self.errors()
        self.say(f"firmware {a.get_version()[1] if hasattr(a, 'get_version') else '?'} · state {a.get_state()[1]} "
                 f"· mode {getattr(a, 'mode', '?')} · error {err} · warning {warn}")
        self.say(f"joints {a.get_servo_angle()[1]}")
        self.say(f"tool pose {self.pose()}  (x, y, z mm; roll, pitch, yaw deg)")
        self.say(f"TCP offset {getattr(a, 'tcp_offset', '?')} (all zero = the flange; set your pointer's length in "
                 "UFACTORY Studio or TROVA_ARM_TCP_MM)")
        try:
            code, rs = a.get_reduced_states()
            if code == 0 and rs:
                self.say(f"controller safety box on: {bool(rs[5]) if len(rs) > 5 else '?'} · box {rs[1]}")
        except Exception:  # noqa: BLE001 - older firmware
            pass
        self.result["connect"] = {"error": err, "warning": warn}

    def step_enable(self) -> None:
        err, warn = self.errors()
        if err or warn:
            self.say(f"The controller reports error {err}, warning {warn} (see the UFACTORY manual or Studio for the text).")
            if not self.yes("Clear errors and enable motion?"):
                raise Abort("left with errors")
        self.enable()
        err, _ = self.errors()
        if err:
            raise Abort(f"error {err} is still active: fix it on the controller / in UFACTORY Studio first")
        self.say("enabled, position mode, ready")

    def step_ready_pose(self) -> list[float]:
        self.say("\nHold the arm. It will go soft (joint teaching mode) so you can move it by hand.")
        if not self.yes("Switch to hand mode now?"):
            raise Abort("no hand mode")
        self.hand_mode(True)
        self.wait_enter("Move the flange about 15 cm above the middle of the table, POINTING STRAIGHT DOWN")
        self.hand_mode(False)
        p = self.pose()
        if abs(abs(p[3]) - 180) > 12 or abs(p[4]) > 12:
            self.say(f"warning: roll {p[3]}, pitch {p[4]}: not pointing straight down (want roll +-180, pitch 0). "
                     "Trova's first move will turn the wrist to vertical.")
        self.result["ready"] = p
        self.say(f"ready pose {p}")
        return p

    def step_home(self) -> list[float]:
        self.hand_mode(True)
        if self.camera_on_arm:
            self.wait_enter("Move the arm by hand to the SURVEY pose: the camera on the arm sees the WHOLE work area "
                            "(watch Trova's live view or the camera preview)")
        else:
            self.wait_enter("Move the arm by hand to its PARK pose: out of the camera's view, flange down if you can")
        self.hand_mode(False)
        p = self.pose()
        self.result["home"] = p
        code, q = self.api.get_servo_angle()
        if code == 0 and q:
            self.result["home_joints"] = [round(float(v), 2) for v in q[:5]]
        self.say(f"{'survey' if self.camera_on_arm else 'park'} pose {p[:3]}, joints {self.result.get('home_joints')}")
        return p

    def step_corners(self) -> list[list[float]]:
        self.say("\nNow the work area: touch the 4 corners of the area on the table where objects will be.")
        pts = []
        self.hand_mode(True)
        for name in ("far left", "far right", "near right", "near left"):
            self.wait_enter(f"Flange (or pointer tip) lightly ON the table at the {name} corner")
            pts.append(self.pose()[:3])
            self.say(f"  {name}: {pts[-1]}")
        self.wait_enter("Lift the arm back above the table")
        self.hand_mode(False)
        self.result["corners"] = pts
        return pts

    def step_test_move(self, start: list[float]) -> None:
        self.say(f"\nTest move: straight up 30 mm at {SPEED:.0f} mm/s, then back. Same orientation.")
        if not self.yes("Staff at the e-stop, area clear. Move?"):
            raise Abort("test move skipped")
        up = list(start)
        up[2] += 30
        for target, what in ((up, "up"), (start, "back")):
            code = self.move(target)
            got = self.pose()
            off = sum((a - b) ** 2 for a, b in zip(got[:3], target[:3], strict=True)) ** 0.5
            self.say(f"  {what}: code {code}, at {got[:3]}, {off:.1f} mm from target")
            if code != 0 or off > 3:
                raise Abort(f"move {what} failed (code {code}, error {self.errors()[0]})")
        self.result["test_move"] = "ok"

    def step_stop(self, start: list[float]) -> None:
        self.say("\nSTOP test: a slow 100 mm move up, stopped by software after about 1 second.")
        if not self.yes("Move?"):
            raise Abort("stop test skipped")
        target = list(start)
        target[2] += 100
        self.move(target, speed=20, wait=False)
        self.sleep(1.2)
        self.stop()
        self.sleep(0.4)
        a = self.pose()
        self.sleep(1.0)
        b = self.pose()
        moved, drift = a[2] - start[2], abs(b[2] - a[2])
        self.say(f"  stopped after {moved:.0f} mm of 100, drift in the next second {drift:.1f} mm")
        ok = 5 < moved < 60 and drift < 1.5
        self.result["stop_test"] = {"moved_mm": round(moved, 1), "drift_mm": round(drift, 1), "ok": ok}
        if not ok:
            raise Abort("the arm did not stop early and hold: do NOT run Trova on it, tell staff")
        self.say("  STOP works and holds.")
        if self.yes("Go back down to the ready pose?"):
            self.enable()
            self.move(start)

    def settings(self) -> dict:
        corners, home = self.result["corners"], self.result["home"]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        z_table = min(p[2] for p in corners)
        lim = [min(xs) - MARGIN_XY, max(xs) + MARGIN_XY, min(ys) - MARGIN_XY, max(ys) + MARGIN_XY,
               z_table + Z_FLOOR, max(home[2], z_table + HOVER) + 80]
        notes = []
        if not (lim[0] <= home[0] <= lim[1] and lim[2] <= home[1] <= lim[3]):   # the park pose must be reachable
            lim = [min(lim[0], home[0] - 20), max(lim[1], home[0] + 20), min(lim[2], home[1] - 20), max(lim[3], home[1] + 20),
                   lim[4], lim[5]]
            notes.append("the safety box was widened to include the park pose")
        env = {"TROVA_ARM": f"xarm:{self.result.get('ip', '<ip>')}",
               "TROVA_ARM_SPEED_MM_S": "50",
               "TROVA_ARM_Z_MM": f"{z_table + HOVER:.0f}",
               "TROVA_ARM_HOME_MM": ",".join(f"{v:.0f}" for v in home[:3]),
               "TROVA_ARM_LIMITS": ",".join(f"{v:.0f}" for v in lim)}
        if self.camera_on_arm and self.result.get("home_joints"):
            env["TROVA_CAMERA_ON_ARM"] = "1"
            env["TROVA_ARM_HOME_JOINTS"] = ",".join(f"{v:g}" for v in self.result["home_joints"])
            notes.append("camera on the arm: home is the survey pose (joints); calibrate with ArUco markers recorded by the arm")
        return {"env": env, "table_z_mm": round(z_table, 1), "notes": notes}

    def run(self, ip: str, out: Path | None) -> dict:
        self.result["ip"] = ip
        self.say("Trova xArm bring-up. Staff next to the e-stop. Ctrl+C stops the script; the arm holds.\n")
        self.step_connect()
        self.step_enable()
        ready = self.step_ready_pose()
        self.step_home()
        self.step_corners()
        self.say("\nBack to the ready pose by hand first.")
        self.hand_mode(True)
        self.wait_enter("Flange above the middle of the table again, pointing down")
        self.hand_mode(False)
        ready = self.pose()
        self.step_test_move(ready)
        self.step_stop(ready)
        s = self.settings()
        self.result["settings"] = s
        self.say("\nPut these lines in .env, then restart Trova:\n")
        for k, v in s["env"].items():
            self.say(f"{k}={v}")
        for n in s["notes"]:
            self.say(f"# note: {n}")
        if self.camera_on_arm:
            self.say("\nThen in Trova: print 4 ArUco markers, tape them on the table, touch each centre with the sphere and "
                     "press Record marker from arm, then Calibrate (ArUco) -> preflight. Teach the look pose for the room scan.")
        else:
            self.say("\nThen in Trova: Setup -> Freedrive on -> Calibrate with arm (click the tip at 4-6 spots) -> preflight.")
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.result, indent=2))
            self.say(f"saved {out}")
        return self.result


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: python -m trova.arm.bringup <xarm ip> [--camera-on-arm]   (e.g. 192.168.1.185)")
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        sys.exit("pip install xarm-python-sdk   (or: pip install -e '.[robots]')")
    ip = args[0]
    api = XArmAPI(ip, is_radian=False)
    b = Bringup(api, camera_on_arm="--camera-on-arm" in sys.argv)
    try:
        b.run(ip, Path("data") / "xarm_bringup.json")
    except (Abort, KeyboardInterrupt) as e:
        try:
            b.stop()
        except Exception as stop_error:  # noqa: BLE001
            print(f"\nSTOP FAILED: {stop_error}. Use the physical E-stop.")
        else:
            print(f"\nStopped: {e or 'Ctrl+C'}. Controller STOP accepted; verify hold with staff.")
    finally:
        try:
            api.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
