"""Set the xArm 5 controller's safety settings from Python instead of clicking through UFACTORY Studio.

    .venv/bin/python -m trova.arm.xarm_config 192.168.1.111            # read-only: print the current settings
    .venv/bin/python -m trova.arm.xarm_config 192.168.1.111 --apply \\
        --limits xmin,xmax,ymin,ymax,zmin,zmax --payload-kg 0.25 --payload-cog 0,0,40 [--tcp x,y,z]

It never moves the arm: only settings calls, then save_conf (kept after a reboot), then everything is read back.
Run it with staff, with the arm standing still. Defaults follow docs/first-supervised-run.md (xArm 5 manual
pp. 67, 80, 81, 93): collision sensitivity 4 (the manual says never below 3), self-collision on, rebound off,
reduced mode at 50 mm/s and 20 deg/s, and the safety boundary (fence) = TROVA_ARM_LIMITS.
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def _nums(s: str, n: int, what: str) -> list[float]:
    v = [float(x) for x in s.split(",")]
    if len(v) != n or not all(math.isfinite(x) for x in v):
        raise SystemExit(f"{what}: give {n} comma-separated numbers")
    return v


def read(api) -> dict:
    """Everything the run depends on, as the controller reports it."""
    code, position = api.get_position()
    if code != 0 or len(position) < 6 or not all(math.isfinite(v) for v in position):
        raise SystemExit("Cannot read the current tool position")
    code, red = api.get_reduced_states()
    return {"position": position, "tcp_offset": api.tcp_offset, "tcp_load": api.tcp_load,
            "collision_sensitivity": api.collision_sensitivity, "collision_rebound": api.is_collision_rebound,
            "self_collision": api.self_collision_params, "fence_mode": api.is_fence_mode,
            "reduced_states": red if code == 0 else f"code {code}", "mount_direction": api.get_mount_direction()}


def plan(a, position) -> list[tuple[str, str, tuple]]:
    """(description, SDK method, args). Refuses values the manual or the run do not allow."""
    if not 3 <= a.sensitivity <= 5:
        raise SystemExit("collision sensitivity must be 3-5 (manual p. 67: not below 3)")
    if not (0 < a.tcp_speed <= 250 and 0 < a.joint_speed <= 60):
        raise SystemExit("reduced speeds out of range for a first run (tcp <= 250 mm/s, joints <= 60 deg/s)")
    if not a.limits:
        raise SystemExit("--limits (or TROVA_ARM_LIMITS) is required: the safety boundary from the bring-up")
    x0, x1, y0, y1, z0, z1 = _nums(a.limits, 6, "--limits")
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise SystemExit("--limits must be xmin,xmax,ymin,ymax,zmin,zmax with each min < max")
    x, y, z = position[:3]
    if not (x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1):
        raise SystemExit(f"the tool is at {x:.0f},{y:.0f},{z:.0f}, outside these limits: enabling the boundary "
                         "would stop the arm. Jog it inside first (or fix the limits)")
    steps = [("collision sensitivity", "set_collision_sensitivity", (a.sensitivity,)),
             ("self-collision detection on", "set_self_collision_detection", (True,)),
             ("collision rebound off", "set_collision_rebound", (False,)),
             (f"reduced max TCP speed {a.tcp_speed} mm/s", "set_reduced_max_tcp_speed", (a.tcp_speed,)),
             (f"reduced max joint speed {a.joint_speed} deg/s", "set_reduced_max_joint_speed", (a.joint_speed,)),
             (f"boundary x {x0}..{x1} y {y0}..{y1} z {z0}..{z1}", "set_reduced_tcp_boundary", ([x1, x0, y1, y0, z1, z0],)),
             ("reduced mode on", "set_reduced_mode", (True,)),
             ("safety boundary (fence) on", "set_fence_mode", (True,))]
    if a.payload_kg is not None:
        if not 0 < a.payload_kg <= 3:
            raise SystemExit("xArm 5 payload must be finite and 0 < kg <= 3")
        cog = _nums(a.payload_cog, 3, "--payload-cog")
        steps.insert(0, (f"payload {a.payload_kg} kg at {cog} mm", "set_tcp_load", (a.payload_kg, cog)))
    if a.tcp:
        steps.insert(0, (f"TCP offset {a.tcp} mm", "set_tcp_offset", (_nums(a.tcp, 3, "--tcp") + [0, 0, 0],)))
    return steps


def apply(api, steps) -> None:
    for what, method, args in steps:
        if method == "set_fence_mode":
            boundary = next(a[0] for _, m, a in steps if m == "set_reduced_tcp_boundary")
            code, pos = api.get_position()
            if code != 0 or len(pos) < 3 or not all(math.isfinite(v) for v in pos[:3]) or not all(
                    boundary[2*i+1] <= v <= boundary[2*i] for i, v in enumerate(pos[:3])):
                raise SystemExit("Tool outside limits after TCP change (or position unreadable): fence refused, nothing saved")
        # SDK's high-level sensitivity setter restarts state 0. Its public core command only writes the setting.
        code = api.core.set_collis_sens(*args) if method == "set_collision_sensitivity" else getattr(api, method)(*args)
        code = code[0] if isinstance(code, (list, tuple)) and code else code
        if type(code) is not int or code != 0:
            raise SystemExit(f"FAILED: {what} ({method} returned {code}); nothing after it was set, nothing saved")
        print("set:", what)
    code = api.save_conf()
    code = code[0] if isinstance(code, (list, tuple)) and code else code
    if type(code) is not int or code != 0:
        raise SystemExit(f"save_conf returned {code}: settings are active but will be lost on reboot")
    print("saved on the controller")


def main(argv=None, api_factory=None, ask=input) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ip")
    p.add_argument("--apply", action="store_true", help="change settings (default: only read)")
    p.add_argument("--limits", default=os.getenv("TROVA_ARM_LIMITS", ""), help="xmin,xmax,ymin,ymax,zmin,zmax mm")
    p.add_argument("--tcp", help="TCP offset x,y,z mm (measured, then checked with the yaw test)")
    p.add_argument("--payload-kg", type=float)
    p.add_argument("--payload-cog", default="0,0,0", help="payload centre of gravity x,y,z mm from the flange")
    p.add_argument("--sensitivity", type=int, default=4)
    p.add_argument("--tcp-speed", type=float, default=50)
    p.add_argument("--joint-speed", type=float, default=20)
    a = p.parse_args(argv)
    if api_factory is None:
        from xarm.wrapper import XArmAPI
        api_factory = lambda ip: XArmAPI(ip, is_radian=False)   # noqa: E731
    api = api_factory(a.ip)
    try:
        before = read(api)
        for k, v in before.items():
            print(f"{k:22} {v}")
        if not a.apply:
            return
        code, state = api.get_state()
        if code != 0 or state not in (0, 2, 4):
            raise SystemExit("the arm is moving (or its state cannot be read): settings refused")
        steps = plan(a, before["position"])
        print("\nWill set:\n  " + "\n  ".join(s[0] for s in steps))
        if ask("Staff present and the arm standing still? Type YES to apply: ").strip() != "YES":
            raise SystemExit("not applied")
        apply(api, steps)
        print("\nRead back:")
        for k, v in read(api).items():
            print(f"{k:22} {v}")
    finally:
        try:
            api.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
