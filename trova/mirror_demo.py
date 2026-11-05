"""Operator-requested mirror or Y-shuttle demo through the running app's jog gateway.

Default: read-only plan and controller IK queries. --execute explicitly sends jogs.
Never imports a motion driver or calls an SDK actuator method.
"""

import argparse
import json
import math
from pathlib import Path

import requests

from .arm.cartesian import Limits


def build_path(start, travel_z, limits):
    x, y, z = map(float, start)
    if y >= 0 or x < 0:
        raise ValueError("This demo requires a starting point on the +X / -Y side")
    if not math.isfinite(travel_z) or travel_z <= z:
        raise ValueError("Travel Z must be finite and above the starting Z")
    radius = math.hypot(x, y)
    angle = math.atan2(y, x)
    n = max(1, math.ceil((-2 * angle) / math.radians(2)))
    path = [(x, y, z), (x, y, travel_z)]
    for index in range(1, n + 1):
        a = angle + (-2 * angle) * index / n
        target = (x, -y, travel_z) if index == n else (radius * math.cos(a), radius * math.sin(a), travel_z)
        previous = path[-1]
        # Axis-aligned small steps use the existing relative-jog endpoint.
        # The intermediate corner is checked too, not just the arc endpoints.
        path.extend([(target[0], previous[1], travel_z), target])
    path.append((x, -y, z))
    for point in path:
        error = limits.check(*point)
        if error:
            raise ValueError(error)
    return path


def build_y_shuttle(start, stroke_mm, cycles, limits):
    if not math.isfinite(stroke_mm) or not 1 <= abs(stroke_mm) <= 300:
        raise ValueError("Y stroke must be 1..300 mm in magnitude")
    if not isinstance(cycles, int) or not 1 <= cycles <= 10:
        raise ValueError("Cycles must be an integer from 1 to 10")
    x, y, z = map(float, start)
    origin = (x, y, z)
    other = (x, y + stroke_mm, z)
    for point in (origin, other):
        error = limits.check(*point)
        if error:
            raise ValueError(error)
    # Each app jog waits up to 30 seconds. At 5 mm/s, 100 mm stays below that timeout.
    count = math.ceil(abs(stroke_mm) / 100)
    leg = [(x, y + stroke_mm * i / count, z) for i in range(count + 1)]
    return [origin] + [point for _ in range(cycles) for point in (leg[1:] + list(reversed(leg[:-1])))]


def api(base, path, body=None):
    response = (requests.get(base + path, timeout=10) if body is None else
                requests.post(base + path, json=body, timeout=45))
    response.raise_for_status()
    return response.json()


def stationary_pose(base):
    state = api(base, "/api/state")
    robot = state.get("robot") or {}
    if (robot.get("source") != "telemetry" or robot.get("simulated") or robot.get("moving") is not False
            or robot.get("error") or not robot.get("pose6")):
        raise RuntimeError("Need a stationary, healthy telemetry pose")
    if state.get("arm", {}).get("state") not in ("idle",):
        raise RuntimeError("Arm is stopped, busy, or not idle; aborting")
    return robot


def check_driver_orientation(robot):
    target = robot.get("cartesian_target_orientation_deg")
    pose = robot["pose6"]
    if target is None:
        raise RuntimeError("App does not expose its commanded orientation; restart the updated server before planning")
    if any(abs((a-b+180) % 360 - 180) >= 0.1 for a, b in zip(pose[3:], target)):
        raise RuntimeError("Current orientation differs from the driver's commanded orientation; no movement sent")


def plan(base, travel_z, *, pattern="mirror", stroke_mm=200, cycles=2, speed_mm_s=5):
    safety = api(base, "/api/safety")
    if not math.isfinite(speed_mm_s) or not 0 < speed_mm_s <= 30:
        raise ValueError("Operator demo speed must be finite, above zero, and at most 30 mm/s")
    if safety.get("workspace_limits") not in ("cylinder:500,200,1000", "-500,500,-500,500,200,1000"):
        raise RuntimeError("Expected the operator-specified 500 mm XY / 200..1000 mm Z workspace")
    robot = stationary_pose(base)
    check_driver_orientation(robot)
    pose = robot["pose6"]
    if abs(abs(pose[3]) - 180) >= 1 or abs(pose[4]) >= 1:
        raise RuntimeError("Flange is not in the driver's pointing orientation")
    limits = Limits.parse(safety["workspace_limits"])
    if pattern == "y-shuttle":
        path = build_y_shuttle(pose[:3], stroke_mm, cycles, limits)
        travel_z = pose[2]
    elif pattern == "mirror":
        path = build_path(pose[:3], travel_z, limits)
    else:
        raise ValueError("Unknown demo pattern")
    checked = validate_ik_path(path, pose, robot["joints"])
    return {"pattern": pattern, "start_pose": pose, "path": path, "ik_samples": checked,
            "travel_z_mm": travel_z, "final_xyz": path[-1], "workspace_limits": safety["workspace_limits"],
            "speed_cap_mm_s": min(speed_mm_s, safety["max_speed_mm_s"]),
            "distance_mm": sum(math.dist(a, b) for a, b in zip(path, path[1:])),
            "collision_avoidance": False, "physical_clearance_source": "operator reports area checked"}


def validate_ik_path(path, pose, previous):
    from .arm.telemetry import open_reader
    reader = open_reader("xarm:192.168.1.111")
    checked = 0
    try:
        for start, end in zip(path, path[1:]):
            n = max(1, math.ceil(math.dist(start, end) / 5))
            for index in range(1, n + 1):
                xyz = [a + (b - a) * index / n for a, b in zip(start, end)]
                result = reader.inverse(xyz + pose[3:], ref_joints=previous)
                if not result.get("ok"):
                    raise RuntimeError(f"IK rejected {xyz}: {result.get('reason')}")
                joints = result["joints"]
                if any(abs(a - b) > 10 for a, b in zip(previous, joints)):
                    raise RuntimeError("IK branch changed abruptly; no movement will be sent")
                previous = joints
                checked += 1
    finally:
        reader.close()
    return checked


def resume_plan(base, source, checkpoint):
    """Deliberate operator continuation; never clears STOP or repositions to a checkpoint."""
    if not 0 < checkpoint < len(source["path"]) - 1:
        raise ValueError("Checkpoint must leave a remaining path")
    robot = stationary_pose(base)
    check_driver_orientation(robot)
    pose = robot["pose6"]
    if math.dist(pose[:3], source["path"][checkpoint]) > 1:
        raise RuntimeError("Robot is not at the selected checkpoint; no movement sent")
    if any(abs((a-b+180) % 360 - 180) > 0.1 for a,b in zip(pose[3:],source["start_pose"][3:])):
        raise RuntimeError("Checkpoint orientation changed; no movement sent")
    safety = api(base, "/api/safety")
    if safety["workspace_limits"] != source["workspace_limits"]:
        raise RuntimeError("Workspace changed since original plan")
    path = source["path"][checkpoint:]
    limits = Limits.parse(safety["workspace_limits"])
    for point in path:
        if limits.check(*point):
            raise RuntimeError("Remaining path leaves the workspace")
    for a,b in zip(path,path[1:]):
        if sum(abs(x-y)>1e-6 for x,y in zip(a,b)) != 1 or math.dist(a,b)>100+1e-6:
            raise RuntimeError("Remaining path requires subdivision")
    speed = source["speed_cap_mm_s"]
    if not math.isfinite(speed) or not 0 < speed <= 30:
        raise RuntimeError("Invalid saved demo speed")
    checked = validate_ik_path(path, pose, robot["joints"])
    return source | {"start_pose": pose, "path": path, "ik_samples": checked,
                     "resume_checkpoint": checkpoint,
                     "original_start_pose": source.get("original_start_pose", source["start_pose"]),
                     "speed_cap_mm_s": min(speed,safety["max_speed_mm_s"]),
                     "distance_mm": sum(math.dist(a,b) for a,b in zip(path,path[1:]))}


def execute(base, reviewed):
    # Recheck the stationary start and configuration after all read-only planning.
    robot = stationary_pose(base)
    check_driver_orientation(robot)
    initial = robot["pose6"]
    if math.dist(initial[:3], reviewed["start_pose"][:3]) > 1:
        raise RuntimeError("Robot changed position after planning; no movement sent")
    if any(abs((a-b+180) % 360 - 180) > 0.1 for a, b in zip(initial[3:], reviewed["start_pose"][3:])):
        raise RuntimeError("Robot changed orientation after planning; no movement sent")
    safety = api(base, "/api/safety")
    if safety.get("mode") != "hardware" or safety.get("workspace_limits") != reviewed["workspace_limits"]:
        raise RuntimeError("Hardware mode or workspace changed; no movement sent")
    # Operator explicitly requested relaxing the 3 mm jog cap for this demo.
    configured = api(base, "/api/safety", {"confirm": True, "jog_max_mm": 300,
                                          "speed_mm_s": reviewed["speed_cap_mm_s"]})
    if not configured.get("ok"):
        raise RuntimeError("Could not configure the operator-requested jog cap")
    directions = (("forward", "backward"), ("left", "right"), ("up", "down"))
    completed = 0
    try:
        for expected, target in zip(reviewed["path"], reviewed["path"][1:]):
            robot = stationary_pose(base)
            actual = robot["pose6"]
            if math.dist(actual[:3], expected) > 1:
                raise RuntimeError("Unexpected tool position; aborting remaining path")
            if any(abs((a - b + 180) % 360 - 180) > 1 for a, b in zip(actual[3:], initial[3:])):
                raise RuntimeError("Unexpected tool orientation; aborting remaining path")
            delta = [b - a for a, b in zip(actual[:3], target)]
            # Correct only the intended axis; residual error on the other axes must remain small.
            axis = max(range(3), key=lambda i: abs(target[i] - expected[i]))
            distance = abs(delta[axis])
            if distance < 1:  # SDK's minimum relative jog is 1 mm; no distance is rounded up.
                continue
            if math.dist(expected, target) > 100 + 1e-6 or distance > 101:
                raise RuntimeError("Demo segment exceeds 100 mm; subdivide before executing to stay within the API timeout")
            direction = directions[axis][0 if delta[axis] > 0 else 1]
            result = api(base, "/api/arm/jog", {"direction": direction, "distance_mm": distance})
            if result.get("status") != "executed" or not result.get("ok"):
                raise RuntimeError("Move refused or stopped: " + str(result.get("reason")))
            reached = stationary_pose(base)["pose6"]
            if math.dist(reached[:3], target) > 1:
                raise RuntimeError("Tool did not reach checkpoint; aborting remaining path")
            completed += 1
            print(json.dumps({"segment": completed, "xyz_mm": reached[:3]}), flush=True)
        final = stationary_pose(base)["pose6"]
        if math.dist(final[:3], reviewed["final_xyz"]) > 1:
            raise RuntimeError("Final position was not verified")
        print(json.dumps({"status": "completed", "final_pose": final, "segments": completed}), flush=True)
    except BaseException:
        # Stop is through the same app. No retry, reset, home or automatic return.
        api(base, "/api/arm/stop", {})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--travel-z", type=float, default=300)
    parser.add_argument("--pattern", choices=("mirror", "y-shuttle"), default="mirror")
    parser.add_argument("--stroke", type=float, default=200, help="Signed Y stroke in mm for the shuttle")
    parser.add_argument("--cycles", type=int, default=2, help="Number of out-and-back cycles")
    parser.add_argument("--speed", type=float, default=5, help="Finite demo speed in mm/s, maximum 30")
    parser.add_argument("--output", default="/tmp/trova-mirror-plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-plan", help="Original saved plan, for an explicitly requested continuation")
    parser.add_argument("--checkpoint", type=int, help="Verified completed checkpoint in that original plan")
    args = parser.parse_args()
    if args.resume_plan:
        if args.checkpoint is None:
            parser.error("--resume-plan needs --checkpoint")
        reviewed = resume_plan(args.base, json.loads(Path(args.resume_plan).read_text()), args.checkpoint)
    else:
        reviewed = plan(args.base, args.travel_z, pattern=args.pattern, stroke_mm=args.stroke, cycles=args.cycles,
                        speed_mm_s=args.speed)
    Path(args.output).write_text(json.dumps(reviewed, indent=2))
    print(json.dumps({k: v for k, v in reviewed.items() if k != "path"}), flush=True)
    if args.execute:
        execute(args.base, reviewed)


if __name__ == "__main__":
    main()
