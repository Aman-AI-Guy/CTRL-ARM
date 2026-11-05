"""Skill loop for an object-directed robot task.

Grok picks a skill name. It does not pick a Cartesian target, a joint angle, a speed, or a
safety setting. Local geometry builds a target only from a valid calibration. MotionGateway
checks it. A rejected move is latched for the rest of the task: a later skill cannot dodge
the rejection by asking for a different distance or direction.

Image centering is reported on its own. It is not a verified 3D position.
"""

from __future__ import annotations

import os
import re
import time

from .finder import find as grok_find
from .geom import VisionStore, pose_changed
from .grok import GrokError, register_mock
from .motion import sample_segment
from .scene import TrackState, adopt_found, decide_text, from_finder, question_for
from .workflow import receipt, receipt_error

MAX_STEPS = 8
APPROACH_SPEED_MM_S = 50.0
REACHED_MM = 15.0
SKILLS = (
    "inspect_scene", "list_objects", "select_object", "locate_object", "observe_object",
    "plan_approach", "move_towards_object", "move_above_object", "inspect_from_another_angle",
    "center_object_in_view", "capture_image", "evaluate_goal", "request_clarification",
    "report_status", "start_scan", "plan_orbit", "observe_from_viewpoint", "capture_viewpoint",
    "evaluate_coverage", "choose_next_view", "reconstruct_object", "inspect_reconstruction",
    "request_object_reorientation", "finish_task",
)
MOVE_SKILLS = {"move_towards_object", "move_above_object", "observe_from_viewpoint"}
SNAPSHOT_MAX_AGE_S = 1.5
_COORD = re.compile(r"(?i)\b(x|y|z|xyz)\s*[:=]\s*-?\d")

SCHEMA = {
    "type": "object",
    "properties": {
        "skill": {"type": "string", "enum": list(SKILLS)},
        "argument": {"type": "string"},
        "stop": {"type": "boolean"},
    },
    "required": ["skill", "argument", "stop"],
    "additionalProperties": False,
}

PROMPT = """Choose the next skill for this robot task. You choose a skill name only.
Goal: {goal}
Phase: {phase}
Calibration valid: {calibration_valid}
Selected object: {selected}
Last reason: {last}
Motion lock: {lock}

Rules:
- argument is the person's words or a track id in words. Never output coordinates, joint angles, a speed, or a distance.
- inspect_scene, list_objects, select_object, locate_object, observe_object, capture_image, evaluate_goal, request_clarification, report_status, center_object_in_view and inspect_from_another_angle do not by themselves command a move.
- plan_approach, move_towards_object and move_above_object use a pose computed by local code. You do not supply that pose.
- center_object_in_view and inspect_from_another_angle do not invent a distance and do not change a safety limit.
- If motion lock is set, choose report_status. Do not ask for another direction, another angle, or another distance.
- Scanning uses start_scan, plan_orbit, observe_from_viewpoint, capture_viewpoint, evaluate_coverage, choose_next_view, reconstruct_object and inspect_reconstruction. Local code supplies every viewpoint; never coordinates. Movement requires confirmation of a specific validated proposal. After a moved orbit, capture then evaluate coverage; never infer complete surfaces. Reorientation requires a separate group and reselection.
- Do not repeat the same skill. After inspect_scene, choose select_object.
- If the phase is clarify, choose request_clarification.
- If the goal is achieved or the phase is held, blocked, evaluated or reported, choose report_status and set stop true.
"""


def _coordinates(text: str) -> bool:
    if _COORD.search(text or ""):
        return True
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return len(numbers) >= 3 and sum(abs(float(number)) > 20 for number in numbers) >= 3


def _dist(a, b) -> float:
    return float(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5)


def _result(skill: str, ok: bool, reason: str, task: dict, **extra) -> dict:
    calibration = task.get("calibration_valid")
    return {
        "ok": ok, "success": ok, "skill": skill, "reason": reason,
        "confidence": extra.pop("confidence", None),
        "calibration_valid": bool(calibration),
        "rejection": extra.pop("rejection", None),
        "task_state": {"phase": task.get("phase"), "goal": task.get("goal"),
                       "selected": None if not task.get("selected") else task["selected"].get("track_id"),
                       "motion_lock": task.get("motion_lock")},
        **extra,
    }


def _mock_skill(prompt: str, images: list, ctx: dict) -> dict:
    phase = (ctx or {}).get("phase") or "start"
    nxt = {
        "start": "inspect_scene", "inspected": "select_object", "selected": "locate_object",
        "located": "plan_approach", "planned": "move_above_object", "moved": "capture_image",
        "observed": "evaluate_goal", "evaluated": "report_status", "lost": "inspect_scene",
        "held": "report_status", "blocked": "report_status", "clarify": "request_clarification",
        "reported": "report_status",
    }.get(phase, "report_status")
    if (ctx or {}).get("lock"):
        nxt = "report_status"
    return {"skill": nxt, "argument": (ctx or {}).get("goal") or "", "stop": nxt in ("report_status", "request_clarification")}


register_mock("trova_skill", _mock_skill)


def choose_skill(grok, task: dict) -> dict:
    selected = task.get("selected")
    prompt = PROMPT.format(
        goal=task.get("goal") or "", phase=task.get("phase") or "start",
        calibration_valid=bool(task.get("calibration_valid")),
        selected="" if not selected else f"track {selected.get('track_id')} {selected.get('name', '')}",
        last=task.get("last_reason") or "", lock=task.get("motion_lock") or "",
    )
    try:
        result = grok.json(name="trova_skill", prompt=prompt, schema=SCHEMA, images=[],
                           mock_context={"phase": task.get("phase"), "goal": task.get("goal"),
                                         "lock": task.get("motion_lock")})
    except GrokError as exc:
        return {"ok": False, "skill": "report_status", "argument": "", "stop": True,
                "reason": str(exc)}
    data = result.data
    if data.get("skill") not in SKILLS:
        return {"ok": False, "skill": "report_status", "argument": "", "stop": True,
                "reason": "the skill name is not one of the known skills"}
    if _coordinates(data.get("argument") or ""):
        return {"ok": False, "skill": "report_status", "argument": "", "stop": True,
                "reason": "Grok returned coordinates. A skill does not accept a pose."}
    data["grok_mode"] = "offline" if result.offline else ("mock" if result.mock else "live")
    data["ok"] = True
    return data


def _calibration_flag(host, task) -> bool:
    size = getattr(host, "image_size", None)
    report = host.vision.assess(size)
    task["calibration_valid"] = bool(report["valid"])
    task["calibration"] = report
    return report["valid"]


def qualification_blockers(host, task) -> list[str]:
    """Measurements and settings the operator still has to supply. This does not invent a target."""
    reasons = []
    report = task.get("calibration")
    if not isinstance(report, dict) or "valid" not in report:
        report = host.vision.assess(getattr(host, "image_size", None))
    task["calibration"] = report
    task["calibration_valid"] = bool(report.get("valid"))
    if not report.get("valid"):
        reasons.extend(report.get("reasons") or ["camera calibration is not valid"])
        if report.get("providers"):
            reasons.append("Review /api/vision/providers: verify the imported mount/TCP/base/table binding and independent references, or use the existing current calibration procedure")
        else:
            reasons.append("capture checkerboard images, solve intrinsics, capture at least 6 hand-eye "
                           "views from robot telemetry, and solve the table plane")
    if getattr(host, "observe_clearance_mm", None) is None:
        reasons.append("set TROVA_OBSERVE_CLEARANCE_MM to the measured gap in millimetres from the table plane to the tool")
    gateway = getattr(host, "gateway", None)
    if gateway is None:
        reasons.append("this arm has no motion gateway")
    else:
        if gateway.limits is None:
            reasons.append("set TROVA_ARM_LIMITS to the measured workspace box xmin,xmax,ymin,ymax,zmin,zmax")
        if gateway.camera is None:
            reasons.append("set TROVA_CAMERA_BODY_MM to the measured camera offset and radius; autonomous approach stays blocked")
    if not task.get("authoritative"):
        mode = task.get("grok_mode") or "not live"
        reasons.append(f"recognition is {mode}, so it cannot authorize motion")
    if getattr(host, "camera_kind", "") in ("demo", "armsim"):
        reasons.append("the picture is synthetic, so it cannot authorize motion")
    robot = {}
    try:
        robot = host.robot() or {}
    except Exception as exc:  # noqa: BLE001 - a dead controller is a blocker, not a pose
        reasons.append(f"robot telemetry failed: {exc}")
    source = robot.get("source")
    if source != "telemetry":
        reasons.append(f"the pose source is {source or 'unavailable'}; record it from the controller, not a simulation")
    if not robot.get("pose6"):
        reasons.append("the robot did not report a tool pose with roll, pitch and yaw")
    if robot.get("moving"):
        reasons.append("the robot is still moving")
    if robot.get("error_code"):
        reasons.append(f"controller error code {robot.get('error_code')}")
    if gateway is not None and gateway.mode == "hardware" and not task.get("planned_joints") and not task.get("ik_verified"):
        reasons.append("controller inverse kinematics has not confirmed the observation pose")
    if gateway is not None and gateway.mode == 'hardware' and not task.get('planned_joints'):
        orientation = robot.get('cartesian_target_orientation_deg')
        pose = robot.get('pose6')
        if not orientation or not pose:
            reasons.append('the driver target orientation is unavailable')
        elif (abs(abs(pose[3]) - abs(orientation[0])) > .1 or abs(pose[4] - orientation[1]) > .1
              or abs((pose[5] - orientation[2] + 180) % 360 - 180) >= .1):
            reasons.append('the Cartesian driver would reorient the tool; establish the pointing orientation before planning')
        if not task.get('path_ik_verified'):
            reasons.append('controller inverse kinematics has not validated the sampled path')
    if gateway is not None and gateway.mode == "hardware" and not getattr(gateway, "keepout_assessed", False):
        reasons.append("keep-out regions are not assessed; set TROVA_ARM_KEEP_OUT to measured boxes, "
                       "or to none after checking the path")
    if getattr(host, "read_only_robot", False) and (gateway is None or gateway.simulated):
        reasons.append("robot telemetry is read-only and the motion driver is simulated, "
                       "so this process will not command the physical arm")
    return reasons


def execution_block(host, task) -> str | None:
    """Why an object approach must not be sent, even when the gateway would accept the segment."""
    reasons = qualification_blockers(host, task)
    return "; ".join(reasons) if reasons else None


def _robot(host) -> dict:
    try:
        return host.robot() or {}
    except Exception as exc:  # noqa: BLE001
        return {"source": "unavailable", "error": str(exc), "pose6": None, "xyz_mm": None}


def _target_from_location(host, task) -> tuple[dict, str | None]:
    location = task.get("location") or {}
    if not location.get("ok"):
        return {}, location.get("reason") or "the object has no plane intersection"
    robot = task.get("_robot") or _robot(host)
    pose = robot.get("pose6")
    yaw = None if not pose or len(pose) < 6 else float(pose[5])
    goal = host.vision.observation_target(location.get("point_mm"), host.observe_clearance_mm, yaw_deg=yaw)
    if task.get("scan_viewpoint") and yaw is not None:
        from .scanning import view_goal
        goal = view_goal(host, location, task["scan_viewpoint"], yaw)
    if not goal.get("ok"):
        return goal, goal.get("reason")
    return goal, None


def _authorize(host, task, target: list, skill: str, start=None):
    speed = min(APPROACH_SPEED_MM_S, float(host.gateway.max_speed_mm_s))
    requested = {"source": "vision_approach", "skill": skill, "goal": task.get("goal"),
                 "track_id": None if not task.get("selected") else task["selected"].get("track_id"),
                 "verified_tool_position": False, "approximate_observation_goal": True}
    return host.gateway.authorize_cartesian(target, speed, source="vision_approach", requested=requested,
                                            start=start)


def _thin_points(points, limit: int = 8) -> list:
    if len(points) <= limit:
        return [[round(float(v), 1) for v in point] for point in points]
    indexes = [round(i * (len(points) - 1) / (limit - 1)) for i in range(limit)]
    return [[round(float(v), 1) for v in points[i]] for i in indexes]


def _pack_proposal(decision, goal, robot, start, task=None) -> dict:
    packed = decision.as_dict()
    packed["trajectory_mm"] = []
    proposal = packed.get("proposal") or {}
    begin = proposal.get("start_mm")
    end = proposal.get("target_mm")
    if begin and end:
        packed["trajectory_mm"] = _thin_points(sample_segment(begin, end))
    packed["ik_verified"] = bool(goal.get("ik_verified"))
    packed['path_ik_verified'] = bool((task or {}).get('path_ik_verified'))
    packed["ik_joints_deg"] = None if not goal.get("ik") else goal["ik"].get("joints")
    packed["ik_reason"] = None if not goal.get("ik") else goal["ik"].get("reason")
    packed["geometry_source"] = "telemetry" if start is not None else (robot.get("source") or "unavailable")
    packed["collision_avoidance"] = False
    packed["pose6"] = goal.get("pose6")
    packed["desired_pose6"] = goal.get("pose6")
    packed["current_pose6"] = robot.get("pose6")
    packed["controller_state"] = robot.get("state_code")
    packed["controller_error"] = robot.get("error_code")
    packed["execution_authorized"] = False
    packed["straight_line_samples_are_not_a_joint_path"] = not packed["ik_verified"]
    proposal = packed.get("proposal") or {}
    distance = proposal.get("distance_mm")
    speed = float((packed.get("requested") or {}).get("speed_mm_s") or APPROACH_SPEED_MM_S)
    if distance is not None and speed > 0:
        packed["path_length_mm"] = distance
        packed["duration_s"] = round(float(distance) / speed, 2)
        packed["duration_is_straight_line_estimate"] = True
    else:
        packed["duration_s"] = None
        packed["duration_is_straight_line_estimate"] = False
    location = (task or {}).get("location") or {}
    chosen = (task or {}).get("selected") or {}
    packed["object_id"] = chosen.get("track_id")
    packed["localization_mm"] = location.get("point_mm")
    packed["uncertainty_mm"] = location.get("uncertainty_mm")
    packed["uncertainty_does_not_expand_limits"] = True
    return packed


def _note(host, stage: str, seconds: float) -> None:
    note = getattr(host, "note_stage", None)
    if note:
        note(stage, max(0.0, float(seconds)))


def _uncertainty_block(host, task, target, start) -> str | None:
    """Refuse when the uncertain region leaves the qualified workspace. Limits stay as measured."""
    raw = (task.get("location") or {}).get("uncertainty_mm")
    if raw is None:
        return None
    try:
        radius = float(raw)
    except (TypeError, ValueError):
        return "localization uncertainty is not a number, so the approach stays blocked"
    if radius < 0:
        return "localization uncertainty is negative, so the approach stays blocked"
    if radius == 0 or start is None:
        return None
    gateway = getattr(host, "gateway", None)
    if gateway is None or gateway.limits is None:
        return None
    for dx, dy in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)):
        edge = [float(target[0]) + dx, float(target[1]) + dy, float(target[2])]
        decision = _authorize(host, task, edge, "uncertainty_check", start=start)
        if decision.status == "rejected":
            return (f"localization uncertainty {radius:g} mm puts part of the approach outside "
                    f"the qualified workspace: {decision.reason}")
    return None


def _mark_authorized(host, task, packed) -> dict:
    """Physical execution only. A simulation acceptance does not authorize the robot."""
    gateway = getattr(host, "gateway", None)
    packed["execution_authorized"] = bool(
        packed.get("allow_motion")
        and packed.get("mode") == "hardware"
        and gateway is not None
        and not gateway.simulated
        and not os.environ.get("PYTEST_CURRENT_TEST")
        and (not getattr(host, "confirmation_required", False) or bool(task.get("_confirmed")))
        and execution_block(host, task) is None)
    return packed


def _selection_unreliable(task) -> str | None:
    chosen = task.get("selected") or {}
    if not chosen:
        return None
    if chosen.get("identity_uncertain") or chosen.get("association") == "uncertain":
        return "the viewpoint changed and the selection is no longer reliable; select the object again"
    if chosen.get('label_authoritative') is False:
        return 'the selected object has no authoritative name from live recognition'
    return None


def _refuse_locked(task, skill: str):
    if not task.get("motion_lock"):
        return None
    reason = "a rejected movement cannot be retried by changing the distance or the direction: " + task["motion_lock"]
    task["phase"] = "blocked"
    return _result(skill, False, reason, task, terminal=True, rejection=reason, executed=False)


def _snapshot_is_reusable(host, task) -> dict | None:
    """Reuse the last picture only while its pose, age, and provenance still match."""
    snap = getattr(host, "last_snapshot", None)
    if not snap or task.get("phase") in {"moved", "lost", "observed"}:
        return None
    taken = snap.get("monotonic")
    if taken is None or time.monotonic() - float(taken) > SNAPSHOT_MAX_AGE_S:
        return None
    robot = _robot(host)
    pose = snap.get("pose") or {}
    if robot.get("moving") or pose.get("moving"):
        return None
    if pose_changed(pose.get("pose6"), robot.get("pose6")):
        return None
    if snap.get("provenance") != (pose.get("provenance") or snap.get("provenance")):
        return None
    if snap.get("camera_kind") in {"demo", "armsim"}:
        return None
    return snap


def skill_inspect(host, task, argument: str) -> dict:
    cached = _snapshot_is_reusable(host, task)
    if cached is not None:
        snap = cached
        reused = True
    else:
        try:
            snap = host.snapshot()
        except Exception as exc:  # noqa: BLE001 - a dead camera ends the step; the old picture is not reused
            task["phase"] = "blocked"
            return _result("inspect_scene", False, str(exc), task, terminal=True, rejection=str(exc))
        snap = dict(snap)
        snap["monotonic"] = time.monotonic()
        host.last_snapshot = snap
        reused = False
    task["objects"] = list(snap.get("objects") or [])
    task["grok_mode"] = snap.get("grok_mode")
    task["authoritative"] = snap.get("grok_mode") == "live"
    task["snapshot_id"] = snap.get("id")
    task["snapshot_pose"] = snap.get("pose")
    task["summary"] = snap.get("summary")
    host.image_size = [snap.get("width") or host.image_size[0], snap.get("height") or host.image_size[1]]
    _calibration_flag(host, task)
    task["phase"] = "inspected"
    if task.get("requested_track_id") is not None:
        chosen = next((obj for obj in task["objects"] if obj.get("track_id") == task["requested_track_id"]), None)
        if not chosen or chosen.get("identity_uncertain") or chosen.get("association") == "uncertain":
            task["motion_lock"] = "The selected object's identity is no longer reliable in the fresh image. Select it again."
            task["phase"] = "blocked"
            return _result("inspect_scene", False, task["motion_lock"], task, terminal=True, executed=False)
        previous = task.get("selected") or {}
        reference = previous.get("geometric_reference")
        if reference and reference.get("snapshot_id") == snap.get("id"):
            chosen["geometric_reference"] = reference
        task["selected"] = chosen
        task["location"] = None
        task["phase"] = "selected"
    return _result("inspect_scene", True, snap.get("summary") or f"{len(task['objects'])} objects", task,
                   grok_mode=snap.get("grok_mode"), n_objects=len(task["objects"]),
                   authoritative=task["authoritative"], snapshot_reused=reused)


def skill_list(host, task, argument: str) -> dict:
    objects = task.get("objects") or []
    return _result("list_objects", True, f"{len(objects)} objects in the latest snapshot", task,
                   objects=[{k: o.get(k) for k in ("track_id", "name", "color", "confidence", "association",
                                                   "identity_uncertain", "ambiguous")} for o in objects])


def skill_select(host, task, argument: str) -> dict:
    objects = task.get("objects") or []
    if host.frame is None:
        task["phase"] = "blocked"
        return _result("select_object", False, "Take a snapshot before selecting", task, terminal=True)
    text = (argument or task.get("goal") or "").strip()
    decision = decide_text(text, objects)
    if decision.status == "defer":
        if host.frame is None:
            task["phase"] = "blocked"
            return _result("select_object", False, "the snapshot frame is gone", task, terminal=True)
        found = grok_find(host.grok, host.frame, host.boxes, text)
        if found.status == "found":
            if hasattr(host, "adopt_found"):
                objects = host.adopt_found(found)
            else:
                tracks = getattr(host, "tracks", None) or TrackState()
                host.tracks = tracks
                host.boxes, objects = adopt_found(tracks, host.frame, host.boxes, objects, found)
            task["objects"] = objects
            task["authoritative"] = found.recognition_live
        decision = from_finder(found.status, found.box.id if found.box else None, found.candidate_ids,
                               found.confidence, found.reason, objects, text)
    by_id = {obj["track_id"]: obj for obj in objects}
    if decision.status == "selected" and decision.track_ids:
        chosen = by_id.get(decision.track_ids[0])
        task["selected"] = chosen
        task["phase"] = "selected"
        return _result("select_object", True, decision.reason, task, confidence=decision.confidence,
                       object=chosen, authoritative=task.get("authoritative"), motion_allowed=False)
    if decision.status == "clarify":
        task["phase"] = "clarify"
        task["question"] = decision.question or question_for([by_id[i] for i in decision.track_ids if i in by_id], text)
        task["candidates"] = [by_id[i] for i in decision.track_ids if i in by_id]
        return _result("select_object", True, task["question"], task, question=task["question"],
                       candidates=task["candidates"], motion_allowed=False)
    task["phase"] = "blocked"
    return _result("select_object", False, decision.reason or "not in this snapshot", task, terminal=True,
                   motion_allowed=False)


def skill_locate(host, task, argument: str) -> dict:
    chosen = task.get("selected")
    if not chosen:
        task["phase"] = "blocked"
        return _result("locate_object", False, "no object is selected", task, terminal=True)
    _calibration_flag(host, task)
    robot = _robot(host)
    snap = task.get("snapshot_pose") or {}
    if snap.get("moving") or robot.get("moving"):
        task["phase"] = "blocked"
        task["motion_lock"] = "the picture and the pose are not from a stationary arm"
        return _result("locate_object", False, task["motion_lock"], task, rejection=task["motion_lock"])
    if pose_changed(snap.get("pose6"), robot.get("pose6")):
        task["phase"] = "blocked"
        task["motion_lock"] = "the robot pose changed after the picture; take another picture before localizing"
        return _result("locate_object", False, task["motion_lock"], task, rejection=task["motion_lock"])
    box = chosen.get("box")
    began = time.perf_counter()
    if hasattr(host.vision, "localize_observation"):
        location = host.vision.localize_observation(robot, chosen, host.image_size)
    else:
        location = host.vision.localize(robot.get("pose6"), chosen["center"][0], chosen["center"][1],
                                        host.image_size, box=box)
    _note(host, "geometric_localization", time.perf_counter() - began)
    task["location"] = location
    chosen["metric_localization_state"] = "localized" if location.get("ok") else "blocked"
    if location.get("ok") and location.get("point_mm") is not None:
        remember = getattr(host, "remember", None)
        if remember:
            remember(chosen["track_id"], location["point_mm"])
        task["phase"] = "located"
        return _result("locate_object", True, location["reason"], task, location=location,
                       confidence=chosen.get("confidence"))
    task["phase"] = "blocked"
    task["motion_lock"] = location.get("reason")
    return _result("locate_object", False, location.get("reason") or "not localized", task,
                   rejection=location.get("reason"), location=location)


def _settled(host):
    robot = host.robot()
    if robot.get("moving"):
        return None, "the robot is still moving"
    if host.settle_s:
        if host.cancel.is_set():
            return None, "stopped"
        time.sleep(host.settle_s)
        if host.robot().get("moving"):
            return None, "the robot moved during settle"
    if host.cancel.is_set():
        return None, "STOP or cancellation requested"
    stopped_at = time.time()
    frame, meta = host.read_frame()
    after = host.robot()
    if host.cancel.is_set() or after.get("moving") or pose_changed(robot.get("pose6"), after.get("pose6")):
        return None, "the robot moved or the task was cancelled during capture"
    if after.get("error_code"):
        return None, "controller fault during capture"
    meta = dict(meta or {})
    meta.setdefault("stopped_at", stopped_at)
    requested = meta.get("requested_at")
    if requested is not None and float(requested) + 1e-3 < stopped_at and meta.get("kind") == "pi":
        return None, "the frame was requested before the robot had settled"
    return (frame, meta), None


def skill_observe(host, task, argument: str) -> dict:
    packed, err = _settled(host)
    if err:
        task["phase"] = "blocked"
        return _result("observe_object", False, err, task, rejection=err)
    frame, meta = packed
    objects = host.redetect(frame)
    task["objects"] = objects
    chosen = task.get("selected")
    hit = None if not chosen else next((obj for obj in objects if obj["track_id"] == chosen["track_id"]), None)
    task["phase"] = "observed"
    visible = hit is not None
    return _result("observe_object", True, "visible" if visible else "the selected object is not in this view", task,
                   visible=visible, association=None if not hit else hit.get("association"),
                   identity_uncertain=bool(hit and hit.get("identity_uncertain")), capture=meta, motion_allowed=False)


def skill_plan(host, task, argument: str) -> dict:
    locked = _refuse_locked(task, "plan_approach")
    if locked:
        return locked
    unreliable = _selection_unreliable(task)
    if unreliable:
        task["selected"] = None
        task["motion_lock"] = unreliable
        task["phase"] = "blocked"
        return _result("plan_approach", False, unreliable, task, terminal=True, rejection=unreliable, executed=False)
    if _coordinates(argument):
        task["motion_lock"] = "Grok returned coordinates. The planner ignored them and stopped."
        task["phase"] = "blocked"
        return _result("plan_approach", False, task["motion_lock"], task, terminal=True, rejection=task["motion_lock"])
    _calibration_flag(host, task)
    if not task.get("calibration_valid"):
        blockers = qualification_blockers(host, task)
        reason = "; ".join(blockers) or "camera calibration is not valid"
        task["motion_lock"] = reason
        task["planned_target"] = None
        task["phase"] = "blocked"
        return _result("plan_approach", False, reason, task, rejection=reason, executed=False, blockers=blockers)
    robot = _robot(host)
    task["_robot"] = robot
    began = time.perf_counter()
    goal, err = _target_from_location(host, task)
    _note(host, "motion_planning", time.perf_counter() - began)
    if err:
        task["motion_lock"] = err
        task["phase"] = "blocked"
        return _result("plan_approach", False, err, task, rejection=err, target=goal, executed=False)
    ask = getattr(host, "inverse", None)
    if ask is not None and goal.get("pose6"):
        ik = ask(goal["pose6"], robot.get("joints"))
    else:
        ik = {"ok": False, "ik_verified": False, "joints": None, "motion_sent": False,
              "reason": "inverse kinematics was not asked of the controller"}
    goal["ik"] = ik
    goal["ik_verified"] = bool(ik.get("ik_verified"))
    task["ik_verified"] = goal["ik_verified"]
    task['path_ik_verified'] = False
    if host.gateway is not None and host.gateway.mode == 'hardware' and goal['ik_verified'] and robot.get('xyz_mm'):
        reference = robot.get('joints')
        path_ok = True
        for point in sample_segment(robot['xyz_mm'], goal['target_mm']):
            if host.cancel.is_set():
                path_ok = False
                break
            query = ask([*point, *goal['pose6'][3:]], reference)
            if not query.get('ik_verified'):
                path_ok = False
                break
            reference = query.get('joints')
        task['path_ik_verified'] = path_ok
        goal['path_ik_verified'] = path_ok
    if getattr(host, "gateway", None) is None:
        task["motion_lock"] = "this arm has no motion gateway"
        task["phase"] = "blocked"
        return _result("plan_approach", False, task["motion_lock"], task, rejection=task["motion_lock"],
                       target=goal, executed=False)
    start = None
    if robot.get("source") == "telemetry" and robot.get("xyz_mm") and len(robot["xyz_mm"]) >= 3:
        start = tuple(float(v) for v in robot["xyz_mm"][:3])
    began = time.perf_counter()
    decision = _authorize(host, task, goal["target_mm"], "plan_approach", start=start)
    _note(host, "gateway_validation", time.perf_counter() - began)
    packed = _mark_authorized(host, task, _pack_proposal(decision, goal, robot, start, task))
    task["planned_target"] = list(goal["target_mm"])
    task["proposal"] = packed
    refusals = []
    if host.gateway.camera is None:
        refusals.append("set TROVA_CAMERA_BODY_MM to the measured camera offset and radius; "
                        "the approach is refused because clearance cannot be established")
    if decision.status == "rejected":
        refusals.append(decision.reason)
    else:
        edge = _uncertainty_block(host, task, goal["target_mm"], start)
        if edge:
            refusals.append(edge)
            packed["allow_motion"] = False
            packed["execution_authorized"] = False
    if refusals:
        task["motion_lock"] = "; ".join(refusals)
        task["phase"] = "blocked"
        return _result("plan_approach", False, task["motion_lock"], task, rejection=task["motion_lock"],
                       motion=packed, target=goal, executed=False)
    task["phase"] = "planned"
    return _result("plan_approach", True, decision.reason, task, motion=packed, target=goal, executed=False)


def _move(host, task, skill: str) -> dict:
    if task.get('planned_joints'):
        from .scan3d.skills import observe
        return observe(host,task,'')
    if getattr(host, "confirmation_required", False) and task.get("_confirmed"):
        invalid = receipt_error(host, task, task.get("_confirmed"))
        if invalid:
            task["motion_lock"] = invalid
            task["phase"] = "blocked"
            return _result(skill, False, invalid, task, terminal=True, executed=False)
    if getattr(host, "confirmation_required", False) and not task.get("planned_target"):
        planned = skill_plan(host, task, "")
        if not planned.get("ok"):
            return planned
    locked = _refuse_locked(task, skill)
    if locked:
        return locked
    unreliable = _selection_unreliable(task)
    if unreliable:
        task["selected"] = None
        task["motion_lock"] = unreliable
        task["phase"] = "blocked"
        return _result(skill, False, unreliable, task, terminal=True, rejection=unreliable, executed=False)
    _calibration_flag(host, task)
    goal, err = _target_from_location(host, task)
    if err:
        task["motion_lock"] = err
        task["phase"] = "blocked"
        return _result(skill, False, err, task, rejection=err, executed=False)
    target = list(task["planned_target"]) if task.get("planned_target") else list(goal["target_mm"])
    # Keep the already planned point. A new number from the caller is not applied.
    robot = _robot(host)
    start = None
    if robot.get("source") == "telemetry" and robot.get("xyz_mm") and len(robot["xyz_mm"]) >= 3:
        start = tuple(float(v) for v in robot["xyz_mm"][:3])
    began = time.perf_counter()
    decision = _authorize(host, task, target, skill, start=start)
    _note(host, "gateway_validation", time.perf_counter() - began)
    task["proposal"] = _mark_authorized(host, task, _pack_proposal(decision, goal, robot, start, task))
    edge = None if decision.status in ("rejected", "stopped", "fault") else _uncertainty_block(
        host, task, target, start)
    if edge:
        task["proposal"]["allow_motion"] = False
        task["proposal"]["execution_authorized"] = False
        task["motion_lock"] = edge
        task["phase"] = "blocked"
        return _result(skill, False, edge, task, rejection=edge, motion=task["proposal"], executed=False)
    if decision.status in ("rejected", "stopped", "fault"):
        task["motion_lock"] = decision.reason
        task["phase"] = "blocked"
        return _result(skill, False, decision.reason, task, rejection=decision.reason,
                       motion=task["proposal"], executed=False)
    blocked = execution_block(host, task)
    if blocked or not decision.allow_motion:
        reason = blocked or decision.reason
        task["motion_lock"] = reason
        task["phase"] = "held"
        return _result(skill, False, reason, task, rejection=reason, motion=task["proposal"],
                       executed=False, target={"target_mm": target, "verified_tool_position": False,
                                               "approximate_observation_goal": True})
    if getattr(host, "confirmation_required", False) and not task.get("_confirmed"):
        task["confirmation"] = receipt(host, task)
        task["phase"] = "awaiting_confirmation"
        return _result(skill, True, "A validated proposal is awaiting operator confirmation", task,
                       terminal=True, executed=False)
    if host.cancel.is_set():
        task["phase"] = "blocked"
        return _result(skill, False, "STOP or cancellation requested", task, terminal=True, executed=False)
    task["workflow_state"] = "movement_executing"
    host.active_task = task
    outcome = host.execute(target)
    task.pop("_confirmed", None)
    task.pop("confirmation", None)
    moved = bool(outcome.get("ok"))
    task['movement_status_unknown'] = bool(outcome.get('movement_status_unknown'))
    task["phase"] = "moved" if moved else "blocked"
    if not moved:
        task["motion_lock"] = outcome.get("reason") or "the arm did not complete the approach"
    return _result(skill, moved, outcome.get("reason") or ("moved" if moved else "not moved"), task,
                   motion=task["proposal"], executed=moved or bool(outcome.get('motion_observed')),
                   movement_status_unknown=task['movement_status_unknown'], final_mm=outcome.get("final_mm"),
                   verified_tool_position=False)


def skill_move_towards(host, task, argument: str) -> dict:
    # Same observation pose as move_above. A contact point is not commanded from one image.
    return _move(host, task, "move_towards_object")


def skill_move_above(host, task, argument: str) -> dict:
    return _move(host, task, "move_above_object")


def skill_capture(host, task, argument: str) -> dict:
    packed, err = _settled(host)
    if err:
        task["phase"] = "blocked"
        return _result("capture_image", False, err, task, rejection=err)
    frame, meta = packed
    if task.get("scan_session"):
        objects = host.redetect(frame)
        selected_id = (task.get("selected") or {}).get("track_id")
        hit = next((obj for obj in objects if obj.get("track_id") == selected_id), None)
        meta["quality_required"] = True
        meta["object_visible"] = bool(hit and not hit.get("identity_uncertain")
                                      and hit.get("association") in {"same_view", "geometric"})
        meta["viewpoint_candidate"] = task.get("scan_viewpoint")
        if hit:
            task["selected"] = {**task["selected"], **hit}
    task["last_capture"] = {k: v for k, v in meta.items() if k != "frame"}
    task["phase"] = "observed"
    if getattr(host, "record_capture", None):
        task["acquisition"] = host.record_capture(frame, meta, task.get("selected"))
        if task.get("scan_session") and task.get("acquisition") and not task["acquisition"].get("accepted"):
            task["phase"] = "blocked"
            task["motion_lock"] = task["acquisition"].get("reject_reason") or "scan capture was rejected"
            return _result("capture_image", False, task["motion_lock"], task, terminal=True,
                           acquisition=task["acquisition"], motion_allowed=False)
    return _result("capture_image", True, "fresh frame after settle", task, capture=task["last_capture"],
                   motion_allowed=False)


def skill_evaluate(host, task, argument: str) -> dict:
    packed, err = _settled(host)
    objects = task.get("objects") or []
    if err is None:
        frame, _meta = packed
        objects = host.redetect(frame)
        task["objects"] = objects
    chosen = task.get("selected")
    hit = None if not chosen else next((obj for obj in objects if obj["track_id"] == chosen["track_id"]), None)
    visible = hit is not None if chosen else False
    near = False
    if hit and host.image_size and host.image_size[0] > 1:
        width, height = host.image_size
        near = abs(hit["center"][0] - width / 2) < 0.2 * width and abs(hit["center"][1] - height / 2) < 0.2 * height
    robot = host.robot()
    target = task.get("planned_target")
    reached = None
    if target and robot.get("xyz_mm"):
        reached = _dist(target, robot["xyz_mm"]) <= REACHED_MM
    loc_valid = bool((task.get("location") or {}).get("ok") and task.get("calibration_valid"))
    association = None if not hit else hit.get("association")
    identity_ok = association in ("same_view", "geometric")
    achieved = bool(err is None and not robot.get("moving") and not robot.get("error_code")
                    and reached and visible and loc_valid and identity_ok and not (hit or {}).get("identity_uncertain"))
    if near and not reached:
        achieved = False
    checks = {
        "object_visible": visible, "near_intended_image_location": near,
        "image_centering_is_not_3d_proximity": True,
        "target_lost": bool(chosen) and not visible,
        "robot_reached_planned_position": reached, "localization_valid": loc_valid,
        "identity": association, "identity_uncertain": bool(hit and hit.get("identity_uncertain")),
        "another_observation_necessary": not achieved, "goal_achieved": achieved,
        "capture_error": err,
    }
    task["evaluation"] = checks
    reason = "goal achieved" if achieved else "goal not achieved"
    if near and not reached:
        reason = "the object is near the image centre; that is not a verified 3D position"
    if err:
        reason = err
    lost = bool(chosen) and not visible and err is None
    if lost and int(task.get("replans") or 0) < 1:
        task["replans"] = int(task.get("replans") or 0) + 1
        task["phase"] = "lost"
        reason = "the object left the view; one new observation is allowed, then the loop stops"
        return _result("evaluate_goal", True, reason, task, evaluation=checks, goal_achieved=False)
    if lost:
        task["phase"] = "blocked"
        task["motion_lock"] = "the object was lost again; the loop stopped"
        return _result("evaluate_goal", False, task["motion_lock"], task, terminal=True,
                       rejection=task["motion_lock"], evaluation=checks, goal_achieved=False)
    task["phase"] = "evaluated"
    return _result("evaluate_goal", err is None, reason, task, evaluation=checks, goal_achieved=achieved)


def skill_center(host, task, argument: str) -> dict:
    """Report the pixel offset. A metric move is the same observation pose, not a new nudge."""
    locked = _refuse_locked(task, "center_object_in_view")
    if locked:
        return locked
    chosen = task.get("selected") or {}
    width = (host.image_size or [0, 0])[0]
    height = (host.image_size or [0, 0])[1]
    center = chosen.get("center")
    offset = None
    near = False
    if center and width > 1 and height > 1:
        offset = [round(float(center[0]) - width / 2, 1), round(float(center[1]) - height / 2, 1)]
        near = abs(offset[0]) < 0.2 * width and abs(offset[1]) < 0.2 * height
    if near:
        return _result("center_object_in_view", True,
                       "the object is near the image centre; that is not a verified 3D position", task,
                       executed=False, pixel_offset=offset, image_centering_is_not_3d_proximity=True,
                       motion_allowed=False)
    if not chosen:
        task["phase"] = "blocked"
        return _result("center_object_in_view", False, "no object is selected", task, terminal=True, executed=False)
    planned = skill_plan(host, task, argument)
    planned["skill"] = "center_object_in_view"
    planned["pixel_offset"] = offset
    planned["image_centering_is_not_3d_proximity"] = True
    return planned


def skill_angle(host, task, argument: str) -> dict:
    """Propose the same table point with 20 degrees more yaw. Do not execute it here."""
    locked = _refuse_locked(task, "inspect_from_another_angle")
    if locked:
        return locked
    unreliable = _selection_unreliable(task)
    if unreliable:
        task["selected"] = None
        task["motion_lock"] = unreliable
        task["phase"] = "blocked"
        return _result("inspect_from_another_angle", False, unreliable, task, terminal=True,
                       rejection=unreliable, executed=False)
    if _coordinates(argument):
        task["motion_lock"] = "Grok returned coordinates. The planner ignored them and stopped."
        task["phase"] = "blocked"
        return _result("inspect_from_another_angle", False, task["motion_lock"], task, terminal=True,
                       rejection=task["motion_lock"])
    _calibration_flag(host, task)
    if not task.get("calibration_valid"):
        blockers = qualification_blockers(host, task)
        reason = "; ".join(blockers) or "camera calibration is not valid"
        task["motion_lock"] = reason
        task["planned_target"] = None
        task["phase"] = "blocked"
        return _result("inspect_from_another_angle", False, reason, task, rejection=reason,
                       executed=False, blockers=blockers)
    robot = _robot(host)
    pose = robot.get("pose6")
    if not pose or len(pose) < 6:
        reason = "yaw was not read from telemetry, so another viewing angle was not invented"
        return _result("inspect_from_another_angle", False, reason, task, executed=False, rejection=reason)
    yaw = float(pose[5]) + 20.0
    location = task.get("location") or {}
    goal = host.vision.observation_target(location.get("point_mm"), host.observe_clearance_mm, yaw_deg=yaw)
    if not goal.get("ok"):
        return _result("inspect_from_another_angle", False, goal.get("reason") or "no observation pose", task,
                       executed=False, rejection=goal.get("reason"))
    ask = getattr(host, "inverse", None)
    ik = ask(goal["pose6"], robot.get("joints")) if ask is not None else {
        "ok": False, "ik_verified": False, "joints": None, "motion_sent": False,
        "reason": "inverse kinematics was not asked of the controller"}
    goal["ik"] = ik
    goal["ik_verified"] = bool(ik.get("ik_verified"))
    if not goal["ik_verified"]:
        reason = ("the controller did not confirm a pose 20 degrees further in yaw; "
                  "the xArm 5 may not reach that orientation. No move was sent. " + (ik.get("reason") or ""))
        return _result("inspect_from_another_angle", False, reason.strip(), task, executed=False,
                       target=goal, ik_verified=False, motion_allowed=False)
    if getattr(host, "gateway", None) is None:
        return _result("inspect_from_another_angle", False, "this arm has no motion gateway", task,
                       executed=False, target=goal)
    start = None
    if robot.get("source") == "telemetry" and robot.get("xyz_mm"):
        start = tuple(float(v) for v in robot["xyz_mm"][:3])
    decision = _authorize(host, task, goal["target_mm"], "inspect_from_another_angle", start=start)
    packed = _mark_authorized(host, task, _pack_proposal(decision, goal, robot, start, task))
    packed["yaw_offset_deg"] = 20.0
    packed["orientation_is_not_a_collision_model"] = True
    task["proposal"] = packed
    if decision.status == "rejected" or host.gateway.camera is None:
        reason = decision.reason if decision.status == "rejected" else (
            "clearance cannot be established for the alternate view")
        task["motion_lock"] = reason
        task["phase"] = "blocked"
        return _result("inspect_from_another_angle", False, reason, task, rejection=reason,
                       motion=packed, executed=False)
    edge = _uncertainty_block(host, task, goal["target_mm"], start)
    if edge:
        packed["allow_motion"] = False
        packed["execution_authorized"] = False
        task["motion_lock"] = edge
        task["phase"] = "blocked"
        return _result("inspect_from_another_angle", False, edge, task, rejection=edge,
                       motion=packed, executed=False)
    task["planned_target"] = list(goal["target_mm"])
    task["phase"] = "planned"
    return _result("inspect_from_another_angle", True,
                   "alternate yaw checked by the gateway; this skill does not execute it", task,
                   motion=packed, target=goal, executed=False, motion_allowed=False)


def skill_scan_plan(host, task, argument):
    if hasattr(host,'orbits'):
        from .scan3d.skills import plan
        return plan(host,task,argument)
    from .scanning import rank_views
    book = getattr(host, "scan_book", None)
    if book is None or not book.active:
        task["phase"] = "blocked"
        return _result("plan_scan_view", False, "No scan session is open", task, terminal=True)
    task["scan_session"] = book.active
    session = book._session(None)
    expected_track = str(session.get('orientation_track_id', session['object_id']))
    if expected_track != str((task.get('selected') or {}).get('track_id')) or session.get('needs_reselection'):
        task['phase'] = 'blocked'
        return _result('plan_scan_view', False, 'The selected object does not match this scan orientation group.',
                       task, terminal=True, executed=False)
    views = [v for v in book.list_views() if v.get("orientation") == book._session(None)["orientation"]]
    report = rank_views(host, task, views)
    task["scan_plan"] = report
    if not report.get("next"):
        task["phase"] = "blocked"
        task["motion_lock"] = report["reason"]
        return _result("plan_scan_view", False, report["reason"], task, terminal=True,
                       scan_plan=report, executed=False)
    task["scan_viewpoint"] = report["next"]["id"]
    return skill_plan(host, task, "")


def skill_clarify(host, task, argument: str) -> dict:
    task["phase"] = "clarify"
    question = task.get("question") or "Which object?"
    return _result("request_clarification", True, question, task, terminal=True, question=question,
                   motion_allowed=False)


def skill_report(host, task, argument: str) -> dict:
    task["phase"] = "reported"
    evaluation = task.get("evaluation") or {}
    reason = task.get("motion_lock") or task.get("question") or task.get("summary") or "task finished"
    if evaluation:
        reason = "goal achieved" if evaluation.get("goal_achieved") else "goal not achieved: " + reason
    return _result("report_status", True, reason, task, terminal=True, evaluation=evaluation,
                   motion=task.get("proposal"), executed=False)


from .scan3d import skills as scan_skills

SKILL_FN = {
    'start_scan':scan_skills.start_scan, 'plan_orbit':scan_skills.plan,
    'observe_from_viewpoint':scan_skills.observe, 'capture_viewpoint':skill_capture,
    'evaluate_coverage':scan_skills.coverage, 'choose_next_view':scan_skills.plan,
    'reconstruct_object':scan_skills.reconstruct, 'inspect_reconstruction':scan_skills.inspect,
    'request_object_reorientation':scan_skills.reorient, 'finish_task':scan_skills.finish,
    "plan_scan_view": skill_scan_plan,
    "inspect_scene": skill_inspect, "list_objects": skill_list, "select_object": skill_select,
    "locate_object": skill_locate, "observe_object": skill_observe, "plan_approach": skill_plan,
    "move_towards_object": skill_move_towards, "move_above_object": skill_move_above,
    "inspect_from_another_angle": skill_angle, "center_object_in_view": skill_center,
    "capture_image": skill_capture, "evaluate_goal": skill_evaluate,
    "request_clarification": skill_clarify, "report_status": skill_report,
}


def _timing_summary(rows: list[dict]) -> dict:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["stage"], []).append(float(row["seconds"]))
    summary = {}
    for stage, values in buckets.items():
        summary[stage] = {"n": len(values), "last_s": round(values[-1], 3),
                          "min_s": round(min(values), 3), "max_s": round(max(values), 3),
                          "mean_s": round(sum(values) / len(values), 3)}
    return summary


def _resume_task(task: dict) -> dict:
    keys = ("goal", "phase", "selected", "objects", "location", "planned_target", "proposal",
            "motion_lock", "evaluation", "question", "authoritative", "calibration_valid",
            "calibration", "last_reason", "steps", "replans", "grok_mode", "snapshot_pose",
            "snapshot_id", "ik_verified", "timings", "summary", "confirmation", "workflow_state", "workflow_events",
            "scan_session", "scan_viewpoint", "scan_plan", "path_ik_verified", "planned_joints",
            "orbit_geometry", "orbit_speed", "orbit_reached", "reconstruction_job", "coverage", "orbit_anchor_joints")
    return {key: task.get(key) for key in keys}


def run(host, text: str, *, step: bool = False, continued: dict | None = None,
        confirm_id: str | None = None, selected_context: dict | None = None,
        skill_sequence: list[str] | None = None) -> dict:
    """Bounded observe-plan-act-observe loop. Hardware moves only when every qualification passes.

    step=True runs one skill and returns. The caller may pass the previous resume dict.
    """
    goal = text.strip()
    reusable = ((step or confirm_id) and isinstance(continued, dict) and continued.get("goal") == goal
                and continued.get("phase") not in {"reported", "blocked", None})
    if reusable:
        task = dict(continued)
        task["steps"] = list(continued.get("steps") or [])
        task["timings"] = list(continued.get("timings") or [])
        task.setdefault("motion_lock", None)
        task.setdefault("replans", 0)
    else:
        task = {
            "goal": goal, "phase": "start", "selected": None, "objects": [], "location": None,
            "planned_target": None, "proposal": None, "motion_lock": None, "evaluation": None,
            "question": None, "authoritative": False, "calibration_valid": False, "last_reason": "",
            "steps": [], "timings": [], "replans": 0,
        }
        if selected_context:
            task.update(selected_context)
            task["requested_track_id"] = (task.get("selected") or {}).get("track_id")
            task["phase"] = "start"
    if confirm_id:
        task["_confirmed"] = confirm_id
        task["phase"] = "planned"
        if task.get('planned_joints') and skill_sequence is None:
            # One accepted motion is followed by observation and another proposal,
            # never by an unconfirmed actuator command. Re-localize from a new scene.
            task['_scan_continuation'] = True
            task['requested_track_id'] = (task.get('selected') or {}).get('track_id')
            skill_sequence = ['observe_from_viewpoint','capture_viewpoint','evaluate_coverage',
                              'inspect_scene','locate_object','plan_orbit','observe_from_viewpoint']
    _calibration_flag(host, task)
    task["workflow_events"] = list(task.get("workflow_events") or [])
    started = time.monotonic()
    status = "completed"
    limit = 1 if step else MAX_STEPS
    for _step in range(limit):
        if host.cancel.is_set():
            status = "stopped"
            break
        if time.monotonic() - started > float(getattr(host, "timeout_s", 90)):
            status = "timeout"
            task["last_reason"] = "the task reached its time limit"
            break
        # Confirmation executes the bound proposal; the model cannot substitute another skill.
        if task.get('phase') == 'scan_ready_for_reconstruction':
            choice = {'skill':'reconstruct_object','argument':'','ok':True}
        elif skill_sequence and _step < len(skill_sequence):
            choice = {"skill": skill_sequence[_step], "argument": "", "ok": True}
        elif task.get("requested_track_id") is not None and task["phase"] == "start":
            choice = {"skill": "inspect_scene", "argument": "", "ok": True}
        elif task.get("_confirmed"):
            choice = {"skill": "observe_from_viewpoint" if task.get("planned_joints") else "move_above_object", "argument": "", "ok": True}
        elif getattr(host, "confirmation_required", False) and task["phase"] in {"moved", "observed"}:
            choice = {"skill": "capture_image" if task["phase"] == "moved" else "evaluate_goal", "argument": ""}
        else:
            choice = choose_skill(host.grok, task)
        if not choice.get("ok", True):
            task["steps"].append(_result(choice.get("skill") or "report_status", False, choice.get("reason", ""), task,
                                         terminal=True, rejection=choice.get("reason")))
            task["phase"] = "blocked"
            status = "blocked"
            break
        # A semantic choice must pass metric localization before any approach planning.
        if task["phase"] == "selected" and not task.get("location"):
            choice = {"skill": "locate_object", "argument": "", "ok": True}
        name = choice["skill"]
        if host.cancel.is_set():
            status = "stopped"
            break
        if time.monotonic() - started > float(getattr(host, "timeout_s", 90)):
            status = "timeout"
            task["last_reason"] = "the task reached its time limit before executing the skill"
            break
        if name in MOVE_SKILLS and task.get("motion_lock"):
            reason = "a rejected movement cannot be retried by changing the distance or the direction: " + task["motion_lock"]
            task["steps"].append(_result(name, False, reason, task, terminal=True, rejection=reason, executed=False))
            task["phase"] = "blocked"
            status = "blocked"
            break
        phase_before = task["phase"]
        if name == "locate_object":
            task["workflow_state"] = "localization_requested"
            task["workflow_events"].append({"state": "localization_requested", "at": time.time()})
        began = time.perf_counter()
        try:
            result = SKILL_FN[name](host, task, choice.get("argument") or "")
        except (GrokError, ValueError, KeyError, RuntimeError) as exc:
            task['motion_lock'] = str(exc)
            result = _result(name, False, str(exc), task, terminal=True, rejection=str(exc))
            task["phase"] = "blocked"
        elapsed = round(time.perf_counter() - began, 3)
        result["seconds"] = elapsed
        task["timings"].append({"stage": name, "seconds": elapsed})
        note = getattr(host, "note_stage", None)
        if note:
            note(name, elapsed)
        task["last_reason"] = result.get("reason")
        task["steps"].append(result)
        event_state = {"select_object": "object_selected", "plan_approach": "trajectory_proposed"}.get(name)
        if event_state and result.get("ok"):
            task["workflow_events"].append({"state": event_state, "at": time.time()})
            if name == "plan_approach":
                task["workflow_events"].append({"state": "trajectory_safety_validated", "at": time.time()})
        aliases = {"select_object": "target_selection", "report_status": "result_reporting",
                   "capture_image": "camera_acquisition"}
        if name in aliases:
            _note(host, aliases[name], elapsed)
        repeated = (len(task["steps"]) >= 2 and task["steps"][-2].get("skill") == name
                    and task["phase"] == phase_before and not result.get("terminal"))
        if repeated:
            blockers = qualification_blockers(host, task)
            task["last_reason"] = "; ".join(blockers) if blockers else "the same skill repeated without progress"
            task["phase"] = "blocked"
            status = "blocked"
            break
        if result.get("terminal") or choice.get("stop") or task["phase"] in {"reported", "blocked"}:
            if task["phase"] == "clarify":
                status = "clarify"
            elif task.get("motion_lock") and task["phase"] in ("blocked", "held"):
                status = "blocked" if task["phase"] == "blocked" else "held"
            elif not result.get("ok") and result.get("terminal"):
                status = "blocked"
            break
        if step:
            status = "step"
            break
    else:
        status = "step_limit"
    if status == "completed":
        if task["phase"] == "clarify":
            status = "clarify"
        elif task.get("motion_lock") and task["phase"] in ("reported", "held"):
            status = "held"
        elif task["phase"] == "blocked" or task.get("motion_lock"):
            status = "blocked"
    evaluation = task.get("evaluation") or {}
    moved = any(row.get("executed") for row in task["steps"])
    simulated = bool(getattr(getattr(host, "gateway", None), "simulated", True))
    proposal = task.get("proposal") or {}
    if status == "stopped":
        state = "movement_refused_or_failed"
        task["last_reason"] = "STOP or cancellation requested"
    elif task["phase"] == "awaiting_confirmation":
        state = "awaiting_operator_confirmation"
        status = "awaiting_confirmation"
    elif moved:
        state = "movement_completed_verified" if evaluation.get("goal_achieved") and not simulated else "movement_unverified"
    elif task.get("motion_lock"):
        state = ("localization_blocked" if not (task.get("location") or {}).get("ok") else
                 "hardware_authorization_missing" if proposal.get("status") == "dry_run" else
                 "movement_refused_or_failed")
    elif task.get("planned_target"):
        state = "trajectory_safety_validated" if proposal.get("ok") else "trajectory_proposed"
    elif task.get("location"):
        state = "localization_requested"
    else:
        state = "object_selected" if task.get("selected") else "localization_requested"
    task["workflow_state"] = state
    task["workflow_events"].append({"state": state, "at": time.time()})
    return {
        "ok": status in ("completed", "clarify", "held", "step") and not (
            status == "held" and task["phase"] == "blocked") and (
            status != "step" or bool(task["steps"] and task["steps"][-1].get("ok"))),
        "status": status, "goal": task["goal"], "phase": task["phase"],
        "selected": task.get("selected"), "question": task.get("question"),
        "calibration_valid": task.get("calibration_valid"), "calibration": task.get("calibration"),
        "location": task.get("location"), "proposal": task.get("proposal"),
        "evaluation": evaluation, "goal_achieved": bool(evaluation.get("goal_achieved")),
        "motion_lock": task.get("motion_lock"), "steps": task["steps"],
        "authoritative": task.get("authoritative"), "grok_mode": task.get("grok_mode"),
        "hardware_moved": moved and not simulated,
        "simulation_moved": moved and simulated,
        "movement_status_unknown": bool(task.get('movement_status_unknown')),
        "workflow_state": state,
        "workflow_events": task["workflow_events"],
        "execution_mode": getattr(getattr(host, "gateway", None), "mode", None),
        "scan_session": task.get("scan_session"), "scan_plan": task.get("scan_plan"),
        "acquisition": task.get("acquisition"), "reconstruction": (task["steps"][-1].get("reconstruction") if task["steps"] else None),
        "confirmation": {k: v for k, v in (task.get("confirmation") or {}).items()
                         if k in {"id", "expires_at", "target_mm", "object_id"}} or None,
        "motion_proposed": task.get("planned_target") is not None,
        "planned_target": task.get("planned_target"),
        "blockers": qualification_blockers(host, task),
        "verified_tool_position": False,
        "reason": task.get("last_reason") or "",
        "collision_avoidance": False, "software_stop_is_not_safety_rated": True,
        "loop": "step" if step else "auto",
        "max_steps": MAX_STEPS,
        "timings": _timing_summary(task.get("timings") or []),
        "realtime_claimed": False,
        "autonomous_execution_allowed": execution_block(host, task) is None and bool(
            getattr(host, "gateway", None) and host.gateway.mode == "hardware" and not host.gateway.simulated),
        "resume": _resume_task(task),
    }


def public_status(vision: VisionStore, image_size) -> dict:
    return vision.assess(image_size)
