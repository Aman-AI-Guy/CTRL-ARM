"""Skill loop for an object-directed robot task.

Grok picks a skill name. It does not pick a Cartesian target, a joint angle, a speed, or a
safety setting. Local geometry builds a target only from a valid calibration. MotionGateway
checks it. A rejected move is latched for the rest of the task: a later skill cannot dodge
the rejection by asking for a different distance or direction.

Image centering is reported on its own. It is not a verified 3D position.
"""

from __future__ import annotations

import re
import time

from .finder import find as grok_find
from .geom import VisionStore, pose_changed
from .grok import GrokError, register_mock
from .motion import sample_segment
from .scene import decide_text, from_finder, question_for

MAX_STEPS = 8
APPROACH_SPEED_MM_S = 50.0
REACHED_MM = 15.0
SKILLS = (
    "inspect_scene", "list_objects", "select_object", "locate_object", "observe_object",
    "plan_approach", "move_towards_object", "move_above_object", "capture_image",
    "evaluate_goal", "request_clarification", "report_status",
)
MOVE_SKILLS = {"move_towards_object", "move_above_object"}
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
- inspect_scene, list_objects, select_object, locate_object, observe_object, capture_image, evaluate_goal, request_clarification and report_status do not move the robot.
- plan_approach, move_towards_object and move_above_object use a pose computed by local code. You do not supply that pose.
- If motion lock is set, choose report_status. Do not ask for another direction or another distance.
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
        "observed": "evaluate_goal", "evaluated": "report_status", "held": "report_status",
        "blocked": "report_status", "clarify": "request_clarification", "reported": "report_status",
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
    if gateway is not None and gateway.mode == "hardware" and not task.get("ik_verified"):
        reasons.append("controller inverse kinematics has not confirmed the observation pose")
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


def _pack_proposal(decision, goal, robot, start) -> dict:
    packed = decision.as_dict()
    packed["trajectory_mm"] = []
    proposal = packed.get("proposal") or {}
    begin = proposal.get("start_mm")
    end = proposal.get("target_mm")
    if begin and end:
        packed["trajectory_mm"] = _thin_points(sample_segment(begin, end))
    packed["ik_verified"] = bool(goal.get("ik_verified"))
    packed["ik_joints_deg"] = None if not goal.get("ik") else goal["ik"].get("joints")
    packed["ik_reason"] = None if not goal.get("ik") else goal["ik"].get("reason")
    packed["geometry_source"] = "telemetry" if start is not None else (robot.get("source") or "unavailable")
    packed["collision_avoidance"] = False
    packed["pose6"] = goal.get("pose6")
    packed["straight_line_samples_are_not_a_joint_path"] = not packed["ik_verified"]
    return packed


def skill_inspect(host, task, argument: str) -> dict:
    try:
        snap = host.snapshot()
    except Exception as exc:  # noqa: BLE001 - a dead camera ends the step; the old picture is not reused
        task["phase"] = "blocked"
        return _result("inspect_scene", False, str(exc), task, terminal=True, rejection=str(exc))
    task["objects"] = list(snap.get("objects") or [])
    task["grok_mode"] = snap.get("grok_mode")
    task["authoritative"] = snap.get("grok_mode") == "live"
    task["snapshot_id"] = snap.get("id")
    task["snapshot_pose"] = snap.get("pose")
    task["summary"] = snap.get("summary")
    host.image_size = [snap.get("width") or host.image_size[0], snap.get("height") or host.image_size[1]]
    _calibration_flag(host, task)
    task["phase"] = "inspected"
    return _result("inspect_scene", True, snap.get("summary") or f"{len(task['objects'])} objects", task,
                   grok_mode=snap.get("grok_mode"), n_objects=len(task["objects"]),
                   authoritative=task["authoritative"])


def skill_list(host, task, argument: str) -> dict:
    objects = task.get("objects") or []
    return _result("list_objects", True, f"{len(objects)} objects in the latest snapshot", task,
                   objects=[{k: o.get(k) for k in ("track_id", "name", "color", "confidence", "association",
                                                   "identity_uncertain", "ambiguous")} for o in objects])


def skill_select(host, task, argument: str) -> dict:
    objects = task.get("objects") or []
    if not objects:
        task["phase"] = "blocked"
        return _result("select_object", False, "Take a snapshot before selecting", task, terminal=True)
    text = (argument or task.get("goal") or "").strip()
    decision = decide_text(text, objects)
    if decision.status == "defer":
        if host.frame is None or not host.boxes:
            task["phase"] = "blocked"
            return _result("select_object", False, "the snapshot frame is gone", task, terminal=True)
        found = grok_find(host.grok, host.frame, host.boxes, text, direct_fallback=False)
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
    location = host.vision.localize(robot.get("pose6"), chosen["center"][0], chosen["center"][1],
                                    host.image_size, box=box)
    task["location"] = location
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
    stopped_at = time.time()
    frame, meta = host.read_frame()
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
    goal, err = _target_from_location(host, task)
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
    if getattr(host, "gateway", None) is None:
        task["motion_lock"] = "this arm has no motion gateway"
        task["phase"] = "blocked"
        return _result("plan_approach", False, task["motion_lock"], task, rejection=task["motion_lock"],
                       target=goal, executed=False)
    start = None
    if robot.get("source") == "telemetry" and robot.get("xyz_mm") and len(robot["xyz_mm"]) >= 3:
        start = tuple(float(v) for v in robot["xyz_mm"][:3])
    decision = _authorize(host, task, goal["target_mm"], "plan_approach", start=start)
    packed = _pack_proposal(decision, goal, robot, start)
    task["planned_target"] = list(goal["target_mm"])
    task["proposal"] = packed
    refusals = []
    if host.gateway.camera is None:
        refusals.append("set TROVA_CAMERA_BODY_MM to the measured camera offset and radius; "
                        "the approach is refused because clearance cannot be established")
    if decision.status == "rejected":
        refusals.append(decision.reason)
    if refusals:
        task["motion_lock"] = "; ".join(refusals)
        task["phase"] = "blocked"
        return _result("plan_approach", False, task["motion_lock"], task, rejection=task["motion_lock"],
                       motion=packed, target=goal, executed=False)
    task["phase"] = "planned"
    return _result("plan_approach", True, decision.reason, task, motion=packed, target=goal, executed=False)


def _move(host, task, skill: str) -> dict:
    if task.get("motion_lock"):
        reason = "a rejected movement cannot be retried by changing the distance or the direction: " + task["motion_lock"]
        task["phase"] = "blocked"
        return _result(skill, False, reason, task, terminal=True, rejection=reason, executed=False)
    _calibration_flag(host, task)
    goal, err = _target_from_location(host, task)
    if err:
        task["motion_lock"] = err
        task["phase"] = "blocked"
        return _result(skill, False, err, task, rejection=err, executed=False)
    target = list(task["planned_target"]) if task.get("planned_target") else list(goal["target_mm"])
    # Keep the already planned point. A new number from the caller is not applied.
    robot = task.get("_robot") or _robot(host)
    start = None
    if robot.get("source") == "telemetry" and robot.get("xyz_mm") and len(robot["xyz_mm"]) >= 3:
        start = tuple(float(v) for v in robot["xyz_mm"][:3])
    decision = _authorize(host, task, target, skill, start=start)
    task["proposal"] = decision.as_dict()
    if decision.status in ("rejected", "stopped", "fault"):
        task["motion_lock"] = decision.reason
        task["phase"] = "blocked"
        return _result(skill, False, decision.reason, task, rejection=decision.reason,
                       motion=decision.as_dict(), executed=False)
    blocked = execution_block(host, task)
    if blocked or not decision.allow_motion:
        reason = blocked or decision.reason
        task["motion_lock"] = reason
        task["phase"] = "held"
        return _result(skill, False, reason, task, rejection=reason, motion=decision.as_dict(),
                       executed=False, target={"target_mm": target, "verified_tool_position": False,
                                               "approximate_observation_goal": True})
    outcome = host.execute(target)
    moved = bool(outcome.get("ok"))
    task["phase"] = "moved" if moved else "blocked"
    if not moved:
        task["motion_lock"] = outcome.get("reason") or "the arm did not complete the approach"
    return _result(skill, moved, outcome.get("reason") or ("moved" if moved else "not moved"), task,
                   motion=decision.as_dict(), executed=moved, final_mm=outcome.get("final_mm"),
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
    task["last_capture"] = {k: v for k, v in meta.items() if k != "frame"}
    task["phase"] = "observed"
    if getattr(host, "record_capture", None):
        task["acquisition"] = host.record_capture(frame, meta, task.get("selected"))
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
    achieved = bool(reached and visible and loc_valid and identity_ok)
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
    task["phase"] = "evaluated"
    reason = "goal achieved" if achieved else "goal not achieved"
    if near and not reached:
        reason = "the object is near the image centre; that is not a verified 3D position"
    if err:
        reason = err
    return _result("evaluate_goal", err is None, reason, task, evaluation=checks, goal_achieved=achieved)


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


SKILL_FN = {
    "inspect_scene": skill_inspect, "list_objects": skill_list, "select_object": skill_select,
    "locate_object": skill_locate, "observe_object": skill_observe, "plan_approach": skill_plan,
    "move_towards_object": skill_move_towards, "move_above_object": skill_move_above,
    "capture_image": skill_capture, "evaluate_goal": skill_evaluate,
    "request_clarification": skill_clarify, "report_status": skill_report,
}


def run(host, text: str) -> dict:
    """Bounded observe-plan-act-observe loop. Hardware moves only when every qualification passes."""
    task = {
        "goal": text.strip(), "phase": "start", "selected": None, "objects": [], "location": None,
        "planned_target": None, "proposal": None, "motion_lock": None, "evaluation": None,
        "question": None, "authoritative": False, "calibration_valid": False, "last_reason": "",
        "steps": [],
    }
    _calibration_flag(host, task)
    started = time.monotonic()
    status = "completed"
    for _step in range(MAX_STEPS):
        if host.cancel.is_set():
            status = "stopped"
            break
        if time.monotonic() - started > float(getattr(host, "timeout_s", 90)):
            status = "timeout"
            task["last_reason"] = "the task reached its time limit"
            break
        choice = choose_skill(host.grok, task)
        if not choice.get("ok", True):
            task["steps"].append(_result(choice.get("skill") or "report_status", False, choice.get("reason", ""), task,
                                         terminal=True, rejection=choice.get("reason")))
            task["phase"] = "blocked"
            status = "blocked"
            break
        name = choice["skill"]
        if name in MOVE_SKILLS and task.get("motion_lock"):
            reason = "a rejected movement cannot be retried by changing the distance or the direction: " + task["motion_lock"]
            task["steps"].append(_result(name, False, reason, task, terminal=True, rejection=reason, executed=False))
            task["phase"] = "blocked"
            status = "blocked"
            break
        phase_before = task["phase"]
        try:
            result = SKILL_FN[name](host, task, choice.get("argument") or "")
        except GrokError as exc:
            result = _result(name, False, str(exc), task, terminal=True, rejection=str(exc))
            task["phase"] = "blocked"
        task["last_reason"] = result.get("reason")
        task["steps"].append(result)
        repeated = (len(task["steps"]) >= 2 and task["steps"][-2].get("skill") == name
                    and task["phase"] == phase_before and not result.get("terminal"))
        if repeated:
            blockers = qualification_blockers(host, task)
            task["last_reason"] = "; ".join(blockers) if blockers else "the same skill repeated without progress"
            task["phase"] = "blocked"
            status = "blocked"
            break
        if result.get("terminal") or choice.get("stop") or task["phase"] == "reported":
            if task["phase"] == "clarify":
                status = "clarify"
            elif task.get("motion_lock") and task["phase"] in ("blocked", "held"):
                status = "blocked" if task["phase"] == "blocked" else "held"
            elif not result.get("ok") and result.get("terminal"):
                status = "blocked"
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
    return {
        "ok": status in ("completed", "clarify", "held") and not (status == "held" and task["phase"] == "blocked"),
        "status": status, "goal": task["goal"], "phase": task["phase"],
        "selected": task.get("selected"), "question": task.get("question"),
        "calibration_valid": task.get("calibration_valid"), "calibration": task.get("calibration"),
        "location": task.get("location"), "proposal": task.get("proposal"),
        "evaluation": evaluation, "goal_achieved": bool(evaluation.get("goal_achieved")),
        "motion_lock": task.get("motion_lock"), "steps": task["steps"],
        "authoritative": task.get("authoritative"), "grok_mode": task.get("grok_mode"),
        "hardware_moved": any(step.get("executed") for step in task["steps"]),
        "motion_proposed": task.get("planned_target") is not None,
        "planned_target": task.get("planned_target"),
        "blockers": qualification_blockers(host, task),
        "verified_tool_position": False,
        "reason": task.get("last_reason") or "",
        "collision_avoidance": False, "software_stop_is_not_safety_rated": True,
    }


def public_status(vision: VisionStore, image_size) -> dict:
    return vision.assess(image_size)
