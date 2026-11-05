"""Proposal receipts and truthful operator feedback shared by speech and typed tasks."""

import copy
import hashlib
import json
import math
import re
import time
import uuid

from .geom import pose_changed
from .motion import safety_report

PROPOSAL_TTL_S = 60.0


def fingerprint(host):
    payload = {"vision": {k: getattr(host.vision, k, None) for k in
                          ("version", "intrinsics", "hand_eye", "table")},
               "safety": safety_report(host.gateway, host.observe_clearance_mm),
               "camera_kind": host.camera_kind, "image_size": host.image_size,
               "camera_body": repr(getattr(host.gateway, "camera", None)),
               "keepouts": repr(getattr(host.gateway, "keepouts", None)),
               "orbit_settings": getattr(getattr(host,'orbits',None),'public',lambda:None)(),
               "calibration_provider_binding": getattr(host.vision, 'binding_fingerprint', lambda: None)(),
               "accel": getattr(host.gateway, "max_accel_mm_s2", None)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def receipt(host, task):
    return {"id": uuid.uuid4().hex, "expires_at": time.time() + PROPOSAL_TTL_S,
            "fingerprint": fingerprint(host), "robot": copy.deepcopy(host.robot()),
            "target_mm": list(task["planned_target"]),
            "joints_deg": copy.deepcopy(task.get('planned_joints')),
            "orbit_geometry": copy.deepcopy(task.get('orbit_geometry')),
            "object_id": (task.get("selected") or {}).get("track_id")}


def receipt_error(host, task, proposal_id):
    pending = task.get("confirmation") or {}
    if not proposal_id or pending.get("id") != proposal_id:
        return "No matching validated motion proposal. Request a new plan."
    if time.time() >= pending["expires_at"]:
        return "The motion proposal expired. Request a new plan."
    if getattr(host,'requires_fresh_observation',False):
        taken=task.get('scene_received_at')
        if taken is None or not 0 <= time.time()-float(taken) <= 30:
            return 'The source image expired; request a new observation and proposal.'
    if host.cancel.is_set():
        return "STOP or cancellation invalidated this proposal."
    if not host.vision.assess(host.image_size).get("valid"):
        return "Calibration is no longer valid. Request a new plan."
    if fingerprint(host) != pending["fingerprint"]:
        return "Calibration, camera or safety settings changed. Request a new plan."
    robot = host.robot()
    previous = pending["robot"]
    pose = robot.get('pose6')
    try:
        valid_pose = bool(pose) and len(pose) == 6 and all(math.isfinite(float(value)) for value in pose)
    except (TypeError, ValueError, OverflowError):
        valid_pose = False
    if not valid_pose:
        return 'The controller returned an invalid tool pose. Request a new plan.'
    if (robot.get("moving") or robot.get("error_code") or robot.get("error")
            or robot.get("source") != previous.get("source")
            or not robot.get("pose6")
            or pose_changed(previous.get("pose6"), robot.get("pose6"), mm=1.0, deg=0.1)):
        return "The robot pose or controller state changed. Request a new plan."
    if previous.get('state_code') is not None and robot.get('state_code') != previous['state_code']:
        return 'The controller state changed. Request a new plan.'
    before_joints, now_joints = previous.get('joints'), robot.get('joints')
    try:
        changed_joints = bool(before_joints) and (not now_joints or len(now_joints) != len(before_joints)
                          or any(not math.isfinite(float(b)) or abs(float(a) - float(b)) > .1
                                 for a, b in zip(before_joints, now_joints)))
    except (TypeError, ValueError, OverflowError):
        changed_joints = True
    if changed_joints:
        return 'The robot joints changed. Request a new plan.'
    if (pending["target_mm"] != task.get("planned_target")
            or pending["object_id"] != (task.get("selected") or {}).get("track_id")):
        return "The selected object or target changed. Request a new plan."
    if pending.get('joints_deg') != task.get('planned_joints') or pending.get('orbit_geometry') != task.get('orbit_geometry'):
        return 'The orbit proposal changed. Request a new plan.'
    return None


def refers_to_selection(text):
    if re.search(r"(?i)\b(trova|find|seleziona|select)\b", text):
        return False
    return bool(re.search(r"(?i)(\b(scansionalo|scan it|selected object|oggetto selezionato|continua|continue)\b|"
                          r"\b(posizionati|mettiti|vai|move|hover|position yourself)\s+(sopra|above|over)"
                          r"\s*(it|di esso|di lui|quello)?[.!?\s]*$|\b(above|over) it[.!?\s]*$)", text))


def feedback(result, text=""):
    """Do not turn task completion or a simulated move into physical success."""
    it = bool(re.search(r"(?i)\b(grok|ora|posizionati|seleziona|sopra|scansionalo|procedi|trova)\b", text))
    selected = result.get("selected") or {}
    name = selected.get("name") or ("l'oggetto" if it else "the object")
    state = result.get("workflow_state")
    reason = result.get("reason") or ""
    if result.get("status") in {"stopped", "refused"}:
        message = ("Movimento rifiutato o interrotto. " if it else "Movement refused or stopped. ") + reason
    elif state == "awaiting_operator_confirmation":
        simulation = result.get("execution_mode") == "simulation"
        scope = (" in simulazione" if it else " in simulation") if simulation else ""
        message = (f"La traiettoria per {name} è validata{scope}. Confermi questa proposta?" if it else
                   f"The trajectory for {name} is validated{scope}. Confirm this proposal?")
    elif state == "localization_blocked" and not result.get("calibration_valid"):
        message = (f"Ho selezionato {name}, ma non posso ancora muovermi sopra di esso perché manca la "
                   "calibrazione che permette di determinarne la posizione fisica." if it else
                   f"I selected {name}, but cannot move above it because the calibration needed to determine "
                   "its physical position is missing.")
        if reason:
            message += " " + reason
    elif state == "hardware_authorization_missing":
        message = ("La traiettoria è validata, ma manca l'autorizzazione hardware. Il robot non si è mosso." if it else
                   "The trajectory is validated, but hardware authorization is missing. The robot has not moved.")
    elif state == "movement_completed_verified":
        message = ("Movimento completato e verificato con telemetria e una nuova immagine." if it else
                   "Movement completed and verified with telemetry and a fresh image.")
    elif result.get('movement_status_unknown'):
        message = ("Non posso verificare se il robot si è mosso: la posizione finale non è disponibile. " if it else
                   "I cannot verify whether the robot moved: its final pose is unavailable. ") + reason
    elif result.get("simulation_moved"):
        message = ("Movimento eseguito in simulazione. Il robot fisico non si è mosso." if it else
                   "Movement executed in simulation. The physical robot has not moved.")
        if result.get('motion_lock'):
            message += ' ' + reason
    elif state == "movement_executing":
        message = ("Movimento in simulazione." if it else "Movement executing in simulation.") if result.get("execution_mode") == "simulation" else ("Movimento in esecuzione." if it else "Movement is executing.")
    elif result.get("hardware_moved"):
        message = ("Il movimento è terminato, ma il risultato non è ancora verificato." if it else
                   "The movement finished, but its result is not yet verified.")
        if result.get('motion_lock'):
            message += ' ' + reason
    elif state == "object_selected":
        message = f"Ho selezionato {name}." if it else f"Selected {name}."
    else:
        message = (("Movimento non eseguito. " if it else "Movement not executed. ") + reason)
    custom = result.get('spoken_message') if state in {'reconstruction_queued', 'object_reoriented'} else None
    result["spoken_message"] = result.get("question") or custom or message
    result["blocker"] = reason if state in {"localization_blocked", "movement_refused_or_failed",
                                            "hardware_authorization_missing"} or result.get('motion_lock') else None
    return result
