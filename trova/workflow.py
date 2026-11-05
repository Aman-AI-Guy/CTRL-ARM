"""Proposal receipts and truthful operator feedback shared by speech and typed tasks."""

import copy
import hashlib
import json
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
               "accel": getattr(host.gateway, "max_accel_mm_s2", None)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def receipt(host, task):
    return {"id": uuid.uuid4().hex, "expires_at": time.time() + PROPOSAL_TTL_S,
            "fingerprint": fingerprint(host), "robot": copy.deepcopy(host.robot()),
            "target_mm": list(task["planned_target"]),
            "object_id": (task.get("selected") or {}).get("track_id")}


def receipt_error(host, task, proposal_id):
    pending = task.get("confirmation") or {}
    if not proposal_id or pending.get("id") != proposal_id:
        return "No matching validated motion proposal. Request a new plan."
    if time.time() >= pending["expires_at"]:
        return "The motion proposal expired. Request a new plan."
    if host.cancel.is_set():
        return "STOP or cancellation invalidated this proposal."
    if not host.vision.assess(host.image_size).get("valid"):
        return "Calibration is no longer valid. Request a new plan."
    if fingerprint(host) != pending["fingerprint"]:
        return "Calibration, camera or safety settings changed. Request a new plan."
    robot = host.robot()
    previous = pending["robot"]
    if (robot.get("moving") or robot.get("error_code") or robot.get("error")
            or robot.get("source") != previous.get("source")
            or not robot.get("pose6")
            or pose_changed(previous.get("pose6"), robot.get("pose6"), mm=1.0, deg=0.1)):
        return "The robot pose or controller state changed. Request a new plan."
    if (pending["target_mm"] != task.get("planned_target")
            or pending["object_id"] != (task.get("selected") or {}).get("track_id")):
        return "The selected object or target changed. Request a new plan."
    return None


def refers_to_selection(text):
    return bool(re.search(r"(?i)(\b(scansionalo|scan it|selected object|oggetto selezionato)\b|"
                          r"\b(posizionati|mettiti|move|hover|position yourself)\s+(sopra|above|over)"
                          r"\s*(it|di esso)?[.!?\s]*$)", text))


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
    elif state == "hardware_authorization_missing":
        message = ("La traiettoria è validata, ma manca l'autorizzazione hardware. Il robot non si è mosso." if it else
                   "The trajectory is validated, but hardware authorization is missing. The robot has not moved.")
    elif state == "movement_completed_verified":
        message = ("Movimento completato e verificato con telemetria e una nuova immagine." if it else
                   "Movement completed and verified with telemetry and a fresh image.")
    elif result.get("simulation_moved"):
        message = ("Movimento eseguito in simulazione. Il robot fisico non si è mosso." if it else
                   "Movement executed in simulation. The physical robot has not moved.")
    elif state == "movement_executing":
        message = ("Movimento in simulazione." if it else "Movement executing in simulation.") if result.get("execution_mode") == "simulation" else ("Movimento in esecuzione." if it else "Movement is executing.")
    elif result.get("hardware_moved"):
        message = ("Il movimento è terminato, ma il risultato non è ancora verificato." if it else
                   "The movement finished, but its result is not yet verified.")
    elif state == "object_selected":
        message = f"Ho selezionato {name}." if it else f"Selected {name}."
    else:
        message = (("Movimento non eseguito. " if it else "Movement not executed. ") + reason)
    result["spoken_message"] = result.get("question") or message
    result["blocker"] = reason if state in {"localization_blocked", "movement_refused_or_failed",
                                            "hardware_authorization_missing"} else None
    return result
