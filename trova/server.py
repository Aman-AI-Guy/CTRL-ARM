"""Trova web app: FastAPI backend + a one-page UI (trova/static/index.html).

Run:  python -m trova      then open http://localhost:8000 (phones: http://<laptop-ip>:8000)
"""

from __future__ import annotations

import hmac
import html
import json
import math
import os
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat

from . import changes as chg
from .arm import CartesianArm, FakeCartesian, FakeJointBackend, SimArm, TaughtGridArm, load_poses, make_arm, save_pose
from .arm.skills import TrayParams, lift_allowed, waypoints as tray_waypoints
from .audit import AuditLog
from .boxes import Box
from .calibration import Calibration, calibrate_aruco, load_marker_positions, make_marker_sheet
from .camera import DemoSource, ImageSource, open_source
from .config import Settings, atomic_json_write, load_settings
from .dog import DogSearch, load_route, make_dog, run_search
from . import voice as voice_cfg
from .finder import find as grok_find, find_all as grok_find_all
from .inventory import describe as describe_table, draw_map
from . import failures as fail_log
from .record_pdf import build as build_record_pdf
from . import history, preflight
from .watch import Watch
from . import room as room_mod
from .light import make_light
from .safety import LiftStore
from . import labels as lbl
from .zones import RULES, Zone, draw_zones, evaluate as eval_zones, load_zones, save_zones
from .grok import GrokClient, GrokError
from .marks import HIGHLIGHT, MARK, draw_marks, jpeg
from .proposals import drop_occluders, find_blobs, is_occluded

STATIC = Path(__file__).parent / "static"
SLOW_S = float(os.getenv("TROVA_SLOW_S", "8"))
FIRST_RUN_MAX_SPEED = 20.0      # mm/s: the first supervised run never goes faster; env may only lower it
FIRST_RUN_MAX_JOINT_SPEED = 5.0  # deg/s for joint moves in that mode (stage 0, homing)
CONFIRM_TTL_S = 60.0            # a first-run confirmation older than this is stale: find again
COLLISION_WARNING = "Collision avoidance: NOT PRESENT. Hold the physical E-stop."
# Captured before route locks/model calls; remains cancelled even after an operator reset.
MOTION_TICKET: ContextVar[threading.Event | None] = ContextVar("motion_ticket", default=None)


class ArmBusy(RuntimeError):
    """The arm is doing something that must not be interrupted by a new command (HTTP 409)."""


class CameraError(RuntimeError):
    """No usable picture right now (camera unplugged or frozen): HTTP 503, the UI keeps polling."""


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def read_upload(photo: UploadFile) -> bytes:
    data = photo.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo exceeds the 10 MiB limit")
    return data


def decode_upload(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(400, "The photo is empty")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        raise HTTPException(400, "Could not read the image")
    h, w = img.shape[:2]
    if max(h, w) > 2000:  # phone photos are huge; Grok gets <= 1600 anyway
        s = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# Request bodies (module level: FastAPI must resolve them under postponed annotations)
class Tune(BaseModel):
    thresh: FiniteFloat | None = None
    min_area_frac: FiniteFloat | None = None


class UV(BaseModel):
    u: FiniteFloat
    v: FiniteFloat


class ClickPoint(BaseModel):
    u: FiniteFloat
    v: FiniteFloat
    x_mm: FiniteFloat
    y_mm: FiniteFloat


class Point(BaseModel):
    x_mm: FiniteFloat | None = None
    y_mm: FiniteFloat | None = None
    u: FiniteFloat | None = None
    v: FiniteFloat | None = None


class Judge(BaseModel):
    kind: str = "find"
    correct: bool = True
    cause: str = ""          # e.g. "arm_missed": Grok was right but the arm pointed at the wrong spot
    false_alarms: int = Field(default=0, ge=0)
    missed: int = Field(default=0, ge=0)
    note: str = ""


class Label(BaseModel):
    u: FiniteFloat | None = None
    v: FiniteFloat | None = None
    absent: bool = False


class LiftReq(BaseModel):
    place_u: FiniteFloat
    place_v: FiniteFloat
    pick_u: FiniteFloat | None = None     # default: the object of the last find
    pick_v: FiniteFloat | None = None


class WatchReq(BaseModel):
    on: bool = True
    point: bool = False       # the arm points at the problem when the alarm goes off


class VoiceLog(BaseModel):
    who: str
    text: str


class Approach(BaseModel):
    """First supervised run: confirm exactly the shown stage and target."""
    stage: int
    confirm: str
    x_mm: FiniteFloat
    y_mm: FiniteFloat
    speed_mm_s: FiniteFloat | None = None


class StopReset(BaseModel):
    operator: str
    stop_nonce: str
    area_clear: bool = False
    controller_ready: bool = False


class LiftRecovery(StopReset):
    fork_freed: bool = False
    note: str = ""


class State:
    def __init__(self, s: Settings):
        self.s = s
        self.audit = AuditLog(s.data_dir)
        self.configuration_errors: list[str] = []
        self.grok = GrokClient(s.xai_api_key, s.model, s.effort, s.mock,
                               on_call=lambda e: self.audit.event("grok_call", **e),
                               timeout=float(os.getenv("TROVA_GROK_TIMEOUT_S", "60")),
                               offline_fallback=os.getenv("TROVA_OFFLINE_FALLBACK", "1").lower() not in ("0", "false", "no"))
        self.camera_spec, self.view_revision = s.camera, 0
        self.source = open_source(s.camera)
        self.override: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self.background_path: str | None = None   # saved copy, so eval cases can be replayed
        self.calib_path = s.data_dir / "calibration.json"
        self.calib = Calibration.load(self.calib_path)
        self.click_points: list[list[float]] = []     # manual calibration: [u, v, table x_mm, table y_mm]
        self.arm_points: list[list[float]] = []       # arm calibration: [u, v, robot x_mm, robot y_mm]
        self.frame_size: list[int] | None = None      # live camera frame size (not an uploaded photo)
        self.camera_error: str | None = None
        self.arm = make_arm(s.arm, s.data_dir, s.table_w_mm, s.table_h_mm)
        # Simulator exemptions require BOTH an explicit simulator spec and a known simulator implementation.
        self.physical = not (
            (s.arm == "sim" and type(self.arm) is SimArm)
            or (s.arm == "cartesian:fake" and isinstance(self.arm, CartesianArm)
                and type(self.arm.backend) is FakeCartesian)
            or (s.arm == "grid:fake" and isinstance(self.arm, TaughtGridArm)
                and type(self.arm.backend) is FakeJointBackend))
        self.lift_store = LiftStore(s.data_dir) if self.physical else None
        self.persistence_error: str | None = None
        self.operator_name = os.getenv("TROVA_OPERATOR_NAME", "").strip()
        self.operator_token = os.getenv("TROVA_OPERATOR_TOKEN", "").strip()
        # the camera is on the arm (the event's xArm 5): every picture is taken from the taught survey pose
        self.camera_on_arm = os.getenv("TROVA_CAMERA_ON_ARM", "0").lower() in ("1", "true", "yes")
        self.settle_s = float(os.getenv("TROVA_CAMERA_SETTLE_S", "0.6"))
        if hasattr(self.source, "attach"):
            self.source.attach(self.arm, s.data_dir)   # simulated arm camera: renders what the arm would see
        self.arm_thread: threading.Thread | None = None
        self.arm_lock = threading.Lock()        # one motion command at a time
        self.safety_lock = threading.Lock()     # local state only: never held over I/O, joins or model calls
        self.stop_latched = threading.Event()   # independent of command cancellation; only reset clears it
        self.stop_epoch = threading.Event()     # invalidates ALL requests admitted before/during STOP
        self.stops_pending = 0                 # reset cannot overtake a driver.stop() still in flight
        self.stop_sequence = 0
        self.stop_nonce = uuid.uuid4().hex      # fresh across boots AND every STOP; never a reusable reset permission
        self.stop_error: str | None = None
        self.manual_mode = False
        self.cancel = threading.Event()         # set by STOP; a fresh one per command
        self.stop_ts = 0.0                      # wall time of the last STOP
        self.arm_job = ""                       # point | tour | lift | home
        self.lift_interrupted = bool(self.lift_store and self.lift_store.read() != "clear")
        if self.physical:
            # Every boot is a new disarmed session, even if the last lift record is clean.
            self.stop_latched.set()
            self.stop_epoch.set()
            self.cancel.set()
            self.arm.latch_stop()              # local event only: controller recovery is NEVER automatic
        self.thresh, self.min_area = s.blob_thresh, s.min_area_frac
        self.last_find: dict | None = None
        self.pending: dict | None = None
        self.reference: dict | None = None
        self.last_check: dict | None = None
        self.label_wait: dict | None = None
        self.pending_move: dict | None = None
        self.inventory: dict | None = None
        self.tour: dict | None = None
        self.zones_path = s.data_dir / "zones.json"
        try:
            self.zones: list[Zone] = load_zones(self.zones_path)
            if any(not isinstance(z.name, str) or not z.name.strip() or z.rule not in RULES
                   or len(z.polygon) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in z.polygon)
                   or not isinstance(z.allowed, list) or not all(isinstance(a, str) for a in z.allowed) for z in self.zones):
                raise ValueError("invalid zone configuration")
        except (ValueError, OSError, TypeError, KeyError, IndexError) as e:
            self.zones = []
            self.configuration_errors.append(f"zones.json: {e}")
            self.audit.event("error", where="zones_load", error=str(e))
        self.zone_draft: list[list[float]] = []
        self.metrics = {"find_ok": 0, "find_wrong": 0, "find_times": [], "checks": 0,
                        "check_ok": 0, "false_alarms": 0, "missed": 0}
        self.lock = threading.RLock()
        self.cases_dir = s.data_dir / "cases"
        self.cases_dir.mkdir(exist_ok=True)
        self.dog = make_dog(s.dog)
        self.dog_route = load_route(s.data_dir)
        self.dog_search: DogSearch | None = None
        self.dog_stop = threading.Event()
        self.dog_thread: threading.Thread | None = None
        self.watch = Watch(self)
        try:
            self.light = make_light()         # optional LED ring on the arm camera
        except Exception as e:  # noqa: BLE001 - a missing light must never stop Trova
            from .light import NoLight
            self.light = NoLight()
            self.audit.event("error", where="light", error=str(e))
        self.batch_path = s.data_dir / "batch.json"          # feature 11: the batch being cleared for
        self.batch = {"lot": os.getenv("TROVA_BATCH", "")}
        try:
            if self.batch_path.exists():
                batch = json.loads(self.batch_path.read_text())
                if not isinstance(batch, dict) or not isinstance(batch.get("lot"), str) or not isinstance(batch.get("product", ""), str):
                    raise ValueError("invalid batch configuration")
                self.batch = batch
        except (ValueError, OSError, TypeError) as e:
            self.configuration_errors.append(f"batch.json: {e}")
            self.audit.event("error", where="batch_load", error=str(e))
        self.labels: dict | None = None
        self.room: dict | None = None          # feature 14: last room scan (status, views, result)
        # first supervised run on hardware: find -> confirm -> hover -> confirm -> descend -> measure (off by default)
        self.first_run = os.getenv("TROVA_FIRST_RUN", "0").lower() in ("1", "true", "yes")
        self.first_run_speed = min(FIRST_RUN_MAX_SPEED, float(os.getenv("TROVA_FIRST_RUN_SPEED_MM_S", "20")))
        self.first_run_hover = float(os.getenv("TROVA_FIRST_RUN_HOVER_MM", "100"))
        self.miss_mm = float(os.getenv("TROVA_ARM_MISS_MM", "10"))
        if self.first_run and isinstance(self.arm, CartesianArm):
            self.arm.speed = min(self.arm.speed, self.first_run_speed)   # every Cartesian move, home included
            self.arm.joint_speed = min(self.arm.joint_speed, FIRST_RUN_MAX_JOINT_SPEED)

    # ---------- frames, proposals, calibration
    def frame(self) -> np.ndarray:
        try:
            f = self.override.copy() if self.override is not None else self.source.read()
            self.camera_error = None
        except Exception as e:  # noqa: BLE001 - any source failure: report it, never run on a stale picture
            self.camera_error = str(e)
            raise CameraError(str(e)) from e
        h, w = f.shape[:2]
        if self.override is None:
            self.frame_size = [w, h]
        if self.calib is None or (self.calib.source == "default" and self.calib.points[2][:2] != [float(w), float(h)]):
            self.calib = Calibration.default(w, h, self.s.table_w_mm, self.s.table_h_mm)
        return f

    def proposals(self, frame: np.ndarray) -> tuple[list[Box], bool]:
        raw = find_blobs(frame, self.background, self.thresh, self.min_area)
        return drop_occluders(raw, frame.shape), is_occluded(raw, frame.shape)

    def invalidate_targets(self) -> None:
        self.view_revision += 1
        self.pending_move, self.pending = None, None

    def view_stamp(self) -> tuple:
        c = self.calib
        return (self.view_revision, self.camera_spec, id(self.source), id(self.override),
                tuple(self.frame_size or ()), json.dumps(c.as_dict(), sort_keys=True) if c else None,
                c.H.tobytes() if c and c.ok else None)

    def target_valid(self, target: dict) -> bool:
        issued = (target.get("approach") or {}).get("issued", target.get("_issued", 0)) if self.first_run else target.get("_issued", 0)
        return target.get("_view") == self.view_stamp() and 0 <= time.time() - issued <= CONFIRM_TTL_S

    def require_target(self, target: dict) -> None:
        if not self.target_valid(target):
            raise ArmBusy("Stale target refused: camera, calibration or source changed, or older than 60 s; find again")

    def arm_blocked(self, pixels: bool = True, *, interlocks: bool = True) -> str | None:
        """Why the arm must not move right now (None = it may). pixels=False: the target is given in robot mm,
        so only the safety rules apply, not the camera calibration."""
        if interlocks:
            blocked = self.arm_interlock(pixels=pixels)
            if blocked:
                return blocked
        return self.calibration_blocked(pixels)

    def arm_interlock(self, pixels: bool = False) -> str | None:
        if self.stop_latched.is_set():
            return "Software STOP is latched: operator safety checks and /api/arm/reset required" + (
                "; a tray lift was interrupted: recover the fork separately" if self.lift_interrupted else "")
        if self.manual_mode:
            return "The arm is in manual teaching mode: leave Freedrive / compliant mode before automatic motion"
        if self.arm_job == "lift" and self.arm_thread and self.arm_thread.is_alive():
            return "The arm is lifting a tray: wait for it to finish, or press STOP"
        if self.lift_interrupted:
            return ("A tray lift was interrupted, the fork may still be in the slot: free the arm by hand "
                    "with staff, then confirm /api/arm/recover_lift while STOP remains latched")
        if self.persistence_error:
            return "Safety persistence failed: " + self.persistence_error
        if (pixels and self.camera_on_arm and isinstance(self.arm, CartesianArm)
                and (self.arm.home_joints or self.arm.home_mm) and not self.at_survey()):
            return "The arm camera is away from the survey pose: Arm home"
        return None

    def calibration_blocked(self, pixels: bool = True) -> str | None:
        if isinstance(self.arm, (CartesianArm, TaughtGridArm)):
            if not self.arm.limits:
                return "No safety box: set TROVA_ARM_LIMITS=xmin,xmax,ymin,ymax,zmin,zmax (robot mm) with staff, restart"
            if not ((self.arm.home_mm or self.arm.home_joints) if isinstance(self.arm, CartesianArm) else self.arm.home_q):
                return ("No home pose: teach one (Setup: Teach survey pose here) or set TROVA_ARM_HOME_MM=x,y,z, restart"
                        if self.camera_on_arm else "No home pose: set TROVA_ARM_HOME_MM=x,y,z outside the camera view, restart")
            if not pixels:
                return None
            if not self.calib or not self.calib.ok or self.calib.source not in ("arm", "aruco_robot"):
                return "Calibrate with the arm first (Setup: Calibrate with arm, or ArUco with markers recorded by the arm)"
            if self.camera_on_arm and isinstance(self.arm, CartesianArm):
                survey = self.arm.home_joints or self.arm.home_mm
                if not (survey and self.calib.pose and len(survey) == len(self.calib.pose)
                        and max(abs(a-b) for a, b in zip(survey, self.calib.pose)) <= 0.01):
                    return "Calibration belongs to a different survey pose: calibrate again"
            if self.physical and self.calib.camera != self.camera_spec:
                return "Calibration belongs to a different camera source: calibrate again"
            if self.source.info().get("kind") in ("images", "video"):
                return "A recorded image/video is shown: the arm only follows the live camera"
            if self.override is not None:
                return "An uploaded photo is shown: the arm only follows the live camera (Back to live)"
            if self.calib.size and self.frame_size and list(self.calib.size) != list(self.frame_size):
                return (f"Calibrated at {self.calib.size[0]}x{self.calib.size[1]} but the camera now gives "
                        f"{self.frame_size[0]}x{self.frame_size[1]}: calibrate again")
        return None

    def first_run_problems(self) -> list[str]:
        """First supervised run: every item must hold or the mode refuses to move (also listed by preflight)."""
        if not isinstance(self.arm, CartesianArm):
            return ["Needs a Cartesian arm: TROVA_ARM=xarm:<ip> (cartesian:fake to rehearse)"]
        p, c = [], self.calib
        survey = self.arm.home_joints or self.arm.home_mm
        if not c or c.source not in ("arm", "aruco_robot"):
            p.append("Calibration: not made with the arm (touch the ArUco markers with the tip, then Calibrate (ArUco))")
        else:
            if not c.size or not self.frame_size or list(c.size) != list(self.frame_size):
                p.append(f"Calibration: made at image size {c.size}, the camera now gives {self.frame_size}: calibrate again")
            if self.camera_on_arm and not (survey and c.pose and len(c.pose) == len(survey)
                                           and max(abs(a - b) for a, b in zip(c.pose, survey)) <= 0.01):
                p.append("Calibration: made at a different survey pose (or none recorded): Calibrate (ArUco) again")
        if not self.arm.limits:
            p.append("Safety box: TROVA_ARM_LIMITS not set")
        if not survey:
            p.append("Home pose: not set")
        if not (self.operator_name and self.operator_token):
            p.append("Operator credentials: TROVA_OPERATOR_NAME / TROVA_OPERATOR_TOKEN not configured")
        if self.stop_latched.is_set():
            p.append("STOP is latched: operator reset required")
        if self.lift_interrupted:
            p.append("Lift recovery required: staff must free the fork and confirm /api/arm/recover_lift while STOP stays latched")
        if self.persistence_error:
            p.append("Safety persistence failed: " + self.persistence_error)
        if not 0 < self.first_run_speed <= FIRST_RUN_MAX_SPEED or self.first_run_hover <= 0:
            p.append("TROVA_FIRST_RUN_SPEED_MM_S must be in (0, 20] and TROVA_FIRST_RUN_HOVER_MM above 0")
        return p

    def approach_view(self) -> dict | None:
        """What the operator sees before confirming: target, height, distance from the current pose, problems."""
        ap = (self.pending_move or {}).get("approach")
        if not ap:
            return None
        out = {k: v for k, v in ap.items() if k not in ("issued", "seq")}
        try:
            tcp = self.arm.tcp_mm()
            out["distance_mm"] = round(sum((a - b) ** 2 for a, b in zip(tcp, ap["to_mm"])) ** 0.5, 1)
        except Exception:  # noqa: BLE001
            out["distance_mm"] = None
        if ap["stage"] in (0, 1, 2):
            box = self.arm.limits.check(*ap["to_mm"]) if self.arm.limits else None
            out["problems"] = self.first_run_problems() + ([f"Safety box: {box}"] if box else [])
        return out | {"marked": self.pending_move.get("marked"), "warning": COLLISION_WARNING}

    def new_approach(self, stage, target_mm: list[float], **extra) -> dict:
        """A single-use confirmation for exactly this stage and target, dead after STOP or CONFIRM_TTL_S.
        to_mm: where the tip goes in this stage (stage 0: levelled pose by forward kinematics)."""
        z = round(self.arm.z + self.first_run_hover if stage in (0, 1) else self.arm.z, 1)
        out = {"stage": stage, "confirm": uuid.uuid4().hex if stage in (0, 1, 2) else None, "target_mm": target_mm,
               "z_mm": z, "to_mm": [*target_mm, z], "speed_mm_s": self.first_run_speed, "issued": time.time(),
               "seq": self.stop_sequence}
        return out | extra

    def first_approach(self, target_mm: list[float]) -> dict:
        """Stage 0 only when the xArm 5 flange is tilted (camera survey pose): it moves in straight lines only with
        the flange level (manual p. 156), so level it in place first, then stage 1 is a straight line."""
        b = self.arm.backend
        if b.is_level():
            return self.new_approach(1, target_mm)
        q = b.joints()
        ql = b.level_joints(q)
        return self.new_approach(0, target_mm, q_from=[round(v, 2) for v in q], q_level=ql,
                                 to_mm=[round(v, 1) for v in b.fk_mm(ql)])

    def at_joints(self, q: list[float], tol_deg: float = 1.0) -> bool:
        try:
            return max(abs(a - b) for a, b in zip(self.arm.backend.joints(), q)) <= tol_deg
        except Exception:  # noqa: BLE001
            return False

    def at_hover(self, ap: dict, tol_mm: float = 5.0) -> bool:
        try:
            tcp = self.arm.tcp_mm()
        except Exception:  # noqa: BLE001
            return False
        return sum((a - b) ** 2 for a, b in zip(tcp, [*ap["target_mm"], ap["z_mm"] + self.first_run_hover])) ** 0.5 <= tol_mm

    def run_approach(self, pm: dict, ap: dict, speed: float) -> None:
        """Worker for one first-run stage: move, record target vs the pose the controller reports, then hand out
        the next confirmation (stage 2) or ask for the measured error."""
        x, y = ap["target_mm"]
        # start poses are rechecked here, right before moving: nothing else moves while this worker runs
        if ap["stage"] == 2 and not self.at_hover(ap):
            self.arm.state, self.arm.last_error = "refused", "not at this target's hover point: stage 2 refused"
            ok = False
        elif ap["stage"] == 1 and ap.get("q_level") and not self.at_joints(ap["q_level"]):
            self.arm.state, self.arm.last_error = "refused", "not at the levelled pose: stage 1 refused"
            ok = False
        elif ap["stage"] == 0:
            ok = self.arm.go_joints(ap["q_level"])         # safety box via FK, STOP latch, reached check
        else:
            ok = self.arm.approach(x, y, ap["z_mm"], speed)
        try:
            tcp = [round(v, 1) for v in self.arm.tcp_mm()]
        except Exception:  # noqa: BLE001
            tcp = None
        off = round(sum((a - b) ** 2 for a, b in zip(tcp, ap["to_mm"])) ** 0.5, 1) if tcp else None
        self.audit.event("arm_approach", stage=ap["stage"], target_mm=ap["to_mm"], tcp_mm=tcp, offset_mm=off,
                         reached=ok, speed_mm_s=speed, error=getattr(self.arm, "last_error", None), find=pm.get("event_id"))
        with self.safety_lock:
            if self.pending_move is not pm:
                return                             # a newer find replaced this target
            if not ok or self.stop_latched.is_set():
                pm["approach"] = None              # STOP / refusal / miss: start again from a new find
            elif ap["stage"] == 0:
                pm["approach"] = self.new_approach(1, ap["target_mm"], q_level=ap["q_level"])
            elif ap["stage"] == 1:
                pm["approach"] = self.new_approach(2, ap["target_mm"])
            else:
                pm["approach"] = self.new_approach("measure", ap["target_mm"], tcp_mm=tcp, offset_mm=off)

    def motion_ticket(self, ticket=None) -> threading.Event:
        request_ticket = MOTION_TICKET.get()
        # A stored result/alarm may add restrictions, never override a cancelled HTTP request's authority.
        if request_ticket is not None and request_ticket.is_set():
            return request_ticket
        return ticket if ticket is not None else (request_ticket or self.stop_epoch)

    def authorize_motion(self, *, pixels: bool = False, ticket=None, since=None) -> None:
        """The single admission check, used for pointing AND all preparatory/home/room moves."""
        if (ticket is not None and ticket.is_set()) or (since is not None and self.stop_ts >= since):
            raise ArmBusy("STOP cancelled this pending command: submit a new command after operator reset")
        blocked = self.arm_blocked(pixels=pixels)
        if blocked:
            raise ArmBusy(blocked)

    # ---------- arm motion: one command at a time, STOP always wins
    def stop_arm(self) -> None:
        """No arm/model lock: publish cancellation before calling the driver's immediate stop path."""
        with self.safety_lock:
            self.stop_ts = time.time()
            self.stop_sequence += 1
            self.stop_nonce = uuid.uuid4().hex
            self.stop_latched.set()
            self.stop_epoch.set()
            self.cancel.set()
            self.stops_pending += 1
        try:
            self.arm.stop()
        except Exception as e:
            with self.safety_lock:
                self.stop_error = str(e) or type(e).__name__
            raise
        else:
            with self.safety_lock:
                self.stop_error = None
        finally:
            with self.safety_lock:
                self.stops_pending -= 1
            self.light.show("stop")

    def _take_arm(self, job: str, *, pixels: bool, ticket, since) -> threading.Event:
        """Caller holds arm_lock. Never clear a driver's STOP as a side effect of a command."""
        self.authorize_motion(pixels=pixels, ticket=ticket, since=since)
        if self.arm_thread and self.arm_thread.is_alive():
            if self.arm_job == "lift":
                raise ArmBusy("The arm is lifting a tray: wait for it to finish, or press STOP")
            self.arm_thread.join(timeout=3)
            if self.arm_thread.is_alive():
                raise ArmBusy("The arm is still finishing the last move: wait a moment and retry")
        self.authorize_motion(pixels=pixels, ticket=ticket, since=since)
        with self.safety_lock:
            # STOP can run during the checks/join. Publishing the command is atomic with STOP.
            if self.stop_latched.is_set() or ticket.is_set():
                raise ArmBusy("Software STOP cancelled this command: operator reset required")
            self.cancel = threading.Event()
            self.arm_job = job
            return self.cancel

    def start_arm(self, fn, *args, since: float | None = None, job: str = "point",
                  pixels: bool = True, ticket=None, wait: bool = False, guard=None) -> bool:
        """Admit one guarded motion; blocked/cancelled commands raise ArmBusy (HTTP 409)."""
        def finite(value):
            if isinstance(value, (int, float)):
                return math.isfinite(value)
            if isinstance(value, (tuple, list)):
                return all(finite(v) for v in value)
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            return True
        if not finite(args):
            raise ArmBusy("Motion coordinates must be finite")
        if self.first_run and job not in ("home", "approach"):
            raise ArmBusy("First supervised run mode: only the two-stage approach and Arm home may move the arm")
        ticket = self.motion_ticket(ticket)
        with self.arm_lock:
            if guard is not None and not guard():
                raise ArmBusy("The watch worker/generation is no longer authorized")
            cancel = self._take_arm(job, pixels=pixels, ticket=ticket, since=since)
            if guard is not None and not guard():
                raise ArmBusy("The watch worker/generation is no longer authorized")

            def run():
                if (not cancel.is_set() and not ticket.is_set() and not self.stop_latched.is_set()
                        and (guard is None or guard())):
                    fn(*args)

            worker = threading.Thread(target=run, daemon=True)
            self.arm_thread = worker
            worker.start()
        if wait:
            worker.join()                  # STOP never needs arm_lock; reset requires this worker to finish
            self.authorize_motion(pixels=pixels, ticket=ticket, since=since)
        return True

    def authenticate_operator(self, supplied: str, claimed: str = "") -> str:
        if not self.physical:
            return claimed.strip()
        if not self.operator_name or not self.operator_token:
            raise HTTPException(403, "Hardware recovery requires configured TROVA_OPERATOR_NAME and TROVA_OPERATOR_TOKEN")
        if not hmac.compare_digest(supplied.encode(), self.operator_token.encode()):
            raise HTTPException(401, "Operator authentication required (x-trova-operator-token)")
        return self.operator_name

    def operator_reset(self, r: StopReset, *, recover_lift: bool = False, credential: str = "") -> str:
        """Software reset only. Controller/e-stop recovery must already have been performed by staff."""
        identity = self.authenticate_operator(credential, r.operator)
        if not identity or not r.area_clear or not r.controller_ready:
            raise ArmBusy("An operator must confirm area_clear and controller_ready; reset performs no controller recovery")
        stop_sequence = self.stop_sequence
        if not self.arm_lock.acquire(blocking=False):
            raise ArmBusy("A motion is still being admitted: wait for it to finish")
        try:
            with self.safety_lock:
                if r.stop_nonce != self.stop_nonce:
                    raise ArmBusy("Stale STOP confirmation: read /api/state.stop_nonce and confirm the current STOP")
            if self.arm_thread and self.arm_thread.is_alive():
                raise ArmBusy("The cancelled motion/scan must finish before reset or lift recovery")
            if self.manual_mode:
                raise ArmBusy("Leave manual teaching mode before reset or lift recovery")
            if isinstance(self.arm, CartesianArm):
                try:
                    moving = self.arm.backend.is_moving()
                except Exception as e:
                    raise ArmBusy("Cannot verify controller idle: " + str(e)) from e
                if moving:
                    raise ArmBusy("The controller still reports motion: reset refused")
            blocked = self.arm_blocked(pixels=False, interlocks=False)
            if blocked:
                raise ArmBusy(blocked)
            if self.arm.state in ("error", "moving", "refused") or getattr(self.arm, "last_error", None):
                raise ArmBusy("Resolve the controller/driver error separately before software reset")
            if recover_lift:
                if not self.lift_interrupted or not r.fork_freed or not r.note.strip():
                    raise ArmBusy("Confirm fork_freed and document staff's manual lift recovery decision")
                with self.safety_lock:
                    if (self.stops_pending or not self.stop_latched.is_set() or self.stop_sequence != stop_sequence
                            or r.stop_nonce != self.stop_nonce):
                        raise ArmBusy("STOP must remain latched throughout lift recovery")
                if self.lift_store:
                    try:
                        self.lift_store.write("clear")
                    except OSError as e:
                        self.persistence_error = str(e)
                        raise ArmBusy("Cannot persist lift recovery: " + str(e)) from e
            with self.safety_lock:
                if (self.stops_pending or not self.stop_latched.is_set() or self.stop_sequence != stop_sequence
                        or r.stop_nonce != self.stop_nonce):
                    raise ArmBusy("STOP must be latched and its driver stop completed before reset or recovery")
                if self.stop_error:
                    raise ArmBusy("Driver STOP failed: resolve controller communication and retry STOP before reset: "
                                  + self.stop_error)
                if recover_lift:
                    self.lift_interrupted = False    # explicit recovery only; STOP remains latched
                    self.persistence_error = None
                    self.stop_nonce = uuid.uuid4().hex  # recovery confirmation cannot also be replayed as rearming
                else:
                    if self.persistence_error:
                        raise ArmBusy("Resolve safety persistence failure and confirm lift recovery before reset")
                    self.arm.ready()                # event clear only; never a controller recovery or move
                    self.stop_epoch = threading.Event()
                    self.stop_latched.clear()
                    self.arm.state = "idle"
                    self.stop_nonce = uuid.uuid4().hex
            return identity
        finally:
            if recover_lift and self.lift_interrupted and self.lift_store:
                # A concurrent STOP invalidated the recovery decision during disk I/O. Restore dirty evidence.
                try:
                    self.lift_store.write("in_progress")
                except OSError as e:
                    self.persistence_error = str(e)
            self.arm_lock.release()

    def set_manual_mode(self, on: bool, change_mode) -> None:
        """Manual recovery/teaching is explicit, serialized with motion admission, and never resets STOP."""
        with self.arm_lock:
            if self.arm_thread and self.arm_thread.is_alive():
                raise ArmBusy("Wait for the motion thread to finish before changing manual mode")
            if on:
                self.manual_mode = True     # fail closed if the controller mode change fails
            change_mode(on)
            self.manual_mode = on

    def at_survey(self) -> bool:
        """Camera on the arm: is the arm at the survey pose, i.e. does the picture match the calibration?"""
        if not self.camera_on_arm or not isinstance(self.arm, CartesianArm):
            return True
        return self.arm.at_home()

    def clear_view(self, *, move: bool = True, ticket=None) -> None:
        """A real arm parks before every capture so it is not mistaken for an object. With the camera on the arm,
        home is the survey pose: the picture is only valid there, after the image settles."""
        if self.configuration_errors:
            raise ArmBusy("Saved configuration invalid: " + "; ".join(self.configuration_errors) + "; restore with staff and restart")
        if isinstance(self.arm, SimArm):
            return
        ticket = self.motion_ticket(ticket)
        was_there = isinstance(self.arm, CartesianArm) and self.arm.at_home()
        busy = self.arm_thread and self.arm_thread.is_alive()
        if not busy and was_there and not self.manual_mode:
            return                         # already parked: reading a frame requires no motion/reset
        if self.first_run:
            raise ArmBusy("First supervised run: Arm home first; captures never move the arm")
        if not move:
            raise ArmBusy("move=false: no safe parked camera view is available; explicitly home the arm first")
        self.start_arm(self.arm.home, job="home", pixels=False, ticket=ticket, wait=True)
        if self.arm.state != "idle":
            raise ArmBusy("The arm did not reach home: " + (getattr(self.arm, "last_error", None) or self.arm.state))
        if self.camera_on_arm:
            if not self.at_survey():
                raise ArmBusy("The arm did not reach the survey pose: " + (self.arm.last_error or self.arm.state))
            if not was_there:
                time.sleep(self.settle_s)      # motion blur and auto exposure settle before the picture

    def run_tour(self, targets: list, pause_s: float = 1.0) -> None:
        """Feature 6: visit each target in turn. STOP ends the whole tour, even during a pause."""
        cancel = self.cancel
        self.tour = {"targets": targets, "index": 0, "status": "running"}
        for i, (x, y) in enumerate(targets):
            self.tour["index"] = i
            if cancel.is_set() or not self.arm.point_at(x, y) or cancel.wait(pause_s):
                self.tour["status"] = "stopped"
                return
        self.tour["status"] = "done"

    def run_room(self, request: str) -> None:
        try:
            self._run_room(request)
        except Exception as e:  # camera/driver failure must terminate the job and remain visible to the operator
            error = f"{type(e).__name__}: {e}"
            try:
                self.stop_arm()
            except Exception as stop_error:
                error += f"; STOP failed: {stop_error}"
            self.room = {**(self.room or {}), "request": request, "status": "error", "error": error}
            self.audit.event("room_scan", **self.room)

    def _run_room(self, request: str) -> None:
        """Feature 14: look around from the taught look pose, let Grok pick the photo and box, turn to face it."""
        cfg = room_mod.pan_settings()
        look = load_poses(self.s.data_dir).get("look_joints")
        self.room = {"status": "scanning", "request": request, "done": 0, "total": len(cfg["angles"])}
        views = []

        def grab():
            f = self.frame()
            views.append(f)
            self.room["done"] = len(views)
            return f
        t0 = time.perf_counter()
        got = room_mod.scan(self.arm, grab, look, cfg["joint"], cfg["angles"], settle_s=self.settle_s)
        if got is None:
            self.room.update(status="stopped", error=self.arm.last_error or "STOP pressed")
            self.audit.event("room_scan", **self.room)
            return
        for v in got:
            v.path = self.audit.save_image(f"room_{v.angle:+.0f}", v.frame)
        try:
            res = room_mod.find(self.grok, got, request)
        except GrokError as e:
            self.room.update(status="error", error=str(e))
            self.audit.event("room_scan", **self.room)
            return
        out = res.as_dict() | {"sheet": self.audit.save_image("room_sheet", room_mod.sheet(got))}
        if res.status == "found":
            res.px_per_deg, res.px_per_deg_source = room_mod.measure_px_per_deg(got, cfg["hfov"], cfg["sign"])
            res.angle = room_mod.aim_angle(res, res.px_per_deg)
            out |= res.as_dict() | {"marked": self.audit.save_image("room_found", room_mod.draw_found(res))}
            q = list(look)
            q[cfg["joint"]] = look[cfg["joint"]] + res.angle
            if not self.cancel.is_set() and self.arm.go_joints(q):
                out["facing"] = True
                out["facing_view"] = self.audit.save_image("room_facing", self.frame())
            else:
                out["facing"], out["error"] = False, self.arm.last_error or "STOP pressed"
        out["total_s"] = round(time.perf_counter() - t0, 1)
        self.light.show("found" if res.status == "found" else "not_found")
        self.room = out
        self.audit.event("room_scan", **self.room)

    def run_lift(self, wps: list) -> None:
        """Feature 7: if the lift does not finish, block automatic moves until the operator frees the arm."""
        self.lift_interrupted = True
        cancel = self.cancel
        try:
            if self.lift_store:
                self.lift_store.write("in_progress")  # MUST reach durable storage before the FIRST waypoint
            if cancel.is_set() or self.stop_latched.is_set():
                return
            completed = self.arm.run_waypoints(wps)
            if completed and not cancel.is_set() and not self.stop_latched.is_set():
                if self.lift_store:
                    self.lift_store.write("clear")
                with self.safety_lock:
                    interrupted = cancel.is_set() or self.stop_latched.is_set()
                    if not interrupted:
                        self.lift_interrupted = False
                if interrupted and self.lift_store:
                    self.lift_store.write("in_progress")
        except Exception as e:  # retain the dirty marker and interlock on persistence/driver failure
            self.persistence_error = str(e)
            self.stop_arm()

    def capture_clean(self, attempts: int = 6) -> tuple[np.ndarray, list[Box], bool]:
        for _ in range(attempts):
            f = self.frame()
            boxes, occl = self.proposals(f)
            if not occl:
                return f, boxes, False
            time.sleep(0.4)
        return f, boxes, True


def create_app(settings: Settings | None = None) -> FastAPI:
    st = State(settings or load_settings())
    app = FastAPI(title="Trova")
    app.state.trova = st
    stop_limiter = anyio.CapacityLimiter(1)
    app.mount("/files", StaticFiles(directory=str(st.s.data_dir)), name="files")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.middleware("http")
    async def _same_origin(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin") or request.headers.get("referer")
            def identity(url):
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("Invalid browser origin")
                return parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                foreign = origin is not None and identity(origin) != identity(str(request.url))
            except ValueError:
                foreign = True
            if foreign or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"error": "Cross-origin state-changing request refused"}, 403)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request, exc):
        # Validation inputs can contain NaN/Infinity; do not echo those into a JSON error response.
        return JSONResponse({"detail": [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]}, 422)

    @app.middleware("http")
    async def _motion_request_ticket(request, call_next):
        token = MOTION_TICKET.set(st.stop_epoch)
        try:
            return await call_next(request)
        finally:
            MOTION_TICKET.reset(token)

    token = os.getenv("TROVA_TOKEN", "").strip()
    if token:   # the venue Wi-Fi is shared: without the token nobody else can move the arm or spend the key
        @app.middleware("http")
        async def _require_token(request, call_next):
            path = request.url.path
            given = (request.query_params.get("token") or request.headers.get("x-trova-token")
                     or request.cookies.get("trova_token") or "")
            if path.startswith(("/api/", "/files/")) and not hmac.compare_digest(given.encode(), token.encode()):
                return JSONResponse({"error": "Open Trova with the link that ends in ?token=... (TROVA_TOKEN)"}, 401)
            resp = await call_next(request)
            if request.query_params.get("token") == token:
                resp.set_cookie("trova_token", token, httponly=True, samesite="lax", max_age=7 * 24 * 3600)
            return resp

    @app.exception_handler(ArmBusy)
    def _arm_busy(request, e: ArmBusy):
        return JSONResponse({"error": str(e), "busy": True}, status_code=409)

    @app.exception_handler(CameraError)
    def _camera_error(request, e: CameraError):
        return JSONResponse({"error": str(e), "camera": True}, status_code=503)

    def fail(e: Exception, where: str):
        st.audit.event("error", where=where, error=str(e))
        return JSONResponse({"error": str(e), "where": where}, status_code=502 if isinstance(e, GrokError) else 400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/jury", response_class=HTMLResponse)
    def jury():
        """Feature 4: full-screen live scoreboard for the projector."""
        return (STATIC / "jury.html").read_text()

    @app.post("/api/metrics/reset")
    def api_metrics_reset():
        st.metrics = {"find_ok": 0, "find_wrong": 0, "find_times": [], "checks": 0, "check_ok": 0, "false_alarms": 0,
                      "missed": 0}
        st.audit.event("metrics_reset")
        return {"ok": True}

    # ------------------------------------------------------------------ live view
    @app.get("/api/frame.jpg")
    def frame_jpg(boxes: int = 1, max_w: int = Query(1280, ge=1, le=4096)):
        f = st.frame()
        img = f.copy()
        survey = st.at_survey()
        if st.camera_on_arm and not survey:      # the arm camera is somewhere else: nothing calibrated to draw
            s = max(0.6, f.shape[1] / 1600)
            cv2.putText(img, "arm camera: moving view (not the survey pose)", (20, int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        s, (0, 200, 255), 2, cv2.LINE_AA)
            if img.shape[1] > max_w:
                img = cv2.resize(img, (max_w, max(1, int(img.shape[0] * max_w / img.shape[1]))), interpolation=cv2.INTER_AREA)
            return Response(jpeg(img, 80), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        if boxes:
            props, _ = st.proposals(f)
            img = draw_marks(img, props)
        lf = st.last_find
        if lf and lf.get("box") and time.time() - lf["ts"] < 30:
            b = lf["box"]
            cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, max(4, f.shape[1] // 250))
        arm = st.arm.status()
        if st.calib and st.calib.ok and arm.get("tip_mm") and not st.camera_on_arm:
            u, v = st.calib.table_to_pixel(*arm["tip_mm"])
            if 0 <= u < f.shape[1] and 0 <= v < f.shape[0]:
                r = max(10, f.shape[1] // 90)
                cv2.circle(img, (int(u), int(v)), r, (255, 255, 0), 3, cv2.LINE_AA)
                cv2.line(img, (int(u) - 2 * r, int(v)), (int(u) + 2 * r, int(v)), (255, 255, 0), 2)
                cv2.line(img, (int(u), int(v) - 2 * r), (int(u), int(v) + 2 * r), (255, 255, 0), 2)
        if st.zones or st.zone_draft:
            img = draw_zones(img, st.zones, (st.last_check or {}).get("result", {}).get("zones"), st.zone_draft)
        for p in st.click_points + st.arm_points:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), MARK, cv2.MARKER_TILTED_CROSS, 30, 3)
        if img.shape[1] > max_w:
            s = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        return Response(jpeg(img, 80), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        try:
            f = st.frame()
            size = [f.shape[1], f.shape[0]]
        except CameraError:
            size = st.frame_size or [1280, 800]
        m = st.metrics
        judged = m["find_ok"] + m["find_wrong"]
        lf = {k: v for k, v in (st.last_find or {}).items() if k not in ("box",)}
        return {
            "mock": st.grok.mock, "grok": st.grok.stats(), "source": st.source.info(),
            "override": st.override is not None, "frame_size": size, "camera_error": st.camera_error, "configuration_errors": st.configuration_errors,
            "background": st.background is not None, "calibration": st.calib.as_dict() if st.calib else None,
            "click_points": st.click_points, "arm_points": len(st.arm_points), "arm": st.arm.status(),
            "arm_blocked": st.arm_blocked(), "lift_interrupted": st.lift_interrupted,
            "stop_latched": st.stop_latched.is_set(),
            "stop_error": st.stop_error,
            "stop_nonce": st.stop_nonce,
            "physical_arm": st.physical, "safety_persistence_error": st.persistence_error,
            "camera_on_arm": st.camera_on_arm, "at_survey": st.at_survey(), "poses": load_poses(st.s.data_dir),
            "reference": None if not st.reference else {k: st.reference[k] for k in ("time", "path", "n_objects")},
            "pending_question": st.pending["question"] if st.pending else None,
            "pending_candidates": st.pending.get("candidates", []) if st.pending else [],
            "label_wait": st.label_wait is not None,
            "last_find": lf or None,
            "last_check": None if not st.last_check else {k: v for k, v in st.last_check.items() if k != "objects"},
            "tuning": {"thresh": st.thresh, "min_area_frac": st.min_area},
            "lang": st.s.lang, "voice_available": not st.grok.mock, "inventory": st.inventory, "tour": st.tour,
            "lift_allowed": lift_allowed(), "watch": st.watch.as_dict(), "batch": st.batch, "room": st.room,
            "light": st.light.status(), "first_run": st.first_run, "approach": st.approach_view() if st.first_run else None,
            "labels": None if not st.labels else {k: v for k, v in st.labels.items() if k != "codes"},
            "dog": None if st.dog is None else {"name": st.dog.name, "route": [x.name for x in st.dog_route],
                                                "search": st.dog_search.as_dict() if st.dog_search else None},
            "metrics": {**{k: v for k, v in m.items() if k != "find_times"}, "find_judged": judged,
                        "avg_find_s": round(sum(m["find_times"]) / len(m["find_times"]), 2) if m["find_times"] else None,
                        "median_find_s": round(sorted(m["find_times"])[len(m["find_times"]) // 2], 2) if m["find_times"] else None},
        }

    # ------------------------------------------------------------------ find
    def do_find(request: str, query_img, history, move: bool = True, reuse: dict | None = None, ticket=None):
        ticket = st.motion_ticket(ticket)
        with st.lock:
            t0, asked = time.perf_counter(), time.time()
            if reuse:                       # a clarify answer: same frame and box numbers as the question
                frame, boxes, occluded = reuse["frame"], reuse["boxes"], reuse["occluded"]
            else:
                st.clear_view(move=move, ticket=ticket)
                frame = st.frame()
                boxes, occluded = st.proposals(frame)
            res = grok_find(st.grok, frame, boxes, request, query_img, history)
            hi = res.box.id if res.box and res.box.source != "grok" else None
            marked = draw_marks(frame, boxes, highlight=hi)
            if res.box is not None and res.box.source == "grok":
                b = res.box
                cv2.rectangle(marked, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, 5)
            paths = {"frame": st.audit.save_image("frame", frame), "marked": st.audit.save_image("find", marked),
                     "query": st.audit.save_image("query", query_img) if query_img is not None else None}
            table_mm, arm_blocked = None, None
            if res.status == "found" and st.calib and st.calib.ok:
                table_mm = [round(v, 1) for v in st.calib.pixel_to_table(*res.box.center)]
                arm_blocked = st.arm_blocked()
                if not arm_blocked and move and not st.first_run:
                    try:
                        st.start_arm(st.arm.point_at, *table_mm, since=asked, ticket=ticket)
                    except ArmBusy as e:
                        arm_blocked = str(e)
            total = time.perf_counter() - t0
            if res.status == "found":
                st.metrics["find_times"].append(total)
            ev = st.audit.event("find", request=request, has_photo=query_img is not None, history=history,
                                result=res.as_dict(), table_mm=table_mm, occluded=occluded, total_s=round(total, 2),
                                **paths)
            st.last_find = {"ts": time.time(), "event_id": ev["id"], "request": request, "result": res.as_dict(),
                            "table_mm": table_mm, "arm_blocked": arm_blocked, "total_s": round(total, 2), "box": res.box,
                            "occluded": occluded, "moved": bool(table_mm and move and not arm_blocked and not st.first_run),
                            "boxes": [b.as_dict() for b in boxes], **paths}
            st.pending_move = {**st.last_find, "_ticket": ticket, "_view": st.view_stamp(), "_issued": time.time()} \
                if res.status == "found" and table_mm and not arm_blocked else None
            if st.first_run and table_mm and st.pending_move and isinstance(st.arm, CartesianArm):
                try:
                    st.pending_move["approach"] = st.first_approach(table_mm)
                except Exception as e:  # noqa: BLE001 - no plan, no confirmation: the arm cannot be moved
                    st.last_find["arm_blocked"] = f"First run: cannot plan the approach: {e}"
                    st.audit.event("error", where="first_run_plan", error=str(e))
            st.light.show({"found": "found", "not_found": "not_found"}.get(res.status, "idle"))
            st.pending = None
            if res.status == "clarify":
                st.pending = {"request": request, "query_image": query_img, "history": history, "move": move,
                              "question": res.clarifying_question, "candidates": res.candidate_ids,
                              "reuse": {"frame": frame, "boxes": boxes, "occluded": occluded}, "ticket": ticket}
            st.label_wait = None
            return {k: v for k, v in st.last_find.items() if k != "box"} | (
                {"approach": st.approach_view()} if st.first_run else {})

    @app.post("/api/find")
    def api_find(text: str = Form(""), photo: UploadFile | None = File(None), move: bool = Form(True)):
        q = decode_upload(read_upload(photo)) if photo is not None and photo.filename else None
        if not text.strip() and q is None:
            raise HTTPException(400, "Say what to find, or send a photo")
        try:
            return do_find(text.strip(), q, [], move)
        except (ArmBusy, CameraError):
            raise
        except (GrokError, ValueError, RuntimeError) as e:
            return fail(e, "find")

    @app.post("/api/find_all")
    def api_find_all(text: str = Form(...), move: bool = Form(True)):
        """Feature 6: "point at all the pens": every match, then the arm visits each in turn."""
        ticket = st.motion_ticket()
        with st.lock:
            t0, asked = time.perf_counter(), time.time()
            st.clear_view(move=move, ticket=ticket)
            frame = st.frame()
            boxes, occluded = st.proposals(frame)
            try:
                res = grok_find_all(st.grok, frame, boxes, text.strip())
            except GrokError as e:
                return fail(e, "find_all")
            marked = draw_marks(frame, boxes)
            for b in res.boxes:
                cv2.rectangle(marked, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, 5)
            targets = [[round(v, 1) for v in st.calib.pixel_to_table(*b.center)] for b in res.boxes] \
                if st.calib and st.calib.ok else []
            blocked = st.arm_blocked() if targets else None
            moved = bool(targets and move and not blocked)
            if moved:
                try:
                    st.start_arm(st.run_tour, targets, since=asked, job="tour", ticket=ticket)
                except ArmBusy as e:
                    moved, blocked = False, str(e)
            out = {"result": res.as_dict(), "targets": targets, "moved": moved, "arm_blocked": blocked,
                   "marked": st.audit.save_image("find_all", marked), "request": text.strip(),
                   "total_s": round(time.perf_counter() - t0, 2), "occluded": occluded}
            ev = st.audit.event("find_all", **out)
            st.pending_move = {"result": {"label": res.label}, "targets": targets, "_ticket": ticket,
                               "_view": st.view_stamp(), "_issued": time.time()} if targets and not blocked else None
            return out | {"event_id": ev["id"]}

    @app.post("/api/clarify")
    def api_clarify(answer: str = Form(...)):
        with st.lock:                      # one answer per question, even if the button is double-clicked
            p, st.pending = st.pending, None
            if not p:
                raise HTTPException(400, "No question pending")
            try:
                return do_find(p["request"], p["query_image"], p["history"] + [(p["question"], answer)],
                               p.get("move", True), reuse=p.get("reuse"), ticket=p["ticket"])
            except (ArmBusy, CameraError):
                raise
            except (GrokError, ValueError, RuntimeError) as e:
                return fail(e, "clarify")

    # ------------------------------------------------------------------ what changed
    @app.post("/api/reference")
    def api_reference(describe: bool = Form(True)):
        with st.lock:
            st.clear_view()
            f, boxes, occl = st.capture_clean()
            names, summary = {}, ""
            if describe and boxes:   # name the objects once, so changes can say WHICH object moved or went missing
                try:
                    inv = describe_table(st.grok, f, boxes, st.s.lang)
                    names, summary = inv.names(), inv.summary
                except GrokError as e:
                    st.audit.event("error", where="reference_inventory", error=str(e))
            path = st.audit.save_image("reference", draw_marks(f, boxes))
            st.reference = {"frame": f, "boxes": boxes, "time": time.strftime("%H:%M:%S"), "path": path,
                            "raw": st.audit.save_image("reference_raw", f), "n_objects": len(boxes), "names": names}
            st.audit.event("reference", n_objects=len(boxes), occluded=occl, path=path, raw=st.reference["raw"],
                           names=names)
            st.watch.alert, st.watch._acked = None, None      # a new approved state: old alarms no longer apply
            st.watch._hist.clear()
            return {"n_objects": len(boxes), "path": path, "occluded": occl, "names": names, "summary": summary}

    @app.post("/api/changes")
    def api_changes(verify: bool = Form(True)):
        if not st.reference:
            raise HTTPException(400, "Save a reference first")
        with st.lock:
            st.clear_view()
            frames, per, occl = [], [], False
            for _ in range(8):
                f = st.frame()
                b, o = st.proposals(f)
                if o:
                    occl = True
                    time.sleep(0.4)
                    continue
                frames.append(f)
                per.append(b)
                if len(frames) == 3:
                    break
                time.sleep(0.25)
            if not frames:
                return JSONResponse({"error": "Hand or arm in view: clear the table view and retry"}, 409)
            try:
                res = chg.check(st.grok if verify else None, st.reference["frame"], st.reference["boxes"],
                                frames, per, occluded=occl and len(frames) < 3, lang=st.s.lang,
                                ref_names=st.reference.get("names"))
            except GrokError as e:
                return fail(e, "changes")
            result = res.as_dict()
            if st.zones:
                names = {}
                if any(z.rule == "allowed" for z in st.zones):   # only "allowed" zones need names from Grok
                    try:
                        names = describe_table(st.grok, frames[-1], per[-1], st.s.lang).names()
                    except GrokError as e:
                        st.audit.event("error", where="zone_names", error=str(e))
                zres = eval_zones(st.zones, per[-1], names)
                result["zones"] = zres
                bad = [p for z in zres for p in z["problems"]]
                if bad:
                    result["line_clear"] = False
                    result["summary"] = (result["summary"] + " Zones: " + "; ".join(bad) + ".").strip()
                res.cur_marked = draw_zones(res.cur_marked, st.zones, zres)
            if st.batch.get("lot"):          # feature 11: every label must belong to the current batch
                try:
                    lr = lbl.read_labels(st.grok, frames[-1], per[-1], st.batch["lot"])
                    result["batch"] = lr.as_dict()
                    if lr.problems:
                        result["line_clear"] = False
                        result["summary"] = (result["summary"] + " Labels: " + "; ".join(p["text"] for p in lr.problems) + ".").strip()
                except GrokError as e:
                    result["batch"] = {"lot": st.batch["lot"], "error": str(e), "ok": False, "items": [], "problems": []}
                    result["line_clear"] = False
                    result["summary"] = (result["summary"] + " Labels could not be read: check them by hand.").strip()
            paths = {"ref_marked": st.audit.save_image("ref_changes", res.ref_marked),
                     "cur_marked": st.audit.save_image("cur_changes", res.cur_marked),
                     "cur_raw": st.audit.save_image("current_raw", res.current)}
            ev = st.audit.event("change_check", result=result, reference=st.reference["raw"], **paths)
            st.metrics["checks"] += 1
            st.last_check = {"event_id": ev["id"], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "result": result, "reference": st.reference["path"], "signoff": None,
                             "model": st.grok.stats()["model"], "mock": st.grok.mock, **paths}
            return st.last_check

    @app.post("/api/signoff")
    def api_signoff(check_id: str = Form(...), operator: str = Form(...), decision: str = Form("approve"), note: str = Form("")):
        with st.lock:
            if not st.last_check:
                raise HTTPException(400, "Run a check first")
            if check_id != st.last_check["event_id"]:
                raise HTTPException(409, "A different check is current: review it before signing")
            if st.last_check.get("signoff"):
                raise HTTPException(409, "This check has already been signed")
            if decision not in ("approve", "reject"):
                raise HTTPException(400, "decision must be approve or reject")
            if not operator.strip():
                raise HTTPException(400, "Operator name first")
            if decision == "approve" and not st.last_check["result"]["line_clear"] and not note.strip():
                raise HTTPException(400, "The line is NOT clear: approving it anyway needs a written reason")
            so = {"operator": operator.strip(), "decision": decision, "note": note.strip(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            st.last_check["signoff"] = so
            st.audit.event("signoff", check=st.last_check["event_id"], **so)
            return so

    @app.get("/api/report", response_class=HTMLResponse)
    def api_report():
        c = st.last_check
        if not c:
            return "<p>No check yet.</p>"
        r = c["result"]
        rows = "".join(
            f"<tr><td>C{x['id']}</td><td>{html.escape(x['kind'])}</td><td>{html.escape(x['label'])}</td>"
            f"<td>{html.escape(x['risk'])}</td><td>{html.escape(x['explanation'])}</td></tr>" for x in r["changes"])
        filt = "".join(f"<li>C{x['id']} {html.escape(x['kind'])}: {html.escape(x['explanation'])}</li>"
                       for x in r["filtered"]) or "<li>none</li>"
        so = c.get("signoff")
        so_html = (f"<p><b>Signed off by {html.escape(so['operator'])}</b> ({html.escape(so['decision'])}) at "
                   f"{so['time']}. {html.escape(so['note'])}</p>") if so else "<p><b>Not signed off yet.</b></p>"
        verdict = "LINE CLEAR" if r["line_clear"] else "NOT CLEAR"
        zones_html = ""
        if r.get("zones"):
            zr = "".join(f"<tr><td>{html.escape(z['name'])}</td><td>{'empty' if z['rule'] == 'empty' else 'only ' + html.escape(', '.join(z['allowed']))}</td>"
                         f"<td>{'OK' if z['ok'] else 'VIOLATION'}</td><td>{html.escape('; '.join(z['problems']) or ', '.join(o['name'] for o in z['objects']) or 'nothing inside')}</td></tr>"
                         for z in r["zones"])
            zones_html = f"<h2>Zones</h2><table><tr><th>Zone</th><th>Rule</th><th>Status</th><th>Details</th></tr>{zr}</table>"
        if r.get("batch"):
            b = r["batch"]
            lr = "".join(f"<tr><td>{it['box_id']}</td><td>{html.escape(it['lot'])}</td><td>{html.escape(it['expiry'])}</td>"
                         f"<td>{html.escape(it['code'] or '')}</td><td>{html.escape(it['text'])}</td></tr>" for it in b.get("items", []))
            pr = "".join(f"<li><b>{html.escape(p['kind'].replace('_', ' '))}</b>: {html.escape(p['text'])}</li>"
                         for p in b.get("problems", [])) or "<li>none</li>"
            zones_html += (f"<h2>Labels (current batch LOT {html.escape(b['lot'])})</h2>"
                           + (f"<p><b>{html.escape(b['error'])}</b></p>" if b.get("error") else "")
                           + f"<table><tr><th>Object</th><th>Lot</th><th>Expiry</th><th>Code</th><th>Print</th></tr>{lr or '<tr><td colspan=5>no labels seen</td></tr>'}</table>"
                           f"<p>Problems:</p><ul>{pr}</ul>")
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Line clearance record</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#111;background:#fff}}
img{{width:100%;border:1px solid #ccc;border-radius:6px}}.g{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}
.v{{font-size:22px;font-weight:700;color:{'#137333' if r['line_clear'] else '#b3261e'}}}</style></head><body>
<h1>Line clearance record</h1><p>Check {c['event_id']} at {c['time']} · model {html.escape(str(c['model']))}{' (mock)' if c['mock'] else ''}</p>
<p class="v">{verdict}</p><p>{html.escape(r['summary'])}</p>
<div class="g"><div><b>Reference</b><img src="/files/{c['ref_marked']}"></div><div><b>Current</b><img src="/files/{c['cur_marked']}"></div></div>
<h2>Changes</h2><table><tr><th>#</th><th>Kind</th><th>Object</th><th>Risk</th><th>Why it matters</th></tr>{rows or '<tr><td colspan=5>none</td></tr>'}</table>
{zones_html}
<h3>Candidates rejected as false alarms</h3><ul>{filt}</ul>
<h2>Sign-off</h2>{so_html}
<p style="color:#555">Trova assists the operator; the operator decides. Every decision above is in the audit log (data/events.jsonl).</p>
</body></html>"""

    @app.get("/api/report.pdf")
    def api_report_pdf():
        """Feature 3: the clearance record as a PDF; a copy is kept in data/records/."""
        c = st.last_check
        if not c:
            raise HTTPException(400, "Run a check first")
        pdf = build_record_pdf(c, st.s.data_dir)
        out = st.s.data_dir / "records"
        out.mkdir(exist_ok=True)
        (out / f"clearance_{c['event_id']}.pdf").write_bytes(pdf)
        st.audit.event("record_pdf", check=c["event_id"], signed=bool(c.get("signoff")))
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f"inline; filename=clearance_{c['event_id']}.pdf"})

    # ------------------------------------------------------------------ setup
    @app.post("/api/background")
    def api_background(mode: str = Form("capture")):
        if mode == "clear":
            st.background, st.background_path = None, None
        elif mode == "capture":
            st.clear_view()
            st.background = st.source.empty() if isinstance(st.source, DemoSource) and st.override is None \
                else st.frame()
            st.background_path = st.audit.save_image("empty_table", st.background)
        else:
            raise HTTPException(400, "mode must be capture or clear")
        st.audit.event("background", mode=mode)
        return {"background": st.background is not None}

    @app.post("/api/source")
    def api_source(spec: str = Form(...)):
        with st.lock:
            try:
                new = open_source(spec)
            except Exception as e:  # noqa: BLE001 - report any camera error to the UI
                return fail(e, "source")
            st.invalidate_targets()
            st.camera_spec = spec
            st.calib, st.reference = None, None
            st.click_points, st.arm_points = [], []
            st.calib_path.unlink(missing_ok=True)
            st.watch.stop()
            old, st.source = st.source, new
            old.close()
            if hasattr(new, "attach"):
                new.attach(st.arm, st.s.data_dir)
            st.override, st.background, st.background_path = None, None, None
            st.audit.event("source", spec=spec)
            return st.source.info()

    @app.post("/api/source/step")
    def api_step(delta: int = Form(1)):
        with st.lock:
            if not isinstance(st.source, ImageSource):
                raise HTTPException(400, "Only for an image folder source")
            st.invalidate_targets()
            st.source.step(delta)
            return st.source.info()

    @app.post("/api/upload_scene")
    def api_upload_scene(photo: UploadFile = File(...)):
        with st.lock:
            st.invalidate_targets()
            st.override = decode_upload(read_upload(photo))
            st.background, st.background_path = None, None
            st.audit.event("upload_scene", size=list(st.override.shape[:2]))
            return {"override": True}

    @app.post("/api/upload_scene/clear")
    def api_upload_clear():
        with st.lock:
            st.invalidate_targets()
            st.override = None
            return {"override": False}

    @app.post("/api/tune")
    def api_tune(t: Tune):
        if t.thresh is not None:
            st.thresh = t.thresh
        if t.min_area_frac is not None:
            st.min_area = t.min_area_frac
        return {"thresh": st.thresh, "min_area_frac": st.min_area}

    # ------------------------------------------------------------------ calibration
    @app.post("/api/calibration/aruco")
    def api_cal_aruco():
        with st.lock:
            if st.override is not None:
                raise HTTPException(409, "An uploaded photo is shown: calibrate on the live camera (Back to live)")
            st.invalidate_targets()
            st.clear_view()            # park (camera on the arm: go to the survey pose) before looking for the markers
            try:
                mpath = st.s.data_dir / "markers.json"
                pos = load_marker_positions(mpath, st.s.table_w_mm, st.s.table_h_mm)
                c = calibrate_aruco(st.frame(), pos)
                if mpath.exists() and json.loads(mpath.read_text()).get("_frame") == "robot":
                    c.source = "aruco_robot"
                    c.fit()
                if st.camera_on_arm and isinstance(st.arm, CartesianArm) and st.at_survey():
                    c.pose = list(st.arm.home_joints or st.arm.home_mm)   # this mapping is only valid here
            except ValueError as e:
                return fail(e, "calibration")
            c.camera = st.camera_spec
            st.calib = c
            st.calib.save(st.calib_path)
            st.audit.event("calibration", **st.calib.as_dict())
            return st.calib.as_dict()

    @app.post("/api/calibration/click")
    def api_cal_click(p: ClickPoint):
        with st.lock:
            st.invalidate_targets()
            st.click_points.append([p.u, p.v, p.x_mm, p.y_mm])
            out = {"points": len(st.click_points)}
            if len(st.click_points) >= 4:
                f = st.frame()
                c = Calibration(points=[list(x) for x in st.click_points], source="clicks", size=[f.shape[1], f.shape[0]])
                try:
                    out["error_mm"] = round(c.fit(), 2)
                except ValueError as e:
                    st.click_points.pop()          # drop the point that broke the fit; the user clicks again
                    return fail(ValueError(f"{e} (that point was not kept)"), "calibration")
                st.calib = c
                c.save(st.calib_path)
                st.audit.event("calibration", **c.as_dict())
            return out

    @app.post("/api/calibration/reset")
    def api_cal_reset():
        with st.lock:
            st.invalidate_targets()
            st.click_points, st.arm_points, st.calib = [], [], None
            if st.calib_path.exists():
                st.calib_path.unlink()
            return {"ok": True}

    @app.post("/api/calibration/markers")
    def api_markers():
        paths = make_marker_sheet(st.s.data_dir / "markers")
        return {"files": [f"/files/markers/{p.name}" for p in paths]}

    def _need_cartesian() -> CartesianArm:
        if not isinstance(st.arm, CartesianArm):
            raise HTTPException(400, "Needs a real Cartesian arm: TROVA_ARM=ur:<ip> | xarm:<ip> | elite:<ip> | cartesian:fake")
        return st.arm

    @app.post("/api/calibration/arm_click")
    def api_cal_arm_click(p: UV):
        with st.lock:
            """Put the arm tip on the table (freedrive or pendant), then click the tip in the live view.
            Robot-frame points live in their own list: they are never mixed with table-mm clicks."""
            arm = _need_cartesian()
            if st.override is not None:
                raise HTTPException(409, "An uploaded photo is shown: calibrate on the live camera (Back to live)")
            st.invalidate_targets()
            x, y, _ = arm.tcp_mm()
            st.arm_points.append([p.u, p.v, x, y])
            pts = st.arm_points
            out = {"points": len(pts), "tcp_mm": [round(x, 1), round(y, 1)]}
            if len(pts) >= 4:
                c = Calibration(points=[list(q) for q in pts], source="clicks", size=st.frame_size)
                try:
                    out["error_mm"] = round(c.fit(), 2)
                except ValueError as e:
                    st.arm_points.pop()
                    return fail(ValueError(f"{e} (that point was not kept)"), "calibration")
                if len(pts) < 5:
                    out.pop("error_mm", None)
                    return out | {"need_points": 5}
                c.source, c.camera = "arm", st.camera_spec
                if st.camera_on_arm:
                    c.pose = list(arm.home_joints or arm.home_mm or ())
                st.calib = c
                c.save(st.calib_path)
                st.audit.event("calibration", **c.as_dict())
            return out

    @app.post("/api/calibration/marker_from_arm")
    def api_marker_from_arm(marker_id: int = Form(...)):
        """Touch the centre of ArUco marker <id> with the tip: its robot position is saved, so later
        'Calibrate (ArUco)' gives robot coordinates in seconds, even after the camera is bumped."""
        arm = _need_cartesian()
        x, y, _ = arm.tcp_mm()
        mpath = st.s.data_dir / "markers.json"
        d = json.loads(mpath.read_text()) if mpath.exists() else {}
        if d.get("_frame") != "robot":
            d = {"_frame": "robot"}
        d[str(marker_id)] = [round(x, 1), round(y, 1)]
        mpath.write_text(json.dumps(d, indent=2))
        st.audit.event("marker_from_arm", marker=marker_id, x_mm=x, y_mm=y)
        return {"markers": {k: v for k, v in d.items() if k != "_frame"}}

    @app.post("/api/arm/teach_pose")
    def api_arm_teach_pose(name: str = Form("home_joints")):
        with st.lock:
            """Record the arm's current joint angles as a named pose: home_joints (with the camera on the arm: the
            survey pose, where the camera sees the whole table) or look_joints (the room scan starts here)."""
            arm = _need_cartesian()
            if name not in ("home_joints", "look_joints"):
                raise HTTPException(400, "name must be home_joints or look_joints")
            try:
                q = arm.backend.joints()
            except NotImplementedError as e:
                return fail(e, "teach_pose")
            st.invalidate_targets()
            poses = save_pose(st.s.data_dir, name, q)
            if name == "home_joints":
                arm.home_joints = poses[name]
                if st.camera_on_arm and st.calib and st.calib.source in ("arm", "aruco_robot"):
                    st.calib = None                # the view changed: the old pixel -> robot mapping is wrong now
                    if st.calib_path.exists():
                        st.calib_path.unlink()
            st.audit.event("teach_pose", name=name, joints=poses[name])
            return {"poses": poses, "note": "survey pose changed: calibrate again (ArUco)" if name == "home_joints"
                    and st.camera_on_arm else ""}

    @app.post("/api/arm/clear_errors")
    def api_arm_clear_errors(request: Request):
        """Controller recovery only; never resets the software STOP or interrupted-lift interlock."""
        arm = _need_cartesian()
        if st.physical:
            st.authenticate_operator(request.headers.get("x-trova-operator-token", ""))
        try:
            with st.arm_lock:
                if st.arm_thread and st.arm_thread.is_alive():
                    raise ArmBusy("Wait for the motion worker to finish before controller recovery")
                if st.physical and not st.stop_latched.is_set():
                    raise ArmBusy("Latch software STOP before explicit controller recovery")
                arm.backend.clear_errors()
        except ArmBusy:
            raise
        except Exception as e:  # noqa: BLE001 - show the SDK message
            return fail(e, "clear_errors")
        arm.state, arm.last_error = "idle", None
        st.audit.event("arm_clear_errors")
        return arm.status()

    @app.post("/api/arm/freedrive")
    def api_arm_freedrive(request: Request, on: bool = Form(...)):
        arm = _need_cartesian()
        if st.physical:
            st.authenticate_operator(request.headers.get("x-trova-operator-token", ""))
        try:
            st.set_manual_mode(on, arm.freedrive)
        except NotImplementedError as e:
            return fail(e, "freedrive")
        st.audit.event("arm_freedrive", on=on)
        return {"freedrive": on}

    # ------------------------------------------------------------------ arm
    def _to_mm(p: Point) -> tuple[float, float]:
        if p.u is not None and p.v is not None:
            if st.calib is None:
                st.frame()                 # right after a reset: rebuilds the default calibration
            xy = st.calib.pixel_to_table(p.u, p.v)
        elif p.x_mm is None or p.y_mm is None:
            raise HTTPException(400, "Give x_mm,y_mm or u,v")
        else:
            xy = (p.x_mm, p.y_mm)
        if not all(math.isfinite(v) for v in xy):
            raise HTTPException(400, "Coordinates must be finite")
        return xy

    @app.post("/api/arm/stop")
    async def api_arm_stop():
        return await anyio.to_thread.run_sync(_stop_response, limiter=stop_limiter)

    def _stop_response():
        try:
            st.stop_arm()
        except Exception as e:  # software latch/cancellation remain set even when the driver cannot acknowledge STOP
            return fail(e, "arm_stop")
        st.audit.event("arm_stop")
        return st.arm.status()

    @app.post("/api/arm/home")
    def api_arm_home():
        """Homing is motion: it cannot reset STOP or clear an interrupted lift."""
        st.start_arm(st.arm.home, job="home", pixels=False)
        st.audit.event("arm_home")
        return st.arm.status()

    @app.post("/api/arm/reset")
    def api_arm_reset(r: StopReset, request: Request):
        identity = st.operator_reset(r, credential=request.headers.get("x-trova-operator-token", ""))
        st.audit.event("arm_reset", operator=identity)
        return {"stop_latched": st.stop_latched.is_set(), "lift_interrupted": st.lift_interrupted, "moved": False}

    @app.post("/api/arm/recover_lift")
    def api_arm_recover_lift(r: LiftRecovery, request: Request):
        identity = st.operator_reset(r, recover_lift=True, credential=request.headers.get("x-trova-operator-token", ""))
        st.audit.event("arm_lift_recovery", operator=identity, note=r.note.strip())
        return {"stop_latched": st.stop_latched.is_set(), "lift_interrupted": st.lift_interrupted, "moved": False}

    @app.post("/api/arm/point")
    def api_arm_point(p: Point):
        blocked = st.arm_blocked(pixels=p.u is not None)
        if blocked:
            raise HTTPException(409, blocked)
        x, y = _to_mm(p)
        if isinstance(st.arm, (CartesianArm, TaughtGridArm)) and st.arm.limits:
            err = st.arm.limits.check(x, y, st.arm.z)
            if err:
                raise ArmBusy(err)
        st.start_arm(st.arm.point_at, x, y, pixels=p.u is not None)
        st.audit.event("arm_point", x_mm=x, y_mm=y)
        return {"x_mm": round(x, 1), "y_mm": round(y, 1)}

    @app.post("/api/arm/lift_tray")
    def api_lift_tray(r: LiftReq):
        with st.lock:
            """Feature 7 (experimental): lift the slotted tray under an object with the fork and set it down elsewhere."""
            if not lift_allowed():
                return JSONResponse({"error": "Tray lifting is off. Measure TROVA_TRAY_Z_SLOT_MM and TROVA_TRAY_APPROACH "
                                              "on the real tray with staff, then set TROVA_ALLOW_LIFT=1."}, 403)
            if not hasattr(st.arm, "run_waypoints"):
                raise HTTPException(400, f"Arm '{st.arm.name}' has no waypoint support")
            if st.arm_blocked():
                return JSONResponse({"error": st.arm_blocked(), "blocked": True}, 409)
            if (r.pick_u is None) != (r.pick_v is None):
                raise HTTPException(400, "Give both pick_u and pick_v")
            if r.pick_u is None:
                lf = st.pending_move
                if not lf or not lf.get("table_mm"):
                    raise HTTPException(400, "Find the object first, or give pick_u, pick_v")
                st.require_target(lf)
                pick = tuple(lf["table_mm"])
            else:
                pick = st.calib.pixel_to_table(r.pick_u, r.pick_v)
            place = st.calib.pixel_to_table(r.place_u, r.place_v)
            z_hover = getattr(st.arm, "z", 150.0)
            wps = tray_waypoints(pick, place, z_hover, TrayParams.from_env())
            limits = getattr(st.arm, "limits", None)
            if limits is None and not (st.s.arm == "sim" and type(st.arm) is SimArm):
                raise ArmBusy("No safety box for tray lift")
            for label, x, y, z in wps:
                err = limits.check(x, y, z) if limits else None
                if err:
                    raise ArmBusy(f"{label}: {err}")
            stamp = st.view_stamp()
            st.start_arm(st.run_lift, wps, job="lift", guard=lambda: stamp == st.view_stamp())
            st.audit.event("arm_lift_tray", pick=[round(v, 1) for v in pick], place=[round(v, 1) for v in place],
                           waypoints=[[w[0]] + [round(v, 1) for v in w[1:]] for w in wps])
            return {"ok": True, "pick_mm": [round(v, 1) for v in pick], "place_mm": [round(v, 1) for v in place],
                    "waypoints": len(wps)}

    @app.post("/api/arm/teach")
    def api_arm_teach(p: Point):
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        x, y = _to_mm(p)
        return st.arm.record(x, y)

    @app.post("/api/arm/teach_home")
    def api_arm_teach_home():
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        return {"home_q": st.arm.record_home()}

    @app.post("/api/arm/compliant")
    def api_arm_compliant(request: Request, on: bool = Form(...)):
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        if st.physical:
            st.authenticate_operator(request.headers.get("x-trova-operator-token", ""))
        st.set_manual_mode(on, st.arm.backend.set_compliant)
        st.audit.event("arm_compliant", on=on)
        return {"compliant": on}

    # ------------------------------------------------------------------ jury mode + eval cases
    @app.post("/api/judge")
    def api_judge(j: Judge):
        if j.kind == "find":
            lf = st.last_find
            if not lf:
                raise HTTPException(400, "No find to judge")
            if lf["result"]["status"] == "clarify":
                raise HTTPException(409, "Answer the clarifying question, then judge the completed find")
            if lf.get("judged"):
                raise HTTPException(409, "This find is already judged: ask for the next object")
            lf["judged"] = True
            st.metrics["find_ok" if j.correct else "find_wrong"] += 1
            case = {"frame": lf["frame"], "request": lf["request"], "query": lf["query"],
                    "background": st.background_path,
                    "expected": lf["result"]["box"]["box"] if (j.correct and lf["result"]["box"]) else None,
                    "absent": j.correct and not lf["result"]["box"], "note": j.note}
            if j.correct:
                _save_case(case)
                if lf.get("total_s", 0) > SLOW_S:
                    fail_log.record(st.cases_dir / "failures.jsonl", lf, "slow", SLOW_S)
            elif j.cause == "arm_missed":          # the pick was right; the arm was not: no label needed
                case["expected"] = lf["result"]["box"]["box"] if lf["result"]["box"] else None
                _save_case(case)
                fail_log.record(st.cases_dir / "failures.jsonl", lf, "arm_missed", SLOW_S)
            else:
                st.label_wait = case   # next: click the right object, or "not on the table"
            st.audit.event("judge", target=lf["event_id"], correct=j.correct, note=j.note, cause=j.cause)
        else:
            if not st.last_check:
                raise HTTPException(400, "Run a check first")
            if st.last_check.get("judged"):
                raise HTTPException(409, "This check is already judged")
            st.last_check["judged"] = True
            st.metrics["check_ok"] += int(j.correct)
            st.metrics["false_alarms"] += j.false_alarms
            st.metrics["missed"] += j.missed
            st.audit.event("judge_check", target=(st.last_check or {}).get("event_id"), correct=j.correct,
                           false_alarms=j.false_alarms, missed=j.missed, note=j.note)
        return st.metrics | {"label_wait": st.label_wait is not None}

    @app.post("/api/label")
    def api_label(lb: Label):
        if not st.label_wait:
            raise HTTPException(400, "Nothing waiting for a label")
        if not lb.absent and (lb.u is None or lb.v is None):
            raise HTTPException(400, "Give u,v or absent=true")
        case = st.label_wait
        unboxed = False
        if lb.absent:
            case["expected"], case["absent"] = None, True
        else:
            inside = [b for b in (st.last_find or {}).get("boxes", [])
                      if Box.from_list(b["box"]).contains(lb.u, lb.v)]
            if inside:
                case["expected"] = min(inside, key=lambda b: Box.from_list(b["box"]).area)["box"]
            else:   # the object the judge meant never got a box: a proposals problem, not a Grok one
                unboxed = True
                case["expected"], case["expected_point"] = None, [lb.u, lb.v]
        cause = fail_log.classify(st.last_find, case.get("expected"), bool(case.get("absent")), unboxed)
        case["cause"] = cause
        if not unboxed:
            _save_case(case)
        entry = fail_log.record(st.cases_dir / "failures.jsonl", st.last_find, cause, SLOW_S,
                                {"expected_point": case.get("expected_point")})
        st.audit.event("failure", **entry)
        st.label_wait = None
        return case | {"failure": entry}

    @app.get("/api/failures")
    def api_failures():
        return fail_log.summary(st.cases_dir / "failures.jsonl")

    def _save_case(case: dict) -> None:
        with (st.cases_dir / "finds.jsonl").open("a") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ feature 1: inventory map
    @app.post("/api/inventory")
    def api_inventory():
        with st.lock:
            st.clear_view()
            f, boxes, occl = st.capture_clean(attempts=3)
            try:
                inv = describe_table(st.grok, f, boxes, st.s.lang)
            except GrokError as e:
                return fail(e, "inventory")
            out = inv.as_dict() | {"map": st.audit.save_image("inventory_map", draw_map(f, boxes, inv)),
                                   "n_objects": len(boxes), "occluded": occl}
            st.inventory = out
            st.audit.event("inventory", **{k: v for k, v in out.items()})
            return out

    # ------------------------------------------------------------------ feature 2: zones
    @app.get("/api/zones")
    def api_zones():
        return {"zones": [z.__dict__ for z in st.zones], "draft": st.zone_draft}

    @app.post("/api/zones/point")
    def api_zone_point(p: UV):
        st.zone_draft.append([p.u, p.v])
        return {"draft": st.zone_draft}

    @app.post("/api/zones/finish")
    def api_zone_finish(name: str = Form(...), rule: str = Form("empty"), allowed: str = Form("")):
        if len(st.zone_draft) < 3:
            raise HTTPException(400, "Click at least 3 corners first")
        if rule not in RULES:
            raise HTTPException(400, f"rule must be one of {RULES}")
        z = Zone(name=name.strip() or f"zone {len(st.zones) + 1}", polygon=st.zone_draft, rule=rule,
                 allowed=[a.strip() for a in allowed.split(",") if a.strip()])
        st.zones = [x for x in st.zones if x.name != z.name] + [z]
        st.zone_draft = []
        save_zones(st.zones_path, st.zones)
        st.audit.event("zone", **z.__dict__)
        return {"zones": [x.__dict__ for x in st.zones]}

    @app.post("/api/zones/cancel")
    def api_zone_cancel():
        st.zone_draft = []
        return {"draft": []}

    @app.delete("/api/zones/{name}")
    def api_zone_delete(name: str):
        st.zones = [z for z in st.zones if z.name != name]
        save_zones(st.zones_path, st.zones)
        return {"zones": [x.__dict__ for x in st.zones]}

    # ------------------------------------------------------------------ voice (Grok Voice realtime)
    @app.post("/api/arm/point_last")
    def api_point_last(a: Approach | None = None):
        with st.lock:
            """Move to the object of the last find (used after a spoken 'yes').
            First supervised run: stage 1 hovers above it, stage 2 descends; each needs its own shown confirmation."""
            lf = getattr(st, "pending_move", None)
            if not lf or not (lf.get("table_mm") or lf.get("targets")):
                if st.first_run:
                    raise ArmBusy("First supervised run refused: no current target; find again")
                raise ArmBusy("No current target: find again before pointing")
            st.require_target(lf)
            if st.first_run:
                return _approach(lf, a)
            blocked = st.arm_blocked()
            if blocked:
                return JSONResponse({"error": blocked, "blocked": True}, 409)
            if lf.get("targets"):                                   # the last request was "point at all ..."
                st.start_arm(st.run_tour, lf["targets"], job="tour", ticket=lf["_ticket"], guard=lambda: st.target_valid(lf))
                st.audit.event("arm_tour", targets=lf["targets"], after="voice confirmation")
                return {"ok": True, "label": lf["result"]["label"], "targets": lf["targets"]}
            st.start_arm(st.arm.point_at, *lf["table_mm"], ticket=lf["_ticket"], guard=lambda: st.target_valid(lf))
            st.audit.event("arm_point", x_mm=lf["table_mm"][0], y_mm=lf["table_mm"][1], after="voice confirmation")
            return {"ok": True, "label": lf["result"]["label"], "table_mm": lf["table_mm"]}

    def _approach(pm: dict, a: Approach | None):
        problems = st.first_run_problems()
        if problems:
            raise ArmBusy("First supervised run refused: " + "; ".join(problems))
        if a is None:
            raise ArmBusy("First supervised run: confirm the shown stage and target (stage, confirm, x_mm, y_mm)")
        if a.speed_mm_s is not None and not 0 < a.speed_mm_s <= st.first_run_speed:
            raise ArmBusy(f"Speed {a.speed_mm_s} mm/s refused: the first-run cap is {st.first_run_speed} mm/s")
        with st.safety_lock:
            ap = pm.get("approach")
            if not ap or ap["stage"] not in (0, 1, 2) or not ap["confirm"] or not hmac.compare_digest(a.confirm.encode(), ap["confirm"].encode()):
                raise ArmBusy(f"No valid stage-{a.stage} confirmation: use the one shown for the current stage")
            ap["confirm"] = None                    # single use, even if a check below refuses
            if a.stage != ap["stage"]:
                raise ArmBusy(f"This confirmation is for stage {ap['stage']}, not stage {a.stage}: find again")
            if [round(a.x_mm, 1), round(a.y_mm, 1)] != ap["target_mm"]:
                raise ArmBusy("This confirmation belongs to a different target: find again")
            if ap["seq"] != st.stop_sequence or time.time() - ap["issued"] > CONFIRM_TTL_S:
                raise ArmBusy("Stale confirmation (STOP, or older than 60 s): find again")
        x, y, z = ap["to_mm"]
        box = st.arm.limits.check(x, y, z)
        if box:
            raise ArmBusy("Outside the safety box: " + box)
        if a.stage == 2 and not st.at_hover(ap):    # descend only straight down from this target's hover point
            raise ArmBusy("The arm is not at this target's hover point: stage 2 refused")
        if a.stage == 1 and ap.get("q_level") and not st.at_joints(ap["q_level"]):
            raise ArmBusy("The arm is not at the levelled pose from stage 0: stage 1 refused")
        speed = a.speed_mm_s or st.first_run_speed
        # the picture (and so the target) is only valid at the survey pose: the first stage must start there
        first = a.stage == 0 or (a.stage == 1 and not ap.get("q_level"))
        st.start_arm(st.run_approach, pm, ap, speed, job="approach", pixels=first, ticket=pm["_ticket"], guard=lambda: st.target_valid(pm))
        st.audit.event("arm_approach_confirmed", stage=a.stage, target_mm=[x, y, z], speed_mm_s=speed)
        return {"ok": True, "stage": a.stage, "target_mm": [x, y, z], "speed_mm_s": speed, "warning": COLLISION_WARNING}

    @app.post("/api/arm/measured_error")
    def api_measured_error(error_mm: float = Form(...)):
        """First supervised run: the ruler error after stage 2. Above TROVA_ARM_MISS_MM it is an arm_missed failure."""
        pm = st.pending_move or {}
        with st.safety_lock:
            ap = pm.get("approach")
            if not ap or ap["stage"] != "measure":
                raise HTTPException(400, "Nothing to measure: finish stage 2 first")
            if not 0 <= error_mm < 1000:
                raise HTTPException(400, "error_mm must be between 0 and 1000")
            pm["approach"] = None                   # one measurement per approach
        target = [*ap["target_mm"], ap["z_mm"]]
        missed = error_mm > st.miss_mm
        st.audit.event("arm_accuracy", target_mm=target, tcp_mm=ap.get("tcp_mm"), offset_mm=ap.get("offset_mm"),
                       measured_error_mm=error_mm, threshold_mm=st.miss_mm, missed=missed, find=pm.get("event_id"))
        if missed:
            fail_log.record(st.cases_dir / "failures.jsonl", pm, "arm_missed", SLOW_S,
                            {"error_mm": error_mm, "target_mm": target, "tcp_mm": ap.get("tcp_mm")})
        return {"error_mm": error_mm, "missed": missed, "threshold_mm": st.miss_mm}

    @app.get("/api/voice/config")
    def api_voice_config(confirm: bool = True):
        return {"model": st.s.voice_model, "available": not st.grok.mock, "stop_words": voice_cfg.STOP_WORDS,
                "session": voice_cfg.session_config(st.s.voice, confirm, st.dog is not None)}

    @app.post("/api/voice/session")
    def api_voice_session():
        if st.grok.mock:
            return JSONResponse({"error": "Voice needs XAI_API_KEY in .env (mock mode has no voice)"}, 400)
        try:
            return voice_cfg.mint_token(st.s.xai_api_key)
        except Exception as e:  # noqa: BLE001
            return fail(e, "voice")

    @app.post("/api/voice/log")
    def api_voice_log(v: VoiceLog):
        st.audit.event("voice", who=v.who, text=v.text[:500])
        return {"ok": True}

    # ------------------------------------------------------------------ optional dog
    @app.post("/api/dog/search")
    def api_dog_search(description: str = Form(...), wait: bool = Form(True)):
        if st.dog is None:
            raise HTTPException(400, "No dog in this setup (TROVA_DOG=off)")
        if st.dog_thread and st.dog_thread.is_alive():
            raise HTTPException(409, "The dog is already searching")
        st.dog_stop.clear()
        st.dog_search = DogSearch(request=description)

        def go():
            try:
                run_search(st.dog, st.grok, st.dog_route, description, st.dog_stop, st.audit.save_image, st.dog_search)
            except Exception as e:  # noqa: BLE001
                st.dog_search.status = "error"
                st.dog_search.where = str(e)
            st.audit.event("dog_search", **st.dog_search.as_dict())

        st.dog_thread = threading.Thread(target=go, daemon=True)
        st.dog_thread.start()
        if wait:
            st.dog_thread.join(timeout=90)
        return st.dog_search.as_dict()

    @app.post("/api/dog/stop")
    def api_dog_stop():
        st.dog_stop.set()
        if st.dog:
            st.dog.stop()
        st.audit.event("dog_stop")
        return {"stopped": True}

    @app.get("/api/log")
    def api_log(n: int = 40, kind: str | None = None):
        return st.audit.recent(n, kind)

    preflight.register(app, st)      # feature 9: GET /api/preflight
    history.register(app, st, STATIC)   # feature 13: /history, /api/history(.csv), /api/history/verify

    # ------------------------------------------------------------------ feature 14: look around the room
    @app.post("/api/room/find")
    def api_room_find(text: str = Form(...)):
        """Scan the room from the taught look pose and face the requested object (runs in the arm thread; poll state.room)."""
        arm = _need_cartesian()
        if not load_poses(st.s.data_dir).get("look_joints"):
            raise HTTPException(400, "Teach the look pose first: Setup, move the arm by hand so the camera looks at the room, "
                                     "Teach look pose here")
        blocked = st.arm_blocked(pixels=False)
        if blocked:
            raise HTTPException(409, blocked)
        try:
            arm.backend.joints()
        except NotImplementedError as e:
            return fail(e, "room")
        if not st.start_arm(st.run_room, text.strip(), job="room", pixels=False):
            raise HTTPException(409, "STOP was pressed")
        st.audit.event("room_find", request=text.strip())
        return {"started": True, "views": len(room_mod.pan_settings()["angles"])}

    # ------------------------------------------------------------------ feature 11: labels and batch
    @app.post("/api/batch")
    def api_batch(lot: str = Form(""), product: str = Form("")):
        st.batch = {"lot": lot.strip().upper(), "product": product.strip()}
        atomic_json_write(st.batch_path, st.batch)
        st.audit.event("batch", **st.batch)
        return st.batch

    @app.post("/api/labels")
    def api_labels():
        with st.lock:
            st.clear_view()
            f, boxes, occl = st.capture_clean(attempts=3)
            try:
                lr = lbl.read_labels(st.grok, f, boxes, st.batch.get("lot", ""))
            except GrokError as e:
                return fail(e, "labels")
            out = lr.as_dict() | {"image": st.audit.save_image("labels", lbl.draw_labels(f, boxes, lr)), "occluded": occl}
            st.labels = out
            st.audit.event("labels", **out)
            return out

    @app.post("/api/labels/sheet")
    def api_label_sheet():
        p = lbl.label_sheet(st.s.data_dir / "labels", lot=st.batch.get("lot") or "A123")
        return {"file": f"/files/labels/{p.name}"}

    # ------------------------------------------------------------------ feature 10: watch the line
    @app.post("/api/watch")
    def api_watch(w: WatchReq):
        if w.on:
            try:
                st.watch.start(point=w.point)
            except ValueError as e:
                return fail(e, "watch")
        else:
            st.watch.stop()
        st.audit.event("watch", on=w.on, point=w.point)
        return st.watch.as_dict()

    @app.post("/api/watch/ack")
    def api_watch_ack(who: str = Form("")):
        a = st.watch.ack(who.strip())
        motion_blocked = None
        if a and a.get("pointed") and not isinstance(st.arm, SimArm):
            try:
                st.start_arm(st.arm.home, job="home", pixels=False, ticket=st.watch.motion_ticket_for(a),
                             guard=st.watch.motion_guard_for(a))
            except ArmBusy as e:
                motion_blocked = str(e)
        return {"acknowledged": a["id"] if a else None, "watch": st.watch.as_dict(), "motion_blocked": motion_blocked}

    # ------------------------------------------------------------------ demo helpers
    @app.post("/api/demo/tamper")
    def api_tamper(kind: str = Form("")):
        if not isinstance(st.source, DemoSource):
            raise HTTPException(400, "Only in demo mode")
        return st.source.table.tamper(kind or None)

    @app.post("/api/demo/hand")
    def api_hand(on: bool = Form(...)):
        if not isinstance(st.source, DemoSource):
            raise HTTPException(400, "Only in demo mode")
        st.source.hand = on
        return {"hand": on}

    return app
