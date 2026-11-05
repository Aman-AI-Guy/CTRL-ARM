"""Trova web app: FastAPI backend + a one-page UI (trova/static/index.html).

Run:  python -m trova      then open http://localhost:8000 (phones: http://<laptop-ip>:8000)
"""

from __future__ import annotations

import hmac
import html
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from . import changes as chg
from .arm import CartesianArm, SimArm, TaughtGridArm, load_poses, make_arm, save_pose
from .arm.skills import TrayParams, lift_allowed, waypoints as tray_waypoints
from .audit import AuditLog
from .boxes import Box
from .calibration import Calibration, calibrate_aruco, load_marker_positions, make_marker_sheet
from .camera import DemoSource, ImageSource, open_source
from .config import Settings, load_settings
from .dog import DogSearch, load_route, make_dog, run_search
from . import voice as voice_cfg
from .finder import FindResult, find as grok_find, find_all as grok_find_all
from .inventory import describe as describe_table, draw_map
from .scene import (
    Decision, TrackState, assign_tracks, build_objects, decide_text, from_finder, pack, question_for,
    remember_world, adopt_found,
)
from .geom import VisionStore, annotate_checkerboard, pose_changed, reference_error
from .motion import apply_operator_safety, safety_report
from .localization import make_router, finite, stationary
from .arm.telemetry import observation_record, open_reader, XArmTelemetry
from .acquire import ScanBook
from .reconstruction import ReconstructionJobs
from . import agent as agent_mod
from .workflow import feedback, receipt_error, refers_to_selection
from . import failures as fail_log
from .record_pdf import build as build_record_pdf
from . import history, preflight
from .watch import Watch
from . import room as room_mod
from .light import make_light
from . import labels as lbl
from .zones import RULES, Zone, draw_zones, evaluate as eval_zones, load_zones, save_zones
from .grok import GrokClient, GrokError
from .marks import HIGHLIGHT, MARK, draw_marks, jpeg
from .proposals import drop_occluders, find_blobs, is_occluded

STATIC = Path(__file__).parent / "static"
SLOW_S = float(os.getenv("TROVA_SLOW_S", "8"))


class ArmBusy(RuntimeError):
    """The arm is doing something that must not be interrupted by a new command (HTTP 409)."""


class CameraError(RuntimeError):
    """No usable picture right now (camera unplugged or frozen): HTTP 503, the UI keeps polling."""


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
    thresh: float | None = None
    min_area_frac: float | None = None


class UV(BaseModel):
    u: float
    v: float


class ClickPoint(BaseModel):
    u: float
    v: float
    x_mm: float
    y_mm: float


class Point(BaseModel):
    x_mm: float | None = None
    y_mm: float | None = None
    u: float | None = None
    v: float | None = None


class Judge(BaseModel):
    kind: str = "find"
    correct: bool = True
    cause: str = ""          # e.g. "arm_missed": Grok was right but the arm pointed at the wrong spot
    false_alarms: int = 0
    missed: int = 0
    note: str = ""


class Label(BaseModel):
    u: float | None = None
    v: float | None = None
    absent: bool = False


class LiftReq(BaseModel):
    place_u: float
    place_v: float
    pick_u: float | None = None     # default: the object of the last find
    pick_v: float | None = None


class WatchReq(BaseModel):
    on: bool = True
    point: bool = False       # the arm points at the problem when the alarm goes off


class JogReq(BaseModel):
    """Spoken or typed relative move. Mode, speed, and limits are not fields: Grok cannot set them."""
    model_config = ConfigDict(extra="forbid")
    direction: str
    distance_mm: float


class VoiceLog(BaseModel):
    who: str
    text: str


class BoardReq(BaseModel):
    """Checkerboard capture. square_mm is the measured square. The arm is not moved."""
    model_config = ConfigDict(extra="forbid")
    cols: int
    rows: int
    square_mm: float


class HandEyeReq(BoardReq):
    on_table: bool = False


class PreviewReq(BaseModel):
    """Look for a checkerboard. This stores nothing and does not move the arm."""
    model_config = ConfigDict(extra="forbid")
    cols: int
    rows: int


class AgentReq(BaseModel):
    """A spoken or typed task. Mode, limits, speed and unlock are not fields."""
    model_config = ConfigDict(extra="forbid")
    text: str
    step: bool = False


class ConfirmReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_id: str


class DropReq(BaseModel):
    """Remove one accepted calibration sample. This does not move the arm."""
    model_config = ConfigDict(extra="forbid")
    kind: str
    index: int


class ReferenceReq(BaseModel):
    """Operator-measured table point, millimetres in the robot base. u and v are optional pixels."""
    model_config = ConfigDict(extra="forbid")
    x_mm: float
    y_mm: float
    z_mm: float
    u: float | None = None
    v: float | None = None
    track_id: int | None = None


class SafetyReq(BaseModel):
    """Measured safety values. This model has no mode and no unlock."""
    model_config = ConfigDict(extra="forbid")
    confirm: bool = False
    limits: str | None = None
    keep_out: str | None = None
    speed_mm_s: float | None = None
    jog_max_mm: float | None = None
    observe_clearance_mm: float | None = None
    camera_body_mm: str | None = None
    joint_limits: str | None = None


class ScanStartReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str = ""


class OrbitReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: bool = False
    settings: dict


class OrientReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orientation: str


def _observe_clearance() -> float | None:
    raw = os.getenv("TROVA_OBSERVE_CLEARANCE_MM", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class State:
    def __init__(self, s: Settings):
        self.s = s
        self.audit = AuditLog(s.data_dir)
        self.grok = GrokClient(s.xai_api_key, s.model, s.effort, s.mock,
                               on_call=lambda e: self.audit.event("grok_call", **e),
                               timeout=float(os.getenv("TROVA_GROK_TIMEOUT_S", "60")),
                               offline_fallback=os.getenv("TROVA_OFFLINE_FALLBACK", "1").lower() not in ("0", "false", "no"))
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
        # Pose reads stay off the motion driver. Empty means "use whatever the arm driver reports".
        self.reader = open_reader(os.getenv("TROVA_ARM_READ", ""))
        # the camera is on the arm (the event's xArm 5): every picture is taken from the taught survey pose
        self.camera_on_arm = os.getenv("TROVA_CAMERA_ON_ARM", "0").lower() in ("1", "true", "yes")
        self.settle_s = float(os.getenv("TROVA_CAMERA_SETTLE_S", "0.6"))
        if hasattr(self.source, "attach"):
            self.source.attach(self.arm, s.data_dir)   # simulated arm camera: renders what the arm would see
        self.arm_thread: threading.Thread | None = None
        self.arm_lock = threading.Lock()        # one motion command at a time
        self.cancel = threading.Event()         # set by STOP; a fresh one per command
        self.stop_ts = 0.0                      # wall time of the last STOP
        self.arm_job = ""                       # point | tour | lift | home
        self.lift_interrupted = False           # the fork may still be in a slot: no automatic moves
        self.thresh, self.min_area = s.blob_thresh, s.min_area_frac
        self.last_find: dict | None = None
        self.pending: dict | None = None
        self.reference: dict | None = None
        self.last_check: dict | None = None
        self.label_wait: dict | None = None
        self.pending_move: dict | None = None
        self.inventory: dict | None = None
        self.tracks = TrackState()             # stable ids across snapshots of the same view
        self.scene_frame: np.ndarray | None = None
        self.scene_boxes: list[Box] | None = None
        self.select_history: list[tuple[str, str]] = []
        self.tour: dict | None = None
        self.zones_path = s.data_dir / "zones.json"
        self.zones: list[Zone] = load_zones(self.zones_path)
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
        self.batch = json.loads(self.batch_path.read_text()) if self.batch_path.exists() else {"lot": os.getenv("TROVA_BATCH", "")}
        self.labels: dict | None = None
        self.room: dict | None = None          # feature 14: last room scan (status, views, result)
        self.vision = make_router(VisionStore.load(s.data_dir / "vision_calib.json"), s.data_dir,
                                  os.getenv("TROVA_CALIBRATION_CANDIDATE") or None)
        self.scans = ScanBook(s.data_dir / "scans")
        from .scan3d.service import OrbitStore
        self.orbits = OrbitStore(s.data_dir / 'orbit_settings.json')
        self.reconstruction = ReconstructionJobs(s.data_dir / "reconstructions", on_done=lambda job:
            self.audit.event('scan', action='reconstruction_finished', job=job['id'], status=job['status'],
                             reconstructed=job['reconstructed'], textured_mesh=job['textured_mesh'], reason=job['reason']))
        self.last_agent: dict | None = None
        self.task_resume: dict | None = None
        self.last_reference: dict | None = None
        self.observe_clearance_mm = _observe_clearance()
        self.stage_samples: list[dict] = []
        self.task_cancel = threading.Event()
        self.agent_lock = threading.Lock()
        self.selected_context: dict | None = None
        self.pending_motion: dict | None = None

    def note_stage(self, stage: str, seconds: float) -> None:
        self.stage_samples.append({"stage": stage, "seconds": round(float(seconds), 3), "at": time.time()})
        del self.stage_samples[:-200]

    def invalidate_pending(self, reason):
        self.pending_motion = None
        self.task_resume = None
        if self.last_agent and self.last_agent.get('confirmation'):
            self.last_agent.update(confirmation=None, status='refused',
                                   workflow_state='movement_refused_or_failed', reason=reason)
            feedback(self.last_agent, self.last_agent.get('goal', ''))

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

    def arm_blocked(self, pixels: bool = True) -> str | None:
        """Why the arm must not move right now (None = it may). pixels=False: the target is given in robot mm,
        so only the safety rules apply, not the camera calibration."""
        if self.arm_job == "lift" and self.arm_thread and self.arm_thread.is_alive():
            return "The arm is lifting a tray: wait for it to finish, or press STOP"
        if self.lift_interrupted:
            return ("A tray lift was interrupted, the fork may still be in the slot: free the arm by hand "
                    "(Freedrive) or with the pendant, then press Arm home")
        if isinstance(self.arm, CartesianArm):
            if not self.arm.limits:
                return "No safety box: set TROVA_ARM_LIMITS=xmin,xmax,ymin,ymax,zmin,zmax (robot mm) with staff, restart"
            if not (self.arm.home_mm or self.arm.home_joints):
                return ("No home pose: teach one (Setup: Teach survey pose here) or set TROVA_ARM_HOME_MM=x,y,z, restart"
                        if self.camera_on_arm else "No home pose: set TROVA_ARM_HOME_MM=x,y,z outside the camera view, restart")
            if not pixels:
                return None
            if self.camera_on_arm and not self.at_survey():
                return "The arm camera is not at the survey pose, so the picture does not match the calibration: Arm home"
            if not self.calib or self.calib.source not in ("arm", "aruco_robot"):
                return "Calibrate with the arm first (Setup: Calibrate with arm, or ArUco with markers recorded by the arm)"
            if self.override is not None:
                return "An uploaded photo is shown: the arm only follows the live camera (Back to live)"
            if self.calib.size and self.frame_size and list(self.calib.size) != list(self.frame_size):
                return (f"Calibrated at {self.calib.size[0]}x{self.calib.size[1]} but the camera now gives "
                        f"{self.frame_size[0]}x{self.frame_size[1]}: calibrate again")
        return None

    # ---------- arm motion: one command at a time, STOP always wins
    def stop_arm(self) -> None:
        """Never waits for anything: mark the time, cancel the running command, stop the driver."""
        self.stop_ts = time.time()
        self.cancel.set()
        self.task_cancel.set()
        self.pending_motion = None
        self.task_resume = None
        self.arm.stop()
        self.light.show("stop")
        if self.last_agent:
            self.last_agent.update(status="stopped", workflow_state="movement_refused_or_failed",
                                   reason="STOP requested", confirmation=None)
            feedback(self.last_agent, self.last_agent.get("goal", ""))

    def _take_arm(self, job: str) -> threading.Event:
        """Caller holds arm_lock. End the running command (unless it must not be cut), then arm a fresh cancel."""
        if self.arm_thread and self.arm_thread.is_alive():
            if self.arm_job == "lift":
                raise ArmBusy("The arm is lifting a tray: wait for it to finish, or press STOP")
            self.cancel.set()
            self.arm.stop()
            self.arm_thread.join(timeout=3)
            if self.arm_thread.is_alive():
                raise ArmBusy("The arm is still finishing the last move: wait a moment and retry")
        self.cancel = threading.Event()
        self.arm.ready()
        self.arm_job = job
        return self.cancel

    def start_arm(self, fn, *args, since: float | None = None, job: str = "point") -> bool:
        """Run a motion in the arm thread. `since`: when the user asked; a STOP after that cancels it.
        Returns False when it was cancelled before starting."""
        since = time.time() if since is None else since
        with self.arm_lock:
            cancel = self._take_arm(job)
            if self.stop_ts >= since:          # STOP pressed while Trova was still thinking
                cancel.set()
                self.arm.stop()
                return False
            self.arm_thread = threading.Thread(target=fn, args=args, daemon=True)
            self.arm_thread.start()
            return True

    def at_survey(self) -> bool:
        """Camera on the arm: is the arm at the survey pose, i.e. does the picture match the calibration?"""
        if not self.camera_on_arm or not isinstance(self.arm, CartesianArm):
            return True
        return self.arm.at_home()

    def clear_view(self) -> None:
        """A real arm parks before every capture so it is not mistaken for an object. With the camera on the arm,
        home is the survey pose: the picture is only valid there, after the image settles."""
        if isinstance(self.arm, SimArm):
            return
        if self.lift_interrupted:
            raise ArmBusy(self.arm_blocked())
        with self.arm_lock:
            was_there = self.at_survey() if self.camera_on_arm else False
            self._take_arm("home")
            self.arm.home()
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
        self.lift_interrupted = not self.arm.run_waypoints(wps)

    def capture_clean(self, attempts: int = 6) -> tuple[np.ndarray, list[Box], bool]:
        for _ in range(attempts):
            f = self.frame()
            boxes, occl = self.proposals(f)
            if not occl:
                return f, boxes, False
            time.sleep(0.4)
        return f, boxes, True

    def controller_reader(self):
        if self.reader is not None:
            return self.reader
        backend = getattr(self.arm, 'backend', None)
        if type(backend).__name__ == 'XArmBackend':
            # Reuse the operator-opened connection through the same read-only guard.
            # Never create another driver or disconnect the shared SDK.
            queries = XArmTelemetry(backend.arm)
            queries._lock = backend.lock
            return queries
        return None

    def robot_telemetry(self) -> dict:
        """Read the pose. Does not enable motors, clear errors, or send a motion command."""
        backend = getattr(self.arm, "backend", None)
        target_orientation = ([180.0, 0.0, float(backend.yaw)]
                              if type(backend).__name__ == "XArmBackend" else None)
        reader = self.controller_reader()
        if reader is not None:
            reading = reader.read()
            reading["cartesian_target_orientation_deg"] = target_orientation
            reading["arm_state"] = "read-only" if self.reader is not None else ('moving' if reading.get('moving') else 'idle')
            reading["tip_mm"] = reading.get("xyz_mm")
            reading["status_tcp_mm"] = reading.get("xyz_mm")
            # This connection never enables motion. process_can_move is the motion driver, separately.
            reading["movements_enabled"] = False
            reading["motion_sent"] = False
            reading["process_can_move"] = self._process_can_move()
            reading["read_only"] = not reading["process_can_move"]
            return reading
        out = {"xyz_mm": None, "pose6": None, "joints": None, "moving": None,
               "cartesian_target_orientation_deg": target_orientation,
               "source": "unavailable", "simulated": None, "error": None,
               "motion_sent": False, "movements_enabled": False, "motors_enabled_by_trova": False,
               "tcp_changed": False, "sync_precision_claimed": False,
               "process_can_move": False, "read_only": True}
        try:
            status = self.arm.status()
        except Exception as exc:  # noqa: BLE001 - telemetry is optional; a dead link is reported
            out["error"] = str(exc)
            return out
        out["moving"] = status.get("state") == "moving"
        out["arm_state"] = status.get("state")
        out["tip_mm"] = status.get("tip_mm")
        out["status_tcp_mm"] = status.get("tcp_mm")
        out["simulated"] = bool(status.get("simulated"))
        backend = getattr(self.arm, "backend", None)
        try:
            if backend is not None and hasattr(backend, "tcp_pose"):
                pose = [round(float(v), 3) for v in backend.tcp_pose()]
                if len(pose) >= 6:
                    out["pose6"] = pose[:6]
                    out["xyz_mm"] = pose[:3]
                    out["source"] = "simulated" if out["simulated"] else "telemetry"
                else:
                    out["xyz_mm"] = pose[:3]
                    out["source"] = "xyz_only"
            elif status.get("tcp_mm"):
                out["xyz_mm"] = [float(v) for v in status["tcp_mm"]]
                out["source"] = "simulated" if out["simulated"] else "xyz_only"
            elif status.get("tip_mm") and len(status["tip_mm"]) >= 2:
                out["xyz_mm"] = [float(status["tip_mm"][0]), float(status["tip_mm"][1]),
                                 float(getattr(self.arm, "z", 0.0))]
                out["source"] = "simulated"
            if backend is not None and hasattr(backend, "joints"):
                out["joints"] = [round(float(v), 3) for v in backend.joints()]
            if backend is not None and hasattr(backend, "is_moving"):
                out["moving"] = bool(backend.is_moving())
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
            out["pose6"] = None
        # A live controller pose is telemetry, and that driver can be asked to move.
        # A simulated pose must not be labeled as a read-only physical connection.
        out["movements_enabled"] = out.get("source") == "telemetry" and not out.get("simulated")
        out["process_can_move"] = self._process_can_move()
        out["read_only"] = not out["process_can_move"]
        return out

    def _process_can_move(self) -> bool:
        """True only when this process is in hardware mode on a non-simulated driver."""
        gateway = getattr(getattr(self, "arm", None), "gateway", None)
        if gateway is None:
            return False
        return gateway.mode == "hardware" and not bool(getattr(gateway, "simulated", True))

    def adopt_found(self, found):
        """Extend the current frozen scene; invalidate any motion receipt before selection changes."""
        snap = self.inventory["snapshot"]
        self.invalidate_pending("Image region selection changed.")
        self.scene_boxes, objects = adopt_found(self.tracks, self.scene_frame, self.scene_boxes,
                                                snap["objects"], found)
        snap["objects"] = objects
        self.inventory["n_objects"] = len(objects)
        self.audit.event("image_region", snapshot_id=snap["id"], box=found.box.as_dict(),
                         localization_provenance=found.box.extra.get("localization_provenance", "opencv_blob"),
                         motion_allowed=False)
        return objects

    def snapshot_scene(self, *, home: bool = False) -> dict:
        """One fresh picture, proposals, names, and stable ids. home=True parks a real arm first (existing inventory).
        An agent snapshot leaves the arm where it is."""
        with self.lock:
            self.invalidate_pending('A new scene snapshot invalidated the previous proposal.')
            if home:
                self.clear_view()
            pose_before = self.robot_telemetry()
            capture_s = 0.0
            proposals_s = 0.0
            frame, boxes, occl = None, [], True
            for _attempt in range(3):
                t_capture = time.perf_counter()
                frame = self.frame()
                capture_s += time.perf_counter() - t_capture
                t_prop = time.perf_counter()
                boxes, occl = self.proposals(frame)
                proposals_s += time.perf_counter() - t_prop
                if not occl:
                    break
                time.sleep(0.4)
            pose = self.robot_telemetry()
            capture_stationary = bool(
                pose_before.get('source') == 'telemetry' and pose.get('source') == 'telemetry'
                and not pose_before.get('moving') and not pose.get('moving')
                and not pose_before.get('error') and not pose.get('error')
                and pose_before.get('error_code') == 0 and pose.get('error_code') == 0
                and pose_before.get('joints') and pose.get('joints')
                and not pose_changed(pose_before.get('pose6'), pose.get('pose6'), mm=1, deg=.1)
                and np.max(np.abs(np.subtract(pose_before['joints'], pose['joints']))) <= .1)
            if pose.get('source') == 'telemetry' and not capture_stationary:
                pose = dict(pose, moving=True)
            size = [int(frame.shape[1]), int(frame.shape[0])]
            if self.override is None:
                self.frame_size = size
            if hasattr(self.vision, "set_context"):
                provider_camera = self.source.info() if self.override is None else {'kind': 'upload'}
                self.vision.set_context(provider_camera, pose)
            vision_ok = self.vision.assess(size)["valid"]
            reset = None
            pose_mm = pose.get("pose6") or pose.get("xyz_mm")
            if self.camera_on_arm and not self.at_survey() and not pose.get("pose6"):
                reset = "arm camera is not at the survey pose"
            boxes, reset = assign_tracks(self.tracks, boxes, frame, reset, pose_mm=pose_mm)
            t_grok = time.perf_counter()
            inv = describe_table(self.grok, frame, boxes, self.s.lang)
            grok_s = inv.latency_s or (time.perf_counter() - t_grok)
            objects = build_objects(boxes, frame, inv.items, self.tracks)
            image = self.audit.save_image("scene", frame)
            drawn = self.audit.save_image("inventory_map", draw_map(frame, boxes, inv))
            info = self.source.info() if self.override is None else {'kind': 'upload', 'provenance': 'uploaded_image'}
            snapshot = {
                "id": image, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "image": image, "map": drawn, "width": size[0], "height": size[1],
                "at_survey": self.at_survey(), "camera_on_arm": self.camera_on_arm,
                "pose": {"state": pose.get("arm_state"), "tip_mm": pose.get("tip_mm"),
                         "tcp_mm": pose.get("status_tcp_mm"), "pose6": pose.get("pose6"),
                         "source": pose.get("source"), "moving": pose.get("moving"),
                         "joints": pose.get("joints")},
                "grok_mode": inv.grok_mode, "tracks_reset": reset,
                "occluded": occl, "objects": objects,
                "authoritative": inv.grok_mode == "live", "motion_allowed": False,
                "camera_profile": info,
                "provenance": info.get("provenance") or info.get("kind"),
                "camera_kind": info.get("kind"),
                "vision_valid": vision_ok, "capture_stationary": capture_stationary,
                "sync_precision_claimed": False,
            }
            out = inv.as_dict() | {"map": drawn, "n_objects": len(boxes), "occluded": occl, "snapshot": snapshot}
            self.scene_frame, self.scene_boxes = frame, boxes
            self.select_history = []
            self.inventory = out
            self.audit.event("inventory", summary=inv.summary, n_objects=len(boxes), occluded=occl,
                             map=drawn, image=image, snapshot_id=image, tracks=objects,
                             grok_mode=inv.grok_mode, authoritative=snapshot["authoritative"],
                             tracks_reset=reset, provenance=snapshot["provenance"], motion_allowed=False)
            for stage, seconds in (("camera_acquisition", capture_s), ("object_proposals", proposals_s),
                                   ("grok_vision", grok_s)):
                self.note_stage(stage, seconds)
            out["timings"] = {"camera_acquisition_s": round(capture_s, 3),
                              "object_proposals_s": round(proposals_s, 3),
                              "grok_vision_s": round(grok_s, 3), "realtime_claimed": False}
            return out


class AgentHost:
    """What the skill loop may touch. Motion goes through the gateway and start_arm."""

    def __init__(self, st: State):
        self.st = st
        self.grok = st.grok
        self.vision = st.vision
        self.gateway = getattr(st.arm, "gateway", None)
        self.settle_s = st.settle_s
        self.cancel = st.task_cancel
        self.timeout_s = float(os.getenv("TROVA_TASK_TIMEOUT_S", "90"))
        info = st.source.info()
        self.camera_kind = 'upload' if st.override is not None else (info.get("kind") or "")
        self.image_size = list(st.frame_size or info.get("size") or [0, 0])
        self.observe_clearance_mm = getattr(st, "observe_clearance_mm", None)
        if self.observe_clearance_mm is None:
            self.observe_clearance_mm = _observe_clearance()
        self.read_only_robot = getattr(st, "reader", None) is not None
        self.frame = st.scene_frame
        self.boxes = st.scene_boxes
        self._jpeg: bytes | None = None
        self.confirmation_required = True
        self.started_at = time.time()
        self.scan_book = st.scans
        self.orbits = st.orbits
        self.reconstruction = st.reconstruction

    def publish(self, task):
        self.st.last_agent = feedback({"status": "executing", "workflow_state": "movement_executing",
                                       "selected": task.get("selected"), "hardware_moved": False,
                                       "execution_mode": self.gateway.mode}, task["goal"])

    def snapshot(self) -> dict:
        out = self.st.snapshot_scene(home=False)
        self.frame = self.st.scene_frame
        self.boxes = self.st.scene_boxes
        snap = out["snapshot"]
        self.image_size = [snap["width"], snap["height"]]
        return snap

    def adopt_found(self, found):
        objects = self.st.adopt_found(found)
        self.boxes = self.st.scene_boxes
        return objects

    def robot(self) -> dict:
        robot = self.st.robot_telemetry()
        if hasattr(self.vision, "set_context"):
            camera = self.st.source.info() if self.st.override is None else {'kind': 'upload'}
            self.vision.set_context(camera, robot)
        return robot

    def note_stage(self, stage: str, seconds: float) -> None:
        self.st.note_stage(stage, seconds)

    def inverse(self, pose6, ref_joints=None) -> dict:
        reader = self.st.controller_reader()
        if reader is None or not hasattr(reader, "inverse"):
            return {"ok": False, "ik_verified": False, "joints": None, "motion_sent": False,
                    "reason": "inverse kinematics was not asked of the controller"}
        return reader.inverse(pose6, ref_joints)

    def read_frame(self):
        kind = self.st.source.info().get("kind")
        if kind == "pi":
            packed = self.st.source.read_capture()
            self._jpeg = packed["jpeg"]
            meta = dict(packed["meta"])
            meta["kind"] = "pi"
        else:
            frame = self.st.frame()
            self.frame = frame
            self._jpeg = None
            meta = {"kind": kind, "requested_at": time.time(), "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "provenance": "synthetic" if kind in ("demo", "armsim") else kind,
                    "sync": "local_read", "sync_precision_claimed": False}
            robot = self.robot()
            meta["moving"] = robot.get("moving")
            meta["pose6"] = robot.get("pose6")
            meta["robot_tcp_mm"] = robot.get("xyz_mm")
            meta["joints_deg"] = robot.get("joints")
            return frame, meta
        robot = self.robot()
        meta["moving"] = robot.get("moving")
        meta["pose6"] = robot.get("pose6")
        meta["robot_tcp_mm"] = robot.get("xyz_mm")
        meta["joints_deg"] = robot.get("joints")
        if self.vision.hand_eye:
            meta["tool_camera"] = self.vision.hand_eye["matrix"]
        meta["calibration_version"] = self.vision.version
        return packed["frame"], meta

    def redetect(self, frame):
        boxes, _occl = self.st.proposals(frame)
        pose = self.robot()
        vision_ok = self.vision.assess([frame.shape[1], frame.shape[0]])["valid"]
        boxes, _reset = assign_tracks(self.st.tracks, boxes, frame, None,
                                      pose_mm=pose.get("xyz_mm") if vision_ok else None)
        self.frame, self.boxes = frame, boxes
        return build_objects(boxes, frame, {})

    def remember(self, track_id, point) -> None:
        remember_world(self.st.tracks, track_id, point)

    def execute(self, target) -> dict:
        gateway = self.gateway
        if gateway is None:
            return {"ok": False, "reason": "this arm has no motion gateway"}
        holder: dict = {}
        start = gateway.current_pose()
        if start is None:
            return {'ok': False, 'reason': 'Current pose is unavailable; approach not started.'}
        target_distance = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
        deadline = 30 + target_distance / max(1.0, gateway.max_speed_mm_s)

        def go():
            try:
                holder["ok"] = bool(gateway._execute_cartesian(tuple(target)))
                holder["error"] = getattr(self.st.arm, "last_error", None)
            except Exception as exc:
                holder['ok'], holder['error'] = False, str(exc)
            finally:
                holder["final"] = gateway.current_pose()

        if self.cancel.is_set() or not self.st.start_arm(go, job="approach", since=self.started_at):
            return {"ok": False, "reason": "STOP is latched. The approach was not started."}
        if getattr(self, 'active_task', None) and holder.get('ok') is not False:
            self.publish(self.active_task)
        if self.st.arm_thread is not None:
            self.st.arm_thread.join(timeout=deadline)
        if "ok" not in holder:
            self.st.stop_arm()
            final = gateway.current_pose()
            return {"ok": False, "reason": holder.get("error") or "the approach did not finish",
                    'motion_observed': final is not None and float(np.linalg.norm(np.asarray(final) - start)) > 1,
                    'movement_status_unknown': final is None}
        return {"ok": bool(holder["ok"]), "final_mm": holder.get("final"),
                'motion_observed': holder.get('final') is not None and float(np.linalg.norm(np.asarray(holder['final']) - start)) > 1,
                'movement_status_unknown': holder.get('final') is None,
                "reason": "moved" if holder["ok"] else (holder.get("error") or "the arm did not complete the approach")}

    def execute_orbit(self, task, geometry):
        gateway = self.gateway
        start = self.robot()
        current = start.get('joints')
        if current is None or self.cancel.is_set():
            return {'ok':False,'reason':'Stationary joints unavailable or task cancelled.'}
        holder={}
        deadline=time.monotonic()+30+max(abs(a-b) for a,b in zip(current,task['planned_joints']))/max(.1,task['orbit_speed'])
        def go():
            try:
                # Revalidate after start_arm owns the driver, immediately before execution.
                decision=gateway.authorize_joints(task['planned_joints'],task['orbit_speed'],source='scan_orbit',geometry=geometry)
                if not decision.allow_motion:
                    holder.update(ok=False,reason=decision.reason)
                    return
                self.publish(task)
                holder.update(gateway.execute_joints(task['planned_joints'],task['orbit_speed'],geometry=geometry))
            except Exception as exc:
                holder.update(ok=False,reason=str(exc))
        if not self.st.start_arm(go,job='approach',since=self.started_at):
            return {'ok':False,'reason':'STOP, busy controller or cancelled command.'}
        while self.st.arm_thread and self.st.arm_thread.is_alive():
            if self.cancel.is_set() or time.monotonic()>deadline:
                self.st.stop_arm()
                holder.update(ok=False,reason='STOP/cancellation or orbit movement deadline exceeded.')
                break
            self.st.arm_thread.join(timeout=.1)
        actual=self.robot()
        final=actual.get('joints')
        reached=bool(final and max(abs(a-b) for a,b in zip(final,task['planned_joints']))<=1 and
                     not actual.get('moving') and not actual.get('error_code'))
        tcp=actual.get('xyz_mm')
        reached=reached and tcp is not None and np.linalg.norm(np.asarray(tcp)-task['planned_target'])<=2
        observed=bool(final and max(abs(a-b) for a,b in zip(final,current))>.1)
        return {'ok':bool(holder.get('ok') and reached),'motion_observed':observed,
                'movement_status_unknown':final is None,
                'reason':holder.get('reason') if not holder.get('ok') else
                         'Joint/TCP target reached; fresh visual verification is still required.' if reached else 'Final telemetry did not verify the orbit pose.'}

    def record_capture(self, frame, meta, selected):
        if self.st.scans.active is None:
            return None
        data = self._jpeg
        if not data:
            data = jpeg(frame, 90)
        detection = {}
        if selected:
            detection = {k: selected.get(k) for k in ("track_id", "name", "box", "association", "confidence")}
        packed = dict(meta or {})
        robot = self.robot()
        packed.setdefault("pose6", robot.get("pose6"))
        packed.setdefault("robot_tcp_mm", robot.get("xyz_mm"))
        packed.setdefault("robot_rpy_deg", None if not robot.get("pose6") else list(robot["pose6"][3:]))
        packed.setdefault("joints_deg", robot.get("joints"))
        packed.setdefault("moving", robot.get("moving"))
        packed.setdefault("pose_source", robot.get("source"))
        packed.setdefault("error_code", robot.get("error_code"))
        packed.setdefault("state_code", robot.get("state_code"))
        packed.setdefault("read_finished_at", robot.get("read_finished_at"))
        if self.vision.hand_eye:
            packed.setdefault("tool_camera", self.vision.hand_eye["matrix"])
        packed.setdefault("calibration_version", self.vision.version)
        packed.setdefault("calibration_valid", bool(self.vision.assess(self.image_size)["valid"]))
        record = self.st.scans.capture(data, packed, detection)
        self.st.audit.event('scan', action='capture', session=record['session_id'], viewpoint=record['viewpoint_id'],
                            accepted=record['accepted'], reason=record['reject_reason'])
        return {k: record.get(k) for k in ("viewpoint_id", "accepted", "reject_reason", "session_id")}


class ProviderReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    pixel: list[float]
    measured_mm: list[float]
    measurement_note: str


class ObjectReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    track_id: int
    pixel: list[float]
    kind: str
    confirm: bool = False


class RegionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    box: list[float] | None = None
    point: list[float] | None = None
    label: str = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    st = State(settings or load_settings())
    app = FastAPI(title="Trova")
    app.state.trova = st
    app.mount("/files", StaticFiles(directory=str(st.s.data_dir)), name="files")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    token = os.getenv("TROVA_TOKEN", "").strip()
    if token:   # the venue Wi-Fi is shared: without the token nobody else can move the arm or spend the key
        @app.middleware("http")
        async def _require_token(request, call_next):
            path = request.url.path
            given = (request.query_params.get("token") or request.headers.get("x-trova-token")
                     or request.cookies.get("trova_token") or "")
            if path.startswith(("/api/", "/files/")) and not hmac.compare_digest(given, token):
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
    def frame_jpg(boxes: int = 1, max_w: int = 1280):
        f = st.frame()
        img = f.copy()
        survey = st.at_survey()
        if st.camera_on_arm and not survey:      # the arm camera is somewhere else: nothing calibrated to draw
            s = max(0.6, f.shape[1] / 1600)
            cv2.putText(img, "arm camera: moving view (not the survey pose)", (20, int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        s, (0, 200, 255), 2, cv2.LINE_AA)
            if img.shape[1] > max_w:
                img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
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
            img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return Response(jpeg(img, 80), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        receipt = (st.pending_motion or {}).get('confirmation') or {}
        if receipt and time.time() >= receipt.get('expires_at', 0):
            st.invalidate_pending('The motion proposal expired. Request a new plan.')
        info = st.source.info()
        # A Pi snapshot waits for a new frame on the camera. The status poll must not take one.
        if info.get("kind") == "pi":
            size = info.get("size") or st.frame_size or [0, 0]
        else:
            try:
                f = st.frame()
                size = [f.shape[1], f.shape[0]]
            except CameraError:
                size = st.frame_size or [1280, 800]
        m = st.metrics
        judged = m["find_ok"] + m["find_wrong"]
        lf = {k: v for k, v in (st.last_find or {}).items() if k not in ("box",)}
        reconstruction = (st.last_agent or {}).get('reconstruction')
        if reconstruction and reconstruction.get('id') in st.reconstruction.jobs:
            current = st.reconstruction.get(reconstruction['id'])
            st.last_agent['reconstruction'] = current
            st.last_agent['status'] = 'reconstruction_' + current['status']
            st.last_agent['spoken_message'] = ('Ricostruzione: ' + current['status'] + '. ' + current['reason'])
        return {
            "mock": st.grok.mock, "grok": st.grok.stats(), "source": st.source.info(),
            "override": st.override is not None, "frame_size": size, "camera_error": st.camera_error,
            "background": st.background is not None, "calibration": st.calib.as_dict() if st.calib else None,
            "click_points": st.click_points, "arm_points": len(st.arm_points), "arm": st.arm.status(),
            "motion": st.arm.gateway.public() if hasattr(st.arm, "gateway") else None,
            "arm_blocked": st.arm_blocked(), "lift_interrupted": st.lift_interrupted,
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
            "light": st.light.status(),
            "vision": st.vision.assess(size),
            "vision_progress": st.vision.progress(),
            "robot": st.robot_telemetry(),
            "safety": safety_report(getattr(st.arm, "gateway", None), st.observe_clearance_mm),
            "reference_check": st.last_reference,
            "latency": agent_mod._timing_summary(st.stage_samples) | {
                "realtime_claimed": False, "n": len(st.stage_samples)},
            "agent": st.last_agent,
            "scan": None if not st.scans.active else st.scans.public(st.scans.sessions[st.scans.active]),
            "labels": None if not st.labels else {k: v for k, v in st.labels.items() if k != "codes"},
            "dog": None if st.dog is None else {"name": st.dog.name, "route": [x.name for x in st.dog_route],
                                                "search": st.dog_search.as_dict() if st.dog_search else None},
            "metrics": {**{k: v for k, v in m.items() if k != "find_times"}, "find_judged": judged,
                        "avg_find_s": round(sum(m["find_times"]) / len(m["find_times"]), 2) if m["find_times"] else None,
                        "median_find_s": round(sorted(m["find_times"])[len(m["find_times"]) // 2], 2) if m["find_times"] else None},
        }

    # ------------------------------------------------------------------ find
    def do_find(request: str, query_img, history, move: bool = True, reuse: dict | None = None):
        if not st.agent_lock.acquire(blocking=False):
            raise ArmBusy('An autonomous task is running; cancel it before changing the target')
        try:
            return _do_find(request, query_img, history, move, reuse)
        finally:
            st.agent_lock.release()

    def _do_find(request, query_img, history, move, reuse):
        with st.lock:
            st.invalidate_pending('A new find invalidated the previous proposal.')
            st.selected_context = None
            t0, asked = time.perf_counter(), time.time()
            if reuse:                       # a clarify answer: same frame and box numbers as the question
                frame, boxes, occluded = reuse["frame"], reuse["boxes"], reuse["occluded"]
            else:
                st.clear_view()
                frame = st.frame()
                boxes, occluded = st.proposals(frame)
            captured_pose = (reuse or {}).get('pose') or st.robot_telemetry()
            res = grok_find(st.grok, frame, boxes, request, query_img, history)
            hi = res.box.id if res.box and res.box.source != "grok" else None
            marked = draw_marks(frame, boxes, highlight=hi)
            if res.box is not None and res.box.source == "grok":
                b = res.box
                cv2.rectangle(marked, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, 5)
            paths = {"frame": st.audit.save_image("frame", frame), "marked": st.audit.save_image("find", marked),
                     "query": st.audit.save_image("query", query_img) if query_img is not None else None}
            if res.status == 'found' and res.box is not None:
                candidates = [res.box] if res.box.source == 'grok' else boxes
                tracked, reset = assign_tracks(st.tracks, candidates, frame,
                                               pose_mm=captured_pose.get('xyz_mm') if st.vision.assess(st.frame_size)['valid'] else None)
                live = not st.grok.mock and res.mode != 'offline'
                labels = {res.box.id: {'name': res.label, 'category': 'other', 'note': res.reason,
                                       'name_from_model': bool(res.label), 'label_authoritative': live}}
                objects = build_objects(tracked, frame, labels, st.tracks)
                chosen = next((obj for obj in objects if obj['box_id'] == res.box.id), None)
                snapshot = {'id': paths['frame'], 'image': paths['frame'], 'map': paths['marked'],
                            'width': frame.shape[1], 'height': frame.shape[0], 'pose': captured_pose,
                            'objects': objects, 'grok_mode': 'live' if live else ('mock' if st.grok.mock else 'offline'),
                            'authoritative': live, 'motion_allowed': False, 'tracks_reset': reset}
                st.scene_frame, st.scene_boxes = frame, tracked
                st.inventory = {'snapshot': snapshot, 'objects': objects, 'summary': res.label, 'map': paths['marked']}
                st.selected_context = {'selected': chosen, 'objects': objects, 'snapshot_id': snapshot['id'],
                                       'snapshot_pose': captured_pose, 'grok_mode': snapshot['grok_mode'],
                                       'authoritative': live and chosen.get('label_authoritative', False), 'phase': 'selected'} if chosen else None
            table_mm, arm_blocked = None, None
            if res.status == "found" and st.calib and st.calib.ok:
                table_mm = [round(v, 1) for v in st.calib.pixel_to_table(*res.box.center)]
                arm_blocked = st.arm_blocked()
                if not arm_blocked and move and not st.start_arm(st.arm.point_at, *table_mm, since=asked):
                    arm_blocked = "STOP was pressed while Trova was looking: the arm stays put"
            total = time.perf_counter() - t0
            if res.status == "found":
                st.metrics["find_times"].append(total)
            ev = st.audit.event("find", request=request, has_photo=query_img is not None, history=history,
                                result=res.as_dict(), table_mm=table_mm, occluded=occluded, total_s=round(total, 2),
                                **paths)
            st.last_find = {"ts": time.time(), "event_id": ev["id"], "request": request, "result": res.as_dict(),
                            "table_mm": table_mm, "arm_blocked": arm_blocked, "total_s": round(total, 2), "box": res.box,
                            "occluded": occluded, "moved": bool(table_mm and move and not arm_blocked),
                            "boxes": [b.as_dict() for b in boxes], **paths}
            st.pending_move = st.last_find if res.status == "found" else None
            st.light.show({"found": "found", "not_found": "not_found"}.get(res.status, "idle"))
            st.pending = None
            if res.status == "clarify":
                st.pending = {"request": request, "query_image": query_img, "history": history, "move": move,
                              "question": res.clarifying_question, "candidates": res.candidate_ids,
                              "reuse": {"frame": frame, "boxes": boxes, "occluded": occluded, 'pose': captured_pose}}
            st.label_wait = None
            return {k: v for k, v in st.last_find.items() if k != "box"}

    @app.post("/api/find")
    def api_find(text: str = Form(""), photo: UploadFile | None = File(None), move: bool = Form(True)):
        q = decode_upload(photo.file.read()) if photo is not None and photo.filename else None
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
        with st.lock:
            t0, asked = time.perf_counter(), time.time()
            st.clear_view()
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
            if moved and not st.start_arm(st.run_tour, targets, since=asked, job="tour"):
                moved, blocked = False, "STOP was pressed while Trova was looking: the arm stays put"
            out = {"result": res.as_dict(), "targets": targets, "moved": moved, "arm_blocked": blocked,
                   "marked": st.audit.save_image("find_all", marked), "request": text.strip(),
                   "total_s": round(time.perf_counter() - t0, 2), "occluded": occluded}
            ev = st.audit.event("find_all", **out)
            st.pending_move = {"result": {"label": res.label}, "targets": targets} if targets else None
            return out | {"event_id": ev["id"]}

    @app.post("/api/clarify")
    def api_clarify(answer: str = Form(...)):
        with st.lock:                      # one answer per question, even if the button is double-clicked
            p, st.pending = st.pending, None
            if not p:
                raise HTTPException(400, "No question pending")
            try:
                return do_find(p["request"], p["query_image"], p["history"] + [(p["question"], answer)],
                               p.get("move", True), reuse=p.get("reuse"))
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
    def api_signoff(operator: str = Form(...), decision: str = Form("approve"), note: str = Form("")):
        if not st.last_check:
            raise HTTPException(400, "Run a check first")
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
        try:
            new = open_source(spec)
        except Exception as e:  # noqa: BLE001 - report any camera error to the UI
            return fail(e, "source")
        old, st.source = st.source, new
        old.close()
        if hasattr(new, "attach"):
            new.attach(st.arm, st.s.data_dir)
        st.override, st.background, st.background_path = None, None, None
        st.audit.event("source", spec=spec)
        return st.source.info()

    @app.post("/api/source/step")
    def api_step(delta: int = Form(1)):
        if not isinstance(st.source, ImageSource):
            raise HTTPException(400, "Only for an image folder source")
        st.source.step(delta)
        return st.source.info()

    @app.post("/api/upload_scene")
    def api_upload_scene(photo: UploadFile = File(...)):
        st.override = decode_upload(photo.file.read())
        st.background, st.background_path = None, None
        st.audit.event("upload_scene", size=list(st.override.shape[:2]))
        return {"override": True}

    @app.post("/api/upload_scene/clear")
    def api_upload_clear():
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
        st.clear_view()            # park (camera on the arm: go to the survey pose) before looking for the markers
        try:
            mpath = st.s.data_dir / "markers.json"
            pos = load_marker_positions(mpath, st.s.table_w_mm, st.s.table_h_mm)
            st.calib = calibrate_aruco(st.frame(), pos)
            if mpath.exists() and json.loads(mpath.read_text()).get("_frame") == "robot":
                st.calib.source = "aruco_robot"
        except ValueError as e:
            return fail(e, "calibration")
        st.calib.save(st.calib_path)
        st.audit.event("calibration", **st.calib.as_dict())
        return st.calib.as_dict()

    @app.post("/api/calibration/click")
    def api_cal_click(p: ClickPoint):
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
        """Put the arm tip on the table (freedrive or pendant), then click the tip in the live view.
        Robot-frame points live in their own list: they are never mixed with table-mm clicks."""
        arm = _need_cartesian()
        if st.override is not None:
            raise HTTPException(409, "An uploaded photo is shown: calibrate on the live camera (Back to live)")
        x, y, _ = arm.tcp_mm()
        st.arm_points.append([p.u, p.v, x, y])
        pts = st.arm_points
        out = {"points": len(pts), "tcp_mm": [round(x, 1), round(y, 1)]}
        if len(pts) >= 4:
            c = Calibration(points=[list(q) for q in pts], source="arm", size=st.frame_size)
            try:
                out["error_mm"] = round(c.fit(), 2)
            except ValueError as e:
                st.arm_points.pop()
                return fail(ValueError(f"{e} (that point was not kept)"), "calibration")
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
        """Record the arm's current joint angles as a named pose: home_joints (with the camera on the arm: the
        survey pose, where the camera sees the whole table) or look_joints (the room scan starts here)."""
        arm = _need_cartesian()
        if name not in ("home_joints", "look_joints"):
            raise HTTPException(400, "name must be home_joints or look_joints")
        try:
            q = arm.backend.joints()
        except NotImplementedError as e:
            return fail(e, "teach_pose")
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
    def api_arm_clear_errors():
        """After a collision stop or a protective stop: clear the controller error and enable motion again."""
        arm = _need_cartesian()
        try:
            arm.backend.clear_errors()
        except Exception as e:  # noqa: BLE001 - show the SDK message
            return fail(e, "clear_errors")
        arm.state, arm.last_error = "idle", None
        st.audit.event("arm_clear_errors")
        return arm.status()

    @app.post("/api/arm/freedrive")
    def api_arm_freedrive(on: bool = Form(...)):
        arm = _need_cartesian()
        try:
            arm.freedrive(on)
        except NotImplementedError as e:
            return fail(e, "freedrive")
        st.audit.event("arm_freedrive", on=on)
        return {"freedrive": on}

    # ------------------------------------------------------------------ arm
    def _to_mm(p: Point) -> tuple[float, float]:
        if p.u is not None and p.v is not None:
            if st.calib is None:
                st.frame()                 # right after a reset: rebuilds the default calibration
            return st.calib.pixel_to_table(p.u, p.v)
        if p.x_mm is None or p.y_mm is None:
            raise HTTPException(400, "Give x_mm,y_mm or u,v")
        return p.x_mm, p.y_mm

    @app.post("/api/arm/stop")
    def api_arm_stop():
        st.stop_arm()
        st.audit.event("arm_stop")
        return st.arm.status()

    @app.post("/api/arm/home")
    def api_arm_home():
        """Explicit operator command: also the way out after an interrupted tray lift."""
        st.lift_interrupted = False
        st.start_arm(st.arm.home, job="home")
        st.audit.event("arm_home")
        return st.arm.status()

    @app.post("/api/arm/jog")
    def api_arm_jog(body: JogReq):
        """Relative jog in the robot base frame. The gateway keeps the requested millimetres and says why if it refuses."""
        gateway = getattr(st.arm, "gateway", None)
        if gateway is None:
            raise HTTPException(400, "this arm has no motion gateway")
        planned = gateway.plan_relative(body.direction, body.distance_mm, execute=False)
        if not planned.allow_motion:
            st.audit.event("arm_jog", direction=body.direction, distance_mm=body.distance_mm,
                           status=planned.status, ok=planned.ok, mode=planned.mode, reason=planned.reason)
            return planned.as_dict()
        holder: dict = {}

        def go():
            holder["decision"] = gateway.plan_relative(body.direction, body.distance_mm, execute=True)

        if not st.start_arm(go, job="jog"):
            planned.ok, planned.status, planned.allow_motion = False, "stopped", False
            planned.reason = "STOP is latched. A software stop holds position; it is not the physical emergency stop."
            st.audit.event("arm_jog", direction=body.direction, distance_mm=body.distance_mm,
                           status=planned.status, ok=False, mode=planned.mode, reason=planned.reason)
            return planned.as_dict()
        if st.arm_thread is not None:
            deadline = (st.arm.motion_timeout_s(abs(body.distance_mm)) + 1
                        if isinstance(st.arm, CartesianArm) else 30)
            st.arm_thread.join(timeout=max(30, deadline))
        decision = holder.get("decision", planned)
        if "decision" not in holder:
            decision.ok, decision.status, decision.allow_motion = False, "error", False
            decision.reason = decision.reason or "the jog did not finish"
        st.audit.event("arm_jog", direction=body.direction, distance_mm=body.distance_mm,
                       status=decision.status, ok=decision.ok, mode=decision.mode, reason=decision.reason)
        return decision.as_dict()

    @app.post("/api/arm/point")
    def api_arm_point(p: Point):
        blocked = st.arm_blocked(pixels=p.u is not None)
        if blocked:
            raise HTTPException(409, blocked)
        x, y = _to_mm(p)
        st.start_arm(st.arm.point_at, x, y)
        st.audit.event("arm_point", x_mm=x, y_mm=y)
        return {"x_mm": round(x, 1), "y_mm": round(y, 1)}

    @app.post("/api/arm/lift_tray")
    def api_lift_tray(r: LiftReq):
        """Feature 7 (experimental): lift the slotted tray under an object with the fork and set it down elsewhere."""
        if not lift_allowed():
            return JSONResponse({"error": "Tray lifting is off. Measure TROVA_TRAY_Z_SLOT_MM and TROVA_TRAY_APPROACH "
                                          "on the real tray with staff, then set TROVA_ALLOW_LIFT=1."}, 403)
        if not hasattr(st.arm, "run_waypoints"):
            raise HTTPException(400, f"Arm '{st.arm.name}' has no waypoint support")
        if st.arm_blocked():
            return JSONResponse({"error": st.arm_blocked(), "blocked": True}, 409)
        if r.pick_u is None:
            lf = st.last_find
            if not lf or not lf.get("table_mm"):
                raise HTTPException(400, "Find the object first, or give pick_u, pick_v")
            pick = tuple(lf["table_mm"])
        else:
            pick = st.calib.pixel_to_table(r.pick_u, r.pick_v)
        place = st.calib.pixel_to_table(r.place_u, r.place_v)
        z_hover = getattr(st.arm, "z", 150.0)
        wps = tray_waypoints(pick, place, z_hover, TrayParams.from_env())
        st.start_arm(st.run_lift, wps, job="lift")
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
    def api_arm_compliant(on: bool = Form(...)):
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        st.arm.backend.set_compliant(on)
        st.audit.event("arm_compliant", on=on)
        return {"compliant": on}

    # ------------------------------------------------------------------ jury mode + eval cases
    @app.post("/api/judge")
    def api_judge(j: Judge):
        if j.kind == "find":
            lf = st.last_find
            if not lf:
                raise HTTPException(400, "No find to judge")
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

    # ------------------------------------------------------------------ feature 1: inventory map + M1 snapshot
    @app.post("/api/inventory")
    def api_inventory():
        try:
            return st.snapshot_scene(home=True)
        except GrokError as e:
            return fail(e, "inventory")

    def _select_object(text: str, track_id: str, box_id: str):
        """Pick one object in the current snapshot. This route never calls the arm."""
        with st.lock:
            snap = (st.inventory or {}).get("snapshot")
            if not snap or st.scene_frame is None or st.scene_boxes is None:
                raise HTTPException(400, "Take a snapshot first")
            objects = snap["objects"]
            authoritative = snap.get("grok_mode") == "live"

            def _opt(raw: str, label: str) -> int | None:
                raw = (raw or "").strip()
                if not raw:
                    return None
                try:
                    return int(raw)
                except ValueError:
                    raise HTTPException(400, f"{label} must be an integer") from None

            tid, bid = _opt(track_id, "track_id"), _opt(box_id, "box_id")
            if tid is None and bid is None and not text.strip():
                raise HTTPException(400, "Say which object, or pass track_id")
            if tid is not None or bid is not None:
                hits = objects
                if tid is not None:
                    hits = [obj for obj in hits if obj["track_id"] == tid]
                if bid is not None:
                    hits = [obj for obj in hits if obj["box_id"] == bid]
                if len(hits) == 1:
                    decision = decide_text(f"oggetto {hits[0]['track_id']}", objects)
                elif not hits:
                    decision = Decision("not_found", reason="That object is not in this snapshot")
                else:
                    decision = Decision("clarify", [obj["track_id"] for obj in hits], question_for(hits, text),
                                        reason="several objects match that id")
            else:
                decision = decide_text(text, objects)
                if decision.status == "defer":
                    history = list(st.select_history)
                    if history and history[-1][1] == "":
                        history[-1] = (history[-1][0], text.strip())
                    try:
                        found = grok_find(st.grok, st.scene_frame, st.scene_boxes, text.strip(),
                                          history=history or None)
                    except GrokError as e:
                        return fail(e, "select")
                    if found.status == "found":
                        objects = st.adopt_found(found)
                        authoritative = found.recognition_live
                    decision = from_finder(found.status, found.box.id if found.box else None, found.candidate_ids,
                                           found.confidence, found.reason, objects, text)
            if decision.status == "clarify" and decision.question:
                st.select_history = [(decision.question, "")]
            else:
                st.select_history = []
            out = pack(decision, objects, authoritative, snap["id"])
            st.invalidate_pending('Selection changed. Request a new plan for this object.')
            st.selected_context = ({"selected": out["object"], "snapshot_id": snap["id"],
                                    "snapshot_pose": snap.get("pose"), "objects": objects,
                                    "authoritative": out["authoritative"], "grok_mode": snap.get("grok_mode"),
                                    "phase": "selected"} if out.get("object") else None)
            if out.get('object') and st.scans.active:
                session = st.scans._session(None)
                if session.get('needs_reselection'):
                    session['orientation_track_id'] = str(out['object']['track_id'])
                    session['needs_reselection'] = False
                    st.scans._write_index(session)
            out["spoken_message"] = feedback({"workflow_state": "object_selected", "selected": out.get("object"),
                                               "question": out.get("question"), "reason": out.get("reason")}, text)["spoken_message"]
            st.audit.event("select", status=out["status"], snapshot_id=snap["id"],
                           track_id=None if not out["object"] else out["object"]["track_id"],
                           authoritative=out["authoritative"], motion_allowed=False)
            return out

    @app.post("/api/select")
    def api_select(text: str = Form(""), track_id: str = Form(""), box_id: str = Form("")):
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "A task is running; cancel it before changing selection")
        try:
            return _select_object(text, track_id, box_id)
        finally:
            st.agent_lock.release()

    @app.get("/api/vision/providers")
    def api_providers():
        st.vision.set_context(st.source.info(), st.robot_telemetry())
        return st.vision.assess(st.frame_size)

    @app.post("/api/vision/provider/reference")
    def api_provider_reference(body: ProviderReference):
        """Store a current independent ruler/reference comparison; never select/activate a provider."""
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "A task is running")
        try:
            with st.lock:
                snap = (st.inventory or {}).get('snapshot') or {}
                historical = st.vision.historical
                if historical is None:
                    raise HTTPException(400, "No imported candidate is available")
                if snap.get('id') != body.snapshot_id:
                    raise HTTPException(409, "Snapshot changed; capture a fresh stationary scene")
                try:
                    measured = finite(body.measured_mm, 3)
                    pixel = finite(body.pixel, 2)
                    if not body.measurement_note.strip():
                        raise ValueError('Describe the independent physical measurement')
                    if not (0 <= pixel[0] < snap['width'] and 0 <= pixel[1] < snap['height']):
                        raise ValueError('Pixel is outside this snapshot')
                    robot = st.robot_telemetry()
                    stationary(robot)
                    if st.override is not None or not snap.get('capture_stationary'):
                        raise ValueError('Reference requires a fresh live snapshot with stationary telemetry before and after capture')
                    old = snap.get('pose') or {}
                    if old.get('source') != 'telemetry' or old.get('moving') or pose_changed(old.get('pose6'), robot.get('pose6'), mm=1, deg=.1):
                        raise ValueError('Reference image and stationary robot pose no longer match')
                    if not old.get('joints') or np.max(np.abs(np.subtract(old['joints'], robot['joints']))) > .1:
                        raise ValueError('Reference image and robot joints no longer match')
                    if [snap['width'], snap['height']] != [historical.cal.width, historical.cal.height]:
                        raise ValueError('Current image resolution differs from the imported optical profile')
                    profile = snap.get('camera_profile') or {}
                    if profile.get('kind') != 'pi' or profile.get('rotate_reported') != 180 or profile.get('extra_rotate') != 0:
                        raise ValueError('Verify the Pi 180-degree server rotation and zero extra rotation first')
                    from .scan3d.evidence import sha256
                    row = {'calibration_sha256': historical.digest,
                           'binding_sha256': historical.binding_hash(historical.station()),
                           'image': snap['image'], 'image_sha256': sha256(st.s.data_dir / snap['image']),
                           'image_size': [snap['width'], snap['height']], 'robot': robot,
                           'pixel': pixel.tolist(), 'measured_mm': measured.tolist(),
                           'measurement_note': body.measurement_note, 'recorded_at': time.time(),
                           'camera_profile': profile, 'provenance': 'operator_independent_reference'}
                    predicted = historical.cal.pixel_to_plane(*pixel, robot['joints'], float(measured[2]))
                    projection = historical.cal.robot_to_pixel(measured, robot['joints'])
                except (ValueError, TypeError, KeyError) as exc:
                    raise HTTPException(400, str(exc)) from None
                path = st.s.data_dir / 'calibration_references.jsonl'
                with path.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                st.invalidate_pending('New independent reference evidence recorded; request a fresh proposal.')
                result = {'diagnostic_point_mm': predicted.tolist(), 'measured_mm': measured.tolist(),
                          'error_mm': float(np.linalg.norm(predicted-measured)),
                          'reprojection_error_px': float(np.linalg.norm(np.asarray(projection)-pixel)),
                          'metric_permitted': False, 'motion_allowed': False, 'provider_activated': False}
                st.audit.event('calibration_reference', **row, comparison=result)
                return result
        finally:
            st.agent_lock.release()

    @app.post("/api/select/reference")
    def api_object_reference(body: ObjectReference):
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "A task is running")
        try:
            with st.lock:
                snap = (st.inventory or {}).get('snapshot') or {}
                context = st.selected_context or {}
                selected = context.get('selected') or {}
                if snap.get('id') != body.snapshot_id or selected.get('track_id') != body.track_id:
                    raise HTTPException(409, "Snapshot or selected track changed")
                if not body.confirm or body.kind != 'support_plane_point':
                    raise HTTPException(400, "Confirm a visible point on the support plane; unknown object height is not a top-plane measurement")
                try:
                    pixel = finite(body.pixel, 2)
                    if not Box.from_list(selected['box']).contains(*pixel):
                        raise ValueError('Reference must lie inside the selected region')
                except (ValueError, TypeError) as exc:
                    raise HTTPException(400, str(exc)) from None
                selected['geometric_reference'] = {'kind': body.kind, 'pixel': pixel.tolist(),
                                                    'snapshot_id': body.snapshot_id,
                                                    'provenance': 'operator_marked_support_point'}
                st.invalidate_pending('Object geometric reference changed.')
                st.audit.event('object_reference', snapshot_id=body.snapshot_id, track_id=body.track_id,
                               reference=selected['geometric_reference'], motion_allowed=False)
                return {'selected': selected, 'motion_allowed': False, 'metric_localization_state': 'not_localized'}
        finally:
            st.agent_lock.release()

    @app.post("/api/scene")
    def api_scene():
        # Unlike the historical inventory endpoint, never parks the arm.
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "A task is running; wait before taking a snapshot")
        try:
            return st.snapshot_scene(home=False)
        finally:
            st.agent_lock.release()

    @app.post("/api/select/region")
    def api_select_region(body: RegionSelection):
        from .regions import validated_region
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "A task is running; wait before changing selection")
        try:
            with st.lock:
                snap = (st.inventory or {}).get("snapshot")
                if not snap or snap["id"] != body.snapshot_id or st.scene_frame is None:
                    raise HTTPException(409, "Snapshot changed; open a fresh snapshot")
                if (body.box is None) == (body.point is None):
                    raise HTTPException(400, "Provide either a box or a point")
                if body.point is not None:
                    if len(body.point) != 2 or not np.isfinite(body.point).all():
                        raise HTTPException(400, "Point requires two finite pixel coordinates")
                    hits = [b for b in st.scene_boxes if b.contains(*body.point)]
                    if not hits:
                        raise HTTPException(400, "No region at this point. Draw a box around the object.")
                    return _select_object("", "", str(min(hits, key=lambda b: b.area).id))
                try:
                    box = validated_region(body.box, st.scene_frame, label=body.label)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(400, str(exc)) from None
                box.extra["localization_provenance"] = "operator_drawn_region"
                found = FindResult(status="found", box=box, label=body.label, mode="direct",
                                   reason="Operator drew this image region; semantic label is operator supplied.")
                objects = st.adopt_found(found)
                obj = next(o for o in objects if o["box_id"] == found.box.id)
                return _select_object("", str(obj["track_id"]), "")
        finally:
            st.agent_lock.release()

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
    def api_point_last():
        """Move to the object of the last find (used after a spoken 'yes')."""
        lf = getattr(st, "pending_move", None)
        if not lf or not (lf.get("table_mm") or lf.get("targets")):
            raise HTTPException(400, "Nothing found yet to point at")
        blocked = st.arm_blocked()
        if blocked:
            return JSONResponse({"error": blocked, "blocked": True}, 409)
        if lf.get("targets"):                                   # the last request was "point at all ..."
            st.start_arm(st.run_tour, lf["targets"], job="tour")
            st.audit.event("arm_tour", targets=lf["targets"], after="voice confirmation")
            return {"ok": True, "label": lf["result"]["label"], "targets": lf["targets"]}
        st.start_arm(st.arm.point_at, *lf["table_mm"])
        st.audit.event("arm_point", x_mm=lf["table_mm"][0], y_mm=lf["table_mm"][1], after="voice confirmation")
        return {"ok": True, "label": lf["result"]["label"], "table_mm": lf["table_mm"]}

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
        if not st.start_arm(st.run_room, text.strip(), job="room"):
            raise HTTPException(409, "STOP was pressed")
        st.audit.event("room_find", request=text.strip())
        return {"started": True, "views": len(room_mod.pan_settings()["angles"])}

    # ------------------------------------------------------------------ feature 11: labels and batch
    @app.post("/api/batch")
    def api_batch(lot: str = Form(""), product: str = Form("")):
        st.batch = {"lot": lot.strip().upper(), "product": product.strip()}
        st.batch_path.write_text(json.dumps(st.batch))
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
        if a and a.get("pointed") and not isinstance(st.arm, SimArm):
            st.start_arm(st.arm.home, job="home")      # the arm pointed at the problem: park it again
        return {"acknowledged": a["id"] if a else None, "watch": st.watch.as_dict()}

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

    # ------------------------------------------------------------------ arm camera, vision task, scan sessions
    @app.get("/api/camera/status")
    def api_camera_status():
        info = st.source.info()
        if info.get("kind") == "pi":
            try:
                live = st.source.cam.status()
            except Exception as exc:  # noqa: BLE001 - the UI shows the camera as offline
                return info | {"online": False, "error": str(exc), "live_status": None}
            info = info | {"online": live.get("camera") != "paused", "live_status": live,
                           "error": None if live.get("camera") != "paused" else (live.get("message") or "cooling")}
        return info

    @app.get("/api/camera/stream")
    def api_camera_stream():
        """MJPEG for the live view. Localization uses /snapshot.jpg, not this queue."""
        if st.source.info().get("kind") != "pi":
            raise HTTPException(404, "The MJPEG stream is the Pi camera. This source has no stream.")
        url = st.source.stream_url()

        def chunks():
            request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
            try:
                with urllib.request.urlopen(request, timeout=8) as response:
                    while True:
                        block = response.read(8192)
                        if not block:
                            break
                        yield block
            except Exception:  # noqa: BLE001 - the browser retries the image
                return

        return StreamingResponse(chunks(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    def _fresh_frame():
        """A picture for calibration or a scan. Does not move the arm."""
        if st.source.info().get("kind") == "pi":
            packed = st.source.read_capture()
            return packed["frame"], packed["jpeg"], packed["meta"]
        frame = st.frame()
        return frame, jpeg(frame, 90), {
            "kind": st.source.info().get("kind"), "requested_at": time.time(),
            "width": int(frame.shape[1]), "height": int(frame.shape[0]),
            "provenance": st.source.info().get("provenance") or st.source.info().get("kind"),
            "sync": "local_read", "sync_precision_claimed": False,
        }

    def _capture_meta(frame, meta: dict) -> dict:
        packed = dict(meta)
        packed["width"] = int(frame.shape[1])
        packed["height"] = int(frame.shape[0])
        return packed

    def _reported_size(frame, meta: dict) -> str | None:
        width, height = meta.get("width_reported"), meta.get("height_reported")
        if not width or not height:
            return None
        if (int(width), int(height)) != (int(frame.shape[1]), int(frame.shape[0])):
            return (f"decoded image is {frame.shape[1]}x{frame.shape[0]} but the camera reported "
                    f"{width}x{height}")
        return None

    @app.get("/api/vision/calibration")
    def api_vision_calibration():
        size = st.frame_size or st.source.info().get("size")
        report = st.vision.as_dict()
        report["status"] = st.vision.assess(size)
        report["progress"] = st.vision.progress()
        report["moved"] = False
        return report

    @app.post("/api/vision/preview")
    def api_vision_preview(body: PreviewReq):
        if body.cols < 2 or body.rows < 2:
            raise HTTPException(400, "cols and rows are the inner corners of the checkerboard")
        frame, _data, meta = _fresh_frame()
        meta = _capture_meta(frame, meta)
        robot = st.robot_telemetry()
        drawn, info = annotate_checkerboard(frame, body.cols, body.rows)
        path = st.audit.save_image("board", drawn)
        return info | {
            "image": path, "moved": False, "persisted": False,
            "size_conflict": _reported_size(frame, meta),
            "observation": observation_record(meta, robot, info),
        }

    @app.post("/api/vision/intrinsics")
    def api_vision_intrinsics(body: BoardReq):
        if body.square_mm <= 0:
            raise HTTPException(400, "square_mm has to be the measured checkerboard square")
        frame, _data, meta = _fresh_frame()
        meta = _capture_meta(frame, meta)
        conflict = _reported_size(frame, meta)
        drawn, info = annotate_checkerboard(frame, body.cols, body.rows)
        path = st.audit.save_image("board", drawn)
        meta["preview"] = path
        if conflict:
            out = {"ok": False, "reason": conflict, "count": len(st.vision.intrinsic_views), "corners": 0}
        else:
            out = st.vision.add_intrinsics(frame, body.cols, body.rows, body.square_mm, image_meta=meta)
        if not out["ok"]:
            st.vision.note_rejection("intrinsics", out["reason"], image_path=path)
        st.audit.event("vision_capture", step="intrinsics", ok=out["ok"], reason=out["reason"],
                       count=out.get("count"), moved=False)
        return out | {"moved": False, "image": path, "found": info["found"],
                      "observation": observation_record(meta, st.robot_telemetry(), info),
                      "progress": st.vision.progress()}

    @app.post("/api/vision/intrinsics/solve")
    def api_vision_intrinsics_solve():
        report = st.vision.solve()
        st.audit.event("vision_capture", step="solve", version=report["version"],
                       status=report["status"]["status"], moved=False)
        return report | {"moved": False, "progress": st.vision.progress()}

    @app.post("/api/vision/handeye")
    def api_vision_handeye(body: HandEyeReq):
        if body.square_mm <= 0:
            raise HTTPException(400, "square_mm has to be the measured checkerboard square")
        before = st.robot_telemetry()
        frame, _data, meta = _fresh_frame()
        meta = _capture_meta(frame, meta)
        after = st.robot_telemetry()
        drawn, info = annotate_checkerboard(frame, body.cols, body.rows)
        path = st.audit.save_image("board", drawn)
        conflict = _reported_size(frame, meta)
        if pose_changed(before.get("pose6"), after.get("pose6")):
            out = {"ok": False, "corners": info["corners"], "count": len(st.vision.handeye_views),
                   "reason": "the arm moved while the picture was taken, so the pose was not stored"}
        elif conflict:
            out = {"ok": False, "corners": 0, "count": len(st.vision.handeye_views), "reason": conflict}
        elif after.get("source") != "telemetry":
            if not after.get("pose6"):
                reason = "the controller did not report roll, pitch and yaw"
            else:
                reason = (f"the pose source is {after.get('source') or 'unavailable'}, not telemetry; "
                          "the view was not stored")
            out = {"ok": False, "corners": info["corners"], "count": len(st.vision.handeye_views), "reason": reason}
        else:
            meta["preview"] = path
            out = st.vision.add_handeye(
                frame, after.get("pose6"), body.cols, body.rows, body.square_mm,
                on_table=body.on_table, pose_source="telemetry", robot=after, image_meta=meta,
                moving=bool(after.get("moving")))
        if not out["ok"]:
            st.vision.note_rejection("handeye", out["reason"], image_path=path, pose_source=after.get("source"))
        st.audit.event("vision_capture", step="handeye", ok=out["ok"], reason=out["reason"],
                       count=out.get("count"), pose_source=after.get("source"), moved=False)
        return out | {"moved": False, "pose_source": after.get("source"), "image": path,
                      "found": info["found"], "observation": observation_record(meta, after, info),
                      "progress": st.vision.progress()}

    @app.post("/api/vision/handeye/solve")
    def api_vision_handeye_solve():
        report = st.vision.solve()
        st.audit.event("vision_capture", step="solve", version=report["version"],
                       status=report["status"]["status"], moved=False)
        return report | {"moved": False, "progress": st.vision.progress()}

    @app.get("/api/vision/samples")
    def api_vision_samples():
        return st.vision.samples() | {"moved": False}

    @app.post("/api/vision/sample/drop")
    def api_vision_drop(body: DropReq):
        out = st.vision.drop_sample(body.kind, body.index)
        if out["ok"]:
            st.vision.save()
        st.audit.event("vision_capture", step="drop", sample_kind=body.kind, index=body.index,
                       ok=out["ok"], reason=out["reason"], moved=False)
        return out | {"moved": False, "progress": st.vision.progress()}

    @app.post("/api/vision/reference")
    def api_vision_reference(body: ReferenceReq):
        """Compare a measured point with the calibrated ray. Does not move the arm."""
        robot = st.robot_telemetry()
        snap = (st.inventory or {}).get("snapshot") or {}
        objects = snap.get("objects") or []
        image_time = snap.get("time")
        size = st.frame_size or st.source.info().get("size") or [0, 0]
        u, v = body.u, body.v
        pose_snap = snap.get("pose") or {}
        pose = pose_snap.get("pose6")
        stale = None
        if robot.get("moving") or pose_snap.get("moving"):
            stale = "the robot is still moving; take another picture before comparing"
        elif snap and not pose:
            stale = "the picture has no robot pose; take another picture before comparing"
        elif pose and (not robot.get("pose6") or pose_changed(pose, robot.get("pose6"))):
            stale = "the robot pose changed after the picture; take another picture before comparing"
        if u is None or v is None:
            chosen = next((obj for obj in objects if obj.get("track_id") == body.track_id), None)
            if chosen is None or body.track_id is None:
                raise HTTPException(400, "pass u and v, or a track_id from the current snapshot")
            u, v = chosen["center"]
            size = [snap.get("width") or size[0], snap.get("height") or size[1]]
        report = st.vision.assess(size)
        predicted = None
        uncertainty = None
        located_reason = None
        if stale:
            located_reason = stale
        elif report["valid"]:
            located = st.vision.localize(robot.get("pose6"), float(u), float(v), size)
            predicted = located.get("point_mm")
            uncertainty = located.get("uncertainty_mm")
            located_reason = located.get("reason")
        compared = reference_error(
            predicted, [body.x_mm, body.y_mm, body.z_mm], uncertainty_mm=uncertainty,
            calibration_version=st.vision.version, image_time=image_time,
            pose_source=robot.get("source"), calibration_valid=bool(report["valid"]) and not stale)
        if stale or (report["valid"] and not compared.get("estimate_from_calibration")):
            compared["reason"] = located_reason
            compared["estimate_from_calibration"] = False
            compared["error_mm"] = None
        if not report["valid"]:
            compared["missing"] = report["reasons"]
        compared["pixel"] = [u, v]
        compared["moved"] = False
        st.last_reference = compared
        st.audit.event("vision_capture", step="reference", error_mm=compared.get("error_mm"),
                       calibration_valid=compared["calibration_valid"], moved=False)
        return compared

    @app.get("/api/safety")
    def api_safety():
        return safety_report(getattr(st.arm, "gateway", None), st.observe_clearance_mm)

    @app.post("/api/safety")
    def api_safety_set(body: SafetyReq):
        gateway = getattr(st.arm, "gateway", None)
        result = apply_operator_safety(gateway, body.model_dump(), observe_clearance_mm=st.observe_clearance_mm)
        if result.get("ok") and "TROVA_OBSERVE_CLEARANCE_MM" in result.get("applied", []):
            st.observe_clearance_mm = result.get("clearance_mm")
        if result.get("mode") not in (None, getattr(gateway, "mode", None)):
            result["ok"] = False
            result["reason"] = "the safety update changed the motion mode; that is refused"
        st.audit.event("safety", ok=bool(result.get("ok")), applied=result.get("applied"),
                       mode=result.get("mode"), moved=False)
        result["moved"] = False
        result["grok_can_edit"] = False
        return result

    @app.get("/api/latency")
    def api_latency():
        summary = agent_mod._timing_summary(st.stage_samples)
        summary["realtime_claimed"] = False
        summary["n"] = len(st.stage_samples)
        return summary

    def _agent_result(result, text):
        resumed = result.pop("resume", None)
        st.task_resume = resumed
        st.pending_motion = resumed if result.get("confirmation") else None
        if result.get("status") == "clarify" and result.get("question"):
            st.select_history = [(result["question"], "")]
        if result.get("selected") and resumed:
            st.selected_context = {k: resumed.get(k) for k in
                                   ("selected", "objects", "snapshot_pose", "snapshot_id", "authoritative", "grok_mode")}
            st.selected_context["phase"] = "selected"
        feedback(result, text)
        st.last_agent = result
        st.audit.event("vision_task", goal=result.get("goal"), status=result.get("status"),
                       phase=result.get("phase"), hardware_moved=result.get("hardware_moved"),
                       goal_achieved=result.get("goal_achieved"), motion_lock=result.get("motion_lock"),
                       calibration_valid=result.get("calibration_valid"),
                       verified_tool_position=False, reason=result.get("reason"))
        return result

    def _refused(reason, text=""):
        return _agent_result({"ok": False, "status": "refused", "workflow_state": "movement_refused_or_failed",
                              "reason": reason, "hardware_moved": False, "simulation_moved": False,
                              "selected": (st.selected_context or {}).get("selected"),
                              "confirmation": None, "verified_tool_position": False}, text)

    @app.post("/api/agent/task")
    def api_agent_task(body: AgentReq):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Say what to do")
        # A conversational yes can never create or authorize a new trajectory.
        import re
        if re.fullmatch(r"(?i)(s[iì]|yes|ok|procedi|proceed|go ahead)([,! .]+(procedi|proceed))?[.! ]*", text):
            return _refused("Confirm the specific validated proposal using its proposal_id.", text)
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "Another agent task is running")
        try:
            st.pending_motion = None
            st.task_cancel = threading.Event()
            host = AgentHost(st)
            if re.search(r"(?i)\b(ho (capovolto|girato|ruotato)|i (flipped|turned|rotated))\b", text):
                if not st.scans.active:
                    return _refused("No scan session is open.", text)
                changed = api_scan_orientation(OrientReq(orientation='reoriented-' + str(time.time_ns())))
                return _agent_result({"ok": True, "status": "awaiting_reselection", "workflow_state": "object_reoriented",
                                      "reason": changed['reason'], "hardware_moved": False, "confirmation": None,
                                      "spoken_message": "Ho aperto un nuovo gruppo di acquisizione. Ora seleziona di nuovo l'oggetto; la posizione precedente non è più valida."}, text)
            if re.search(r"(?i)(\b(scan\w*|scansion\w*|reconstruct\w*|ricostru\w*)\b|modello\s*3d|3d\s*model)", text):
                if re.search(r"(?i)\b(reconstruct\w*|ricostru\w*)\b", text):
                    if not st.scans.last:
                        return _refused("No scan dataset exists. Acquire overlapping views first.", text)
                    job = st.reconstruction.start(st.scans._session(st.scans.last))
                    return _agent_result({"ok": True, "status": "reconstruction_queued",
                                          "workflow_state": "reconstruction_queued", "reconstruction": job,
                                          "spoken_message": "Ricostruzione avviata; il risultato non è ancora disponibile.",
                                          "reason": "Reconstruction queued; no model is claimed yet.",
                                          "hardware_moved": False, "confirmation": None}, text)
                context = st.selected_context
                needs_selection = not context or not context.get('selected') or bool(re.search(r'(?i)\b(seleziona|select)\b',text))
                if needs_selection:
                    result=agent_mod.run(host,text,skill_sequence=['inspect_scene','select_object','locate_object','start_scan','plan_orbit','observe_from_viewpoint'])
                    return _agent_result(result,text)
                object_id = str(context["selected"]["track_id"])
                active = st.scans._session(st.scans.active) if st.scans.active else None
                if active and str(active.get('orientation_track_id', active['object_id'])) != object_id:
                    return _refused("Finish the open scan before selecting another scan object.", text)
                if not active:
                    session = st.scans.start(object_id)
                    if st.vision.intrinsics:
                        st.scans._session(session['id'])["intrinsics"] = st.vision.intrinsics
                result = agent_mod.run(host, text, selected_context=context,
                                       skill_sequence=["inspect_scene", "locate_object", "start_scan", "plan_orbit", "observe_from_viewpoint"])
                return _agent_result(result, text)
            context = st.selected_context if refers_to_selection(text) else None
            if refers_to_selection(text) and not context:
                return _refused("No retained target is identifiable. Select the object explicitly before continuing.", text)
            result = agent_mod.run(host, text, step=body.step,
                                   continued=st.task_resume if body.step else None, selected_context=context)
            return _agent_result(result, text)
        except GrokError as exc:
            return _refused(str(exc), text)
        except (KeyError, ValueError, RuntimeError) as exc:
            return _refused(str(exc), text)
        finally:
            st.agent_lock.release()

    @app.post("/api/agent/confirm")
    def api_agent_confirm(body: ConfirmReq):
        if not st.agent_lock.acquire(blocking=False):
            raise HTTPException(409, "Another agent task is running")
        try:
            pending = st.pending_motion
            st.pending_motion = None  # Single use, including rejected confirmations.
            if not pending:
                return _refused("No validated motion proposal is awaiting confirmation.", "procedi")
            if pending.get("scan_session") and pending['scan_session'] != st.scans.active:
                return _refused("The scan session changed or finished. Request a new viewpoint.", pending['goal'])
            if pending.get("snapshot_id") != ((st.inventory or {}).get("snapshot") or {}).get("id"):
                return _refused("The scene changed. Select the object and request a new plan.", pending["goal"])
            host = AgentHost(st)
            error = receipt_error(host, pending, body.proposal_id)
            if error:
                return _refused(error, pending["goal"])
            if host.cancel.is_set():
                return _refused("STOP or cancellation requested", pending["goal"])
            result = agent_mod.run(host, pending["goal"], continued=pending, confirm_id=body.proposal_id)
            return _agent_result(result, pending["goal"])
        finally:
            st.agent_lock.release()

    @app.post("/api/agent/cancel")
    def api_agent_cancel():
        st.task_cancel.set()
        st.pending_motion = None
        st.task_resume = None
        if st.arm_job == "approach" and st.arm_thread and st.arm_thread.is_alive():
            st.stop_arm()
        _refused("Task cancelled. Previous motion confirmations are invalid.", "annulla")
        st.audit.event("vision_task", status="cancel_requested")
        return {"cancelled": True}

    def _scan_meta(meta: dict, robot: dict, settled_at: float | None) -> dict:
        packed = dict(meta)
        packed["pose6"] = robot.get("pose6")
        packed["robot_tcp_mm"] = robot.get("xyz_mm")
        packed["robot_rpy_deg"] = None if not robot.get("pose6") else list(robot["pose6"][3:])
        packed["joints_deg"] = robot.get("joints")
        packed["pose_source"] = robot.get("source")
        packed["moving"] = bool(robot.get("moving"))
        packed["state_code"] = robot.get("state_code")
        packed["error_code"] = robot.get("error_code")
        packed["warn_code"] = robot.get("warn_code")
        packed["read_finished_at"] = robot.get("read_finished_at")
        packed["calibration_version"] = st.vision.version
        packed["calibration_valid"] = bool(st.vision.assess(st.frame_size)["valid"])
        if st.vision.hand_eye:
            packed["tool_camera"] = st.vision.hand_eye["matrix"]
        # settled_at is taken before the snapshot, and only when the robot was already still.
        if settled_at is not None:
            packed["stopped_at"] = settled_at
        return packed

    @app.get('/api/integration/evidence')
    def api_m5_evidence():
        from .scan3d.evidence import catalog
        try:
            result = catalog()
            result['current_compatibility'] = st.vision.historical_compatibility(st.frame_size)
            return result
        except FileNotFoundError as exc:
            return {'available':False,'reason':str(exc)}

    @app.get('/api/orbit/settings')
    def api_orbit_settings():
        return st.orbits.public()

    @app.post('/api/orbit/settings')
    def api_orbit_settings_set(body: OrbitReq):
        try:
            result = st.orbits.configure(body.settings,body.confirm)
            st.pending_motion = None
            st.audit.event('setup',action='orbit_settings',settings=result)
            return result
        except (ValueError,TypeError) as exc:
            raise HTTPException(400,str(exc)) from None

    @app.post('/api/orbit/plan')
    def api_orbit_plan():
        context = st.selected_context
        if not context or not context.get('selected'):
            return {'next':None,'candidates':[],'reason':'Select an object first.'}
        host = AgentHost(st)
        task = dict(context)
        located = agent_mod.skill_locate(host,task,'')
        if not located.get('ok'):
            return {'next':None,'candidates':[],'reason':located.get('reason')}
        views = st.scans.list_views() if st.scans.active else []
        return st.orbits.plan(host,task,views)

    @app.post('/api/scan/replay/mouse')
    def api_scan_replay():
        if st.scans.active:
            raise HTTPException(409,'Finish the current scan before importing a recording.')
        try:
            result = st.scans.import_recording()
            st.audit.event('scan',action='recorded_mouse_import',session=result['id'])
            return result
        except (FileNotFoundError,ValueError) as exc:
            raise HTTPException(400,str(exc)) from None

    @app.post("/api/scan/start")
    def api_scan_start(body: ScanStartReq):
        if st.agent_lock.locked():
            raise HTTPException(409, 'Wait for the task to finish or cancel it before changing scan sessions')
        st.invalidate_pending('The scan session changed. Request a new viewpoint.')
        session = st.scans.start(body.object_id)
        if st.vision.intrinsics:
            st.scans.sessions[session["id"]]["intrinsics"] = st.vision.intrinsics
        st.audit.event("scan", action="start", session=session["id"], object_id=body.object_id)
        return session

    @app.post("/api/scan/capture")
    def api_scan_capture():
        try:
            host = AgentHost(st)
            packed, error = agent_mod._settled(host)
            if error:
                raise RuntimeError(error)
            frame, meta = packed
            selected = (st.selected_context or {}).get("selected")
            objects = host.redetect(frame)
            hit = next((obj for obj in objects if selected and obj.get("track_id") == selected.get("track_id")), None)
            meta["quality_required"] = True
            meta["object_visible"] = bool(hit and not hit.get("identity_uncertain")
                                          and hit.get("association") in {"same_view", "geometric"})
            data = host._jpeg or jpeg(frame, 90)
            record = st.scans.capture(data, _scan_meta(meta, host.robot(), meta.get("stopped_at")), hit)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from None
        st.audit.event("scan", action="capture", viewpoint=record["viewpoint_id"],
                       accepted=record["accepted"], reason=record["reject_reason"])
        return {k: record.get(k) for k in ("viewpoint_id", "session_id", "object_id", "accepted",
                                           "reject_reason", "pose_source", "robot_tcp_mm", "joints_deg",
                                           "camera_pose", "calibration_version", "recorded_at",
                                           "sync", "sync_precision_claimed")}

    @app.get("/api/scan")
    def api_scan_list():
        if not st.scans.last or st.scans.last not in st.scans.sessions:
            return {"session": None, "views": []}
        session = st.scans.sessions[st.scans.last]
        return {"session": st.scans.public(session), "views": st.scans.list_views(session["id"])}

    @app.get("/api/scan/candidates")
    def api_scan_candidates():
        point = ((st.last_agent or {}).get("location") or {}).get("point_mm")
        try:
            return st.scans.candidates(point, _observe_clearance())
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/scan/orientation")
    def api_scan_orientation(body: OrientReq):
        if st.arm.status().get('state') == 'moving':
            raise HTTPException(409, 'Stop the arm before reorienting the object')
        try:
            result = st.scans.mark_orientation(body.orientation)
            st.selected_context = None
            st.invalidate_pending('Object reorientation invalidated the proposal. Select the object again.')
            st.task_cancel.set()
            result['requires_reselection'] = True
            result['old_object_frame_valid'] = False
            result['reason'] = 'Object reorientation invalidated localization. Capture the scene and select the object again.'
            st.audit.event('scan', action='orientation', orientation=body.orientation, registration='not run')
            return result
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/scan/finish")
    def api_scan_finish():
        if st.arm_job == 'approach' and st.arm_thread and st.arm_thread.is_alive():
            raise HTTPException(409, 'Wait for the approach to finish or cancel it before finishing the scan')
        try:
            done = st.scans.finish()
            st.invalidate_pending('The scan session finished. Its pending viewpoint is no longer valid.')
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None
        st.audit.event("scan", action="finish", session=done["id"], n_views=done["n_views"],
                       reconstructed=False)
        return done

    @app.post("/api/scan/export")
    def api_scan_export():
        try:
            return st.scans.export_colmap()
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/scan/next")
    def api_scan_next():
        return api_agent_task(AgentReq(text="Scansionalo: pianifica la prossima vista."))

    @app.get("/api/reconstruction/capabilities")
    def api_reconstruction_capabilities():
        return st.reconstruction.capabilities()

    @app.post("/api/reconstruction/start")
    def api_reconstruction_start():
        try:
            job = st.reconstruction.start(st.scans._session(None))
            st.audit.event('scan', action='reconstruction_start', job=job['id'], reconstructed=False)
            return job
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post('/api/reconstruction/replay/mouse')
    def api_reconstruction_replay_mouse():
        if not st.scans.last:
            raise HTTPException(400,'Import the recorded mouse dataset first.')
        session=st.scans._session(st.scans.last)
        if session.get('recorded_benchmark')!='mouse-60':
            raise HTTPException(400,'This replay requires the retained mouse dataset.')
        if session['orientation']!='as_found':
            raise HTTPException(400,'Recorded orientations must remain separate; select the original group.')
        try:
            return st.reconstruction.start(dict(session,postprocessing_replay=True))
        except RuntimeError as exc:
            raise HTTPException(409,str(exc)) from None

    @app.get('/api/reconstruction/jobs')
    def api_reconstruction_jobs():
        return [st.reconstruction.public(job) for job in st.reconstruction.jobs.values()]

    @app.get("/api/reconstruction/{job_id}")
    def api_reconstruction_status(job_id: str):
        try:
            return st.reconstruction.get(job_id)
        except KeyError:
            raise HTTPException(404, 'Unknown reconstruction job') from None

    @app.post("/api/reconstruction/{job_id}/cancel")
    def api_reconstruction_cancel(job_id: str):
        try:
            return st.reconstruction.cancel(job_id)
        except KeyError:
            raise HTTPException(404, 'Unknown reconstruction job') from None

    @app.get("/api/reconstruction/{job_id}/artifacts/{name}")
    def api_reconstruction_artifact(job_id: str, name: str):
        try:
            return FileResponse(st.reconstruction.artifact(job_id, name), filename=name)
        except KeyError:
            raise HTTPException(404, 'Unknown reconstruction artifact') from None

    return app
