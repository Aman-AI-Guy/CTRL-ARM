"""One motion gateway. Every arm move is accepted or rejected here before a driver runs.

Modes, read only from the operator environment (never from a Grok tool or a request field):

- dry_run: validate and return a proposal. The default for a real controller.
- simulation: validate and move a simulated backend only.
- hardware: move a real controller only when the operator set TROVA_MOTION_MODE=hardware
  and TROVA_HARDWARE_UNLOCK=MOVE. Pytest cannot select this mode.

Grok never sets the mode, the workspace, the speed, or the joint limits. A software STOP
holds position; it is not a safety-rated e-stop. The physical e-stop is the real stop.

There is no robot-body collision mesh. A camera sphere is checked only when
TROVA_CAMERA_BODY_MM is set. Otherwise clearance is reported as not configured.
"""

from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

# Hard ceilings. An environment variable may only make these smaller.
HARD_JOG_MAX_MM = 300.0          # the voice script's real cap; its comment used to say 3 mm
JOG_MIN_MM = 1.0
HARD_SEGMENT_MM = 700.0          # one Cartesian command, hardware only (xArm 5 reach)
HARD_SPEED_MM_S = 250.0
HARD_ACCEL_MM_S2 = 1000.0
HARD_JOINT_SPEED_DEG_S = 80.0
SAMPLE_MM = 5.0
SAMPLE_DEG = 2.0
BRINGUP_MAX_SPEED_MM_S = 30.0

# xArm 5 joint ranges in degrees (UFACTORY). A config may shrink a range, never widen it.
XARM5_JOINT_LIMITS_DEG = (
    (-360.0, 360.0),
    (-118.0, 120.0),
    (-225.0, 11.0),
    (-97.0, 180.0),
    (-360.0, 360.0),
)

# Robot base frame. 'Left' is +Y of the base, not the operator's left, until the station is checked.
DIRECTION_AXIS_SIGN = {
    "forward": (0, 1.0),
    "backward": (0, -1.0),
    "left": (1, 1.0),
    "right": (1, -1.0),
    "up": (2, 1.0),
    "down": (2, -1.0),
}

_tls = threading.local()


@dataclass
class MotionTicket:
    max_speed_mm_s: float
    max_accel_mm_s2: float
    max_joint_speed_deg_s: float
    max_joint_accel_deg_s2: float = 400.0


@contextmanager
def motion_ticket(ticket: MotionTicket):
    previous = getattr(_tls, "ticket", None)
    _tls.ticket = ticket
    try:
        yield ticket
    finally:
        _tls.ticket = previous


def require_ticket() -> MotionTicket:
    """Real drivers call this before any motion command. A direct SDK call has no ticket."""
    ticket = getattr(_tls, "ticket", None)
    if ticket is None:
        raise RuntimeError("motion refused: the driver only moves when the motion gateway has approved it")
    return ticket


def approve_bringup_speed(speed_mm_s: float) -> None:
    """Operator bring-up has no workspace yet. The only gateway rule that applies is the speed cap."""
    if float(speed_mm_s) > BRINGUP_MAX_SPEED_MM_S:
        raise RuntimeError(f"bring-up speed {speed_mm_s:g} mm/s exceeds the cap of {BRINGUP_MAX_SPEED_MM_S:g} mm/s")


@dataclass
class KeepOut:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def contains(self, point) -> bool:
        return (self.x[0] <= point[0] <= self.x[1] and self.y[0] <= point[1] <= self.y[1]
                and self.z[0] <= point[2] <= self.z[1])

    def expanded(self, radius: float) -> "KeepOut":
        return KeepOut((self.x[0] - radius, self.x[1] + radius), (self.y[0] - radius, self.y[1] + radius),
                       (self.z[0] - radius, self.z[1] + radius))


@dataclass
class CameraBody:
    """Sphere centre relative to the TCP, in the base frame, while the tool points down."""
    offset_mm: tuple[float, float, float]
    radius_mm: float

    def centre(self, tcp) -> tuple[float, float, float]:
        return tuple(tcp[i] + self.offset_mm[i] for i in range(3))


@dataclass
class MotionResult:
    ok: bool
    status: str
    mode: str
    reason: str
    requested: dict
    proposal: dict
    safety: dict
    allow_motion: bool = False
    software_stop_is_not_safety_rated: bool = True
    collision_avoidance: bool = False
    final_mm: list | None = None
    ticket: MotionTicket | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "status": self.status, "mode": self.mode, "reason": self.reason,
            "requested": self.requested, "proposal": self.proposal, "safety": self.safety,
            "final_mm": self.final_mm, "allow_motion": self.allow_motion,
            "software_stop_is_not_safety_rated": True, "collision_avoidance": False,
        }


def resolve_mode(simulated: bool, *, environ: dict | None = None, under_pytest: bool | None = None) -> tuple[str, str]:
    """Operator environment only. A caller-supplied mode string is not an argument here."""
    env = os.environ if environ is None else environ
    if under_pytest is None:
        under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    raw = (env.get("TROVA_MOTION_MODE") or "").strip().lower()
    unlock = env.get("TROVA_HARDWARE_UNLOCK", "")
    if raw == "hardware":
        if under_pytest:
            return ("simulation" if simulated else "dry_run"), "hardware mode is disabled under pytest"
        if unlock != "MOVE":
            return ("simulation" if simulated else "dry_run"), "hardware mode needs TROVA_HARDWARE_UNLOCK=MOVE, set by the operator"
        if simulated:
            return "dry_run", "hardware mode is refused for a simulated backend"
        return "hardware", ""
    if raw == "dry_run":
        return "dry_run", ""
    if raw == "simulation":
        if not simulated:
            return "dry_run", "simulation mode does not send commands to a real controller"
        return "simulation", ""
    if raw:
        return ("simulation" if simulated else "dry_run"), f"unknown TROVA_MOTION_MODE {raw!r}"
    if simulated:
        return "simulation", ""
    return "dry_run", "a real controller defaults to dry_run"


def _jog_max(env: dict) -> float:
    raw = (env.get("TROVA_JOG_MAX_MM") or "").strip()
    if not raw:
        return HARD_JOG_MAX_MM
    try:
        asked = float(raw)
    except ValueError:
        return HARD_JOG_MAX_MM
    if asked <= 0:
        return HARD_JOG_MAX_MM
    return min(asked, HARD_JOG_MAX_MM)


def _capped(env: dict, name: str, hard: float, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return min(default, hard)
    try:
        asked = float(raw)
    except ValueError:
        return min(default, hard)
    if asked <= 0:
        return min(default, hard)
    return min(asked, hard)


def _parse_keepouts(raw: str) -> list[KeepOut]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "none":
        return []
    boxes = []
    for part in raw.split(";"):
        values = [float(v) for v in part.split(",")]
        if len(values) != 6:
            raise ValueError("TROVA_ARM_KEEP_OUT boxes are xmin,xmax,ymin,ymax,zmin,zmax separated by ';'")
        boxes.append(KeepOut((values[0], values[1]), (values[2], values[3]), (values[4], values[5])))
    return boxes


def _parse_camera(raw: str) -> CameraBody | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    values = [float(v) for v in raw.split(",")]
    if len(values) != 4 or values[3] < 0:
        raise ValueError("TROVA_CAMERA_BODY_MM is x,y,z,radius (mm from the TCP)")
    return CameraBody((values[0], values[1], values[2]), values[3])


def _shrink_joints(raw: str) -> tuple[tuple[float, float], ...] | None:
    raw = (raw or "").strip()
    if not raw:
        return XARM5_JOINT_LIMITS_DEG
    values = [float(v) for v in raw.split(",")]
    if len(values) != 10:
        raise ValueError("TROVA_ARM_JOINT_LIMITS is ten numbers, min,max for joints 1 to 5, inside the xArm 5 ranges")
    out = []
    for i, (hard_lo, hard_hi) in enumerate(XARM5_JOINT_LIMITS_DEG):
        lo, hi = values[2 * i], values[2 * i + 1]
        if lo > hi or lo < hard_lo or hi > hard_hi:
            raise ValueError(f"joint {i + 1} limits must sit inside [{hard_lo:g}, {hard_hi:g}]")
        out.append((lo, hi))
    return tuple(out)


UNCHECKED_OBSTACLES = (
    "The robot links, the dress pack, and the camera cable are not in the model.",
    "Objects on the table are not treated as obstacles.",
    "The table edge beyond the measured plane is not checked.",
    "People in the workspace are not detected.",
    "collision_avoidance stays false. A path whose clearance cannot be checked is refused.",
)

_SAFETY_FIELDS = {"confirm", "limits", "keep_out", "speed_mm_s", "jog_max_mm",
                  "observe_clearance_mm", "camera_body_mm", "joint_limits"}


def safety_report(gateway, observe_clearance_mm) -> dict:
    """What is still missing. This does not invent a measurement."""
    missing = []
    if gateway is None:
        missing.append("motion gateway")
    else:
        if gateway.limits is None:
            missing.append("TROVA_ARM_LIMITS")
        if not getattr(gateway, "keepout_assessed", False):
            missing.append("TROVA_ARM_KEEP_OUT")
        if gateway.camera is None:
            missing.append("TROVA_CAMERA_BODY_MM")
    if observe_clearance_mm is None:
        missing.append("TROVA_OBSERVE_CLEARANCE_MM")
    mode = None if gateway is None else gateway.mode
    keep = None
    if gateway is not None:
        keep = ("not_assessed" if not gateway.keepout_assessed
                else ("none" if not gateway.keepouts else f"{len(gateway.keepouts)} boxes"))
    return {
        "missing": missing, "unchecked": list(UNCHECKED_OBSTACLES),
        "collision_avoidance": False, "mode": mode, "grok_can_edit": False,
        "mode_changed_here": False, "software_stop_is_not_safety_rated": True,
        "physical_estop": "independent of this software",
        "workspace": None if gateway is None else ("configured" if gateway.limits is not None else "not_configured"),
        "workspace_limits": None if gateway is None or gateway.limits is None else gateway.limits.describe(),
        "workspace_frame": "robot_base",
        "workspace_applies_to": "TCP and configured camera sphere, not all robot links",
        "keep_out": keep,
        "clearance_mm": observe_clearance_mm,
        "camera": "configured" if gateway is not None and gateway.camera else "not_configured",
        "max_speed_mm_s": None if gateway is None else gateway.max_speed_mm_s,
        "jog_max_mm": None if gateway is None else gateway.jog_max_mm,
        "joint_limits": None if gateway is None or not gateway.joint_limits else [
            [lo, hi] for lo, hi in gateway.joint_limits],
    }


def apply_operator_safety(gateway, data: dict, *, observe_clearance_mm=None) -> dict:
    """Store measurements the operator typed. The motion mode and the unlock are not fields."""
    unknown = sorted(set(data) - _SAFETY_FIELDS)
    mode = None if gateway is None else gateway.mode
    if unknown:
        return {"ok": False, "applied": [], "mode": mode, "mode_unchanged": True, "collision_avoidance": False,
                "reason": "unknown safety fields: " + ", ".join(unknown)}
    if not data.get("confirm"):
        return {"ok": False, "applied": [], "mode": mode, "mode_unchanged": True, "collision_avoidance": False,
                "reason": "confirm must be true after the values are measured; nothing was changed"}
    if gateway is None:
        return {"ok": False, "applied": [], "mode": mode, "collision_avoidance": False,
                "reason": "no motion gateway"}
    parsed: dict = {}
    try:
        if data.get("limits") is not None:
            from .arm.cartesian import Limits
            limits = Limits.parse(str(data["limits"]))
            if limits is None:
                raise ValueError("TROVA_ARM_LIMITS needs six measured numbers")
            parsed["limits"] = limits
        if data.get("keep_out") is not None:
            raw = str(data["keep_out"]).strip()
            if not raw:
                raise ValueError("keep-out was empty; send measured boxes or the word none")
            parsed["keepouts"] = _parse_keepouts(raw)
        if data.get("speed_mm_s") is not None:
            speed = float(data["speed_mm_s"])
            if not 0 < speed <= HARD_SPEED_MM_S:
                raise ValueError(f"speed must be above 0 and at most {HARD_SPEED_MM_S:g} mm/s")
            parsed["speed"] = speed
        if data.get("jog_max_mm") is not None:
            jog = float(data["jog_max_mm"])
            if not JOG_MIN_MM <= jog <= HARD_JOG_MAX_MM:
                raise ValueError(f"jog max must be from {JOG_MIN_MM:g} to {HARD_JOG_MAX_MM:g} mm")
            parsed["jog"] = jog
        if data.get("camera_body_mm") is not None:
            camera = _parse_camera(str(data["camera_body_mm"]))
            if camera is None:
                raise ValueError("TROVA_CAMERA_BODY_MM needs x,y,z,radius")
            parsed["camera"] = camera
        if data.get("joint_limits") is not None:
            raw = str(data["joint_limits"]).strip()
            if not raw:
                raise ValueError("joint limits were empty; send ten measured numbers inside the xArm 5 ranges")
            parsed["joints"] = _shrink_joints(raw)
        if data.get("observe_clearance_mm") is not None:
            clearance = float(data["observe_clearance_mm"])
            if clearance <= 0:
                raise ValueError("TROVA_OBSERVE_CLEARANCE_MM has to be a measured gap above 0")
            parsed["clearance"] = clearance
    except (TypeError, ValueError) as exc:
        return {"ok": False, "applied": [], "reason": str(exc), "mode": gateway.mode,
                "mode_unchanged": gateway.mode == mode, "collision_avoidance": False}
    applied = []
    if "limits" in parsed:
        gateway.limits = parsed["limits"]
        if gateway.arm is not None and hasattr(gateway.arm, "limits"):
            gateway.arm.limits = parsed["limits"]
        applied.append("TROVA_ARM_LIMITS")
    if "keepouts" in parsed:
        gateway.keepouts = parsed["keepouts"]
        gateway.keepout_assessed = True
        applied.append("TROVA_ARM_KEEP_OUT")
    if "speed" in parsed:
        gateway.max_speed_mm_s = parsed["speed"]
        if gateway.arm is not None and hasattr(gateway.arm, "speed"):
            gateway.arm.speed = min(float(getattr(gateway.arm, "speed", parsed["speed"])), parsed["speed"])
        applied.append("TROVA_ARM_SPEED_MM_S")
    if "jog" in parsed:
        gateway.jog_max_mm = parsed["jog"]
        applied.append("TROVA_JOG_MAX_MM")
    if "camera" in parsed:
        gateway.camera = parsed["camera"]
        applied.append("TROVA_CAMERA_BODY_MM")
    if "joints" in parsed:
        gateway.joint_limits = parsed["joints"]
        applied.append("TROVA_ARM_JOINT_LIMITS")
    clearance = observe_clearance_mm
    if "clearance" in parsed:
        clearance = parsed["clearance"]
        applied.append("TROVA_OBSERVE_CLEARANCE_MM")
    if gateway.mode != mode:
        gateway.mode = mode
    if not applied:
        return {"ok": False, "applied": [], "reason": "send at least one measured value",
                "mode": gateway.mode, "mode_unchanged": True, "collision_avoidance": False}
    report = safety_report(gateway, clearance)
    report.update({"ok": True, "applied": applied, "clearance_mm": clearance, "mode_unchanged": True,
                   "reason": "operator measurements stored in this process"})
    return report


def sample_segment(start, end, step_mm: float = SAMPLE_MM) -> list[tuple[float, float, float]]:
    start, end = tuple(start), tuple(end)
    dist = math.dist(start, end)
    steps = max(1, math.ceil(dist / step_mm)) if dist else 1
    return [tuple(start[i] + (end[i] - start[i]) * k / steps for i in range(3)) for k in range(steps + 1)]


def sample_joints(start, end, step_deg: float = SAMPLE_DEG) -> list[list[float]]:
    span = max((abs(a - b) for a, b in zip(start, end, strict=True)), default=0.0)
    steps = max(1, math.ceil(span / step_deg)) if span else 1
    return [[start[i] + (end[i] - start[i]) * k / steps for i in range(len(start))] for k in range(steps + 1)]


def _round_point(point) -> list[float] | None:
    if point is None:
        return None
    return [round(float(v), 3) for v in point]


def backend_is_simulated(arm) -> bool:
    if arm is None or any(cls.__name__ == "SimArm" for cls in type(arm).__mro__):
        return True
    backend = getattr(arm, "backend", None)
    if backend is None:
        return True
    # Test doubles subclass the fakes (a stop that races the send, a late xArm start).
    if any(cls.__name__ in {"FakeCartesian", "FakeJointBackend"} for cls in type(backend).__mro__):
        return True
    name = type(backend).__name__
    if name in {"URBackend", "EliteBackend"}:
        ip = str(getattr(backend, "ip", ""))
        return ip in {"127.0.0.1", "localhost", "::1"} or ip.startswith("127.")
    if name == "XArmBackend":
        sdk = getattr(backend, "arm", None)
        module = type(sdk).__module__ if sdk is not None else ""
        return not module.startswith("xarm")
    return False


class MotionGateway:
    def __init__(self, arm, *, mode: str, simulated: bool, limits, jog_max_mm: float,
                 max_speed_mm_s: float, max_accel_mm_s2: float, max_joint_speed_deg_s: float,
                 joint_limits, keepouts: list[KeepOut], camera: CameraBody | None, note: str = "",
                 keepout_assessed: bool = False):
        self.arm = arm
        self.simulated = simulated
        self.limits = limits
        self.jog_max_mm = min(float(jog_max_mm), HARD_JOG_MAX_MM)
        self.max_speed_mm_s = min(float(max_speed_mm_s), HARD_SPEED_MM_S)
        self.max_accel_mm_s2 = min(float(max_accel_mm_s2), HARD_ACCEL_MM_S2)
        self.max_joint_speed_deg_s = min(float(max_joint_speed_deg_s), HARD_JOINT_SPEED_DEG_S)
        self.joint_limits = joint_limits
        self.keepouts = list(keepouts)
        self.keepout_assessed = bool(keepout_assessed)
        self.joint_geometry = None  # Operator-qualified robot/head geometry, never supplied by Grok.
        self.camera = camera
        self.note = note
        if mode == "hardware" and (simulated or os.environ.get("PYTEST_CURRENT_TEST")):
            mode = "dry_run" if not simulated else "simulation"
            self.note = (note + "; " if note else "") + "hardware execution is refused in this process"
        if mode not in {"dry_run", "simulation", "hardware"}:
            raise ValueError(f"unknown motion mode {mode!r}")
        self.mode = mode

    @classmethod
    def for_arm(cls, arm, *, mode: str | None = None, environ: dict | None = None) -> "MotionGateway":
        env = os.environ if environ is None else environ
        simulated = backend_is_simulated(arm)
        # The test suite always drives simulated arms. An operator .env must not freeze them in dry_run.
        if os.environ.get("PYTEST_CURRENT_TEST") and simulated and mode is None and environ is None:
            chosen, note = "simulation", ""
        elif mode is None:
            chosen, note = resolve_mode(simulated, environ=env)
        else:
            chosen, note = mode, ""
        limits = getattr(arm, "limits", None)
        speed = float(getattr(arm, "speed", None) or env.get("TROVA_ARM_SPEED_MM_S") or 100)
        backend = getattr(arm, "backend", None)
        # xArm accel is mm/s^2. UR and Elite store m/s^2 under the same attribute name.
        accel = float(backend.accel) if type(backend).__name__ == "XArmBackend" else 500.0
        joint_speed = float(getattr(arm, "joint_speed", None) or env.get("TROVA_ARM_JOINT_SPEED_DEG_S") or 20)
        joints = None
        if not simulated and type(getattr(arm, "backend", None)).__name__ == "XArmBackend":
            joints = _shrink_joints(env.get("TROVA_ARM_JOINT_LIMITS", ""))
        # Pytest must not inherit a developer's keep-out or camera sphere. A test sets those on the gateway.
        testing = bool(os.environ.get("PYTEST_CURRENT_TEST")) and environ is None
        keep_raw = "" if testing else (env.get("TROVA_ARM_KEEP_OUT") or "")
        return cls(
            arm, mode=chosen, simulated=simulated, limits=limits, jog_max_mm=_jog_max({} if testing else env),
            max_speed_mm_s=_capped({} if testing else env, "TROVA_ARM_MAX_SPEED_MM_S", HARD_SPEED_MM_S, speed),
            max_accel_mm_s2=_capped({} if testing else env, "TROVA_ARM_MAX_ACCEL_MM_S2", HARD_ACCEL_MM_S2, accel),
            max_joint_speed_deg_s=_capped(
                {} if testing else env, "TROVA_ARM_MAX_JOINT_SPEED_DEG_S", HARD_JOINT_SPEED_DEG_S, joint_speed),
            joint_limits=joints, keepouts=[] if testing else _parse_keepouts(keep_raw),
            keepout_assessed=bool(keep_raw.strip()) and not testing,
            camera=None if testing else _parse_camera(env.get("TROVA_CAMERA_BODY_MM", "")), note=note,
        )

    @classmethod
    def standalone(cls, *, environ: dict | None = None) -> "MotionGateway":
        """No robot attached. Used by the voice CLI before any connection."""
        env = os.environ if environ is None else environ
        mode, note = resolve_mode(False, environ=env)
        keep_raw = env.get("TROVA_ARM_KEEP_OUT") or ""
        return cls(
            None, mode=mode, simulated=False, limits=None, jog_max_mm=_jog_max(env),
            max_speed_mm_s=_capped(env, "TROVA_ARM_MAX_SPEED_MM_S", HARD_SPEED_MM_S, 100),
            max_accel_mm_s2=_capped(env, "TROVA_ARM_MAX_ACCEL_MM_S2", HARD_ACCEL_MM_S2, 50),
            max_joint_speed_deg_s=_capped(env, "TROVA_ARM_MAX_JOINT_SPEED_DEG_S", HARD_JOINT_SPEED_DEG_S, 20),
            joint_limits=XARM5_JOINT_LIMITS_DEG, keepouts=_parse_keepouts(keep_raw),
            keepout_assessed=bool(keep_raw.strip()),
            camera=_parse_camera(env.get("TROVA_CAMERA_BODY_MM", "")), note=note,
        )

    def public(self) -> dict:
        return {
            "mode": self.mode, "note": self.note, "jog_max_mm": self.jog_max_mm,
            "max_speed_mm_s": self.max_speed_mm_s, "collision_avoidance": False,
            "clearance": "camera_sphere" if self.camera else "not_configured",
            "workspace": "configured" if self.limits is not None else "not_configured",
            "keep_out": ("not_assessed" if not self.keepout_assessed
                         else ("none" if not self.keepouts else f"{len(self.keepouts)} boxes")),
            "software_stop_is_not_safety_rated": True,
        }

    def current_pose(self):
        arm = self.arm
        if arm is None:
            return None
        backend = getattr(arm, "backend", None)
        if backend is not None and hasattr(backend, "tcp_mm"):
            try:
                return tuple(float(v) for v in backend.tcp_mm())
            except Exception:  # noqa: BLE001 - a dead controller is "no pose", and the move is refused
                return None
        tip = getattr(arm, "tip", None)
        if tip is not None and len(tip) >= 2:
            return (float(tip[0]), float(tip[1]), float(getattr(arm, "z", 0.0)))
        return None

    def current_joints(self):
        backend = getattr(self.arm, "backend", None)
        if backend is None or not hasattr(backend, "joints"):
            return None
        try:
            return [float(v) for v in backend.joints()]
        except Exception:  # noqa: BLE001
            return None

    def _ticket(self, speed_mm_s: float, joint_speed: float | None = None) -> MotionTicket:
        cap = self.max_speed_mm_s if self.mode == "hardware" else max(float(speed_mm_s), self.max_speed_mm_s)
        joint_cap = self.max_joint_speed_deg_s if self.mode == "hardware" else max(
            float(joint_speed or self.max_joint_speed_deg_s), self.max_joint_speed_deg_s)
        return MotionTicket(cap, self.max_accel_mm_s2 if self.mode == "hardware" else max(self.max_accel_mm_s2, 1e6),
                            joint_cap)

    def _base(self, requested: dict) -> dict:
        return {
            "workspace": "configured" if self.limits is not None else "not_configured",
            "jog_max_mm": self.jog_max_mm, "max_speed_mm_s": self.max_speed_mm_s,
            "max_accel_mm_s2": self.max_accel_mm_s2, "clearance": "not_configured" if self.camera is None else "camera_sphere",
            "keep_out": ("not_assessed" if not self.keepout_assessed
                         else ("none" if not self.keepouts else f"{len(self.keepouts)} boxes")),
            "collision_avoidance": False, "software_stop_is_not_safety_rated": True,
            "joint_limits": "xarm5" if self.joint_limits else "not_applied", "note": self.note,
        }

    def _finish(self, *, ok: bool, status: str, reason: str, requested: dict, proposal: dict,
                safety: dict, allow: bool, speed: float, joint_speed: float | None = None) -> MotionResult:
        ticket = self._ticket(speed, joint_speed) if allow else None
        return MotionResult(ok, status, self.mode, reason, requested, proposal, safety, allow, ticket=ticket)

    def _stopped_or_fault(self, requested: dict) -> MotionResult | None:
        stop = getattr(self.arm, "_stop", None)
        if stop is not None and stop.is_set():
            return self._finish(ok=False, status="stopped", speed=0,
                                reason="STOP is latched. A software stop holds position; it is not the physical emergency stop.",
                                requested=requested, proposal={}, safety=self._base(requested), allow=False)
        if getattr(self.arm, "state", "") == "error":
            return self._finish(ok=False, status="fault", speed=0,
                                reason=getattr(self.arm, "last_error", None) or "the controller is in a fault state",
                                requested=requested, proposal={}, safety=self._base(requested), allow=False)
        return None

    def _limits_and_body(self, samples: list[tuple], safety: dict, *, check_camera: bool = True) -> str | None:
        if self.limits is not None:
            for point in samples:
                err = self.limits.check(*point)
                if err:
                    where = ", ".join(f"{n}={v:.0f}" for n, v in zip("xyz", point, strict=True))
                    safety["workspace"] = "failed"
                    return f"{err} (sampled along the path at {where})"
            safety["workspace"] = "passed"
        elif self.mode == "hardware":
            safety["workspace"] = "not_configured"
            return "workspace limits are not configured (set TROVA_ARM_LIMITS); hardware motion is refused"
        for box in self.keepouts:
            for point in samples:
                if box.contains(point):
                    where = ", ".join(f"{n}={v:.0f}" for n, v in zip("xyz", point, strict=True))
                    safety["keep_out"] = "failed"
                    return f"the path enters a keep-out at {where}"
        if self.keepouts:
            safety["keep_out"] = "passed"
        if not check_camera:
            # Rotated link/head spheres were checked by the joint-geometry validator.
            safety['clearance'] = 'rotated_CAD_head_spheres'
            return None
        if self.camera is None:
            safety["clearance"] = "not_configured"
            return None
        centres = [self.camera.centre(point) for point in samples]
        if self.limits is not None:
            radius = self.camera.radius_mm
            for centre in centres:
                if self.limits.radius_mm is not None and math.hypot(*centre[:2]) + radius > self.limits.radius_mm:
                    safety["clearance"] = "failed"
                    return "the camera sphere exceeds the cylindrical workspace radius"
                for name, value, (lo, hi) in (("x", centre[0], self.limits.x), ("y", centre[1], self.limits.y),
                                              ("z", centre[2], self.limits.z)):
                    if not lo + radius <= value <= hi - radius:
                        safety["clearance"] = "failed"
                        return (f"camera body {name}={value:.0f} mm is outside the workspace inset by "
                                f"{radius:.0f} mm [{lo + radius:.0f}, {hi - radius:.0f}]")
        for box in self.keepouts:
            grown = box.expanded(self.camera.radius_mm)
            for centre in centres:
                if grown.contains(centre):
                    safety["clearance"] = "failed"
                    return "the camera sphere intersects a keep-out"
        safety["clearance"] = "camera_sphere"
        return None

    def authorize_cartesian(self, target, speed_mm_s: float, *, start=None, source: str = "cartesian",
                            requested: dict | None = None) -> MotionResult:
        target = tuple(float(v) for v in target)
        speed_mm_s = float(speed_mm_s)
        requested = dict(requested or {})
        requested.setdefault("target_mm", _round_point(target))
        requested.setdefault("speed_mm_s", speed_mm_s)
        requested.setdefault("source", source)
        blocked = self._stopped_or_fault(requested)
        if blocked:
            return blocked
        safety = self._base(requested)
        if self.mode == "hardware" and speed_mm_s > self.max_speed_mm_s:
            return self._finish(ok=False, status="rejected", allow=False, speed=speed_mm_s, safety=safety,
                                requested=requested, proposal={"target_mm": _round_point(target)},
                                reason=(f"requested speed {speed_mm_s:g} mm/s exceeds the hardware limit "
                                        f"{self.max_speed_mm_s:g} mm/s"))
        if start is None:
            start = self.current_pose()
        if start is None and self.mode == "hardware":
            return self._finish(ok=False, status="rejected", allow=False, speed=speed_mm_s, safety=safety,
                                requested=requested, proposal={"target_mm": _round_point(target)},
                                reason="no current pose, so the trajectory cannot be checked")
        samples = sample_segment(start, target) if start is not None else [target]
        safety["trajectory_samples"] = len(samples)
        if start is not None and self.mode == "hardware" and math.dist(start, target) > HARD_SEGMENT_MM:
            return self._finish(ok=False, status="rejected", allow=False, speed=speed_mm_s, safety=safety,
                                requested=requested, proposal={"start_mm": _round_point(start),
                                                              "target_mm": _round_point(target)},
                                reason=(f"requested move of {math.dist(start, target):.0f} mm exceeds the "
                                        f"{HARD_SEGMENT_MM:.0f} mm per-command limit"))
        err = self._limits_and_body(samples, safety)
        proposal = {"start_mm": _round_point(start), "target_mm": _round_point(target),
                    "samples": len(samples), "source": source,
                    "distance_mm": None if start is None else round(math.dist(start, target), 3)}
        if err:
            return self._finish(ok=False, status="rejected", reason=err, requested=requested, proposal=proposal,
                                safety=safety, allow=False, speed=speed_mm_s)
        if self.mode == "dry_run":
            return self._finish(ok=True, status="dry_run", allow=False, speed=speed_mm_s, safety=safety,
                                requested=requested, proposal=proposal,
                                reason="dry-run: the path passed the configured checks and was not sent to the robot")
        return self._finish(ok=True, status="accepted", allow=True, speed=speed_mm_s, safety=safety,
                            requested=requested, proposal=proposal, reason="accepted")

    def authorize_polyline(self, points, speed_mm_s: float, *, source: str = "waypoints") -> MotionResult:
        points = [tuple(float(v) for v in p) for p in points]
        if len(points) < 2:
            return self.authorize_cartesian(points[0] if points else (0, 0, 0), speed_mm_s, source=source)
        last = None
        requested = {"source": source, "waypoints": [_round_point(p) for p in points]}
        for index in range(len(points) - 1):
            last = self.authorize_cartesian(points[index + 1], speed_mm_s, start=points[index], source=source,
                                           requested=requested)
            # A dry-run still has to walk the whole path: a later segment may leave the box.
            if last.status != "dry_run" and not last.allow_motion:
                return last
        return last

    def authorize_joints(self, target_q, speed_deg_s: float, *, source: str = "joints", geometry=None,
                         start_q=None) -> MotionResult:
        target_q = [float(v) for v in target_q]
        speed_deg_s = float(speed_deg_s)
        requested = {"joints_deg": [round(v, 3) for v in target_q], "joint_speed_deg_s": speed_deg_s, "source": source}
        if not all(math.isfinite(v) for v in target_q) or not math.isfinite(speed_deg_s) or speed_deg_s <= 0:
            return self._finish(ok=False,status='rejected',reason='Joint target/speed must be finite, with positive speed.',
                                requested=requested,proposal={},safety=self._base(requested),allow=False,speed=0)
        blocked = self._stopped_or_fault(requested)
        if blocked:
            return blocked
        safety = self._base(requested)
        if self.mode == "hardware" and speed_deg_s > self.max_joint_speed_deg_s:
            return self._finish(ok=False, status="rejected", allow=False, speed=0, joint_speed=speed_deg_s,
                                requested=requested, proposal={}, safety=safety,
                                reason=(f"requested joint speed {speed_deg_s:g} deg/s exceeds the limit "
                                        f"{self.max_joint_speed_deg_s:g} deg/s"))
        geometry = geometry if geometry is not None else self.joint_geometry
        if source == 'scan_orbit' and geometry is None:
            return self._finish(ok=False, status='rejected', reason='Scan orbit has no qualified robot geometry.',
                                requested=requested, proposal={}, safety=safety, allow=False, speed=0)
        current = self.current_joints() if start_q is None else start_q
        if source == 'scan_orbit' and current is None:
            return self._finish(ok=False, status='rejected', reason='Current joints are unavailable.',
                                requested=requested, proposal={}, safety=safety, allow=False, speed=0)
        samples_q = sample_joints(current, target_q) if current is not None and len(current) == len(target_q) else [target_q]
        if any(not math.isfinite(value) for pose in samples_q for value in pose):
            return self._finish(ok=False, status='rejected', reason='Joint path contains non-finite values.',
                                requested=requested, proposal={}, safety=safety, allow=False, speed=0)
        if geometry is not None:
            if speed_deg_s > geometry.settings.joint_speed_deg_s:
                return self._finish(ok=False, status='rejected', reason='Orbit exceeds its measured joint speed setting.',
                                    requested=requested, proposal={}, safety=safety, allow=False, speed=0)
            if current is None or len(current) != 5 or len(target_q) != 5:
                return self._finish(ok=False, status='rejected', reason='Orbit requires five current and target joints.',
                                    requested=requested, proposal={}, safety=safety, allow=False, speed=0)
            jump = max(abs(a-b) for a,b in zip(current,target_q))
            reference = geometry.cable_reference_deg if geometry.cable_reference_deg is not None else current[4]
            if jump > geometry.settings.max_joint_step_deg or abs(target_q[4]-reference) > geometry.settings.cable_span_deg:
                return self._finish(ok=False, status='rejected', reason='Orbit exceeds the joint jump or cable budget.',
                                    requested=requested, proposal={}, safety=safety, allow=False, speed=0)
            for pose in samples_q:
                hits = geometry.check(pose)
                if hits:
                    safety['robot_link_geometry'] = 'failed'
                    return self._finish(ok=False, status='rejected', reason='Robot/head geometry: ' + ', '.join(hits),
                                        requested=requested, proposal={}, safety=safety, allow=False, speed=0)
            safety['robot_link_geometry'] = 'CAD-derived spheres, sampled 2 deg; limited self-pair set'
            safety['continuous_collision_checked'] = False
        if self.joint_limits is not None:
            if len(target_q) != len(self.joint_limits):
                return self._finish(ok=False, status="rejected", allow=False, speed=0, joint_speed=speed_deg_s,
                                    requested=requested, proposal={}, safety=safety,
                                    reason=f"expected {len(self.joint_limits)} joints, got {len(target_q)}")
            for pose in samples_q:
                for i, (angle, (lo, hi)) in enumerate(zip(pose, self.joint_limits, strict=True)):
                    if not lo <= angle <= hi:
                        safety["joint_limits"] = "failed"
                        return self._finish(
                            ok=False, status="rejected", allow=False, speed=0, joint_speed=speed_deg_s,
                            requested=requested, proposal={"joints_deg": [round(v, 2) for v in pose]}, safety=safety,
                            reason=f"joint pose: joint {i + 1} = {angle:.0f} deg is outside [{lo:.0f}, {hi:.0f}]")
            safety["joint_limits"] = "passed"
        backend = getattr(self.arm, "backend", None)
        fk = getattr(backend, "fk_mm", None)
        cartesian_samples = []
        if fk is not None:
            try:
                cartesian_samples = [tuple(float(v) for v in fk(pose)[:3]) for pose in samples_q]
            except NotImplementedError:
                fk = None
            except Exception as exc:  # noqa: BLE001
                return self._finish(ok=False, status="rejected", allow=False, speed=0, joint_speed=speed_deg_s,
                                    requested=requested, proposal={}, safety=safety,
                                    reason=f"joint pose: forward kinematics failed: {exc}")
        if fk is None and self.mode == "hardware":
            return self._finish(ok=False, status="rejected", allow=False, speed=0, joint_speed=speed_deg_s,
                                requested=requested, proposal={}, safety=safety,
                                reason="forward kinematics is not available, so the joint path cannot be checked")
        if cartesian_samples:
            if geometry is not None:
                from .scan3d.flange import flange_pose
                expected = [(flange_pose(q) @ geometry.tcp)[:3,3] for q in samples_q]
                if self.mode == 'hardware' and any(math.dist(a,b) > 2 for a,b in zip(expected,cartesian_samples)):
                    return self._finish(ok=False,status='rejected',reason='Controller/local FK or TCP offset differs by more than 2 mm.',
                                        requested=requested,proposal={},safety=safety,allow=False,speed=0)
                ratios = [math.dist(a,b) / max(1e-9,max(abs(x-y) for x,y in zip(q0,q1)))
                          for a,b,q0,q1 in zip(expected,expected[1:],samples_q,samples_q[1:])]
                if ratios and max(ratios) * speed_deg_s > self.max_speed_mm_s + .001:
                    return self._finish(ok=False,status='rejected',reason='Sampled TCP velocity exceeds the workspace speed limit.',
                                        requested=requested,proposal={},safety=safety,allow=False,speed=0)
                safety['tcp_velocity'] = 'sampled_bound'
            err = self._limits_and_body(cartesian_samples, safety, check_camera=geometry is None)
            if err:
                return self._finish(ok=False, status="rejected", allow=False, speed=0, joint_speed=speed_deg_s,
                                    requested=requested, proposal={"fk_mm": _round_point(cartesian_samples[-1])},
                                    safety=safety, reason=f"joint pose: {err}")
        proposal = {"joints_deg": [round(v, 3) for v in target_q], "samples": len(samples_q), "source": source}
        if geometry is not None:
            proposal['joint_path'] = samples_q
            proposal['geometry'] = geometry.signature()
            if not geometry.qualified:
                return self._finish(ok=True,status='dry_run',allow=False,speed=0,joint_speed=speed_deg_s,
                                    requested=requested,proposal=proposal,safety=safety,
                                    reason='Offline geometry proposal only; current physical model binding is unqualified.')
        if self.mode == "dry_run":
            return self._finish(ok=True, status="dry_run", allow=False, speed=0, joint_speed=speed_deg_s,
                                requested=requested, proposal=proposal, safety=safety,
                                reason="dry-run: the joint path passed the configured checks and was not sent")
        result = self._finish(ok=True, status="accepted", allow=True, speed=0, joint_speed=speed_deg_s,
                              requested=requested, proposal=proposal, safety=safety, reason="accepted")
        if geometry is not None and result.ticket:
            result.ticket.max_joint_accel_deg_s2 = min(result.ticket.max_joint_accel_deg_s2, geometry.settings.joint_accel_deg_s2)
        return result

    def execute_joints(self, target_q, speed_deg_s, *, geometry):
        """Use the existing driver and joint reached/STOP checks, after gateway authorization."""
        decision = self.authorize_joints(target_q,speed_deg_s,source='scan_orbit',geometry=geometry)
        if not decision.allow_motion:
            return {'ok':False,'reason':decision.reason,'motion':decision.as_dict()}
        previous_geometry,self.joint_geometry = self.joint_geometry,geometry
        arm=self.arm
        previous_speed=arm.joint_speed
        arm.joint_speed=min(float(speed_deg_s),self.max_joint_speed_deg_s)
        try:
            ok=bool(arm.go_joints(list(target_q)))
            return {'ok':ok,'reason':('joint viewpoint reached' if ok else arm.last_error or 'Joint movement failed'),
                    'motion':decision.as_dict(),'final_mm':self.current_pose(),'final_joints':self.current_joints()}
        finally:
            arm.joint_speed=previous_speed
            self.joint_geometry=previous_geometry

    def authorize_freedrive(self, on: bool) -> MotionResult:
        requested = {"freedrive": bool(on), "source": "freedrive"}
        blocked = self._stopped_or_fault(requested)
        if blocked and on:
            return blocked
        safety = self._base(requested)
        if self.mode == "dry_run":
            return self._finish(ok=True, status="dry_run", allow=False, speed=0, requested=requested, proposal={},
                                safety=safety, reason="dry-run: freedrive was not enabled")
        if self.mode == "hardware" and self.limits is None:
            return self._finish(ok=False, status="rejected", allow=False, speed=0, requested=requested, proposal={},
                                safety=safety, reason="workspace limits are not configured; freedrive is refused")
        return self._finish(ok=True, status="accepted", allow=True, speed=0, requested=requested, proposal={},
                            safety=safety, reason="accepted")

    def plan_relative(self, direction: str, distance_mm: float, *, allowed_directions=None, pose=None,
                      execute: bool = False, overrides: dict | None = None) -> MotionResult:
        """Validate a spoken relative move. The requested distance is never rewritten."""
        try:
            distance = float(distance_mm)
        except (TypeError, ValueError):
            distance = float("nan")
        requested = {"action": "move", "direction": direction, "distance_mm": distance_mm, "source": "jog"}
        safety = self._base(requested)
        if overrides:
            names = ", ".join(sorted(overrides))
            return self._finish(ok=False, status="rejected", allow=False, speed=0, requested=requested, proposal={},
                                safety=safety, reason=f"safety parameters cannot be set by a command: {names}")
        if direction not in DIRECTION_AXIS_SIGN:
            return self._finish(ok=False, status="rejected", allow=False, speed=0, requested=requested, proposal={},
                                safety=safety, reason=f"unknown direction {direction!r}")
        if allowed_directions is not None and direction not in allowed_directions:
            return self._finish(ok=False, status="rejected", allow=False, speed=0, requested=requested, proposal={},
                                safety=safety,
                                reason=(f"{direction} is not enabled for this command. The first physical jog allows "
                                        "only backward (-X in the robot base frame) until each direction is checked."))
        if not math.isfinite(distance) or distance < JOG_MIN_MM or distance > self.jog_max_mm:
            return self._finish(
                ok=False, status="rejected", allow=False, speed=0, requested=requested, proposal={}, safety=safety,
                reason=(f"requested {distance_mm} mm is outside the jog limit {JOG_MIN_MM:g}.."
                        f"{self.jog_max_mm:g} mm"))
        axis, sign = DIRECTION_AXIS_SIGN[direction]
        delta = [0.0, 0.0, 0.0]
        delta[axis] = sign * distance
        if type(self.arm).__name__ == "SimArm" and axis == 2:
            return self._finish(ok=False, status="rejected", allow=False, speed=0, requested=requested,
                                proposal={"delta_mm": delta, "frame": "base"}, safety=safety,
                                reason="the simulated table arm has no vertical axis")
        if pose is None and self.arm is not None:
            pose = self.current_pose()
        backend = getattr(self.arm, "backend", None)
        if self.mode == "hardware" and type(backend).__name__ == "XArmBackend":
            try:
                actual = backend.tcp_pose()
                yaw_gap = abs((actual[5] - backend.yaw + 180) % 360 - 180)
                aligned = abs(abs(actual[3]) - 180) < 1 and abs(actual[4]) < 1 and yaw_gap < 0.1
            except Exception:
                aligned = False
            if not aligned:
                return self._finish(ok=False, status="rejected", allow=False, speed=0,
                                    requested=requested, proposal={}, safety=safety,
                                    reason="Voice jog refused: this driver would reorient the xArm flange. "
                                    "Use Studio to establish the pointing orientation (roll +/-180, pitch 0, "
                                    "and the yaw read at app start) before jogging; no automatic reorientation.")
        proposal = {"delta_mm": [round(v, 3) for v in delta], "frame": "base",
                    "direction_note": "robot base frame, not the operator's left and right"}
        if pose is None:
            proposal["target_mm"] = None
            if self.mode == "hardware":
                return self._finish(ok=False, status="rejected", allow=False, speed=self.max_speed_mm_s,
                                    requested=requested, proposal=proposal, safety=safety,
                                    reason="no current pose, so the jog cannot be checked against the workspace")
            safety["workspace"] = "not_checked_no_pose"
            return self._finish(ok=True, status="dry_run", allow=False, speed=self.max_speed_mm_s,
                                requested=requested, proposal=proposal, safety=safety,
                                reason=("dry-run: no robot pose was read, so the workspace was not checked. "
                                        f"Proposed {direction} {distance:g} mm in the robot base frame."))
        target = tuple(pose[i] + delta[i] for i in range(3))
        decision = self.authorize_cartesian(target, min(self.max_speed_mm_s, 100.0), start=tuple(pose), source="jog",
                                           requested=requested)
        decision.proposal["delta_mm"] = proposal["delta_mm"]
        decision.proposal["frame"] = "base"
        decision.proposal["direction_note"] = proposal["direction_note"]
        if execute and decision.allow_motion:
            ok = self._execute_cartesian(target)
            decision.final_mm = _round_point(self.current_pose())
            if ok:
                decision.ok, decision.status = True, "executed"
                decision.reason = f"moved {direction} {distance:g} mm"
            else:
                decision.ok, decision.status, decision.allow_motion = False, "error", False
                decision.reason = getattr(self.arm, "last_error", None) or "the arm did not complete the jog"
        return decision

    def _execute_cartesian(self, target) -> bool:
        if self.mode == "hardware" and (self.simulated or os.environ.get("PYTEST_CURRENT_TEST")):
            raise RuntimeError("refusing hardware execution")
        arm = self.arm
        if arm is None:
            return False
        # CartesianArm._go keeps Z. SimArm and the taught grid point in the table plane.
        # TaughtGridArm._go is joint-space and must not be called with millimetres.
        if type(arm).__name__ == "CartesianArm":
            if self.mode == "hardware":
                previous = arm.speed
                arm.speed = min(float(previous), self.max_speed_mm_s)
                try:
                    return bool(arm._go(*target))
                finally:
                    arm.speed = previous
            return bool(arm._go(*target))
        if hasattr(arm, "point_at"):
            return bool(arm.point_at(target[0], target[1]))
        return False
