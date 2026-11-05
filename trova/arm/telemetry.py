"""Read-only pose and joint telemetry for a UFACTORY xArm 5.

This object is not an arm driver. It never enables the motors, clears errors, changes the
TCP offset, or sends a motion command. Motion stays on MotionGateway and the arm driver.
A missing or refused controller is reported; it is not replaced with a simulated pose.
"""

from __future__ import annotations

import socket
import threading
import time

READ_METHODS = frozenset({
    "get_position",
    "get_servo_angle",
    "get_state",
    "get_is_moving",
    "get_err_warn_code",
    "get_inverse_kinematics",
    "get_forward_kinematics",
    "disconnect",
})

# Names that would change the controller. The guard refuses these and every other name.
WRITE_METHODS = frozenset({
    "motion_enable", "set_state", "set_position", "set_servo_angle", "set_servo_angle_j",
    "set_tcp_offset", "clean_error", "clean_warn", "set_mode", "move_gohome", "reset",
    "set_position_aa", "set_servo_cartesian", "clean_conf", "save_conf", "set_tcp_load",
    "set_world_offset", "set_collision_sensitivity", "set_reduced_mode", "set_fence_mode",
    "set_joint_maxacc", "emergency_stop", "set_pause_time",
})


class TelemetryWriteRefused(RuntimeError):
    """A telemetry object was asked to change the robot."""


class GuardedSDK:
    """Forwards a fixed set of reads. Any other attribute raises before the SDK sees it."""

    def __init__(self, raw):
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "calls", [])

    def read_property(self, name):
        if name not in {'mode', 'tcp_offset', 'world_offset'}:
            raise TelemetryWriteRefused(f'{name} is not an allowed telemetry property')
        self.calls.append('read_property:'+name)
        value = getattr(self._raw,name,None)
        return list(value) if isinstance(value,(tuple,list)) else value

    def __getattr__(self, name: str):
        if name not in READ_METHODS:
            def refused(*_args, **_kwargs):
                raise TelemetryWriteRefused(
                    f"{name} is not a telemetry read. This connection does not enable motors, "
                    "change the TCP, clear errors, or move the arm.")
            return refused
        method = getattr(self._raw, name)

        def wrapped(*args, **kwargs):
            self.calls.append(name)
            return method(*args, **kwargs)

        return wrapped


def _pair(result):
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    return 0, result


def _failed(message: str) -> dict:
    return {
        "xyz_mm": None, "pose6": None, "joints": None, "joints_all": None,
        "moving": None, "source": "unavailable", "simulated": False, "error": message,
        "state_code": None, "error_code": None, "warn_code": None,
        "position_code": None, "joints_code": None, "state_read_code": None,
        "read_started_at": None, "read_finished_at": None,
        "motion_sent": False, "motors_enabled_by_trova": False, "tcp_changed": False,
        "movements_enabled": False, "sync_precision_claimed": False,
    }


class XArmTelemetry:
    """One already-open SDK. Tests pass a fake. Live code uses LazyXArmTelemetry."""

    def __init__(self, sdk, ip: str = ""):
        self.sdk = sdk if isinstance(sdk, GuardedSDK) else GuardedSDK(sdk)
        self.ip = ip
        self._lock = threading.Lock()

    def read(self) -> dict:
        """One pose, the joint angles, and the controller state. No actuator call."""
        with self._lock:
            started = time.time()
            try:
                position_code, pose = _pair(self.sdk.get_position())
                joints_code, joints = _pair(self.sdk.get_servo_angle())
                state_code, state = _pair(self.sdk.get_state())
                error_code, err = _pair(self.sdk.get_err_warn_code(show=False))
                moving = bool(self.sdk.get_is_moving())
            except TelemetryWriteRefused:
                raise
            except Exception as exc:  # noqa: BLE001 - the caller shows the controller error
                return _failed(str(exc))
            finished = time.time()
        pose6 = None
        if position_code == 0 and pose is not None and len(list(pose)) >= 6:
            pose6 = [round(float(v), 3) for v in list(pose)[:6]]
        joint_list = None
        if joints_code == 0 and joints is not None:
            joint_list = [round(float(v), 3) for v in list(joints)]
        warn = None
        controller_error = None
        if error_code == 0 and isinstance(err, (list, tuple)) and len(err) >= 2:
            controller_error = int(err[0])
            warn = int(err[1])
        elif error_code == 0 and isinstance(err, (list, tuple)) and err:
            controller_error = int(err[0])
        message = None
        if pose6 is None:
            message = f"xArm get_position code {position_code}"
        if controller_error:
            message = f"controller error code {controller_error}"
        return {
            "xyz_mm": None if pose6 is None else pose6[:3],
            "pose6": pose6,
            "joints": None if joint_list is None else joint_list[:5],
            "joints_all": joint_list,
            "moving": moving,
            "source": "telemetry",
            "simulated": False,
            "error": message,
            "state_code": None if state_code != 0 else int(state),
            "error_code": controller_error,
            "warn_code": warn,
            "position_code": int(position_code),
            "joints_code": int(joints_code),
            "state_read_code": int(state_code),
            "error_read_code": int(error_code),
            "read_started_at": started,
            "read_finished_at": finished,
            "read_s": round(finished - started, 4),
            "ip": self.ip,
            "motion_sent": False,
            "motors_enabled_by_trova": False,
            "tcp_changed": False,
            "movements_enabled": False,
            "sync_precision_claimed": False,
        }

    def configuration(self):
        with self._lock:
            return {name:self.sdk.read_property(name) for name in ('mode','tcp_offset','world_offset')}

    def forward(self, joints):
        import math
        q = [float(v) for v in joints]
        if len(q) != 5 or not all(math.isfinite(v) for v in q):
            raise ValueError('Five finite joint angles in degrees are required')
        with self._lock:
            code, pose = _pair(self.sdk.get_forward_kinematics(q,input_is_radian=False,return_is_radian=False))
        if code != 0 or not pose or len(pose)<6:
            return {'ok':False,'pose6':None,'reason':f'Controller FK query returned {code}','motion_sent':False}
        return {'ok':True,'pose6':[float(v) for v in pose[:6]],'motion_sent':False}

    def inverse(self, pose6, ref_joints=None) -> dict:
        """Ask the controller for joint angles. This does not move the arm."""
        from ..motion import XARM5_JOINT_LIMITS_DEG
        query = [float(v) for v in list(pose6)[:6]]
        ref = None if not ref_joints else [float(v) for v in list(ref_joints)]
        with self._lock:
            try:
                code, angles = _pair(self.sdk.get_inverse_kinematics(
                    query, input_is_radian=False, return_is_radian=False, ref_angles=ref))
            except TelemetryWriteRefused:
                raise
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "ik_verified": False, "joints": None, "code": None,
                        "motion_sent": False, "reason": str(exc)}
        if code != 0 or not angles:
            return {"ok": False, "ik_verified": False, "joints": None, "code": int(code),
                    "motion_sent": False,
                    "reason": f"controller inverse kinematics returned {code}"}
        joints = [round(float(v), 3) for v in list(angles)[:5]]
        if len(joints) < 5:
            return {"ok": False, "ik_verified": False, "joints": joints, "code": int(code),
                    "motion_sent": False,
                    "reason": f"inverse kinematics returned {len(joints)} joints, need 5"}
        for index, (angle, (lo, hi)) in enumerate(zip(joints, XARM5_JOINT_LIMITS_DEG, strict=True)):
            if not lo <= angle <= hi:
                return {"ok": False, "ik_verified": False, "joints": joints, "code": int(code),
                        "motion_sent": False,
                        "reason": f"joint {index + 1} = {angle} deg is outside [{lo:g}, {hi:g}]"}
        return {"ok": True, "ik_verified": True, "joints": joints, "code": int(code),
                "motion_sent": False,
                "reason": "controller inverse kinematics returned a pose inside the xArm 5 joint ranges"}

    def close(self) -> None:
        try:
            self.sdk.disconnect()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass


class LazyXArmTelemetry:
    """Connect on the first read. The TCP check fails in a few seconds when the robot is down."""

    def __init__(self, ip: str):
        self.ip = ip.strip()
        self._inner: XArmTelemetry | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> XArmTelemetry:
        if self._inner is not None:
            return self._inner
        with self._lock:
            if self._inner is not None:
                return self._inner
            try:
                with socket.create_connection((self.ip, 18333), timeout=3):
                    pass
            except OSError as exc:
                raise ConnectionError(
                    f"xArm telemetry port 18333 on {self.ip} did not accept a connection: {exc}") from exc
            from xarm.wrapper import XArmAPI
            # XArmAPI opens the socket. It is not asked to enable motion or set a tool offset.
            raw = XArmAPI(self.ip, is_radian=False)
            self._inner = XArmTelemetry(raw, self.ip)
            return self._inner

    def read(self) -> dict:
        try:
            inner = self._ensure()
        except Exception as exc:  # noqa: BLE001
            return _failed(str(exc))
        return inner.read()

    def configuration(self):
        try:
            return self._ensure().configuration()
        except Exception as exc:  # noqa: BLE001 - inaccessible configuration remains unknown
            return {'mode':None,'tcp_offset':None,'world_offset':None,'error':str(exc)}

    def forward(self, joints):
        return self._ensure().forward(joints)

    def inverse(self, pose6, ref_joints=None) -> dict:
        try:
            inner = self._ensure()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "ik_verified": False, "joints": None, "code": None,
                    "motion_sent": False, "reason": str(exc)}
        return inner.inverse(pose6, ref_joints)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()


def open_reader(spec: str | None):
    """`xarm:<ip>` or a bare IP. Empty, off, and sim open nothing and do not touch the network."""
    raw = (spec or "").strip()
    if not raw or raw.lower() in {"off", "none", "sim"}:
        return None
    if raw.lower().startswith("xarm:"):
        raw = raw.split(":", 1)[1].strip()
    if not raw:
        return None
    return LazyXArmTelemetry(raw)


def observation_record(image_meta: dict | None, robot: dict | None, detection: dict | None) -> dict:
    """Pair one image with the pose that was read around it.

    skew_s is the wall-clock gap between those two timestamps. It is not a measured
    hardware trigger, so sync_precision_claimed stays false.
    """
    image = dict(image_meta or {})
    arm = dict(robot or {})
    image_at = image.get("received_at")
    pose_at = arm.get("read_finished_at")
    skew = None
    if image_at is not None and pose_at is not None:
        skew = round(abs(float(image_at) - float(pose_at)), 4)
    return {
        "image": {
            "requested_at": image.get("requested_at"),
            "received_at": image.get("received_at"),
            "request_s": image.get("request_s"),
            "width": image.get("width"),
            "height": image.get("height"),
            "rotate_reported": image.get("rotate_reported"),
            "extra_rotate": image.get("extra_rotate"),
            "rotated_again": bool(image.get("rotated_again")),
            "provenance": image.get("provenance"),
            "kind": image.get("kind"),
            "bytes": image.get("bytes"),
        },
        "robot": {
            "source": arm.get("source"),
            "pose6": arm.get("pose6"),
            "joints": arm.get("joints"),
            "state_code": arm.get("state_code"),
            "error_code": arm.get("error_code"),
            "warn_code": arm.get("warn_code"),
            "moving": arm.get("moving"),
            "read_started_at": arm.get("read_started_at"),
            "read_finished_at": arm.get("read_finished_at"),
            "motion_sent": False,
        },
        "detection": dict(detection or {}),
        "skew_s": skew,
        "sync_note": "wall-clock gap between image receive and pose read; not a measured trigger",
        "sync_precision_claimed": False,
    }
