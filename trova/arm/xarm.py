"""UFactory xArm5 via the official pure-Python SDK:  pip install xarm-python-sdk

No pendant: release the e-stop, power the box, put the Mac on its subnet (often 192.168.1.x).
xArm5 is 5-axis: Cartesian moves need roll=180, pitch=0; we keep the yaw we read at connect.
Units: mm and degrees. set_position is non-blocking with wait=False and QUEUES commands; a still-moving
controller is refused. STOP requests SDK stop state 4, checks its result and NEVER enables/restarts motion.
This does not command torque off. Controller recovery/enabling is a separate explicit staff operation.
Set the tool tip with set_tcp_offset (or in UFACTORY Studio) before calibrating: TROVA_ARM_TCP_MM=0,0,<length>
applies it at connect (not saved on the controller). First connection: python -m trova.arm.bringup <ip>.
"""

from __future__ import annotations

import os
import threading

from .cartesian import CartesianBackend


class XArmBackend(CartesianBackend):
    name = "xarm"

    def __init__(self, ip: str, accel_mm_s2: float = 500.0):
        if not ip:
            raise ValueError("TROVA_ARM=xarm:<robot ip>")
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as e:
            raise RuntimeError("pip install xarm-python-sdk") from e
        self.lock = threading.Lock()
        self.accel = accel_mm_s2
        self.arm = XArmAPI(ip, is_radian=False)
        # Connection is not operator authorization to recover faults or enable the controller.
        # Explicit controller recovery is separate; telemetry uses its own read-only connection.
        tcp = os.getenv("TROVA_ARM_TCP_MM", "").strip()
        if tcp:      # the pointer / printed tool: positions then refer to its tip, not the flange
            x, y, z = (float(v) for v in tcp.split(","))
            self._check(self.arm.set_tcp_offset([x, y, z, 0, 0, 0], wait=True), "set_tcp_offset")
        code, pose = self.arm.get_position()
        self._check(code, "get_position")
        if not pose or len(pose) < 6:
            raise RuntimeError("xArm get_position returned no complete pose")
        self.yaw = pose[5]

    @staticmethod
    def _check(code, operation):
        if type(code) is not int or code != 0:
            raise RuntimeError(f"xArm {operation} returned code {code}")

    def move_linear(self, x, y, z, speed_mm_s):
        from ..motion import require_ticket
        ticket = require_ticket()
        speed_mm_s = min(float(speed_mm_s), ticket.max_speed_mm_s)
        accel = min(self.accel, ticket.max_accel_mm_s2)
        with self.lock:
            if self._moving():                  # never recover/re-enable a moving controller implicitly
                raise RuntimeError("xArm controller is still moving: STOP and explicit staff recovery required")
            # xArm 5 moves in straight lines only with the flange parallel to the base (roll +-180, pitch 0). Coming
            # from a taught pose that is not (e.g. a camera survey pose), plan this one move in joint space via IK.
            code, pose = self.arm.get_position()
            self._check(code, "get_position")
            flat = code == 0 and pose and abs(abs(pose[3]) - 180) < 3 and abs(pose[4]) < 3
            extra = {} if flat else {"motion_type": 2}
            code = self.arm.set_position(x=x, y=y, z=z, roll=180, pitch=0, yaw=self.yaw, speed=speed_mm_s,
                                         mvacc=accel, wait=False, **extra)
        if code != 0:
            raise RuntimeError(f"xArm set_position returned code {code} (error {self.arm.error_code})")

    def move_joints(self, q, speed_deg_s):
        from ..motion import require_ticket
        ticket = require_ticket()
        speed_deg_s = min(float(speed_deg_s), ticket.max_joint_speed_deg_s)
        with self.lock:
            if self._moving():
                raise RuntimeError("xArm controller is still moving: STOP and explicit staff recovery required")
            code = self.arm.set_servo_angle(angle=list(q), speed=speed_deg_s, mvacc=speed_deg_s * 5, wait=False)
        if code != 0:
            raise RuntimeError(f"xArm set_servo_angle returned code {code} (error {self.arm.error_code})")

    def fk_mm(self, q):
        with self.lock:
            code, pose = self.arm.get_forward_kinematics(list(q))
        if code != 0 or not pose:
            raise RuntimeError(f"xArm forward kinematics code {code}")
        return tuple(pose[:3])

    def joints(self):
        with self.lock:
            code, q = self.arm.get_servo_angle()
        if code != 0:
            raise RuntimeError(f"xArm get_servo_angle code {code}")
        return [float(v) for v in q[:5]]

    def is_moving(self) -> bool:
        with self.lock:
            return self._moving()

    def _moving(self) -> bool:
        # SDK get_is_moving() discards get_state()'s error code and may return cached state.
        code, state = self.arm.get_state()
        self._check(code, "get_state")
        if state not in (0, 1, 2, 3, 4, 5, 6):
            raise RuntimeError(f"xArm unknown controller state {state}")
        return state == 1

    def _halt(self):
        # SDK documents 4 as stop state, 0 as motion state. No motion_enable(False)/torque-off command.
        self._check(self.arm.set_state(4), "set_state(STOP)")

    def stop(self):
        with self.lock:
            self._halt()

    def tcp_mm(self):
        return self.tcp_pose()[:3]

    def tcp_pose(self):
        """Read-only tool pose: x, y, z millimetres and roll, pitch, yaw degrees. Does not enable motion."""
        with self.lock:
            code, pose = self.arm.get_position()
        if code != 0:
            raise RuntimeError(f"xArm get_position code {code}")
        if pose is None or len(pose) < 6:
            raise RuntimeError("xArm get_position did not return roll, pitch and yaw")
        return tuple(float(v) for v in pose[:6])

    def freedrive(self, on: bool):
        from ..motion import require_ticket
        require_ticket()
        with self.lock:
            self._check(self.arm.set_mode(2 if on else 0), "set_mode")
            self._check(self.arm.set_state(0), "set_state")

    def clear_errors(self):
        with self.lock:
            self._check(self.arm.clean_error(), "clean_error")
            self._check(self.arm.clean_warn(), "clean_warn")
            self._check(self.arm.motion_enable(True), "motion_enable")
            self._check(self.arm.set_mode(0), "set_mode")
            self._check(self.arm.set_state(0), "set_state")

    def close(self):
        try:
            self.arm.disconnect()
        except Exception:  # noqa: BLE001
            pass
