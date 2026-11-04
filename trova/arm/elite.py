"""Elite Robots CS63 / CS66 over plain sockets (the official SDK has no macOS build).

Elite mirrors UR's ports, but its script is Python-style and its dashboard words differ:
  moves, stop, freedrive -> Elite script inside a def ... end envelope, tab-indented, to port 30001
                            (robot must be in remote mode: Settings > System > Remote Control)
  live pose              -> RTSI on 30004 (RTDE-compatible framing, protocol 1)
  protective stop reset  -> dashboard 29999: closeSafetyDialog, unlockProtectiveStop
movel takes a plain list [x, y, z, r1, r2, r3] in metres + radians (no p[...]). The docs disagree on RPY vs
rotation vector, so we keep the orientation we read and change only x, y, z.
UNTESTED on a real CS tonight: run it once on the Elite simulator or with staff before the demo.
"""

from __future__ import annotations

import socket
import threading
import time

from .cartesian import CartesianBackend
from .rtde_reader import RtdeReader


class EliteBackend(CartesianBackend):
    name = "elite"

    def __init__(self, ip: str, script_port: int = 30001, rtsi_port: int = 30004, dash_port: int = 29999,
                 accel_m_s2: float = 0.5):
        if not ip:
            raise ValueError("TROVA_ARM=elite:<robot ip>")
        self.ip, self.script_port, self.dash_port, self.accel = ip, script_port, dash_port, accel_m_s2
        self.reader = RtdeReader(ip, rtsi_port, protocol=1, hz=50.0)
        self.sent_at = 0.0
        self._send_lock = threading.Lock()   # a stop is never overtaken by a move half-sent

    def _send(self, script: str) -> None:
        with self._send_lock, socket.create_connection((self.ip, self.script_port), timeout=2) as s:
            s.sendall((script + "\n").encode())

    def _dash(self, cmd: str) -> str:
        with socket.create_connection((self.ip, self.dash_port), timeout=2) as s:
            s.sendall((cmd + "\n").encode())
            time.sleep(0.3)
            return s.recv(4096).decode(errors="ignore").strip()

    def move_linear(self, x, y, z, speed_mm_s):
        r1, r2, r3 = self.reader.get()[3:]
        self._send("def trova_move():\n\tmovel([%.5f,%.5f,%.5f,%.5f,%.5f,%.5f], a=%.3f, v=%.4f)\nend"
                   % (x / 1000, y / 1000, z / 1000, r1, r2, r3, self.accel, speed_mm_s / 1000))
        self.sent_at = time.time()

    def is_moving(self) -> bool:
        return time.time() - self.sent_at < 0.5 or self.reader.speed_mm_s() > 2.0

    def stop(self):
        self._send("def trova_stop():\n\tstopj(2.0)\nend")
        self.sent_at = 0.0

    def tcp_mm(self):
        x, y, z = self.reader.get()[:3]
        return x * 1000, y * 1000, z * 1000

    def freedrive(self, on: bool):
        self._send("def trova_free():\n\tfreedrive_mode()\n\twhile True:\n\t\tsync()\nend" if on
                   else "def trova_endfree():\n\tend_freedrive_mode()\nend")

    def clear_errors(self):
        self._dash("closeSafetyDialog")
        self._dash("unlockProtectiveStop")

    def close(self):
        self.reader.close()
