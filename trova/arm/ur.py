"""Universal Robots UR3e / UR5e over plain sockets (no ur_rtde: it has no macOS wheels).

  moves, stop, freedrive -> URScript to the secondary interface, port 30002 (robot must be in REMOTE mode)
  live pose              -> RTDE on 30004, protocol 2 (read-only; works in local mode too)
  protective stop reset  -> dashboard server, port 29999

Pendant setup (once): Settings > System > Remote Control > Enable, then switch to Remote (top right).
PolyScope 5.10+: Settings > Security > Services: enable Dashboard, Secondary client, RTDE. Set the TCP of the
printed tool (Installation > General > TCP), or every pose is the flange, not the tip.
URScript poses are metres + rotation vector; we keep the orientation we read and change only x, y, z.
A new program sent to 30002 replaces the running one, so stop = send a program that runs stopl().
"""

from __future__ import annotations

import socket
import time

from .cartesian import CartesianBackend
from .rtde_reader import RtdeReader


class URBackend(CartesianBackend):
    name = "ur"

    def __init__(self, ip: str, script_port: int = 30002, rtde_port: int = 30004, dash_port: int = 29999,
                 accel_m_s2: float = 0.5):
        if not ip:
            raise ValueError("TROVA_ARM=ur:<robot ip>")
        self.ip, self.script_port, self.dash_port, self.accel = ip, script_port, dash_port, accel_m_s2
        self.reader = RtdeReader(ip, rtde_port, protocol=2)
        self.target: tuple | None = None
        self.sent_at = 0.0

    def _send(self, script: str) -> None:
        with socket.create_connection((self.ip, self.script_port), timeout=2) as s:
            s.sendall(script.encode())

    def _dash(self, cmd: str) -> str:
        with socket.create_connection((self.ip, self.dash_port), timeout=2) as s:
            s.recv(1024)                                  # welcome banner
            s.sendall((cmd + "\n").encode())
            return s.recv(1024).decode(errors="ignore").strip()

    def move_linear(self, x, y, z, speed_mm_s):
        rx, ry, rz = self.reader.get()[3:]
        self._send("def trova_move():\n  movel(p[%.5f,%.5f,%.5f,%.5f,%.5f,%.5f], a=%.3f, v=%.4f)\nend\n"
                   % (x / 1000, y / 1000, z / 1000, rx, ry, rz, self.accel, speed_mm_s / 1000))
        self.target, self.sent_at = (x, y, z), time.time()

    def is_moving(self) -> bool:
        if time.time() - self.sent_at < 0.5:              # the program needs a moment to start
            return True
        if self.reader.speed_mm_s() > 2.0:
            return True
        return False

    def stop(self):
        self._send("def trova_stop():\n  stopl(2.0)\nend\n")
        self.sent_at = 0.0

    def tcp_mm(self):
        x, y, z = self.reader.get()[:3]
        return x * 1000, y * 1000, z * 1000

    def freedrive(self, on: bool):
        # freedrive only lasts while its program runs, so the program loops until replaced
        self._send("def trova_free():\n  freedrive_mode()\n  while True:\n    sync()\n  end\nend\n" if on
                   else "def trova_endfree():\n  end_freedrive_mode()\nend\n")

    def clear_errors(self):
        self._dash("close safety popup")
        self._dash("unlock protective stop")            # refused within 5 s of the stop

    def close(self):
        self.reader.close()
