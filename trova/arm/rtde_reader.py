"""Minimal, dependency-free reader of the live TCP pose over RTDE (UR, port 30004, protocol 2) or
RTSI (Elite CS, port 30004, protocol 1, RTDE-compatible framing). Read-only: works in local mode too.

Framing (big-endian): uint16 size (incl. header) + uint8 type + payload.
  'V' request protocol version (uint16)       -> reply 'V' uint8 accepted
  'O' setup outputs: double Hz + "actual_TCP_pose" -> reply 'O' uint8 recipe id + types
  'S' start                                    -> reply 'S' uint8 accepted
  'U' data: uint8 recipe id + 6 doubles (x, y, z in m; orientation as the controller reports it)
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque


class RtdeReader:
    def __init__(self, ip: str, port: int = 30004, protocol: int = 2, hz: float = 125.0):
        self.ip, self.port, self.protocol, self.hz = ip, port, protocol, hz
        self.pose: tuple | None = None
        self.updated = 0.0
        self.error: str | None = None
        self._hist: deque = deque(maxlen=64)
        self._lock = threading.Lock()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -------- protocol
    @staticmethod
    def _rx_exact(s, n):
        b = b""
        while len(b) < n:
            chunk = s.recv(n - len(b))
            if not chunk:
                raise ConnectionError("RTDE connection closed")
            b += chunk
        return b

    def _rx(self, s):
        n, t = struct.unpack(">HB", self._rx_exact(s, 3))
        return chr(t), self._rx_exact(s, n - 3)

    @staticmethod
    def _tx(s, t, payload=b""):
        s.sendall(struct.pack(">HB", 3 + len(payload), ord(t)) + payload)

    def _expect(self, s, want):
        for _ in range(20):          # skip text messages ('M') and anything unexpected
            t, body = self._rx(s)
            if t == want:
                return body
        raise ConnectionError(f"RTDE: no '{want}' reply")

    def _session(self):
        with socket.create_connection((self.ip, self.port), timeout=3) as s:
            self._tx(s, "V", struct.pack(">H", self.protocol))
            if not self._expect(s, "V")[0]:
                raise ConnectionError(f"controller refused RTDE protocol {self.protocol}")
            self._tx(s, "O", struct.pack(">d", self.hz) + b"actual_TCP_pose")
            setup = self._expect(s, "O")
            if b"NOT_FOUND" in setup:
                raise ConnectionError("controller does not publish actual_TCP_pose")
            self._tx(s, "S")
            self._expect(s, "S")
            s.settimeout(2)
            while self._run:
                t, body = self._rx(s)
                if t == "U" and len(body) >= 49:
                    pose = struct.unpack_from(">6d", body, 1)
                    now = time.time()
                    with self._lock:
                        self.pose, self.updated, self.error = pose, now, None
                        self._hist.append((now, pose[:3]))

    def _loop(self):
        while self._run:
            try:
                self._session()
            except Exception as e:  # noqa: BLE001 - keep retrying; the UI shows the error
                with self._lock:
                    self.error = f"{type(e).__name__}: {e}"
                time.sleep(1.0)

    # -------- API
    def get(self, max_age_s: float = 1.0) -> tuple:
        with self._lock:
            if self.pose is None or time.time() - self.updated > max_age_s:
                raise RuntimeError(f"no fresh pose from {self.ip}:{self.port} ({self.error or 'waiting'})")
            return self.pose

    def speed_mm_s(self, window_s: float = 0.15) -> float:
        with self._lock:
            pts = [p for p in self._hist if p[0] >= time.time() - window_s]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1e-3:
            return 0.0
        (t0, a), (t1, b) = pts[0], pts[-1]
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 * 1000 / (t1 - t0)

    def close(self):
        self._run = False
