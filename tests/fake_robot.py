"""A fake UR/Elite controller on localhost: RTDE/RTSI pose stream + a script port that understands
movel / stopl / stopj. Lets us test the socket drivers byte for byte without a robot."""

import re
import socket
import struct
import threading
import time


class FakeController:
    def __init__(self, start_m=(0.3, 0.0, 0.2), rot=(3.14, 0.0, 0.0)):
        self.pos = list(start_m)
        self.rot = rot
        self.target = list(start_m)
        self.speed = 0.1
        self.scripts: list[str] = []
        self.protocols: list[int] = []
        self._lock = threading.Lock()
        self._run = True
        self.rtde = self._listen(self._serve_rtde)
        self.script = self._listen(self._serve_script)
        threading.Thread(target=self._physics, daemon=True).start()

    def _listen(self, handler):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)

        def loop():
            while self._run:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                threading.Thread(target=handler, args=(conn,), daemon=True).start()
        threading.Thread(target=loop, daemon=True).start()
        return srv.getsockname()[1]

    def _physics(self):
        while self._run:
            with self._lock:
                d = [t - p for t, p in zip(self.target, self.pos)]
                dist = sum(v * v for v in d) ** 0.5
                step = self.speed * 0.01
                self.pos = list(self.target) if dist <= step else [p + v / dist * step for p, v in zip(self.pos, d)]
            time.sleep(0.01)

    @staticmethod
    def _rx(c, n):
        b = b""
        while len(b) < n:
            k = c.recv(n - len(b))
            if not k:
                raise ConnectionError
            b += k
        return b

    def _serve_rtde(self, c):
        try:
            while True:
                n, t = struct.unpack(">HB", self._rx(c, 3))
                body = self._rx(c, n - 3)
                t = chr(t)
                if t == "V":
                    self.protocols.append(struct.unpack(">H", body)[0])
                    c.sendall(struct.pack(">HBB", 4, ord("V"), 1))
                elif t == "O":
                    p = b"\x01VECTOR6D"
                    c.sendall(struct.pack(">HB", 3 + len(p), ord("O")) + p)
                elif t == "S":
                    c.sendall(struct.pack(">HBB", 4, ord("S"), 1))
                    while self._run:
                        with self._lock:
                            vals = self.pos + list(self.rot)
                        p = b"\x01" + struct.pack(">6d", *vals)
                        c.sendall(struct.pack(">HB", 3 + len(p), ord("U")) + p)
                        time.sleep(0.01)
        except (ConnectionError, OSError):
            pass

    def _serve_script(self, c):
        data = b""
        try:
            while True:
                k = c.recv(4096)
                if not k:
                    break
                data += k
        except OSError:
            pass
        text = data.decode()
        self.scripts.append(text)
        m = re.search(r"movel\(p?\[([^\]]+)\].*?v=([\d.]+)", text)
        with self._lock:
            if m:
                v = [float(x) for x in m.group(1).split(",")]
                self.target, self.speed = v[:3], float(m.group(2))
            elif "stopl" in text or "stopj" in text:
                self.target = list(self.pos)

    def wait_for(self, text: str, timeout: float = 2.0) -> bool:
        """Scripts are logged when their connection closes, a moment after the driver returns."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.scripts and text in self.scripts[-1]:
                return True
            time.sleep(0.01)
        return False

    def close(self):
        self._run = False
