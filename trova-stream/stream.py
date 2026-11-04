#!/usr/bin/env python3
"""
trova-stream: camera server for the trova-cam Pi (Zero 2 W + IMX219).

rpicam-vid encodes MJPEG to stdout; this server splits it into JPEG frames and
hands the newest one to every viewer. Slow viewers skip frames instead of
lagging, and the camera only runs while someone is using it.

Built-in protection
  * settings are clamped to what the Pi can handle; risky ones show a warning
  * temperature watchdog: at TEMP_LIGHT the camera drops to a light mode, at
    TEMP_PAUSE it pauses until the Pi cools down, then it resumes by itself
  * under-voltage and CPU-throttling warnings from the Pi firmware

Endpoints
  /              viewer page
  /stream.mjpg   MJPEG stream  (?fps=N caps the rate for that viewer)
  /snapshot.jpg  one JPEG captured after the request (?download=1 saves it)
  /status.json   camera state, temperature, warnings

Only the Python standard library and rpicam-apps are needed.
"""
import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# SETTINGS: edit these, then restart (sudo systemctl restart trova-stream).
# Each one can also be overridden for a single run with a flag
# (python3 stream.py --help).
# ---------------------------------------------------------------------------
ROTATE = 180          # 0 or 180. The camera is mounted upside down, so 180 turns
                      # the picture upright everywhere: page, snapshots, OpenCV.
                      # Done by the sensor itself, so it costs nothing.
WIDTH, HEIGHT = 1640, 1232   # the sensor's full field of view (4:3)
FRAMERATE = 10        # frames per second
QUALITY = 50          # JPEG quality: higher is sharper but heavier on WiFi
PORT = 8080           # web port
IDLE_TIMEOUT = 60     # seconds the camera stays on after the last request

# ---------------------------------------------------------------------------
# HARDWARE LIMITS of the Pi Zero 2 W. Values outside a limit are clamped.
# Values above the "comfortable" level still run but show a warning.
# ---------------------------------------------------------------------------
MAX_WIDTH, MAX_HEIGHT = 1920, 1232
BINNED_MAX = (1640, 1232)     # above this the sensor reads out at full resolution
FULL_RES_MAX_FPS = 20         # sensor limit at full resolution
MAX_FRAMERATE, COMFORT_FRAMERATE = 30, 15
MIN_QUALITY, COMFORT_QUALITY, MAX_QUALITY = 20, 70, 85

# Temperature protection, in degrees C, smoothed over about 30 seconds.
# The Pi firmware itself starts slowing the CPU at 80 C and hard-limits at 85 C.
TEMP_LIGHT = 80       # switch to LIGHT_MODE...
TEMP_NORMAL = 72      # ...and back once below this for COOL_HOLD seconds
TEMP_PAUSE = 84       # pause the camera...
TEMP_RESUME = 70      # ...and resume (in light mode) once below this
COOL_HOLD = 60
LIGHT_MODE = (640, 480, 5, 40)   # width, height, fps, quality while hot
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 8      # frames to let auto-exposure settle before a snapshot
STALL_SECONDS = 10     # restart rpicam-vid if it produces nothing for this long
RETRY_SECONDS = 3      # wait between start attempts (e.g. camera busy)
THERMAL_EVERY = 2      # seconds between temperature reads
FLAGS_EVERY = 10       # seconds between firmware throttle checks
TEMP_FILE = "/sys/class/thermal/thermal_zone0/temp"
HERE = os.path.dirname(os.path.abspath(__file__))

Profile = namedtuple("Profile", "width height framerate quality")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --- settings ---------------------------------------------------------------
def clamp_settings(args):
    """Clamp the requested settings to the hardware limits; return (profile, warnings)."""
    warnings = []

    def clamp(name, value, lo, hi):
        if lo <= value <= hi:
            return value
        new = min(max(value, lo), hi)
        warnings.append(f"{name} {value:g} is outside the hardware limit, using {new:g}")
        return new

    width = int(clamp("width", args.width, 320, MAX_WIDTH))
    height = int(clamp("height", args.height, 240, MAX_HEIGHT))
    full_res = width > BINNED_MAX[0] or height > BINNED_MAX[1]
    fps = clamp("framerate", args.framerate, 1, FULL_RES_MAX_FPS if full_res else MAX_FRAMERATE)
    quality = int(clamp("quality", args.quality, MIN_QUALITY, MAX_QUALITY))

    if full_res:
        warnings.append(f"{width}×{height} reads the sensor at full resolution: more CPU, heat and WiFi")
    if fps > COMFORT_FRAMERATE:
        warnings.append(f"{fps:g} fps is more than needed: extra heat and WiFi load")
    if quality > COMFORT_QUALITY:
        warnings.append(f"quality {quality} makes big frames: slow on the Pi's ~10 Mbit/s WiFi")
    return Profile(width, height, fps, quality), warnings


def build_cmd(profile, rotate, extra):
    binned = profile.width <= BINNED_MAX[0] and profile.height <= BINNED_MAX[1]
    cmd = ["rpicam-vid", "-v", "0", "-n", "-t", "0", "--flush",
           "--codec", "mjpeg", "-q", str(profile.quality),
           "--width", str(profile.width), "--height", str(profile.height),
           "--framerate", str(profile.framerate),
           "--mode", "1640:1232:10:P" if binned else "3280:2464:10:P"]
    if rotate == 180:
        cmd += ["--rotation", "180"]
    return cmd + extra + ["-o", "-"]


# --- temperature --------------------------------------------------------------
def next_level(level, avg, cool_since, now):
    """Protection state machine: normal -> light -> paused and back, with hysteresis.

    Returns (new_level, cool_since). cool_since tracks how long a hot Pi has
    been below TEMP_NORMAL before it may go back to normal settings.
    """
    if avg >= TEMP_PAUSE:
        return "paused", None
    if level == "paused":
        return ("light" if avg < TEMP_RESUME else "paused"), None
    if avg >= TEMP_LIGHT:
        return "light", None
    if level == "light":
        if avg >= TEMP_NORMAL:
            return "light", None
        cool_since = now if cool_since is None else cool_since
        if now - cool_since >= COOL_HOLD:
            return "normal", None
        return "light", cool_since
    return "normal", None


class Thermal:
    """SoC temperature (smoothed over ~30 s) and firmware throttle flags."""

    def __init__(self):
        self.temp = self._read_temp() or 0.0
        self.avg = self.temp
        self.flags = 0
        self._flags_due = 0.0

    @staticmethod
    def _read_temp():
        try:
            with open(TEMP_FILE) as f:
                return int(f.read()) / 1000
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_flags():
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                 text=True, timeout=5).stdout
            return int(out.strip().split("=")[1], 16)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return 0

    def update(self, now):
        t = self._read_temp()
        if t is not None:
            self.temp = t
            self.avg += 0.15 * (t - self.avg)
        if now >= self._flags_due:
            self._flags_due = now + FLAGS_EVERY
            self.flags = self._read_flags()

    def warnings(self):
        out = []
        if self.flags & 0x1:
            out.append("Power too low (under-voltage): check the xArm power supply")
        if self.flags & 0xE:
            out.append("The Pi firmware is slowing the CPU (heat or power)")
        return out


# --- camera -------------------------------------------------------------------
def jpeg_end(buf, soi):
    """Index just past the JPEG that starts at buf[soi].

    Returns None if the frame is not complete yet and -1 if it is malformed.
    Header segments are skipped by their length; after start-of-scan every
    0xFF in the entropy data is byte-stuffed, so the next FFD9 is the end.
    """
    i, n = soi + 2, len(buf)
    while i + 4 <= n:
        if buf[i] != 0xFF:
            return -1
        marker = buf[i + 1]
        if marker == 0xFF:            # fill byte
            i += 1
            continue
        seglen = (buf[i + 2] << 8) | buf[i + 3]
        if marker == 0xDA:            # start of scan
            eoi = buf.find(b"\xff\xd9", i + 2 + seglen)
            return None if eoi < 0 else eoi + 2
        i += 2 + seglen
    return None


class Camera:
    """Runs rpicam-vid while there are users, keeps the newest frame, minds the heat."""

    def __init__(self, normal, light, rotate, extra, idle_timeout, config_warnings):
        self.profiles = {"normal": normal, "light": light}
        self.rotate = rotate
        self.extra = extra
        self.idle_timeout = idle_timeout
        self.config_warnings = config_warnings
        self.thermal = Thermal()
        self.level = "normal"          # normal | light | paused
        self.cool_since = None
        self.running_level = None
        self.cond = threading.Condition()
        self.wake = threading.Event()
        self.frame = None
        self.seq = 0            # frames published since the server started
        self.run_frames = 0     # frames since the current rpicam-vid started
        self.users = 0
        self.last_user = time.monotonic()
        self.last_frame_t = 0.0
        self.started_t = 0.0
        self.fps = 0.0
        self.proc = None
        self.error = None
        threading.Thread(target=self._supervise, daemon=True).start()

    # --- users -----------------------------------------------------------
    def acquire(self):
        with self.cond:
            self.users += 1
        self.wake.set()

    def release(self):
        with self.cond:
            self.users -= 1
            self.last_user = time.monotonic()

    def wait_frame(self, after_seq, min_run_frames=0, timeout=5.0):
        """Block until a frame newer than after_seq exists; (None, after_seq) on timeout."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while self.seq <= after_seq or self.run_frames < min_run_frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, after_seq
                self.cond.wait(remaining)
            return self.frame, self.seq

    def status(self):
        with self.cond:
            running = self.proc is not None and self.proc.poll() is None
            if self.level == "paused":
                state = "paused"
            elif running and self.run_frames:
                state = "streaming"
            elif running:
                state = "starting"
            elif self.users and self.error:
                state = "error"
            else:
                state = "idle"
            profile = self.profiles["normal" if self.level == "normal" else "light"]
            avg = self.thermal.avg
            message = {
                "paused": f"Pi is too hot ({avg:.0f} °C): camera paused, "
                          f"resumes by itself below {TEMP_RESUME} °C",
                "starting": "Starting camera…",
                "error": f"Camera error: {self.error}. Retrying…",
            }.get(state)
            if state != "paused" and self.level == "light":
                message = (f"Pi is hot ({avg:.0f} °C): running at "
                           f"{profile.width}×{profile.height} until it cools down")
            return {
                "camera": state,
                "viewers": self.users,
                "fps": round(self.fps, 1) if state == "streaming" else 0,
                "width": profile.width,
                "height": profile.height,
                "framerate": profile.framerate,
                "quality": profile.quality,
                "rotate": self.rotate,
                "temp": round(self.thermal.temp, 1),
                "protection": self.level,
                "message": message,
                "warnings": self.config_warnings + self.thermal.warnings(),
            }

    # --- process management ---------------------------------------------
    def _supervise(self):
        next_try = next_thermal = 0.0
        while True:
            self.wake.wait(0.5)
            self.wake.clear()
            now = time.monotonic()

            if now >= next_thermal:
                next_thermal = now + THERMAL_EVERY
                self.thermal.update(now)
                level, self.cool_since = next_level(self.level, self.thermal.avg,
                                                    self.cool_since, now)
                if level != self.level:
                    log(f"temperature {self.thermal.avg:.1f} C: {self.level} -> {level}")
                    with self.cond:
                        self.level = level

            with self.cond:
                proc, users, want = self.proc, self.users, self.level
                idle_for = now - self.last_user
                quiet_for = now - max(self.last_frame_t, self.started_t)
            running = proc is not None and proc.poll() is None

            if proc is not None and not running:
                code = proc.returncode
                with self.cond:
                    self.proc = None
                    if not self.error:
                        self.error = f"rpicam-vid exited with code {code} (camera busy or disconnected?)"
                log(f"rpicam-vid exited with code {code}")
                next_try = now + RETRY_SECONDS
                running = False

            if running and want == "paused":
                log("pausing camera to cool down")
                self._stop(proc)
            elif running and want != self.running_level:
                log(f"switching camera to {want} settings")
                self._stop(proc)
                next_try = 0.0
                self.wake.set()
            elif users and not running and want != "paused" and now >= next_try:
                self._start(want)
                next_try = now + RETRY_SECONDS
            elif running and not users and idle_for >= self.idle_timeout:
                log("no users, stopping camera")
                self._stop(proc)
            elif running and quiet_for > STALL_SECONDS:
                log(f"no frames for {STALL_SECONDS}s, restarting camera")
                with self.cond:
                    self.error = "camera stalled"
                self._stop(proc)

    def _start(self, level):
        cmd = build_cmd(self.profiles[level], self.rotate, self.extra)
        log("starting camera: " + " ".join(cmd))
        env = dict(os.environ, LIBCAMERA_LOG_LEVELS="*:ERROR")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, env=env)
        except OSError as e:
            with self.cond:
                self.error = f"cannot run rpicam-vid: {e}"
            log(self.error)
            return
        with self.cond:
            self.proc = proc
            self.running_level = level
            self.run_frames = 0
            self.fps = 0.0
            self.last_frame_t = 0.0
            self.started_t = time.monotonic()
            self.error = None
        threading.Thread(target=self._read, args=(proc,), daemon=True).start()

    def _stop(self, proc):
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        with self.cond:
            if self.proc is proc:
                self.proc = None

    def stop(self):
        with self.cond:
            proc = self.proc
        self._stop(proc)

    def _read(self, proc):
        buf = bytearray()
        out = proc.stdout
        while True:
            chunk = out.read1(1 << 16)
            if not chunk:
                break
            buf += chunk
            pos = 0
            while True:
                soi = buf.find(b"\xff\xd8", pos)
                if soi < 0:
                    pos = max(pos, len(buf) - 1)   # keep a trailing 0xFF
                    break
                end = jpeg_end(buf, soi)
                if end is None:
                    pos = soi
                    break
                if end < 0:
                    pos = soi + 2
                    continue
                self._publish(bytes(buf[soi:end]))
                pos = end
            del buf[:pos]
            if len(buf) > 8 << 20:                 # garbage, resync
                buf.clear()
        out.close()

    def _publish(self, frame):
        now = time.monotonic()
        with self.cond:
            if self.last_frame_t:
                dt = now - self.last_frame_t
                if dt > 0:
                    self.fps = (1 / dt) if not self.fps else 0.9 * self.fps + 0.1 / dt
            self.last_frame_t = now
            self.frame = frame
            self.seq += 1
            self.run_frames += 1
            self.cond.notify_all()


# --- web server ---------------------------------------------------------------
class Server(ThreadingHTTPServer):
    """Listens on IPv6 and IPv4 (trova-cam.local often resolves to IPv6)."""
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        with contextlib.suppress(OSError):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    server_version = "trova-stream/2.0"
    timeout = 20            # drop clients whose connection stalls
    cam = None              # set in main()
    page = b""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", self.page)
        elif url.path == "/stream.mjpg":
            self._stream(query)
        elif url.path == "/snapshot.jpg":
            self._snapshot(query)
        elif url.path == "/status.json":
            self._send(200, "application/json", json.dumps(self.cam.status()).encode())
        else:
            self._send(404, "text/plain", b"not found\n")

    def _send(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, query):
        try:
            cap = float(query.get("fps", ["0"])[0])
        except ValueError:
            cap = 0.0
        min_gap = 1.0 / cap if cap > 0 else 0.0
        who = self.client_address[0].removeprefix("::ffff:")
        self.cam.acquire()
        log(f"viewer connected: {who} ({self.cam.users} watching)")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = 0
            while True:
                frame, new_seq = self.cam.wait_frame(seq)
                if frame is None:
                    continue
                seq = new_seq
                t = time.monotonic()
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                    % (len(frame), frame))
                if min_gap:
                    time.sleep(max(0.0, min_gap - (time.monotonic() - t)))
        except OSError:          # client went away (BrokenPipe, reset, timeout)
            pass
        finally:
            self.cam.release()
            log(f"viewer left: {who} ({self.cam.users} watching)")

    def _snapshot(self, query):
        status = self.cam.status()
        if status["camera"] == "paused":
            self._send(503, "text/plain", (status["message"] + "\n").encode())
            return
        self.cam.acquire()
        try:
            with self.cam.cond:
                after = self.cam.seq
            frame, _ = self.cam.wait_frame(after, WARMUP_FRAMES, timeout=12)
        finally:
            self.cam.release()
        if frame is None:
            msg = self.cam.status()["message"] or "no frame from camera"
            self._send(503, "text/plain", (msg + "\n").encode())
            return
        extra = []
        if query.get("download"):
            name = time.strftime("trova-cam-%Y%m%d-%H%M%S.jpg")
            extra.append(("Content-Disposition", f'attachment; filename="{name}"'))
        self._send(200, "image/jpeg", frame, extra)


def main():
    p = argparse.ArgumentParser(
        description="Camera server for trova-cam: live page, snapshots, heat protection.",
        epilog="Other options are passed to rpicam-vid, e.g. --hflip or --shutter 10000. "
               "Defaults come from the SETTINGS block at the top of this file.",
        allow_abbrev=False)
    p.add_argument("--rotate", type=int, default=ROTATE, choices=(0, 180),
                   help=f"0 or 180, done by the camera itself (default {ROTATE})")
    p.add_argument("--width", type=int, default=WIDTH, help=f"default {WIDTH}, max {MAX_WIDTH}")
    p.add_argument("--height", type=int, default=HEIGHT, help=f"default {HEIGHT}, max {MAX_HEIGHT}")
    p.add_argument("--framerate", type=float, default=FRAMERATE,
                   help=f"default {FRAMERATE}, max {MAX_FRAMERATE}")
    p.add_argument("--quality", type=int, default=QUALITY,
                   help=f"JPEG quality {MIN_QUALITY}-{MAX_QUALITY}, default {QUALITY}")
    p.add_argument("--port", type=int, default=PORT, help=f"default {PORT}")
    p.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT,
                   help=f"seconds the camera stays on after the last request, default {IDLE_TIMEOUT}")
    args, extra = p.parse_known_args()

    normal, warnings = clamp_settings(args)
    light = Profile(*LIGHT_MODE)
    Handler.cam = Camera(normal, light, args.rotate, extra, args.idle_timeout, warnings)
    with open(os.path.join(HERE, "index.html"), "rb") as f:
        Handler.page = f.read()

    server = Server(("::", args.port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log(f"trova-stream on http://0.0.0.0:{args.port}/  ({normal.width}x{normal.height} "
        f"@ {normal.framerate:g} fps, q{normal.quality}, rotate {args.rotate}, "
        f"temp {Handler.cam.thermal.temp:.1f} C)")
    for w in warnings:
        log("warning: " + w)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.cam.stop()


if __name__ == "__main__":
    main()
