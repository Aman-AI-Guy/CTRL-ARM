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
  /stream.mjpg   MJPEG stream  (?fps=N caps the rate for that viewer; 2 fps max during a scan)
  /snapshot.jpg  one JPEG captured after the request (?download=1 saves it)
  /status.json   camera state, temperature, warnings, scan state

3D scan mode (temporary profile; the default profile comes back when the scan stops)
  POST   /scan/start          meter auto exposure / white balance, lock them, record every frame to
                              ~/scans/<id>/ (body JSON or query: width height fps quality shutter_max_us
                              bin_s max_s gain_max ae_settle_s skip_frames queue_frames metadata label)
  POST   /scan/stop           stop, restore the default profile, return the summary
  GET    /scan/live?since=N   multipart stream of the sharpest frame per time bin, frame index > N
  GET    /scan/list           scans on the Pi
  GET    /scan/clock          Pi wall and monotonic clock (for latency measurements)
  GET    /scan/<id>/frame/<i> one recorded frame
  GET    /scan/<id>/<file>    meta.json, frames.csv, bins.csv, camera_metadata.json, metering_metadata.json
  GET    /scan/<id>.tar       the whole scan, streamed, uncompressed
  DELETE /scan/<id>           delete a scan

Only the Python standard library and rpicam-apps are needed.
"""
import argparse
import bisect
import contextlib
import json
import os
import queue
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from collections import OrderedDict, namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# SETTINGS: edit these, then restart (sudo systemctl restart trova-stream).
# Each one can also be overridden for a single run with a flag
# (python3 stream.py --help).
# ---------------------------------------------------------------------------
ROTATE = 180          # 0 or 180. The camera is mounted upside down, so 180 turns
                      # the picture upright everywhere: page, snapshots, OpenCV.
                      # Done by the sensor itself, so it costs nothing.
WIDTH, HEIGHT = 1600, 900    # 16:9, the largest that keeps the cool binned sensor mode; the ISP crops
                             # top and bottom, full width kept. (1920, 1080 = full HD but hot; 1640, 1232 = 4:3)
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
LIGHT_MODE = (640, 360, 5, 40)   # width, height, fps, quality while hot (16:9 like the default)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3D SCAN MODE: defaults for POST /scan/start (each one can be overridden per scan).
# Recordings go to the SD card: /tmp is RAM on this Pi.
# ---------------------------------------------------------------------------
SCANS_DIR = os.path.expanduser("~/scans")
SCAN_DEFAULTS = {
    "width": 1640, "height": 1232,
    "fps": 15,
    "quality": 85,
    "shutter_max_us": 8000,   # cap on the locked shutter: shorter = less motion blur, more gain (noise)
    "bin_s": 0.33,            # live push: the largest JPEG (≈ sharpest) of every bin_s seconds
    "max_s": 90,              # the scan stops by itself after this long
    "gain_max": 16.0,         # cap on the total gain (analogue x digital) after capping the shutter
    "ae_settle_s": 2.0,       # seconds of auto exposure / white balance before reading the values to lock
    "skip_frames": 4,         # frames not recorded after the camera restarts locked (gain still ramping)
    "queue_frames": 40,       # writer queue (RAM); when full a frame is counted as dropped, never blocks
    "metadata": 1,            # 1 = save per-frame camera metadata (exposure, gains, FocusFoM) with the scan
    "label": "",              # optional text appended to the scan id
}
SCAN_BOUNDS = {"bin_s": (0.05, 5.0), "max_s": (1, 600), "shutter_max_us": (100, 200000),
               "gain_max": (1.0, 64.0), "ae_settle_s": (0.3, 10.0), "skip_frames": (0, 30),
               "queue_frames": (4, 200), "metadata": (0, 1)}
SCAN_MIN_FREE_MB = 500        # refuse to start a scan below this much free disk
SCAN_VIEWER_FPS = 2           # /stream.mjpg cap for other viewers while a scan runs
LIVE_CACHE = 40               # best frames kept in RAM for /scan/live (older ones are read from disk)
SENSOR_MAX_ANALOGUE_GAIN = 10.667   # IMX219: gain above this is applied digitally by the ISP (more noise)
SCAN_FILES = ("meta.json", "frames.csv", "bins.csv", "camera_metadata.json", "metering_metadata.json")
SCAN_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 8      # frames to let auto-exposure settle before a snapshot
STALL_SECONDS = 10     # restart rpicam-vid if it produces nothing for this long
RETRY_SECONDS = 3      # wait between start attempts (e.g. camera busy)
THERMAL_EVERY = 2      # seconds between temperature reads
FLAGS_EVERY = 10       # seconds between firmware throttle checks
TEMP_FILE = os.environ.get("TROVA_TEMP_FILE", "/sys/class/thermal/thermal_zone0/temp")   # env: tests only
HERE = os.path.dirname(os.path.abspath(__file__))


def _rpicam_vid():
    """The camera program. TROVA_RPICAM_VID (a JSON list) replaces it in tests on a PC."""
    v = os.environ.get("TROVA_RPICAM_VID", "rpicam-vid")
    return json.loads(v) if v.startswith("[") else [v]


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
    cmd = _rpicam_vid() + ["-v", "0", "-n", "-t", "0", "--flush",
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


class CpuMeter:
    """CPU use from /proc: per process in % of one core, and the whole Pi in % of all cores (Linux only)."""

    def __init__(self):
        self.prev = {}
        try:
            self.hz = os.sysconf("SC_CLK_TCK")
        except (AttributeError, ValueError, OSError):
            self.hz = None

    @staticmethod
    def _proc_ticks(pid):
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])          # utime + stime

    @staticmethod
    def _total():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]                          # all, idle + iowait

    def sample(self, pids):
        if not self.hz:
            return {}
        now, out = time.monotonic(), {}
        for name, pid in pids.items():
            try:
                ticks = self._proc_ticks(pid)
            except (OSError, ValueError, IndexError):
                continue
            prev = self.prev.get((name, pid))
            self.prev[(name, pid)] = (now, ticks)
            if prev and now > prev[0]:
                out[name] = round(100.0 * (ticks - prev[1]) / self.hz / (now - prev[0]), 1)
        try:
            total, idle = self._total()
            prev = self.prev.get("total")
            self.prev["total"] = (total, idle)
            if prev and total > prev[0]:
                out["total"] = round(100.0 * (1 - (idle - prev[1]) / (total - prev[0])), 1)
        except (OSError, ValueError, IndexError):
            pass
        return out


def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


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


# --- 3D scan --------------------------------------------------------------------
def parse_scan_params(raw):
    """Scan settings from the request (JSON body or query), typed and range-checked. ValueError if bad."""
    p = dict(SCAN_DEFAULTS)
    for key, value in raw.items():
        if key not in SCAN_DEFAULTS:
            raise ValueError(f"unknown scan parameter '{key}' (known: {', '.join(SCAN_DEFAULTS)})")
        default = SCAN_DEFAULTS[key]
        try:
            if isinstance(default, str):
                p[key] = str(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                p[key] = int(float(value))
            else:
                p[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}") from None
    for key, (lo, hi) in SCAN_BOUNDS.items():
        if not lo <= p[key] <= hi:
            raise ValueError(f"{key} {p[key]:g} is outside {lo:g}..{hi:g}")
    label = re.sub(r"[^0-9A-Za-z_-]", "", p["label"])[:32]
    if label != p["label"]:
        raise ValueError("label may only hold letters, digits, - and _ (max 32)")
    return p


def compute_lock(auto, shutter_max_us, gain_max):
    """Locked shutter / gain / white balance from settled auto-exposure values.

    Brightness ~ exposure x analogue gain x digital gain. If the auto shutter is above the cap, the shutter is
    capped and the gain raised so the brightness stays the same (up to gain_max).
    """
    exp = float(auto["ExposureTime"])
    target = exp * float(auto.get("AnalogueGain", 1.0)) * float(auto.get("DigitalGain", 1.0))
    shutter = min(exp, float(shutter_max_us))
    gain = target / shutter
    gain_capped = gain > gain_max
    gain = max(1.0, min(gain, gain_max))
    r, b = (float(g) for g in auto["ColourGains"][:2])
    return {"shutter_us": int(round(shutter)), "gain": round(gain, 4), "awb_gains": [round(r, 4), round(b, 4)],
            "shutter_capped": exp > shutter_max_us, "gain_capped": gain_capped,
            "brightness_vs_auto": round(shutter * gain / target, 3) if target else None}


class MetaTail:
    """Follows the JSON metadata file rpicam-vid writes (one object per frame) without blocking the camera."""

    FIELDS = ("ExposureTime", "AnalogueGain", "DigitalGain", "FocusFoM", "Lux")

    def __init__(self, path, keep=10):
        self.path = path
        self.pos = 0
        self.buf = ""
        self.recent = []
        self.keep = keep
        self.count = 0
        self.series = {k: [] for k in self.FIELDS}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop.wait(0.25):
            self.poll()

    def poll(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return
        if not data:
            return
        with self.lock:
            self.buf += data.decode("latin-1")
            dec, i = json.JSONDecoder(), 0
            while True:
                j = self.buf.find("{", i)
                if j < 0:
                    i = len(self.buf)
                    break
                try:
                    obj, end = dec.raw_decode(self.buf, j)
                except ValueError:          # object not complete yet
                    i = j
                    break
                self._add(obj)
                i = end
            self.buf = self.buf[i:]

    def _add(self, obj):
        self.count += 1
        self.recent = (self.recent + [obj])[-self.keep:]
        for k in self.FIELDS:
            if isinstance(obj.get(k), (int, float)):
                self.series[k].append(obj[k])

    def settled(self, n=5):
        """Median of the last n frames' exposure, gains and white balance; None without data."""
        with self.lock:
            last = [o for o in self.recent[-n:] if "ExposureTime" in o and "ColourGains" in o]
        if not last:
            return None
        out = {k: statistics.median(o[k] for o in last if k in o) for k in
               ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "ColourTemperature", "FocusFoM")
               if any(k in o for o in last)}
        out["ColourGains"] = [statistics.median(o["ColourGains"][c] for o in last) for c in (0, 1)]
        out["frames"] = self.count
        return out

    def stop(self):
        self._stop.set()
        self.thread.join(2)
        self.poll()

    def summary(self):
        with self.lock:
            s = {"frames": self.count}
            for k, v in self.series.items():
                if not v:
                    continue
                if k == "FocusFoM":
                    q = sorted(v)
                    s["focus_fom"] = {"p10": q[len(q) // 10], "p50": q[len(q) // 2], "p90": q[(9 * len(q)) // 10]}
                else:
                    s[k] = [round(min(v), 4), round(max(v), 4)]
            return s


class ScanRecorder:
    """One scan: every frame to disk through a bounded queue, plus the sharpest frame per time bin for live push."""

    def __init__(self, scan_id, sdir, params, profile, lock, auto, warnings, cmd, free_mb, temp):
        self.id, self.dir, self.params = scan_id, sdir, params
        self.frames_dir = os.path.join(sdir, "frames")
        self.profile, self.lock_values, self.auto, self.warnings, self.cmd = profile, lock, auto, warnings, cmd
        self.state = "recording"            # recording | stopping | done | aborted
        self.reason = None
        self.created = time.time()
        self.t_start = time.monotonic()
        self.t0_ns = self.t_last_ns = None
        self.skip = params["skip_frames"]
        self.started_camera = False
        self.stop_requested = False
        self.n = self.saved = self.bytes_total = self.write_errors = 0
        self.dropped = []
        self.bin_ns = int(params["bin_s"] * 1e9)
        self.cur_bin = self.cur_best = None
        self.live, self.live_idx = [], []
        self.cache = OrderedDict()
        self.saved_set = set()
        self.sent, self.live_clients, self.live_bytes = set(), 0, 0
        self.samples, self.events = [], []
        self.temp_start, self.temp_max, self.throttle_flags = temp, temp, 0
        self.free_mb_start, self.free_mb_end = free_mb, None
        self.camera_restarts = 0
        self.meta_tail = None
        self.finished_t = None
        self.cond = threading.Condition()
        self.add_lock = threading.Lock()
        self.first_frame = threading.Event()
        self.q = queue.Queue(params["queue_frames"])
        os.makedirs(self.frames_dir, exist_ok=True)
        self.csv = open(os.path.join(sdir, "frames.csv"), "w", newline="\n")
        self.csv.write("index,mono_ns,wall,bytes,saved\n")
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()
        self.write_meta()

    @property
    def active(self):
        return self.state in ("recording", "stopping")

    # --- frames in (camera reader thread) ---
    def add(self, frame, mono_ns, wall):
        with self.add_lock:
            if self.state != "recording":
                return
            if self.skip > 0:
                self.skip -= 1
                return
            idx = self.n
            self.n += 1
            if self.t0_ns is None:
                self.t0_ns = mono_ns
                self.first_frame.set()
            self.t_last_ns = mono_ns
            self.bytes_total += len(frame)
            try:
                self.q.put_nowait((idx, mono_ns, wall, frame))
            except queue.Full:
                self.dropped.append((idx, mono_ns, wall, len(frame)))
            b = (mono_ns - self.t0_ns) // self.bin_ns
            if self.cur_bin is not None and b != self.cur_bin:
                self._close_bin()
            self.cur_bin = b
            if self.cur_best is None or len(frame) > len(self.cur_best[3]):
                self.cur_best = (idx, mono_ns, wall, frame)

    def _close_bin(self):
        if self.cur_best is None:
            return
        idx, mono, wall, data = self.cur_best
        entry = {"index": idx, "bin": int(self.cur_bin), "mono_ns": mono, "wall": wall, "bytes": len(data)}
        with self.cond:
            self.live.append(entry)
            self.live_idx.append(idx)
            self.cache[idx] = data
            for k in list(self.cache):
                if len(self.cache) <= LIVE_CACHE:
                    break
                if k in self.saved_set:        # never forget a best frame that is not on disk
                    del self.cache[k]
            self.cond.notify_all()
        self.cur_best = None

    def _write_loop(self):
        n = 0
        while True:
            item = self.q.get()
            if item is None:
                break
            idx, mono, wall, data = item
            ok = 1
            try:
                with open(os.path.join(self.frames_dir, f"{idx:06d}.jpg"), "wb") as f:
                    f.write(data)
            except OSError as e:
                ok = 0
                self.write_errors += 1
                log(f"scan {self.id}: cannot write frame {idx}: {e}")
            self.csv.write(f"{idx},{mono},{wall:.6f},{len(data)},{ok}\n")
            n += 1
            if n % 30 == 0:
                self.csv.flush()
            with self.cond:
                if ok:
                    self.saved += 1
                    self.saved_set.add(idx)

    # --- frames out ---
    def frame_bytes(self, idx):
        with self.cond:
            data = self.cache.get(idx)
        if data is not None:
            return data
        try:
            with open(os.path.join(self.frames_dir, f"{idx:06d}.jpg"), "rb") as f:
                return f.read()
        except OSError:
            return None

    def mark_sent(self, idx, nbytes):
        with self.cond:
            self.sent.add(idx)
            self.live_bytes += nbytes

    # --- housekeeping (supervisor thread) ---
    def sample(self, temp, cpu, mem, flags):
        self.temp_max = max(self.temp_max, temp)
        self.throttle_flags |= flags
        self.samples.append({"t": round(time.monotonic() - self.t_start, 1), "temp": temp, "cpu": cpu,
                             "mem_available_mb": mem})

    def event(self, kind, **info):
        self.events.append({"t": round(time.monotonic() - self.t_start, 1), "event": kind, **info})

    # --- stop ---
    def begin_stop(self):
        """No more frames are recorded; the last bin is published."""
        with self.add_lock:
            if self.state != "recording":
                return False
            self.state = "stopping"
            self._close_bin()
        return True

    def finish(self, aborted, reason):
        self.q.put(None)
        self.writer.join(60)
        for idx, mono, wall, nbytes in self.dropped:
            self.csv.write(f"{idx},{mono},{wall:.6f},{nbytes},0\n")
        self.csv.close()
        with open(os.path.join(self.dir, "bins.csv"), "w", newline="\n") as f:
            f.write("bin,index,mono_ns,bytes,sent\n")
            for e in self.live:
                f.write(f"{e['bin']},{e['index']},{e['mono_ns']},{e['bytes']},{int(e['index'] in self.sent)}\n")
        if self.meta_tail:
            self.meta_tail.stop()
        with contextlib.suppress(OSError):
            self.free_mb_end = shutil.disk_usage(self.dir).free // (1 << 20)
        with self.cond:
            self.state = "aborted" if aborted else "done"
            self.reason = reason
            self.finished_t = time.time()
            self.cond.notify_all()
        self.write_meta()
        log(f"scan {self.id} {self.state} ({reason}): {self.n} frames, {len(self.dropped)} dropped, "
            f"{len(self.live)} bins")

    # --- reports ---
    def duration_s(self):
        if self.t0_ns is None or self.t_last_ns is None:
            return 0.0
        return (self.t_last_ns - self.t0_ns) / 1e9

    def brief(self):
        d = self.duration_s()
        return {"id": self.id, "state": self.state, "reason": self.reason, "frames": self.n,
                "dropped": len(self.dropped), "elapsed_s": round(time.monotonic() - self.t_start, 1)
                if self.active else round(d, 1), "fps": round((self.n - 1) / d, 2) if d > 0 else 0,
                "bins": len(self.live), "bins_sent": len(self.sent), "live_clients": self.live_clients,
                "temp_max": self.temp_max}

    def summary(self):
        d = self.duration_s()
        cpu = {}
        for s in self.samples:
            for k, v in (s.get("cpu") or {}).items():
                cpu.setdefault(k, []).append(v)
        out = {
            "scan_id": self.id, "state": self.state, "reason": self.reason,
            "frames": self.n, "saved": self.saved, "dropped": len(self.dropped), "write_errors": self.write_errors,
            "duration_s": round(d, 2), "real_fps": round((self.n - 1) / d, 2) if d > 0 else 0,
            "mean_kb": round(self.bytes_total / self.n / 1024, 1) if self.n else 0,
            "bins": len(self.live), "bins_sent": len(self.sent), "live_mbytes_sent": round(self.live_bytes / 1e6, 2),
            "temp_start": self.temp_start, "temp_max": self.temp_max,
            "cpu_mean": {k: round(sum(v) / len(v), 1) for k, v in cpu.items()},
            "cpu_max": {k: max(v) for k, v in cpu.items()},
            "mem_available_min_mb": min((s["mem_available_mb"] for s in self.samples
                                         if s.get("mem_available_mb") is not None), default=None),
            "protection_events": [e for e in self.events if e["event"] == "protection"],
            "throttle_flags": hex(self.throttle_flags), "camera_restarts": self.camera_restarts,
            "free_mb_start": self.free_mb_start, "free_mb_end": self.free_mb_end,
            "locked": self.lock_values,
        }
        if self.meta_tail:
            out["camera_metadata"] = self.meta_tail.summary()
        return out

    def start_info(self):
        return {"scan_id": self.id, "state": self.state, "profile": self.profile._asdict(),
                "locked": self.lock_values, "auto": self.auto, "warnings": self.warnings, "params": self.params}

    def write_meta(self):
        meta = {"scan_id": self.id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.created)),
                "created_wall": self.created, "finished_wall": self.finished_t,
                "params": self.params, "profile": self.profile._asdict(), "auto": self.auto,
                "locked": self.lock_values, "warnings": self.warnings, "command": self.cmd,
                "summary": self.summary(), "temperature_series": self.samples, "events": self.events}
        path = os.path.join(self.dir, "meta.json")
        try:
            with open(path + ".tmp", "w") as f:
                json.dump(meta, f, indent=1)
            os.replace(path + ".tmp", path)
        except OSError as e:
            log(f"scan {self.id}: cannot write meta.json: {e}")


class Camera:
    """Runs rpicam-vid while there are users, keeps the newest frame, minds the heat, records scans."""

    def __init__(self, normal, light, rotate, extra, idle_timeout, config_warnings,
                 scans_dir=SCANS_DIR, scan_min_free_mb=SCAN_MIN_FREE_MB):
        self.profiles = {"normal": normal, "light": light}
        self.rotate = rotate
        self.extra = extra
        self.idle_timeout = idle_timeout
        self.config_warnings = config_warnings
        self.thermal = Thermal()
        self.level = "normal"          # normal | light | paused
        self.cool_since = None
        self.running_level = None      # normal | light | meter | scan
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
        # 3D scan: a temporary camera mode used instead of "normal" while the protection level is normal
        self.scans_dir = scans_dir
        self.scan_min_free_mb = scan_min_free_mb
        self.mode = None               # None | meter | scan
        self.mode_profiles = {}
        self.mode_extra = {}
        self.scan = None               # current or last ScanRecorder
        self.scan_lock = threading.Lock()
        self.cpu = CpuMeter()
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

    def scanning(self):
        rec = self.scan
        return self.mode is not None or (rec is not None and rec.active)

    def _profile_for(self, level):
        if level in ("meter", "scan"):
            return self.mode_profiles[level]
        return self.profiles["normal" if level == "normal" else "light"]

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
            want = self.mode if (self.mode and self.level == "normal") else self.level
            profile = self._profile_for(want if want != "paused" else "light")
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
            rec = self.scan
            scan = {"state": "metering"} if self.mode == "meter" else (rec.brief() if rec is not None else None)
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
                "scan": scan,
            }

    # --- process management ---------------------------------------------
    def _supervise(self):
        next_try = next_thermal = 0.0
        while True:
            self.wake.wait(0.5)
            self.wake.clear()
            now = time.monotonic()
            rec = self.scan

            if now >= next_thermal:
                next_thermal = now + THERMAL_EVERY
                self.thermal.update(now)
                level, self.cool_since = next_level(self.level, self.thermal.avg,
                                                    self.cool_since, now)
                if level != self.level:
                    log(f"temperature {self.thermal.avg:.1f} C: {self.level} -> {level}")
                    if rec is not None and rec.active:
                        rec.event("protection", **{"from": self.level, "to": level,
                                                   "temp_avg": round(self.thermal.avg, 1)})
                    with self.cond:
                        self.level = level
                if rec is not None and rec.state == "recording":
                    with self.cond:
                        pids = {"rpicam": self.proc.pid} if self.proc is not None else {}
                    pids["server"] = os.getpid()
                    rec.sample(round(self.thermal.temp, 1), self.cpu.sample(pids), mem_available_mb(),
                               self.thermal.flags)
                    if self.level != "normal" and not rec.stop_requested:
                        rec.stop_requested = True
                        threading.Thread(target=self.scan_stop, daemon=True, kwargs={
                            "reason": f"protection {self.level} ({self.thermal.avg:.0f} C)",
                            "aborted": True}).start()

            if (rec is not None and rec.state == "recording" and not rec.stop_requested
                    and now - rec.t_start >= rec.params["max_s"]):
                rec.stop_requested = True
                threading.Thread(target=self.scan_stop, daemon=True,
                                 kwargs={"reason": f"max_s {rec.params['max_s']:g} s reached"}).start()

            with self.cond:
                proc, users = self.proc, self.users
                want = self.mode if (self.mode and self.level == "normal") else self.level
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
        extra = self.extra + self.mode_extra.get(level, []) if level in ("meter", "scan") else self.extra
        cmd = build_cmd(self._profile_for(level), self.rotate, extra)
        log("starting camera: " + " ".join(cmd))
        rec = self.scan
        if level == "scan" and rec is not None and rec.state == "recording":
            if rec.started_camera:              # a restart in the middle of a scan
                rec.camera_restarts += 1
                rec.event("camera_restart")
                with rec.add_lock:
                    rec.skip = rec.params["skip_frames"]
            rec.started_camera = True
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
        threading.Thread(target=self._read, args=(proc, level), daemon=True).start()

    def _stop(self, proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
        except (ValueError, OSError):      # Windows (tests only) has no SIGINT for child processes
            proc.terminate()
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

    def _read(self, proc, level):
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
                self._publish(bytes(buf[soi:end]), level)
                pos = end
            del buf[:pos]
            if len(buf) > 8 << 20:                 # garbage, resync
                buf.clear()
        out.close()

    def _publish(self, frame, level="normal"):
        now_ns = time.monotonic_ns()
        now = now_ns / 1e9
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
            rec = self.scan
        if level == "scan" and rec is not None:
            rec.add(frame, now_ns, time.time())

    # --- 3D scan -----------------------------------------------------------
    def _wait_running(self, mode, timeout):
        """Wait until rpicam-vid runs in this mode and has produced a frame."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while not (self.running_level == mode and self.run_frames > 0 and self.proc is not None):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.level != "normal":
                    return False
                self.cond.wait(min(remaining, 0.5))
        return True

    def _end_mode(self):
        with self.cond:
            self.mode = None
        self.release()
        self.wake.set()

    def scan_start(self, raw):
        """Meter, lock exposure / gain / white balance, then record. Returns (http_code, json_dict)."""
        if not self.scan_lock.acquire(blocking=False):
            return 409, {"error": "a scan is already starting or stopping"}
        try:
            if self.scan is not None and self.scan.active:
                return 409, {"error": f"scan {self.scan.id} is already running: POST /scan/stop first"}
            if self.level != "normal":
                return 503, {"error": f"protection is '{self.level}' ({self.thermal.avg:.0f} C): "
                                      "wait until the Pi cools down"}
            try:
                p = parse_scan_params(raw)
            except ValueError as e:
                return 400, {"error": str(e)}
            os.makedirs(self.scans_dir, exist_ok=True)
            free_mb = shutil.disk_usage(self.scans_dir).free // (1 << 20)
            if free_mb < self.scan_min_free_mb:
                return 507, {"error": f"only {free_mb} MB free on the Pi (need {self.scan_min_free_mb}): "
                                      "DELETE old scans"}
            profile, warnings = clamp_settings(SimpleNamespace(width=p["width"], height=p["height"],
                                                               framerate=p["fps"], quality=p["quality"]))
            scan_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{p['label']}" if p["label"] else "")
            base, k = scan_id, 2
            while os.path.exists(os.path.join(self.scans_dir, scan_id)):
                scan_id, k = f"{base}-{k}", k + 1
            sdir = os.path.join(self.scans_dir, scan_id)
            os.makedirs(sdir)

            # 1) meter: auto exposure and white balance at the scan profile, metadata to the scan folder
            t_req = time.monotonic()
            meter_path = os.path.join(sdir, "metering_metadata.json")
            tail = MetaTail(meter_path)
            self.mode_profiles = {"meter": profile, "scan": profile}
            self.mode_extra = {"meter": ["--metadata", meter_path, "--metadata-format", "json"]}
            self.acquire()
            with self.cond:
                self.mode = "meter"
            self.wake.set()
            ok = self._wait_running("meter", timeout=15)
            if ok:
                time.sleep(p["ae_settle_s"])
            tail.stop()
            auto = tail.settled()
            if not ok or auto is None or self.level != "normal":
                self._end_mode()
                shutil.rmtree(sdir, ignore_errors=True)       # only the metering file: nothing recorded
                return 503, {"error": "camera did not start or gave no metadata while metering"
                             if self.level == "normal" else f"protection went to '{self.level}'",
                             "camera": self.status()["camera"]}
            lock = compute_lock(auto, p["shutter_max_us"], p["gain_max"])
            if lock["gain"] > SENSOR_MAX_ANALOGUE_GAIN:
                warnings.append(f"locked gain {lock['gain']:g} is above the sensor's analogue maximum "
                                f"{SENSOR_MAX_ANALOGUE_GAIN:g}: the rest is digital gain (more noise). More light helps")
            if lock["gain_capped"]:
                warnings.append(f"gain capped at {p['gain_max']:g}: frames are "
                                f"{100 * (1 - lock['brightness_vs_auto']):.0f}% darker than auto exposure. "
                                "Add light or raise shutter_max_us")

            # 2) record with the locked values
            extra = ["--shutter", str(lock["shutter_us"]), "--gain", f"{lock['gain']:.4f}",
                     "--awbgains", f"{lock['awb_gains'][0]:.4f},{lock['awb_gains'][1]:.4f}"]
            if p["metadata"]:
                extra += ["--metadata", os.path.join(sdir, "camera_metadata.json"), "--metadata-format", "json"]
            self.mode_extra["scan"] = extra
            rec = ScanRecorder(scan_id, sdir, p, profile, lock, auto, warnings,
                               build_cmd(profile, self.rotate, self.extra + extra), free_mb,
                               round(self.thermal.temp, 1))
            if p["metadata"]:
                rec.meta_tail = MetaTail(os.path.join(sdir, "camera_metadata.json"))
            with self.cond:
                self.scan = rec
                self.mode = "scan"
            self.wake.set()
            log(f"scan {scan_id}: auto {auto.get('ExposureTime')} us x{auto.get('AnalogueGain')} -> locked "
                f"{lock['shutter_us']} us x{lock['gain']}, awb {lock['awb_gains']}")
            if not rec.first_frame.wait(15):
                self._finish_scan(rec, aborted=True, reason="no frames after locking the exposure")
                return 503, {"error": "no frames after locking the exposure", "summary": rec.summary()}
            info = rec.start_info()
            info["start_latency_s"] = round(time.monotonic() - t_req, 2)
            rec.event("recording", start_latency_s=info["start_latency_s"])
            return 200, info
        finally:
            self.scan_lock.release()

    def _finish_scan(self, rec, aborted, reason):
        if rec.begin_stop():
            self._end_mode()
            rec.finish(aborted, reason)

    def scan_stop(self, reason="stop requested", aborted=False):
        if not self.scan_lock.acquire(timeout=30):
            return 409, {"error": "a scan is still starting"}
        try:
            rec = self.scan
            if rec is None:
                return 404, {"error": "no scan since the server started"}
            if rec.active:
                self._finish_scan(rec, aborted, reason)
            return 200, rec.summary()
        finally:
            self.scan_lock.release()

    def scan_dir(self, scan_id):
        if not SCAN_ID_RE.match(scan_id or ""):
            return None
        d = os.path.join(self.scans_dir, scan_id)
        return d if os.path.isdir(d) else None

    def scan_list(self):
        out = []
        if not os.path.isdir(self.scans_dir):
            return out
        for name in sorted(os.listdir(self.scans_dir)):
            d = self.scan_dir(name)
            if d is None:
                continue
            item = {"id": name}
            with contextlib.suppress(OSError, ValueError):
                with open(os.path.join(d, "meta.json")) as f:
                    m = json.load(f)
                s = m.get("summary", {})
                item.update(state=s.get("state"), created=m.get("created"), frames=s.get("frames"),
                            duration_s=s.get("duration_s"), real_fps=s.get("real_fps"), bins=s.get("bins"),
                            profile=m.get("profile"))
            if self.scan is not None and self.scan.id == name:
                item["state"] = self.scan.state
            with contextlib.suppress(OSError):
                item["mbytes"] = round(sum(e.stat().st_size for e in os.scandir(os.path.join(d, "frames"))) / 1e6, 1)
            out.append(item)
        return out

    def scan_delete(self, scan_id):
        d = self.scan_dir(scan_id)
        if d is None:
            return 404, {"error": f"no scan {scan_id}"}
        if self.scan is not None and self.scan.id == scan_id and self.scan.active:
            return 409, {"error": "the scan is running: POST /scan/stop first"}
        size = sum(e.stat().st_size for e in os.scandir(os.path.join(d, "frames"))) \
            if os.path.isdir(os.path.join(d, "frames")) else 0
        shutil.rmtree(d)
        return 200, {"deleted": scan_id, "freed_mb": round(size / 1e6, 1)}


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
    server_version = "trova-stream/2.1"
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
        elif url.path.startswith("/scan/"):
            self._scan_get(url.path, query)
        else:
            self._send(404, "text/plain", b"not found\n")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/scan/start":
            try:
                raw = {k: v[-1] for k, v in parse_qs(url.query).items()}
                raw.update(self._json_body())
            except ValueError as e:
                self._json(400, {"error": f"bad request body: {e}"})
                return
            self._json(*self.cam.scan_start(raw))
        elif url.path == "/scan/stop":
            self._json(*self.cam.scan_stop())
        else:
            self._send(404, "text/plain", b"not found\n")

    def do_DELETE(self):
        m = re.match(r"^/scan/([^/]+)$", urlparse(self.path).path)
        if m and m.group(1) not in ("list", "live", "clock", "start", "stop"):
            self._json(*self.cam.scan_delete(m.group(1)))
        else:
            self._send(404, "text/plain", b"not found\n")

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        body = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def _send(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, "application/json", json.dumps(obj).encode())

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
                gap = max(min_gap, 1.0 / SCAN_VIEWER_FPS) if self.cam.scanning() else min_gap
                if gap:
                    time.sleep(max(0.0, gap - (time.monotonic() - t)))
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

    # --- 3D scan endpoints -----------------------------------------------
    def _scan_get(self, path, query):
        if path == "/scan/live":
            self._scan_live(query)
        elif path == "/scan/list":
            self._json(200, {"scans": self.cam.scan_list(), "scans_dir": self.cam.scans_dir,
                             "free_mb": shutil.disk_usage(self.cam.scans_dir).free // (1 << 20)
                             if os.path.isdir(self.cam.scans_dir) else None})
        elif path == "/scan/clock":
            self._json(200, {"wall": time.time(), "mono_ns": time.monotonic_ns()})
        elif path.endswith(".tar") and path.count("/") == 2:
            self._scan_tar(path[len("/scan/"):-len(".tar")])
        else:
            m = re.match(r"^/scan/([^/]+)/frame/(\d+)$", path) or re.match(r"^/scan/([^/]+)/([^/]+)$", path)
            d = self.cam.scan_dir(m.group(1)) if m else None
            if d is None:
                self._send(404, "text/plain", b"no such scan\n")
            elif m.group(2).isdigit():
                self._scan_frame(m.group(1), d, int(m.group(2)))
            elif m.group(2) in SCAN_FILES and os.path.isfile(os.path.join(d, m.group(2))):
                with open(os.path.join(d, m.group(2)), "rb") as f:
                    body = f.read()
                ctype = "text/csv" if m.group(2).endswith(".csv") else "application/json"
                self._send(200, ctype, body)
            else:
                self._send(404, "text/plain", b"not found\n")

    def _scan_frame(self, scan_id, d, idx):
        rec = self.cam.scan
        data = None
        if rec is not None and rec.id == scan_id:
            data = rec.frame_bytes(idx)
        else:
            with contextlib.suppress(OSError):
                with open(os.path.join(d, "frames", f"{idx:06d}.jpg"), "rb") as f:
                    data = f.read()
        if data is None:
            self._send(404, "text/plain", b"frame not recorded or dropped\n")
        else:
            self._send(200, "image/jpeg", data, [("X-Frame-Index", str(idx))])

    def _scan_tar(self, scan_id):
        d = self.cam.scan_dir(scan_id)
        if d is None:
            self._send(404, "text/plain", b"no such scan\n")
            return
        if self.cam.scan is not None and self.cam.scan.id == scan_id and self.cam.scan.active:
            self._send(409, "text/plain", b"the scan is still recording: stop it first\n")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-tar")
        self.send_header("Content-Disposition", f'attachment; filename="{scan_id}.tar"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            with tarfile.open(fileobj=self.wfile, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
                tar.add(d, arcname=scan_id)
        except OSError:
            pass

    def _scan_live(self, query):
        """multipart/mixed: the best frame of each bin, with index / bin / timestamps in the part headers.
        Resume after a WiFi drop with ?since=<last index received>. Ends with a JSON part when the scan stops."""
        rec = self.cam.scan
        want = (query.get("id") or [None])[0]
        if rec is None or (want and want != rec.id):
            self._send(404, "text/plain", b"no such scan in memory: use bins.csv and /scan/<id>/frame/<i>\n")
            return
        try:
            since = int((query.get("since") or ["-1"])[0])
        except ValueError:
            since = -1
        self.send_response(200)
        self.send_header("Content-Type", "multipart/mixed; boundary=trovascan")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Scan-Id", rec.id)
        self.end_headers()
        with rec.cond:
            rec.live_clients += 1
        try:
            while True:
                with rec.cond:
                    k = bisect.bisect_right(rec.live_idx, since)
                    while k >= len(rec.live) and rec.active:
                        rec.cond.wait(1.0)
                        k = bisect.bisect_right(rec.live_idx, since)
                    batch = rec.live[k:]
                    finished = not rec.active
                for e in batch:
                    data = rec.frame_bytes(e["index"])
                    since = e["index"]
                    if data is None:
                        continue
                    self.wfile.write((
                        "--trovascan\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n"
                        "X-Frame-Index: %d\r\nX-Bin: %d\r\nX-Mono-Ns: %d\r\nX-Wall: %.6f\r\nX-Bytes: %d\r\n"
                        "X-Sent-Mono-Ns: %d\r\n\r\n" % (len(data), e["index"], e["bin"], e["mono_ns"], e["wall"],
                                                       e["bytes"], time.monotonic_ns())).encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    rec.mark_sent(e["index"], len(data))
                if finished:
                    end = json.dumps({"end": True, "scan_id": rec.id, "state": rec.state, "reason": rec.reason,
                                      "bins": len(rec.live), "last_index": since}).encode()
                    self.wfile.write(b"--trovascan\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                                     b"%s\r\n--trovascan--\r\n" % (len(end), end))
                    break
        except OSError:          # client went away: it resumes with ?since=
            pass
        finally:
            with rec.cond:
                rec.live_clients -= 1


def main():
    p = argparse.ArgumentParser(
        description="Camera server for trova-cam: live page, snapshots, heat protection, 3D scan recording.",
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
    p.add_argument("--scans-dir", default=SCANS_DIR, help=f"where scans are recorded, default {SCANS_DIR}")
    p.add_argument("--scan-min-free-mb", type=int, default=SCAN_MIN_FREE_MB,
                   help=f"refuse a scan below this much free disk, default {SCAN_MIN_FREE_MB}")
    args, extra = p.parse_known_args()

    normal, warnings = clamp_settings(args)
    light = Profile(*LIGHT_MODE)
    Handler.cam = Camera(normal, light, args.rotate, extra, args.idle_timeout, warnings,
                         scans_dir=args.scans_dir, scan_min_free_mb=args.scan_min_free_mb)
    with open(os.path.join(HERE, "index.html"), "rb") as f:
        Handler.page = f.read()

    server = Server(("::", args.port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log(f"trova-stream on http://0.0.0.0:{args.port}/  ({normal.width}x{normal.height} "
        f"@ {normal.framerate:g} fps, q{normal.quality}, rotate {args.rotate}, "
        f"temp {Handler.cam.thermal.temp:.1f} C, scans in {args.scans_dir})")
    for w in warnings:
        log("warning: " + w)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rec = Handler.cam.scan
        if rec is not None and rec.active:
            Handler.cam.scan_stop(reason="server stopped", aborted=True)
        Handler.cam.stop()


if __name__ == "__main__":
    main()
