"""
trova_cam: grab frames from the trova-cam Pi for object recognition, and drive its 3D-scan mode.

    from trova_cam import TrovaCam
    cam = TrovaCam()              # or TrovaCam("192.168.5.160")
    frame = cam.get_frame()       # OpenCV BGR image, captured after this call

Use get_frame() instead of cv2.VideoCapture on the stream: VideoCapture queues
frames, so right after the arm moves it can return a picture from before the
move. get_frame() always waits for a frame captured after the request.
Let the arm stop before calling it, or the picture will be blurred.

Frames are already upright (the camera does the 180° turn itself).
When the Pi is hot the server drops to 640x360 (16:9); status()["protection"] says so.

3D scan (see trova-stream/stream.py):
    info = cam.scan_start(fps=15, bin_s=0.33)          # locks exposure, starts recording on the Pi
    with cam.scan_live(since=-1) as live:               # sharpest frame per time bin, as they are taken
        for f in live:                                  # LiveFrame(index, bin, ..., data=JPEG bytes)
            ...
    summary = cam.scan_stop()
    cam.scan_frame(scan_id, index), cam.scan_bins(scan_id), cam.scan_tar(scan_id, "x.tar"),
    cam.scan_list(), cam.scan_delete(scan_id), cam.clock_offset()
Every method raises CameraError on network or server errors, like the rest.

Needs:  pip install opencv-python   (get_jpeg(), the scan methods and the command line work without it)
Try:    python trova_cam.py [host]  -> saves trova-frame.jpg and prints the status
"""
import csv
import http.client
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import namedtuple

DEFAULT_HOST = os.environ.get("TROVA_CAM_HOST", "trova-cam.local")
LIVE_BOUNDARY = b"trovascan"

# One frame from /scan/live. wall = Pi wall clock at capture; recv_wall = this PC's clock when it arrived.
LiveFrame = namedtuple("LiveFrame", "index bin mono_ns wall nbytes sent_mono_ns recv_wall data")


class CameraError(RuntimeError):
    pass


def _read_part(fp, boundary):
    """Next part of a multipart body: (headers dict with lower-case keys, body bytes), or None at the end."""
    open_b, close_b = b"--" + boundary, b"--" + boundary + b"--"
    while True:
        line = fp.readline()
        if not line:
            return None
        line = line.strip()
        if line == close_b:
            return None
        if line == open_b:
            break
    headers = {}
    while True:
        h = fp.readline()
        if not h:
            return None
        h = h.strip()
        if not h:
            break
        k, _, v = h.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    n = int(headers["content-length"])
    body = fp.read(n)
    if len(body) != n:
        return None
    fp.readline()
    return headers, body


class LiveStream:
    """Iterator over /scan/live. Ends cleanly (StopIteration, .end = the server's end info) when the scan stops;
    raises CameraError if the connection drops first: reconnect with since=<last index received>."""

    def __init__(self, resp):
        self.resp = resp
        self.scan_id = resp.headers.get("X-Scan-Id")
        self.end = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.end is not None:
            raise StopIteration
        try:
            part = _read_part(self.resp, LIVE_BOUNDARY)
        except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
            self.close()
            raise CameraError(f"live stream dropped: {e}") from None
        if part is None:
            self.close()
            raise CameraError("live stream dropped before the end of the scan")
        h, body = part
        if h.get("content-type", "").startswith("application/json"):
            self.end = json.loads(body)
            self.close()
            raise StopIteration
        return LiveFrame(int(h["x-frame-index"]), int(h["x-bin"]), int(h["x-mono-ns"]), float(h["x-wall"]),
                         int(h["x-bytes"]), int(h.get("x-sent-mono-ns", 0)), time.time(), body)

    def close(self):
        try:
            self.resp.close()
        except (OSError, AttributeError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrovaCam:
    def __init__(self, host=DEFAULT_HOST, port=8080, timeout=15):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _open(self, method, path, body=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace").strip()
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            raise CameraError(f"{path}: HTTP {e.code} {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise CameraError(f"can't reach {self.base}: {reason}") from None

    def _request(self, method, path, body=None, timeout=None):
        with self._open(method, path, body, timeout) as r:
            try:
                return r.read()
            except (OSError, http.client.HTTPException) as e:
                raise CameraError(f"{path}: connection dropped: {e}") from None

    def _get(self, path):
        return self._request("GET", path)

    def _json(self, method, path, body=None, timeout=None):
        return json.loads(self._request(method, path, body, timeout))

    def status(self):
        """Camera state, temperature, protection level, warnings and scan state."""
        return json.loads(self._get("/status.json"))

    def get_jpeg(self):
        """One JPEG (bytes) captured after this call."""
        return self._get("/snapshot.jpg")

    def get_frame(self):
        """One frame as an OpenCV BGR numpy array, captured after this call."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise CameraError("get_frame() needs OpenCV: pip install opencv-python") from None
        img = cv2.imdecode(np.frombuffer(self.get_jpeg(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise CameraError("could not decode the JPEG from the camera")
        return img

    # --- 3D scan -------------------------------------------------------------
    def scan_start(self, **params):
        """Meter and lock exposure, then record on the Pi. Returns scan_id, locked / auto values, warnings.
        Parameters: width height fps quality shutter_max_us bin_s max_s gain_max ae_settle_s skip_frames
        queue_frames metadata label (see stream.py)."""
        return self._json("POST", "/scan/start", params, timeout=max(self.timeout, 40))

    def scan_stop(self):
        """Stop recording, restore the default profile, return the Pi's summary."""
        return self._json("POST", "/scan/stop", {}, timeout=max(self.timeout, 90))

    def scan_live(self, since=-1, scan_id=None, timeout=20):
        """LiveStream of the sharpest frame per bin with index > since (use it with `with`)."""
        q = f"?since={int(since)}" + (f"&id={scan_id}" if scan_id else "")
        return LiveStream(self._open("GET", "/scan/live" + q, timeout=timeout))

    def scan_frame(self, scan_id, index):
        """One recorded frame (JPEG bytes)."""
        return self._get(f"/scan/{scan_id}/frame/{int(index)}")

    def scan_file(self, scan_id, name):
        """meta.json, frames.csv, bins.csv, camera_metadata.json or metering_metadata.json (bytes)."""
        return self._get(f"/scan/{scan_id}/{name}")

    def scan_bins(self, scan_id):
        """The best frame of every bin (written when the scan stops): list of dicts bin, index, mono_ns, bytes, sent."""
        rows = csv.DictReader(io.StringIO(self.scan_file(scan_id, "bins.csv").decode()))
        return [{k: int(v) for k, v in r.items()} for r in rows]

    def scan_tar(self, scan_id, dest, progress=None, chunk=1 << 20):
        """Download the whole scan (every frame, uncompressed tar) to dest. Returns the number of bytes."""
        n = 0
        tmp = str(dest) + ".part"
        with self._open("GET", f"/scan/{scan_id}.tar", timeout=max(self.timeout, 60)) as r, open(tmp, "wb") as f:
            while True:
                try:
                    data = r.read(chunk)
                except (OSError, http.client.HTTPException) as e:
                    raise CameraError(f"tar download dropped after {n} bytes: {e}") from None
                if not data:
                    break
                f.write(data)
                n += len(data)
                if progress:
                    progress(n)
        os.replace(tmp, dest)
        return n

    def scan_list(self):
        return self._json("GET", "/scan/list")["scans"]

    def scan_delete(self, scan_id):
        return self._json("DELETE", f"/scan/{scan_id}")

    def scan_clock(self):
        return self._json("GET", "/scan/clock")

    def clock_offset(self, n=5):
        """(Pi wall clock - this PC's wall clock, round trip) in seconds, from the fastest of n requests."""
        best = None
        for _ in range(n):
            t0 = time.time()
            pi = self.scan_clock()["wall"]
            t1 = time.time()
            if best is None or t1 - t0 < best[1]:
                best = (pi - (t0 + t1) / 2, t1 - t0)
        return best


if __name__ == "__main__":
    cam = TrovaCam(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST)
    try:
        st = cam.status()
        print(f"camera {st['camera']}, {st['width']}x{st['height']}, {st['temp']} C, "
              f"protection {st['protection']}")
        for w in st["warnings"]:
            print("warning:", w)
        t = time.time()
        jpg = cam.get_jpeg()
        with open("trova-frame.jpg", "wb") as f:
            f.write(jpg)
        print(f"saved trova-frame.jpg ({len(jpg) // 1024} KB) in {time.time() - t:.2f} s")
    except CameraError as e:
        sys.exit(f"error: {e}")
