"""HTTP client for the camera on the Raspberry Pi (Sony IMX219, server in trova-stream/).

The Pi serves Wi-Fi on port 8080. It is not the robot's Ethernet API and not a local
/dev/video device. GET /snapshot.jpg is captured after the request: use it after the arm
has stopped. The MJPEG stream is for display only; OpenCV's VideoCapture queue can hand
back a frame from before the move.

The server already rotates the picture 180 degrees. This client does not rotate again
unless TROVA_CAM_EXTRA_ROTATE is set. A failed request raises CameraError and does not
return an older buffer.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_HOST = "trova-cam.local"
DEFAULT_PORT = 8080


class CameraError(RuntimeError):
    pass


def _jpeg_complete(data: bytes) -> bool:
    return len(data) >= 32 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


@dataclass
class Capture:
    jpeg: bytes
    meta: dict
    requested_at: float
    received_at: float


@dataclass
class PiCam:
    """One Pi camera. Construction does not open a connection."""

    host: str = ""
    port: int = DEFAULT_PORT
    timeout: float = 15.0
    retries: int = 2
    status_timeout: float = 3.0
    extra_rotate: int = 0
    base: str = field(init=False)

    def __post_init__(self) -> None:
        host = (self.host or os.environ.get("TROVA_CAM_HOST") or DEFAULT_HOST).strip()
        port = int(self.port or os.environ.get("TROVA_CAM_PORT") or DEFAULT_PORT)
        self.host, self.port = host, port
        self.base = f"http://{host}:{port}"
        extra = self.extra_rotate
        if extra == 0 and os.environ.get("TROVA_CAM_EXTRA_ROTATE"):
            extra = int(os.environ["TROVA_CAM_EXTRA_ROTATE"])
        if extra % 360 not in (0, 90, 180, 270):
            raise CameraError("TROVA_CAM_EXTRA_ROTATE must be 0, 90, 180 or 270")
        self.extra_rotate = extra % 360

    def _get(self, path: str, timeout: float, retries: int) -> bytes:
        last: CameraError | None = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(self.base + path, headers={"Cache-Control": "no-cache"})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace").strip()
                err = CameraError(f"{path}: HTTP {exc.code} {detail}")
                # 503 is the thermal pause (or a snapshot that never arrived). Retrying hides it.
                if exc.code in (500, 502, 504) and attempt < retries:
                    last = err
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise err from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                last = CameraError(f"can't reach {self.base}: {reason}")
                if attempt < retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise last from None
        raise last or CameraError(f"can't reach {self.base}")

    def _open(self, method, path, body=None, timeout=None):
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=payload, method=method,
                                         headers={'Content-Type':'application/json','Cache-Control':'no-cache'})
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except (urllib.error.URLError, OSError) as exc:
            raise CameraError(str(exc)) from None

    def _json(self, method, path, body=None, timeout=None):
        with self._open(method,path,body,timeout) as response:
            return json.loads(response.read())

    def scan_start(self, **params):
        return self._json('POST','/scan/start',params,timeout=max(self.timeout,40))

    def scan_stop(self):
        return self._json('POST','/scan/stop',{},timeout=max(self.timeout,90))

    def scan_live(self, since=-1, scan_id=None, timeout=20):
        from .scan3d.streaming import LiveStream
        from urllib.parse import urlencode
        query = {'since':int(since)}
        if scan_id is not None: query['id'] = scan_id
        return LiveStream(self._open('GET','/scan/live?'+urlencode(query),timeout=timeout))

    def scan_frame(self, scan_id, index):
        self._scan_identifier(scan_id)
        return self._get(f'/scan/{scan_id}/frame/{int(index)}',self.timeout,self.retries)

    @staticmethod
    def _scan_identifier(value):
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',str(value)):
            raise CameraError('Invalid scan identifier')

    def scan_file(self, scan_id, name):
        self._scan_identifier(scan_id)
        if name not in {'meta.json','frames.csv','bins.csv','camera_metadata.json','metering_metadata.json'}:
            raise CameraError('Unsupported scan metadata file')
        return self._get(f'/scan/{scan_id}/{name}',self.timeout,self.retries)

    def scan_bins(self, scan_id):
        import csv, io
        return [{key:int(value) for key,value in row.items()} for row in
                csv.DictReader(io.StringIO(self.scan_file(scan_id,'bins.csv').decode()))]

    def scan_clock(self):
        return self._json('GET','/scan/clock')

    def clock_offset(self, n=5):
        samples=[]
        for _ in range(max(1,min(int(n),10))):
            before=time.time(); wall=self.scan_clock()['wall']; after=time.time()
            samples.append((wall-(before+after)/2,after-before))
        return min(samples,key=lambda value:value[1])

    def status(self) -> dict:
        """Camera state, resolution, temperature, protection level and warnings."""
        raw = self._get("/status.json", self.status_timeout, self.retries)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CameraError(f"/status.json was not JSON: {exc}") from None
        if not isinstance(data, dict):
            raise CameraError("/status.json was not an object")
        return data

    def get_jpeg(self) -> bytes:
        """One JPEG captured after this call. A bad body is an error, not a previous frame."""
        data = self._get("/snapshot.jpg", self.timeout, self.retries)
        if not _jpeg_complete(data):
            raise CameraError("snapshot was not a complete JPEG")
        return data

    def capture(self) -> Capture:
        """Status (best effort) plus a snapshot taken after this call returns."""
        requested = time.time()
        status, status_error = None, None
        try:
            status = self.status()
        except CameraError as exc:
            status_error = str(exc)
        if status and status.get("camera") == "paused":
            message = status.get("message") or "camera paused while it cools down"
            raise CameraError(f"/snapshot.jpg: {message}")
        jpeg = self.get_jpeg()
        received = time.time()
        meta = {
            "kind": "pi",
            "host": self.host,
            "port": self.port,
            "requested_at": requested,
            "received_at": received,
            "request_s": round(received - requested, 3),
            "bytes": len(jpeg),
            "rotate_reported": None if not status else status.get("rotate"),
            "extra_rotate": self.extra_rotate,
            "rotated_again": self.extra_rotate != 0,
            "camera_status": status,
            "status_error": status_error,
            "protection": None if not status else status.get("protection"),
            "temp_c": None if not status else status.get("temp"),
            "width_reported": None if not status else status.get("width"),
            "height_reported": None if not status else status.get("height"),
            "warnings": [] if not status else list(status.get("warnings") or []),
            "provenance": "raspberry_pi",
            "sync": "snapshot_after_request",
            "sync_precision_claimed": False,
        }
        return Capture(jpeg, meta, requested, received)

    def stream_url(self) -> str:
        return self.base + "/stream.mjpg"


TrovaCam = PiCam
