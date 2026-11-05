"""Original scan multipart parser, reused by Trova PiCam; no second camera client."""
import json, http.client, time
from collections import namedtuple
LIVE_BOUNDARY = b"trovascan"
LiveFrame = namedtuple("LiveFrame", "index bin mono_ns wall nbytes sent_mono_ns recv_wall data")
from ..pi_cam import CameraError

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
