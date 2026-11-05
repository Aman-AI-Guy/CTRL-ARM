"""Receive a scan's live frames on this PC as they are taken, then fill in whatever the live push missed.

    rx = Receiver(cam, scan_id, out_dir, clock_offset_s)
    rx.start()                      # background thread: /scan/live -> out_dir/images/<index>.jpg
    ...                             # the arm moves; POST /scan/stop
    report = rx.finish(drain_timeout_s=30)   # wait for the live stream to end, then fetch missing bins

After a WiFi drop the thread reconnects with since=<last index received>, so nothing is sent twice. Each image is
written to a temporary name and renamed, so a reconstruction that starts early never reads half a file.
Per-frame timings go to received.csv: Pi capture time (converted to this PC's clock), arrival time, latency.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from trova_cam import CameraError, TrovaCam  # noqa: E402


@dataclass
class Received:
    index: int
    bin: int
    nbytes: int
    source: str                 # live | backfill
    capture_pc: float | None    # Pi capture time on this PC's clock (wall seconds)
    recv_pc: float              # arrival on this PC (wall seconds)
    latency_s: float | None


def _pct(values, q):
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * len(v)))], 3) if v else None


class Receiver:
    def __init__(self, cam: TrovaCam, scan_id: str, out_dir: str | Path, clock_offset_s: float = 0.0,
                 reconnect_delay_s: float = 0.5, http_timeout_s: float = 20.0, log=print):
        self.cam, self.scan_id = cam, scan_id
        self.out = Path(out_dir)
        self.images = self.out / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.offset = clock_offset_s            # Pi wall - PC wall
        self.reconnect_delay = reconnect_delay_s
        self.http_timeout = http_timeout_s
        self.log = log
        self.frames: dict[int, Received] = {}
        self.last_index = -1
        self.end: dict | None = None
        self.reconnects = 0
        self.errors: list[str] = []
        self.missing: list[int] = []
        self.bins_total: int | None = None
        self.t_drain_done = self.t_done = None
        self._live = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True, name="scan-receiver")

    # ------------------------------------------------------------------ live
    def start(self) -> "Receiver":
        self.thread.start()
        return self

    def _run(self):
        while not self._stop.is_set() and self.end is None:
            try:
                with self.cam.scan_live(since=self.last_index, scan_id=self.scan_id, timeout=self.http_timeout) as live:
                    self._live = live
                    for f in live:
                        self._save(f.index, f.bin, f.data, "live", f.wall, f.recv_wall)
                        if self._stop.is_set():
                            break
                    if live.end is not None:
                        self.end = live.end
            except CameraError as e:
                if self._stop.is_set():
                    break
                self.errors.append(f"{time.strftime('%H:%M:%S')} {e}")
                if "HTTP 404" in str(e):          # the Pi no longer has this scan in memory: backfill covers it
                    break
                self.reconnects += 1
                self.log(f"[receiver] {e}; reconnecting from index {self.last_index}")
                self._stop.wait(self.reconnect_delay)
            finally:
                self._live = None

    def _save(self, index: int, bin_: int, data: bytes, source: str, pi_wall: float | None, recv: float):
        path = self.images / f"{index:06d}.jpg"
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        cap = pi_wall - self.offset if pi_wall else None
        with self._lock:
            self.frames[index] = Received(index, bin_, len(data), source, cap, recv,
                                          round(recv - cap, 4) if cap is not None and source == "live" else None)
            self.last_index = max(self.last_index, index)

    # ------------------------------------------------------------------ after stop
    def stop(self):
        """Stop listening (closes the live connection)."""
        self._stop.set()
        live = self._live
        if live is not None:
            live.close()
        self.thread.join(5)

    def finish(self, drain_timeout_s: float = 30.0) -> dict:
        """Call after POST /scan/stop: wait for the live stream to deliver its backlog and end, then fetch every
        bin it did not deliver from the Pi's recording."""
        self.thread.join(drain_timeout_s)
        if self.thread.is_alive():
            self.errors.append(f"live stream still busy after {drain_timeout_s:g} s: stopped, backfilling")
            self.stop()
        self.t_drain_done = time.time()
        self.backfill()
        self.t_done = time.time()
        return self.report()

    def backfill(self) -> list[int]:
        try:
            bins = self.cam.scan_bins(self.scan_id)
        except CameraError as e:
            self.errors.append(f"bins.csv: {e}")
            return self.missing
        self.bins_total = len(bins)
        for b in bins:
            if b["index"] in self.frames:
                continue
            try:
                data = self.cam.scan_frame(self.scan_id, b["index"])
            except CameraError as e:
                self.errors.append(f"frame {b['index']}: {e}")
                continue
            self._save(b["index"], b["bin"], data, "backfill", None, time.time())
        self.missing = [b["index"] for b in bins if b["index"] not in self.frames]
        return self.missing

    # ------------------------------------------------------------------ reports
    def received(self) -> list[Received]:
        with self._lock:
            return sorted(self.frames.values(), key=lambda r: r.index)

    def live_stats(self, since_pc: float | None = None) -> dict:
        live = [r for r in self.received() if r.source == "live" and (since_pc is None or r.recv_pc >= since_pc)]
        out = {"frames": len(live)}
        if len(live) >= 2:
            span = live[-1].recv_pc - live[0].recv_pc
            if span > 0:
                out["fps"] = round((len(live) - 1) / span, 2)
                out["mbit_s"] = round(sum(r.nbytes for r in live[1:]) * 8 / span / 1e6, 2)
        lat = [r.latency_s for r in live if r.latency_s is not None]
        if lat:
            out["latency_s"] = {"p50": _pct(lat, 0.5), "p90": _pct(lat, 0.9), "max": round(max(lat), 3)}
        return out

    def report(self) -> dict:
        rec = self.received()
        live = [r for r in rec if r.source == "live"]
        return {
            "bins_on_pi": self.bins_total,
            "received_live": len(live),
            "backfilled": len(rec) - len(live),
            "missing_after_backfill": len(self.missing),
            "missing_indices": self.missing,
            "kb_mean": round(sum(r.nbytes for r in rec) / len(rec) / 1024, 1) if rec else 0,
            "live": self.live_stats(),
            "reconnects": self.reconnects,
            "errors": self.errors,
            "last_live_recv_pc": live[-1].recv_pc if live else None,
            "drain_done_pc": self.t_drain_done,
            "done_pc": self.t_done,
            "end": self.end,
        }

    def write_csv(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.out / "received.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(Received.__dataclass_fields__))
            w.writeheader()
            for r in self.received():
                w.writerow(asdict(r))
        return path

