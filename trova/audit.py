"""Append-only audit trail: every request, every Grok call, every decision, every sign-off.

This is the traceability judges ask for, the line-clearance record's source, and our own
failure log (each event is tagged so we can classify what went wrong, Track-03 style).
Images go to data/runs/ and events reference them by path.

Tamper evidence (feature 13): every event carries `hash` = SHA-256 of the previous event's hash plus this event's
canonical JSON. Editing, inserting or deleting any line breaks the chain from that line on; `verify()` finds it.
The end of the chain is anchored outside the log: data/events.anchor (written atomically after every event) holds
the event count, the last hash and the line where the chain starts, so deleting the last lines, stripping every
hash, prepending lines or deleting the whole log is detected too, also across restarts. A last line cut off by a
crash is moved to data/events.partial on start-up, so the next event does not merge into it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class AuditLog:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = Path(data_dir) / "events.jsonl"
        self.anchor_file = Path(data_dir) / "events.anchor"
        self.partial_file = Path(data_dir) / "events.partial"
        self._recent: deque = deque(maxlen=300)
        self._lock = threading.Lock()
        self._last_hash = ""
        self._count = 0                      # lines in the log
        self._chain_start: int | None = None  # index of the first hashed line; unhashed lines after it are rejected
        self._broken: str | None = None       # tampering found at start-up: stays until the anchor is reset by hand
        self._warning: str | None = None
        torn = self._repair_torn_tail()
        lines = self.file.read_text().splitlines() if self.file.exists() else []
        hashes: list[str | None] = []
        for i, line in enumerate(lines):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                hashes.append(None)
                continue
            if i >= len(lines) - 300 and isinstance(e, dict):
                self._recent.append(e)
            hashes.append(e.get("hash") if isinstance(e, dict) else None)
            if hashes[-1]:
                self._last_hash = hashes[-1]
                if self._chain_start is None:
                    self._chain_start = i
        self._count = len(lines)
        self._reconcile(hashes)
        with self._lock:
            self._write_anchor()
        if torn:
            self.event("error", where="audit", error=f"the last line of events.jsonl was cut off ({torn} bytes): "
                                                     "moved to events.partial")
        if self._broken:
            self.event("error", where="audit", error=self._broken)

    def _repair_torn_tail(self) -> int:
        if not self.file.exists():
            return 0
        data = self.file.read_bytes()
        if not data or data.endswith(b"\n"):
            return 0
        cut = data.rfind(b"\n") + 1
        try:
            json.loads(data[cut:])
            with self.file.open("ab") as f:   # a complete event that only lost its newline
                f.write(b"\n")
            return 0
        except json.JSONDecodeError:
            pass
        with self.partial_file.open("ab") as q:
            q.write(data[cut:] + b"\n")
        with self.file.open("r+b") as f:
            f.truncate(cut)
        return len(data) - cut

    def _reconcile(self, hashes: list[str | None]) -> None:
        """Compare the log as it is now with the anchor written by the last run."""
        when = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.anchor_file.exists():
            if self._count:
                self._warning = (f"events.anchor was missing at start-up ({when}) and was rebuilt from the log: "
                                 "lines deleted before then cannot be detected")
            return
        try:
            a = json.loads(self.anchor_file.read_text())
            n, h = int(a.get("count", 0)), str(a.get("last_hash", ""))
        except (ValueError, TypeError, AttributeError, OSError):
            self._broken = f"events.anchor is unreadable ({when}): the end of the log cannot be checked"
            return
        self._broken, self._warning = a.get("broken"), a.get("rebuilt")
        if a.get("chain_start") is not None:
            self._chain_start = int(a["chain_start"])
        anchored = hashes[n - 1] if 0 < n <= len(hashes) else ""
        if n > self._count or (n and anchored != h) or (h and not n):
            self._broken = self._broken or (
                f"found at start-up ({when}): the anchor says {n} events ending in {h[:16]}, the log has "
                f"{self._count}: lines were deleted, prepended or edited while Trova was stopped")

    def _write_anchor(self) -> None:
        tmp = self.anchor_file.parent / (self.anchor_file.name + ".tmp")
        tmp.write_text(json.dumps({"count": self._count, "last_hash": self._last_hash,
                                   "chain_start": self._chain_start, "broken": self._broken,
                                   "rebuilt": self._warning}))
        os.replace(tmp, self.anchor_file)

    @staticmethod
    def _canonical(e: dict) -> str:
        return json.dumps({k: v for k, v in e.items() if k != "hash"}, ensure_ascii=False, default=str, sort_keys=True)

    @staticmethod
    def _chain(prev: str, canonical: str) -> str:
        return hashlib.sha256((prev + canonical).encode()).hexdigest()

    def event(self, kind: str, **payload) -> dict:
        e = {"id": uuid.uuid4().hex[:10], "ts": time.time(),
             "time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **payload}
        with self._lock:
            # hash exactly what verify() reloads: integer dict keys become strings, tuples lists, numbers JSON
            e = json.loads(json.dumps(e, ensure_ascii=False, default=str))
            e["hash"] = self._chain(self._last_hash, self._canonical(e))
            with self.file.open("a") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            if self._chain_start is None:
                self._chain_start = self._count
            self._count += 1
            self._last_hash = e["hash"]
            self._write_anchor()
            self._recent.append(e)
        return e

    def anchor(self) -> dict:
        """The end of the chain, for printing on a record: anyone can later check the log still reaches it."""
        with self._lock:
            return {"events": self._count, "last_hash": self._last_hash[:16]}

    def all(self) -> list[dict]:
        if not self.file.exists():
            return []
        out = []
        for line in self.file.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"kind": "unreadable_line", "raw": line[:200]})
        return out

    def verify(self) -> dict:
        """Recompute the hash chain and compare its end with the anchor. Events written before the chain existed
        (no hash, before the chain's first line) are counted, not checked."""
        with self._lock:
            lines = self.file.read_text().splitlines() if self.file.exists() else []
            count, last, start, broken, warning = (self._count, self._last_hash, self._chain_start, self._broken,
                                                   self._warning)
        prev, n, hashed = "", 0, 0
        for i, line in enumerate(lines):
            n += 1
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1, "reason": "not valid JSON"}
            if "hash" not in e:
                if hashed or (start is not None and i >= start):
                    return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1, "reason": "hash missing"}
                continue
            if not hashed and start is not None and i != start:
                return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1,
                        "reason": "lines were inserted before the start of the chain"}
            if self._chain(prev, self._canonical(e)) != e["hash"]:
                return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1, "id": e.get("id"),
                        "reason": "edited, inserted or a line before it was deleted"}
            prev, hashed = e["hash"], hashed + 1
        if n != count or prev != last:
            return {"ok": False, "events": n, "hashed": hashed, "broken_at": min(n, count) + 1,
                    "reason": f"the log should hold {count} events ending in {last[:16] or '(none)'}: "
                              "lines at the end were deleted or added"}
        if broken:
            return {"ok": False, "events": n, "hashed": hashed, "broken_at": None, "reason": broken}
        out = {"ok": True, "events": n, "hashed": hashed, "last_hash": prev[:16]}
        if warning:
            out["warning"] = warning
        return out

    def save_image(self, name: str, img: np.ndarray) -> str:
        fname = f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}_{name}.jpg"
        cv2.imwrite(str(self.dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return f"runs/{fname}"

    def recent(self, n: int = 50, kind: str | None = None) -> list[dict]:
        with self._lock:
            items = [e for e in self._recent if kind is None or e["kind"] == kind]
        return items[-n:][::-1]
