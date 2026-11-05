"""Append-only audit trail: every request, every Grok call, every decision, every sign-off.

This is the traceability judges ask for, the line-clearance record's source, and our own
failure log (each event is tagged so we can classify what went wrong, Track-03 style).
Images go to data/runs/ and events reference them by path.

Tamper evidence (feature 13): every event carries `hash` = SHA-256 of the previous event's hash plus this event's
canonical JSON. Editing, inserting or deleting any line breaks the chain from that line on; `verify()` finds it.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def read_text_lines(path: Path) -> list[str]:
    """Lines of a UTF-8 log. Lines written by older Windows builds (platform default cp1252) are decoded as such, so
    the same text comes back and the hash chain, which hashes the text, still verifies."""
    out = []
    for raw in Path(path).read_bytes().splitlines():
        try:
            out.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(raw.decode("cp1252", errors="replace"))
    return out


class AuditLog:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = Path(data_dir) / "events.jsonl"
        self._recent: deque = deque(maxlen=300)
        self._lock = threading.Lock()
        self._last_hash = ""
        if self.file.exists():
            for line in read_text_lines(self.file)[-300:]:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._recent.append(e)
                self._last_hash = e.get("hash", self._last_hash)

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
            canonical = self._canonical(e)
            e = json.loads(canonical)                 # exactly what is stored (tuples -> lists, numbers -> JSON)
            e["hash"] = self._chain(self._last_hash, canonical)
            with self.file.open("a", encoding="utf-8") as f:   # never the platform default (cp1252 on Windows)
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._last_hash = e["hash"]
            self._recent.append(e)
        return e

    def all(self) -> list[dict]:
        if not self.file.exists():
            return []
        out = []
        for line in read_text_lines(self.file):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"kind": "unreadable_line", "raw": line[:200]})
        return out

    def verify(self) -> dict:
        """Recompute the hash chain. Events written before the chain existed (no hash) are counted, not checked."""
        prev, n, hashed = "", 0, 0
        for i, line in enumerate(read_text_lines(self.file) if self.file.exists() else []):
            n += 1
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1, "reason": "not valid JSON"}
            if "hash" not in e:
                if hashed:
                    return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1, "reason": "hash missing"}
                continue
            if self._chain(prev, self._canonical(e)) != e["hash"]:
                return {"ok": False, "events": n, "hashed": hashed, "broken_at": i + 1, "id": e.get("id"),
                        "reason": "edited, inserted or a line before it was deleted"}
            prev, hashed = e["hash"], hashed + 1
        return {"ok": True, "events": n, "hashed": hashed, "last_hash": prev[:16]}

    def save_image(self, name: str, img: np.ndarray) -> str:
        fname = f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}_{name}.jpg"
        cv2.imwrite(str(self.dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return f"runs/{fname}"

    def recent(self, n: int = 50, kind: str | None = None) -> list[dict]:
        with self._lock:
            items = [e for e in self._recent if kind is None or e["kind"] == kind]
        return items[-n:][::-1]
