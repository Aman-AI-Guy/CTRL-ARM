"""Append-only audit trail: every request, every Grok call, every decision, every sign-off.

This is the traceability judges ask for, the line-clearance record's source, and our own
failure log (each event is tagged so we can classify what went wrong, Track-03 style).
Images go to data/runs/ and events reference them by path.
"""

from __future__ import annotations

import json
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
        self._recent: deque = deque(maxlen=300)
        self._lock = threading.Lock()
        if self.file.exists():
            for line in self.file.read_text().splitlines()[-300:]:
                try:
                    self._recent.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def event(self, kind: str, **payload) -> dict:
        e = {"id": uuid.uuid4().hex[:10], "ts": time.time(),
             "time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **payload}
        with self._lock:
            with self.file.open("a") as f:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
            self._recent.append(e)
        return e

    def save_image(self, name: str, img: np.ndarray) -> str:
        fname = f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}_{name}.jpg"
        cv2.imwrite(str(self.dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return f"runs/{fname}"

    def recent(self, n: int = 50, kind: str | None = None) -> list[dict]:
        with self._lock:
            items = [e for e in self._recent if kind is None or e["kind"] == kind]
        return items[-n:][::-1]
