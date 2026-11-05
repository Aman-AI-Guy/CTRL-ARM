"""Application safety persistence. This cannot stop a controller after a process/power failure."""

import json
import os
import tempfile
from pathlib import Path


def _replace_durable(src: str, dst: Path) -> None:
    """Atomically replace dst with src and make the rename durable. Any failure raises OSError (blocks motion).

    POSIX: os.replace, then fsync the directory. Windows cannot open a directory with os.open (EACCES), so the
    rename itself is made durable with MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), which
    returns only after the move has been flushed to disk.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        if not move(str(src), str(dst), 0x1 | 0x8):     # MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
            raise ctypes.WinError(ctypes.get_last_error())
        return
    os.replace(src, dst)
    directory = os.open(dst.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class LiftStore:
    """Only a complete, recognized record may establish that a previous lift finished/recovered.

    Writes replace one fsynced file atomically, then fsync the directory. I/O failure must block motion.
    A missing/corrupt/foreign record is UNKNOWN, never evidence that a fork/load is safe.
    """

    def __init__(self, directory: Path):
        self.path = directory / "arm_safety.json"

    def read(self) -> str:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Ambiguous safety record")
                result[key] = value
            return result
        try:
            d = json.loads(self.path.read_text(), object_pairs_hook=unique)
            if (type(d) is dict and set(d) == {"version", "lift"} and type(d["version"]) is int
                    and d["version"] == 1 and d["lift"] in ("clear", "in_progress")):
                return d["lift"]
        except (OSError, ValueError, TypeError):
            pass
        return "unknown"

    def write(self, lift: str) -> None:
        if lift not in ("clear", "in_progress"):
            raise ValueError("Invalid lift safety state")
        fd, name = tempfile.mkstemp(prefix=".arm_safety-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"version": 1, "lift": lift}, f)
                f.flush()
                os.fsync(f.fileno())
            _replace_durable(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
