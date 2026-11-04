"""Review an operator safety file. This does not connect to a robot and does not change the motion mode.

    python -m trova.qualify safety.json

The file is JSON with the measured values. confirm must be true before a running Trova
process will store them. This command only reports what is present and what is invalid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .motion import apply_operator_safety


class _NoArm:
    """A stand-in gateway so the file can be checked without a controller."""

    def __init__(self):
        self.mode = "dry_run"
        self.simulated = True
        self.limits = None
        self.keepouts = []
        self.keepout_assessed = False
        self.camera = None
        self.max_speed_mm_s = 100.0
        self.jog_max_mm = 300.0
        self.joint_limits = None
        self.arm = None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m trova.qualify safety.json", file=sys.stderr)
        return 2
    path = Path(args[0])
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "reason": str(exc), "moved": False}))
        return 1
    if not isinstance(data, dict):
        print(json.dumps({"ok": False, "reason": "the file must be a JSON object", "moved": False}))
        return 1
    report = apply_operator_safety(_NoArm(), data)
    report["moved"] = False
    report["connected"] = False
    report["hardware_mode_set"] = False
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
