"""Feature 5: failure log. Every wrong find gets a cause and an owner, so the team knows what to fix next
(and the jury sees we measure our failures, Track-03 style).

Causes are decided by plain code from what the judge showed us after pressing "wrong":
  missed_box            the judge clicked an object that had no box       -> P2 (proposals / threshold)
  wrong_pick            the right object had a box, Grok chose another    -> P3 (prompt, close-ups)
  said_not_found        the object was boxed, Grok said "not found"       -> P3
  picked_absent_object  the object is not on the table, Grok picked one   -> P3 (never-guess rule)
  asked_unnecessarily   Grok asked a question the judge found pointless   -> P3
  arm_missed            Grok was right, the arm pointed at the wrong spot -> P1 (calibration, TCP offset)
  slow                  any find slower than TROVA_SLOW_S (default 8 s)    -> P3 (model, effort)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OWNER = {"missed_box": "P2 camera and vision", "wrong_pick": "P3 Grok", "said_not_found": "P3 Grok",
         "picked_absent_object": "P3 Grok", "asked_unnecessarily": "P3 Grok", "arm_missed": "P1 arm",
         "slow": "P3 Grok", "other": "team"}
FIX = {"missed_box": "lower TROVA_BLOB_THRESH, capture the empty table again, or rely on Grok's direct box",
       "wrong_pick": "add the confusing pair to the prompt examples; check the close-up sheet",
       "said_not_found": "check the close-up sheet resolution; relax the never-guess wording",
       "picked_absent_object": "strengthen the never-guess rule; lower confidence threshold",
       "asked_unnecessarily": "tell the prompt when one object clearly fits better",
       "arm_missed": "recalibrate with the arm; check the TCP offset and the hover height",
       "slow": "try the non-reasoning model or effort low", "other": "look at the event in the log"}


def classify(last_find: dict, expected_box: list | None, absent: bool, clicked_unboxed: bool) -> str:
    r = last_find["result"]
    if absent:
        return "picked_absent_object" if r["status"] == "found" else "other"
    if clicked_unboxed:
        return "missed_box"
    if r["status"] == "not_found":
        return "said_not_found"
    if r["status"] == "clarify":
        return "asked_unnecessarily"
    if r["status"] == "found" and r.get("box") and expected_box and r["box"]["box"] != expected_box:
        return "wrong_pick"
    return "other"


def record(path: Path, last_find: dict, cause: str, slow_s: float, extra: dict | None = None) -> dict:
    causes = [cause] + (["slow"] if last_find.get("total_s", 0) > slow_s and cause != "slow" else [])
    entry = {"event": last_find.get("event_id"), "request": last_find.get("request"), "causes": causes,
             "owner": OWNER.get(cause, "team"), "fix": FIX.get(cause, ""), "status": last_find["result"]["status"],
             "label": last_find["result"].get("label"), "total_s": last_find.get("total_s"),
             "frame": last_find.get("frame"), **(extra or {})}
    with Path(path).open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def summary(path: Path) -> dict:
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    counts = Counter(c for r in rows for c in r["causes"])
    return {"total": len(rows),
            "by_cause": [{"cause": c, "count": n, "owner": OWNER.get(c, "team"), "fix": FIX.get(c, ""),
                          "examples": [r["request"] for r in rows if c in r["causes"]][:3]}
                         for c, n in counts.most_common()],
            "recent": rows[-10:][::-1]}
