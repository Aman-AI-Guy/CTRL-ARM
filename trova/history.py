"""Feature 13: the audit trail as a page. Every find, check, sign-off, alarm, label reading and arm command,
one plain line each, with its photo; filters, search, CSV export, and a check that the log was not edited.

  GET /history                 the page (link "history" in the header)
  GET /api/history             rows, newest first   (?groups=finds,clearance,...&q=text&n=300)
  GET /api/history.csv         the same rows as CSV (for the record or a spreadsheet)
  GET /api/history/verify      recompute the hash chain of data/events.jsonl
"""

from __future__ import annotations

import csv
import io
from collections import Counter

from fastapi.responses import HTMLResponse, Response

GROUPS = {
    "finds": {"find", "find_all", "judge", "failure", "inventory"},
    "clearance": {"reference", "change_check", "signoff", "record_pdf", "judge_check", "labels", "batch", "zone"},
    "alarms": {"watch", "watch_alert", "watch_ack", "watch_false_alarm"},
    "arm": {"arm_stop", "arm_home", "arm_point", "arm_tour", "arm_lift_tray", "arm_freedrive", "arm_compliant"},
    "setup": {"background", "source", "upload_scene", "calibration", "marker_from_arm", "preflight", "metrics_reset"},
    "voice": {"voice"},
    "dog": {"dog_search", "dog_stop"},
    "errors": {"error", "unreadable_line"},
}
GROUP_OF = {k: g for g, ks in GROUPS.items() for k in ks}


def _txt(v) -> str:
    return "" if v is None else str(v)


def summarize(e: dict) -> dict:
    """One plain line per event: what was asked or done, what came out, and its photo if any."""
    k, what, result, image = e.get("kind", "?"), "", "", None
    r = e.get("result") or {}
    if k == "find":
        what = e.get("request") or "[photo]"
        result = {"found": f"found: {r.get('label', '')}", "clarify": f"asked: {r.get('clarifying_question', '')}",
                  "not_found": "not found"}.get(r.get("status"), r.get("status", ""))
        result += f" ({e.get('total_s')} s)" + (" · offline" if r.get("mode") == "offline" else "")
        image = e.get("marked")
    elif k == "find_all":
        what, result, image = e.get("request", ""), f"{r.get('count', 0)} found: {r.get('label', '')}", e.get("marked")
    elif k == "change_check":
        what = "line clearance check"
        result = ("LINE CLEAR" if r.get("line_clear") else f"NOT CLEAR, {len(r.get('changes', []))} change(s)") + \
            f": {r.get('summary', '')}"
        image = e.get("cur_marked")
    elif k == "signoff":
        what, result = f"sign-off by {e.get('operator', '')}", f"{_txt(e.get('decision')).upper()} {e.get('note', '')}".strip()
    elif k == "reference":
        what, result, image = "reference saved", f"{e.get('n_objects')} objects", e.get("path")
    elif k == "watch_alert":
        what, result, image = "watch alarm", e.get("summary", ""), e.get("image")
    elif k in ("watch_ack", "watch_false_alarm"):
        what = f"alarm {e.get('alert')}"
        result = f"acknowledged by {e.get('who') or '?'}" if k == "watch_ack" else "closed: Grok found it was a false alarm"
    elif k == "watch":
        what, result = "watch the line", ("on" if e.get("on") else "off") + (", arm points" if e.get("point") else "")
    elif k == "labels":
        what = f"labels read (batch {e.get('lot') or 'not set'})"
        probs = e.get("problems") or []
        result = "OK" if not probs else "; ".join(p.get("text", "") for p in probs)
        image = e.get("image")
    elif k == "batch":
        what, result = "current batch set", f"LOT {e.get('lot', '')}"
    elif k == "judge":
        what, result = "jury judged a find", ("correct" if e.get("correct") else "wrong") + (f" ({e['cause']})" if e.get("cause") else "")
    elif k == "judge_check":
        what = "jury judged a check"
        result = "all right" if e.get("correct") else f"false alarms {e.get('false_alarms')}, missed {e.get('missed')}"
    elif k == "failure":
        what, result = e.get("request") or "[photo]", f"cause: {', '.join(e.get('causes', []))} -> {e.get('owner', '')}"
    elif k == "inventory":
        what, result, image = "what is on the table", e.get("summary", ""), e.get("map")
    elif k.startswith("arm_"):
        what = k[4:].replace("_", " ")
        if "x_mm" in e:
            result = f"to {e.get('x_mm'):.0f}, {e.get('y_mm'):.0f} mm" if isinstance(e.get("x_mm"), (int, float)) else ""
        elif "targets" in e:
            result = f"{len(e['targets'])} targets"
        elif "on" in e:
            result = "on" if e["on"] else "off"
        result += f" ({e['after']})" if e.get("after") else ""
    elif k == "calibration":
        what, result = "calibration", f"{e.get('source')}, error {e.get('error_mm')} mm"
    elif k == "preflight":
        what, result = "preflight", e.get("summary", "") + (": " + "; ".join(e.get("problems", [])[:4]) if e.get("problems") else "")
    elif k == "error":
        what, result = f"error in {e.get('where', '?')}", e.get("error", "")
    elif k == "voice":
        what, result = f"voice ({e.get('who')})", e.get("text", "")
    elif k == "dog_search":
        what, result = f"dog: {e.get('request', '')}", f"{e.get('status')} {e.get('spot') or ''}".strip()
        image = e.get("photo")
    else:
        what = k.replace("_", " ")
        result = ", ".join(f"{a}={_txt(b)[:40]}" for a, b in e.items() if a not in ("id", "ts", "time", "kind", "hash"))[:200]
    return {"id": e.get("id", ""), "time": e.get("time", ""), "kind": k, "group": GROUP_OF.get(k, "other"),
            "what": _txt(what)[:300], "result": _txt(result)[:400], "image": image}


def rows(events: list[dict], groups: set[str] | None = None, q: str = "", n: int = 300) -> list[dict]:
    out = []
    q = q.lower().strip()
    for e in reversed(events):
        if e.get("kind") == "grok_call":          # hundreds of them; their latency is in the find rows
            continue
        s = summarize(e)
        if groups and s["group"] not in groups:
            continue
        if q and q not in (s["what"] + " " + s["result"] + " " + s["kind"]).lower():
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def to_csv(rs: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "event id", "kind", "group", "what", "result", "image"])
    for r in rs:
        w.writerow([r["time"], r["id"], r["kind"], r["group"], r["what"], r["result"], r["image"] or ""])
    return buf.getvalue()


def register(app, st, static_dir) -> None:
    @app.get("/history", response_class=HTMLResponse)
    def history_page():
        return (static_dir / "history.html").read_text()

    def _rows(groups: str, q: str, n: int):
        g = {x for x in groups.split(",") if x} or None
        return rows(st.audit.all(), g, q, n)

    @app.get("/api/history")
    def api_history(groups: str = "", q: str = "", n: int = 300):
        events = st.audit.all()
        counts = Counter(GROUP_OF.get(e.get("kind"), "other") for e in events if e.get("kind") != "grok_call")
        return {"rows": rows(events, {x for x in groups.split(",") if x} or None, q, n), "counts": counts,
                "groups": list(GROUPS), "grok_calls": sum(e.get("kind") == "grok_call" for e in events)}

    @app.get("/api/history.csv")
    def api_history_csv(groups: str = "", q: str = "", n: int = 100000):
        return Response(to_csv(_rows(groups, q, n)), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=trova_audit_trail.csv"})

    @app.get("/api/history/verify")
    def api_history_verify():
        return st.audit.verify()
