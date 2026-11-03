"""Trova web app: FastAPI backend + a one-page UI (trova/static/index.html).

Run:  python -m trova      then open http://localhost:8000 (phones: http://<laptop-ip>:8000)
"""

from __future__ import annotations

import html
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import changes as chg
from .arm import CartesianArm, SimArm, TaughtGridArm, make_arm
from .audit import AuditLog
from .boxes import Box
from .calibration import Calibration, calibrate_aruco, load_marker_positions, make_marker_sheet
from .camera import DemoSource, ImageSource, open_source
from .config import Settings, load_settings
from .dog import DogSearch, load_route, make_dog, run_search
from . import voice as voice_cfg
from .finder import find as grok_find
from .inventory import describe as describe_table, draw_map
from .grok import GrokClient, GrokError
from .marks import HIGHLIGHT, MARK, draw_marks, jpeg
from .proposals import drop_occluders, find_blobs, is_occluded

STATIC = Path(__file__).parent / "static"


def decode_upload(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read the image")
    h, w = img.shape[:2]
    if max(h, w) > 2000:  # phone photos are huge; Grok gets <= 1600 anyway
        s = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# Request bodies (module level: FastAPI must resolve them under postponed annotations)
class Tune(BaseModel):
    thresh: float | None = None
    min_area_frac: float | None = None


class ClickPoint(BaseModel):
    u: float
    v: float
    x_mm: float
    y_mm: float


class Point(BaseModel):
    x_mm: float | None = None
    y_mm: float | None = None
    u: float | None = None
    v: float | None = None


class Judge(BaseModel):
    kind: str = "find"
    correct: bool = True
    false_alarms: int = 0
    missed: int = 0
    note: str = ""


class Label(BaseModel):
    u: float | None = None
    v: float | None = None
    absent: bool = False


class State:
    def __init__(self, s: Settings):
        self.s = s
        self.audit = AuditLog(s.data_dir)
        self.grok = GrokClient(s.xai_api_key, s.model, s.effort, s.mock,
                               on_call=lambda e: self.audit.event("grok_call", **e))
        self.source = open_source(s.camera)
        self.override: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self.background_path: str | None = None   # saved copy, so eval cases can be replayed
        self.calib_path = s.data_dir / "calibration.json"
        self.calib = Calibration.load(self.calib_path)
        self.click_points: list[list[float]] = []
        self.arm = make_arm(s.arm, s.data_dir, s.table_w_mm, s.table_h_mm)
        self.arm_thread: threading.Thread | None = None
        self.thresh, self.min_area = s.blob_thresh, s.min_area_frac
        self.last_find: dict | None = None
        self.pending: dict | None = None
        self.reference: dict | None = None
        self.last_check: dict | None = None
        self.label_wait: dict | None = None
        self.pending_move: dict | None = None
        self.inventory: dict | None = None
        self.metrics = {"find_ok": 0, "find_wrong": 0, "find_times": [], "checks": 0,
                        "check_ok": 0, "false_alarms": 0, "missed": 0}
        self.lock = threading.RLock()
        self.cases_dir = s.data_dir / "cases"
        self.cases_dir.mkdir(exist_ok=True)
        self.dog = make_dog(s.dog)
        self.dog_route = load_route(s.data_dir)
        self.dog_search: DogSearch | None = None
        self.dog_stop = threading.Event()
        self.dog_thread: threading.Thread | None = None

    # ---------- frames, proposals, calibration
    def frame(self) -> np.ndarray:
        f = self.override.copy() if self.override is not None else self.source.read()
        h, w = f.shape[:2]
        if self.calib is None or (self.calib.source == "default" and self.calib.points[2][:2] != [float(w), float(h)]):
            self.calib = Calibration.default(w, h, self.s.table_w_mm, self.s.table_h_mm)
        return f

    def proposals(self, frame: np.ndarray) -> tuple[list[Box], bool]:
        raw = find_blobs(frame, self.background, self.thresh, self.min_area)
        return drop_occluders(raw, frame.shape), is_occluded(raw, frame.shape)

    def arm_blocked(self) -> str | None:
        """A real Cartesian arm only moves once pixels are calibrated in ITS coordinates."""
        if isinstance(self.arm, CartesianArm) and (not self.calib or self.calib.source not in ("arm", "aruco_robot")):
            return "Calibrate with the arm first (Setup: Calibrate with arm, or ArUco with markers recorded by the arm)"
        return None

    def clear_view(self) -> None:
        """A real arm parks before every capture so it is not mistaken for an object."""
        if not isinstance(self.arm, SimArm):
            self.arm.home()

    def start_arm(self, fn, *args) -> None:
        if self.arm_thread and self.arm_thread.is_alive():
            self.arm.stop()
            self.arm_thread.join(timeout=2)
        self.arm_thread = threading.Thread(target=fn, args=args, daemon=True)
        self.arm_thread.start()

    def capture_clean(self, attempts: int = 6) -> tuple[np.ndarray, list[Box], bool]:
        for _ in range(attempts):
            f = self.frame()
            boxes, occl = self.proposals(f)
            if not occl:
                return f, boxes, False
            time.sleep(0.4)
        return f, boxes, True


def create_app(settings: Settings | None = None) -> FastAPI:
    st = State(settings or load_settings())
    app = FastAPI(title="Trova")
    app.state.trova = st
    app.mount("/files", StaticFiles(directory=str(st.s.data_dir)), name="files")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    def fail(e: Exception, where: str):
        st.audit.event("error", where=where, error=str(e))
        return JSONResponse({"error": str(e), "where": where}, status_code=502 if isinstance(e, GrokError) else 400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    # ------------------------------------------------------------------ live view
    @app.get("/api/frame.jpg")
    def frame_jpg(boxes: int = 1, max_w: int = 1280):
        f = st.frame()
        img = f.copy()
        if boxes:
            props, _ = st.proposals(f)
            img = draw_marks(img, props)
        lf = st.last_find
        if lf and lf.get("box") and time.time() - lf["ts"] < 30:
            b = lf["box"]
            cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, max(4, f.shape[1] // 250))
        arm = st.arm.status()
        if st.calib and st.calib.ok and arm.get("tip_mm"):
            u, v = st.calib.table_to_pixel(*arm["tip_mm"])
            if 0 <= u < f.shape[1] and 0 <= v < f.shape[0]:
                r = max(10, f.shape[1] // 90)
                cv2.circle(img, (int(u), int(v)), r, (255, 255, 0), 3, cv2.LINE_AA)
                cv2.line(img, (int(u) - 2 * r, int(v)), (int(u) + 2 * r, int(v)), (255, 255, 0), 2)
                cv2.line(img, (int(u), int(v) - 2 * r), (int(u), int(v) + 2 * r), (255, 255, 0), 2)
        for p in st.click_points:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), MARK, cv2.MARKER_TILTED_CROSS, 30, 3)
        if img.shape[1] > max_w:
            s = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return Response(jpeg(img, 80), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        f = st.frame()
        m = st.metrics
        judged = m["find_ok"] + m["find_wrong"]
        lf = {k: v for k, v in (st.last_find or {}).items() if k not in ("box",)}
        return {
            "mock": st.grok.mock, "grok": st.grok.stats(), "source": st.source.info(),
            "override": st.override is not None, "frame_size": [f.shape[1], f.shape[0]],
            "background": st.background is not None, "calibration": st.calib.as_dict() if st.calib else None,
            "click_points": st.click_points, "arm": st.arm.status(),
            "reference": None if not st.reference else {k: st.reference[k] for k in ("time", "path", "n_objects")},
            "pending_question": st.pending["question"] if st.pending else None,
            "pending_candidates": st.pending.get("candidates", []) if st.pending else [],
            "label_wait": st.label_wait is not None,
            "last_find": lf or None,
            "last_check": None if not st.last_check else {k: v for k, v in st.last_check.items() if k != "objects"},
            "tuning": {"thresh": st.thresh, "min_area_frac": st.min_area},
            "lang": st.s.lang, "voice_available": not st.grok.mock, "inventory": st.inventory,
            "dog": None if st.dog is None else {"name": st.dog.name, "route": [x.name for x in st.dog_route],
                                                "search": st.dog_search.as_dict() if st.dog_search else None},
            "metrics": {**{k: v for k, v in m.items() if k != "find_times"}, "find_judged": judged,
                        "avg_find_s": round(sum(m["find_times"]) / len(m["find_times"]), 2) if m["find_times"] else None},
        }

    # ------------------------------------------------------------------ find
    def do_find(request: str, query_img, history, move: bool = True):
        with st.lock:
            t0 = time.perf_counter()
            st.clear_view()
            frame = st.frame()
            boxes, occluded = st.proposals(frame)
            res = grok_find(st.grok, frame, boxes, request, query_img, history)
            hi = res.box.id if res.box and res.box.source != "grok" else None
            marked = draw_marks(frame, boxes, highlight=hi)
            if res.box is not None and res.box.source == "grok":
                b = res.box
                cv2.rectangle(marked, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, 5)
            paths = {"frame": st.audit.save_image("frame", frame), "marked": st.audit.save_image("find", marked),
                     "query": st.audit.save_image("query", query_img) if query_img is not None else None}
            table_mm, arm_blocked = None, None
            if res.status == "found" and st.calib and st.calib.ok:
                table_mm = [round(v, 1) for v in st.calib.pixel_to_table(*res.box.center)]
                arm_blocked = st.arm_blocked()
                if not arm_blocked and move:
                    st.start_arm(st.arm.point_at, *table_mm)
            total = time.perf_counter() - t0
            if res.status == "found":
                st.metrics["find_times"].append(total)
            ev = st.audit.event("find", request=request, has_photo=query_img is not None, history=history,
                                result=res.as_dict(), table_mm=table_mm, occluded=occluded, total_s=round(total, 2),
                                **paths)
            st.last_find = {"ts": time.time(), "event_id": ev["id"], "request": request, "result": res.as_dict(),
                            "table_mm": table_mm, "arm_blocked": arm_blocked, "total_s": round(total, 2), "box": res.box,
                            "occluded": occluded, "moved": bool(table_mm and move and not arm_blocked),
                            "boxes": [b.as_dict() for b in boxes], **paths}
            st.pending_move = st.last_find if res.status == "found" else None
            st.pending = None
            if res.status == "clarify":
                st.pending = {"request": request, "query_image": query_img, "history": history, "move": move,
                              "question": res.clarifying_question, "candidates": res.candidate_ids}
            st.label_wait = None
            return {k: v for k, v in st.last_find.items() if k != "box"}

    @app.post("/api/find")
    def api_find(text: str = Form(""), photo: UploadFile | None = File(None), move: bool = Form(True)):
        q = decode_upload(photo.file.read()) if photo is not None and photo.filename else None
        if not text.strip() and q is None:
            raise HTTPException(400, "Say what to find, or send a photo")
        try:
            return do_find(text.strip(), q, [], move)
        except (GrokError, ValueError, RuntimeError) as e:
            return fail(e, "find")

    @app.post("/api/clarify")
    def api_clarify(answer: str = Form(...)):
        if not st.pending:
            raise HTTPException(400, "No question pending")
        p = st.pending
        try:
            return do_find(p["request"], p["query_image"], p["history"] + [(p["question"], answer)],
                           p.get("move", True))
        except (GrokError, ValueError, RuntimeError) as e:
            return fail(e, "clarify")

    # ------------------------------------------------------------------ what changed
    @app.post("/api/reference")
    def api_reference(describe: bool = Form(True)):
        with st.lock:
            st.clear_view()
            f, boxes, occl = st.capture_clean()
            names, summary = {}, ""
            if describe and boxes:   # name the objects once, so changes can say WHICH object moved or went missing
                try:
                    inv = describe_table(st.grok, f, boxes, st.s.lang)
                    names, summary = inv.names(), inv.summary
                except GrokError as e:
                    st.audit.event("error", where="reference_inventory", error=str(e))
            path = st.audit.save_image("reference", draw_marks(f, boxes))
            st.reference = {"frame": f, "boxes": boxes, "time": time.strftime("%H:%M:%S"), "path": path,
                            "raw": st.audit.save_image("reference_raw", f), "n_objects": len(boxes), "names": names}
            st.audit.event("reference", n_objects=len(boxes), occluded=occl, path=path, raw=st.reference["raw"],
                           names=names)
            return {"n_objects": len(boxes), "path": path, "occluded": occl, "names": names, "summary": summary}

    @app.post("/api/changes")
    def api_changes(verify: bool = Form(True)):
        if not st.reference:
            raise HTTPException(400, "Save a reference first")
        with st.lock:
            st.clear_view()
            frames, per, occl = [], [], False
            for _ in range(8):
                f = st.frame()
                b, o = st.proposals(f)
                if o:
                    occl = True
                    time.sleep(0.4)
                    continue
                frames.append(f)
                per.append(b)
                if len(frames) == 3:
                    break
                time.sleep(0.25)
            if not frames:
                return JSONResponse({"error": "Hand or arm in view: clear the table view and retry"}, 409)
            try:
                res = chg.check(st.grok if verify else None, st.reference["frame"], st.reference["boxes"],
                                frames, per, occluded=occl and len(frames) < 3, lang=st.s.lang,
                                ref_names=st.reference.get("names"))
            except GrokError as e:
                return fail(e, "changes")
            paths = {"ref_marked": st.audit.save_image("ref_changes", res.ref_marked),
                     "cur_marked": st.audit.save_image("cur_changes", res.cur_marked),
                     "cur_raw": st.audit.save_image("current_raw", res.current)}
            ev = st.audit.event("change_check", result=res.as_dict(), reference=st.reference["raw"], **paths)
            st.metrics["checks"] += 1
            st.last_check = {"event_id": ev["id"], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "result": res.as_dict(), "reference": st.reference["path"], "signoff": None,
                             "model": st.grok.stats()["model"], "mock": st.grok.mock, **paths}
            return st.last_check

    @app.post("/api/signoff")
    def api_signoff(operator: str = Form(...), decision: str = Form("approve"), note: str = Form("")):
        if not st.last_check:
            raise HTTPException(400, "Run a check first")
        so = {"operator": operator, "decision": decision, "note": note, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        st.last_check["signoff"] = so
        st.audit.event("signoff", check=st.last_check["event_id"], **so)
        return so

    @app.get("/api/report", response_class=HTMLResponse)
    def api_report():
        c = st.last_check
        if not c:
            return "<p>No check yet.</p>"
        r = c["result"]
        rows = "".join(
            f"<tr><td>C{x['id']}</td><td>{html.escape(x['kind'])}</td><td>{html.escape(x['label'])}</td>"
            f"<td>{html.escape(x['risk'])}</td><td>{html.escape(x['explanation'])}</td></tr>" for x in r["changes"])
        filt = "".join(f"<li>C{x['id']} {html.escape(x['kind'])}: {html.escape(x['explanation'])}</li>"
                       for x in r["filtered"]) or "<li>none</li>"
        so = c.get("signoff")
        so_html = (f"<p><b>Signed off by {html.escape(so['operator'])}</b> ({html.escape(so['decision'])}) at "
                   f"{so['time']}. {html.escape(so['note'])}</p>") if so else "<p><b>Not signed off yet.</b></p>"
        verdict = "LINE CLEAR" if r["line_clear"] else "NOT CLEAR"
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Line clearance record</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#111;background:#fff}}
img{{width:100%;border:1px solid #ccc;border-radius:6px}}.g{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}
.v{{font-size:22px;font-weight:700;color:{'#137333' if r['line_clear'] else '#b3261e'}}}</style></head><body>
<h1>Line clearance record</h1><p>Check {c['event_id']} at {c['time']} · model {html.escape(str(c['model']))}{' (mock)' if c['mock'] else ''}</p>
<p class="v">{verdict}</p><p>{html.escape(r['summary'])}</p>
<div class="g"><div><b>Reference</b><img src="/files/{c['ref_marked']}"></div><div><b>Current</b><img src="/files/{c['cur_marked']}"></div></div>
<h2>Changes</h2><table><tr><th>#</th><th>Kind</th><th>Object</th><th>Risk</th><th>Why it matters</th></tr>{rows or '<tr><td colspan=5>none</td></tr>'}</table>
<h3>Candidates rejected as false alarms</h3><ul>{filt}</ul>
<h2>Sign-off</h2>{so_html}
<p style="color:#555">Trova assists the operator; the operator decides. Every decision above is in the audit log (data/events.jsonl).</p>
</body></html>"""

    # ------------------------------------------------------------------ setup
    @app.post("/api/background")
    def api_background(mode: str = Form("capture")):
        if mode == "clear":
            st.background, st.background_path = None, None
        elif mode == "capture":
            st.clear_view()
            st.background = st.source.empty() if isinstance(st.source, DemoSource) and st.override is None \
                else st.frame()
            st.background_path = st.audit.save_image("empty_table", st.background)
        else:
            raise HTTPException(400, "mode must be capture or clear")
        st.audit.event("background", mode=mode)
        return {"background": st.background is not None}

    @app.post("/api/source")
    def api_source(spec: str = Form(...)):
        try:
            new = open_source(spec)
        except Exception as e:  # noqa: BLE001 - report any camera error to the UI
            return fail(e, "source")
        old, st.source = st.source, new
        old.close()
        st.override, st.background, st.background_path = None, None, None
        st.audit.event("source", spec=spec)
        return st.source.info()

    @app.post("/api/source/step")
    def api_step(delta: int = Form(1)):
        if not isinstance(st.source, ImageSource):
            raise HTTPException(400, "Only for an image folder source")
        st.source.step(delta)
        return st.source.info()

    @app.post("/api/upload_scene")
    def api_upload_scene(photo: UploadFile = File(...)):
        st.override = decode_upload(photo.file.read())
        st.background, st.background_path = None, None
        st.audit.event("upload_scene", size=list(st.override.shape[:2]))
        return {"override": True}

    @app.post("/api/upload_scene/clear")
    def api_upload_clear():
        st.override = None
        return {"override": False}

    @app.post("/api/tune")
    def api_tune(t: Tune):
        if t.thresh is not None:
            st.thresh = t.thresh
        if t.min_area_frac is not None:
            st.min_area = t.min_area_frac
        return {"thresh": st.thresh, "min_area_frac": st.min_area}

    # ------------------------------------------------------------------ calibration
    @app.post("/api/calibration/aruco")
    def api_cal_aruco():
        try:
            mpath = st.s.data_dir / "markers.json"
            pos = load_marker_positions(mpath, st.s.table_w_mm, st.s.table_h_mm)
            st.calib = calibrate_aruco(st.frame(), pos)
            if mpath.exists() and json.loads(mpath.read_text()).get("_frame") == "robot":
                st.calib.source = "aruco_robot"
        except ValueError as e:
            return fail(e, "calibration")
        st.calib.save(st.calib_path)
        st.audit.event("calibration", **st.calib.as_dict())
        return st.calib.as_dict()

    @app.post("/api/calibration/click")
    def api_cal_click(p: ClickPoint):
        st.click_points.append([p.u, p.v, p.x_mm, p.y_mm])
        out = {"points": len(st.click_points)}
        if len(st.click_points) >= 4:
            c = Calibration(points=[list(x) for x in st.click_points], source="clicks")
            try:
                out["error_mm"] = round(c.fit(), 2)
            except ValueError as e:
                return fail(e, "calibration")
            st.calib = c
            c.save(st.calib_path)
            st.audit.event("calibration", **c.as_dict())
        return out

    @app.post("/api/calibration/reset")
    def api_cal_reset():
        st.click_points, st.calib = [], None
        if st.calib_path.exists():
            st.calib_path.unlink()
        return {"ok": True}

    @app.post("/api/calibration/markers")
    def api_markers():
        paths = make_marker_sheet(st.s.data_dir / "markers")
        return {"files": [f"/files/markers/{p.name}" for p in paths]}

    def _need_cartesian() -> CartesianArm:
        if not isinstance(st.arm, CartesianArm):
            raise HTTPException(400, "Needs a real Cartesian arm: TROVA_ARM=ur:<ip> | xarm:<ip> | elite:<ip> | cartesian:fake")
        return st.arm

    @app.post("/api/calibration/arm_click")
    def api_cal_arm_click(p: dict):
        """Put the arm tip on the table (freedrive or pendant), then click the tip in the live view."""
        arm = _need_cartesian()
        x, y, _ = arm.tcp_mm()
        pts = [q for q in st.click_points if len(q) == 4]
        pts.append([float(p["u"]), float(p["v"]), x, y])
        st.click_points = pts
        out = {"points": len(pts), "tcp_mm": [round(x, 1), round(y, 1)]}
        if len(pts) >= 4:
            c = Calibration(points=[list(q) for q in pts], source="arm")
            try:
                out["error_mm"] = round(c.fit(), 2)
            except ValueError as e:
                return fail(e, "calibration")
            st.calib = c
            c.save(st.calib_path)
            st.audit.event("calibration", **c.as_dict())
        return out

    @app.post("/api/calibration/marker_from_arm")
    def api_marker_from_arm(marker_id: int = Form(...)):
        """Touch the centre of ArUco marker <id> with the tip: its robot position is saved, so later
        'Calibrate (ArUco)' gives robot coordinates in seconds, even after the camera is bumped."""
        arm = _need_cartesian()
        x, y, _ = arm.tcp_mm()
        mpath = st.s.data_dir / "markers.json"
        d = json.loads(mpath.read_text()) if mpath.exists() else {}
        if d.get("_frame") != "robot":
            d = {"_frame": "robot"}
        d[str(marker_id)] = [round(x, 1), round(y, 1)]
        mpath.write_text(json.dumps(d, indent=2))
        st.audit.event("marker_from_arm", marker=marker_id, x_mm=x, y_mm=y)
        return {"markers": {k: v for k, v in d.items() if k != "_frame"}}

    @app.post("/api/arm/freedrive")
    def api_arm_freedrive(on: bool = Form(...)):
        arm = _need_cartesian()
        try:
            arm.freedrive(on)
        except NotImplementedError as e:
            return fail(e, "freedrive")
        st.audit.event("arm_freedrive", on=on)
        return {"freedrive": on}

    # ------------------------------------------------------------------ arm
    def _to_mm(p: Point) -> tuple[float, float]:
        if p.u is not None and p.v is not None:
            return st.calib.pixel_to_table(p.u, p.v)
        if p.x_mm is None or p.y_mm is None:
            raise HTTPException(400, "Give x_mm,y_mm or u,v")
        return p.x_mm, p.y_mm

    @app.post("/api/arm/stop")
    def api_arm_stop():
        st.arm.stop()
        st.audit.event("arm_stop")
        return st.arm.status()

    @app.post("/api/arm/home")
    def api_arm_home():
        st.start_arm(st.arm.home)
        return st.arm.status()

    @app.post("/api/arm/point")
    def api_arm_point(p: Point):
        if p.u is not None and st.arm_blocked():
            raise HTTPException(409, st.arm_blocked())
        x, y = _to_mm(p)
        st.start_arm(st.arm.point_at, x, y)
        st.audit.event("arm_point", x_mm=x, y_mm=y)
        return {"x_mm": round(x, 1), "y_mm": round(y, 1)}

    @app.post("/api/arm/teach")
    def api_arm_teach(p: Point):
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        x, y = _to_mm(p)
        return st.arm.record(x, y)

    @app.post("/api/arm/teach_home")
    def api_arm_teach_home():
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        return {"home_q": st.arm.record_home()}

    @app.post("/api/arm/compliant")
    def api_arm_compliant(on: bool = Form(...)):
        if not isinstance(st.arm, TaughtGridArm):
            raise HTTPException(400, "Teaching needs TROVA_ARM=grid:...")
        st.arm.backend.set_compliant(on)
        st.audit.event("arm_compliant", on=on)
        return {"compliant": on}

    # ------------------------------------------------------------------ jury mode + eval cases
    @app.post("/api/judge")
    def api_judge(j: Judge):
        if j.kind == "find":
            lf = st.last_find
            if not lf:
                raise HTTPException(400, "No find to judge")
            st.metrics["find_ok" if j.correct else "find_wrong"] += 1
            case = {"frame": lf["frame"], "request": lf["request"], "query": lf["query"],
                    "background": st.background_path,
                    "expected": lf["result"]["box"]["box"] if (j.correct and lf["result"]["box"]) else None,
                    "absent": j.correct and not lf["result"]["box"], "note": j.note}
            if j.correct:
                _save_case(case)
            else:
                st.label_wait = case   # next: click the right object, or "not on the table"
            st.audit.event("judge", target=lf["event_id"], correct=j.correct, note=j.note)
        else:
            st.metrics["check_ok"] += int(j.correct)
            st.metrics["false_alarms"] += j.false_alarms
            st.metrics["missed"] += j.missed
            st.audit.event("judge_check", target=(st.last_check or {}).get("event_id"), correct=j.correct,
                           false_alarms=j.false_alarms, missed=j.missed, note=j.note)
        return st.metrics | {"label_wait": st.label_wait is not None}

    @app.post("/api/label")
    def api_label(lb: Label):
        if not st.label_wait:
            raise HTTPException(400, "Nothing waiting for a label")
        case = st.label_wait
        if lb.absent:
            case["expected"], case["absent"] = None, True
        else:
            inside = [b for b in (st.last_find or {}).get("boxes", [])
                      if Box.from_list(b["box"]).contains(lb.u, lb.v)]
            if not inside:
                raise HTTPException(400, "No candidate box there; click inside a numbered box")
            case["expected"] = min(inside, key=lambda b: Box.from_list(b["box"]).area)["box"]
        _save_case(case)
        st.label_wait = None
        return case

    def _save_case(case: dict) -> None:
        with (st.cases_dir / "finds.jsonl").open("a") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ feature 1: inventory map
    @app.post("/api/inventory")
    def api_inventory():
        with st.lock:
            st.clear_view()
            f, boxes, occl = st.capture_clean(attempts=3)
            try:
                inv = describe_table(st.grok, f, boxes, st.s.lang)
            except GrokError as e:
                return fail(e, "inventory")
            out = inv.as_dict() | {"map": st.audit.save_image("inventory_map", draw_map(f, boxes, inv)),
                                   "n_objects": len(boxes), "occluded": occl}
            st.inventory = out
            st.audit.event("inventory", **{k: v for k, v in out.items()})
            return out

    # ------------------------------------------------------------------ voice (Grok Voice realtime)
    @app.post("/api/arm/point_last")
    def api_point_last():
        """Move to the object of the last find (used after a spoken 'yes')."""
        lf = getattr(st, "pending_move", None)
        if not lf or not lf.get("table_mm"):
            raise HTTPException(400, "Nothing found yet to point at")
        blocked = st.arm_blocked()
        if blocked:
            return JSONResponse({"error": blocked, "blocked": True}, 409)
        st.start_arm(st.arm.point_at, *lf["table_mm"])
        st.audit.event("arm_point", x_mm=lf["table_mm"][0], y_mm=lf["table_mm"][1], after="voice confirmation")
        return {"ok": True, "label": lf["result"]["label"], "table_mm": lf["table_mm"]}

    @app.get("/api/voice/config")
    def api_voice_config(confirm: bool = True):
        return {"model": st.s.voice_model, "available": not st.grok.mock, "stop_words": voice_cfg.STOP_WORDS,
                "session": voice_cfg.session_config(st.s.voice, confirm, st.dog is not None)}

    @app.post("/api/voice/session")
    def api_voice_session():
        if st.grok.mock:
            return JSONResponse({"error": "Voice needs XAI_API_KEY in .env (mock mode has no voice)"}, 400)
        try:
            return voice_cfg.mint_token(st.s.xai_api_key)
        except Exception as e:  # noqa: BLE001
            return fail(e, "voice")

    class VoiceLog(BaseModel):
        who: str
        text: str

    @app.post("/api/voice/log")
    def api_voice_log(v: VoiceLog):
        st.audit.event("voice", who=v.who, text=v.text[:500])
        return {"ok": True}

    # ------------------------------------------------------------------ optional dog
    @app.post("/api/dog/search")
    def api_dog_search(description: str = Form(...), wait: bool = Form(True)):
        if st.dog is None:
            raise HTTPException(400, "No dog in this setup (TROVA_DOG=off)")
        if st.dog_thread and st.dog_thread.is_alive():
            raise HTTPException(409, "The dog is already searching")
        st.dog_stop.clear()
        st.dog_search = DogSearch(request=description)

        def go():
            try:
                run_search(st.dog, st.grok, st.dog_route, description, st.dog_stop, st.audit.save_image, st.dog_search)
            except Exception as e:  # noqa: BLE001
                st.dog_search.status = "error"
                st.dog_search.where = str(e)
            st.audit.event("dog_search", **st.dog_search.as_dict())

        st.dog_thread = threading.Thread(target=go, daemon=True)
        st.dog_thread.start()
        if wait:
            st.dog_thread.join(timeout=90)
        return st.dog_search.as_dict()

    @app.post("/api/dog/stop")
    def api_dog_stop():
        st.dog_stop.set()
        if st.dog:
            st.dog.stop()
        st.audit.event("dog_stop")
        return {"stopped": True}

    @app.get("/api/log")
    def api_log(n: int = 40, kind: str | None = None):
        return st.audit.recent(n, kind)

    # ------------------------------------------------------------------ demo helpers
    @app.post("/api/demo/tamper")
    def api_tamper(kind: str = Form("")):
        if not isinstance(st.source, DemoSource):
            raise HTTPException(400, "Only in demo mode")
        return st.source.table.tamper(kind or None)

    @app.post("/api/demo/hand")
    def api_hand(on: bool = Form(...)):
        if not isinstance(st.source, DemoSource):
            raise HTTPException(400, "Only in demo mode")
        st.source.hand = on
        return {"hand": on}

    return app
