"""Feature 9: preflight check. One button (or `python -m trova.preflight`) before the jury arrives:
camera, image quality, empty table, calibration, arm and safety box, Grok, voice, disk.

Each item is ok | warn | fail, with the fix in plain words. Nothing here moves the arm.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import cv2
import numpy as np

from .arm import CartesianArm, SimArm
from .arm.skills import lift_allowed
from .camera import DemoSource
from .grok import GrokError, register_mock
from . import voice as voice_cfg

PING_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
               "additionalProperties": False}
register_mock("trova_ping", lambda prompt, images, ctx: {"ok": True})


def _item(name: str, status: str, detail: str, fix: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "fix": fix}


def _camera(st) -> tuple[list[dict], np.ndarray | None]:
    out = []
    try:
        t0 = time.perf_counter()
        frames = []
        for _ in range(5):
            frames.append(st.frame())
            time.sleep(0.05)
        dt = (time.perf_counter() - t0) / 5
    except Exception as e:  # noqa: BLE001 - any camera failure is a preflight failure
        return [_item("Camera", "fail", f"{type(e).__name__}: {e}", "Check the cable / Continuity Camera, then Setup: Switch source")], None
    f = frames[-1]
    h, w = f.shape[:2]
    kind = st.source.info().get("kind", "?") + (" (uploaded photo)" if st.override is not None else "")
    frozen = st.override is None and not isinstance(st.source, DemoSource) and \
        st.source.info().get("kind") in ("webcam", "video") and all(np.array_equal(frames[0], x) for x in frames[1:])
    if frozen:
        out.append(_item("Camera", "fail", f"{kind} {w}x{h}: the picture is frozen", "Reconnect the camera or restart the source"))
    elif isinstance(st.source, DemoSource) and st.override is None:
        out.append(_item("Camera", "warn", f"demo source (synthetic table) {w}x{h}", "TROVA_CAMERA=0 (or 1) for the real camera"))
    elif w < 960:
        out.append(_item("Camera", "warn", f"{kind} {w}x{h}: low resolution", "Use 1920x1080 if the camera allows it"))
    else:
        out.append(_item("Camera", "ok", f"{kind} {w}x{h}, {dt * 1000:.0f} ms per frame"))
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    mean, sharp = float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mean < 50 or mean > 210:
        out.append(_item("Light", "warn", f"mean brightness {mean:.0f}/255", "Too dark or too bright: move the lamp, lock exposure"))
    else:
        out.append(_item("Light", "ok", f"mean brightness {mean:.0f}/255"))
    out.append(_item("Focus", "warn" if sharp < 20 else "ok", f"sharpness {sharp:.0f}",
                     "Picture looks blurry: refocus or clean the lens" if sharp < 20 else ""))
    return out, f


def _vision(st, f) -> list[dict]:
    out = []
    if st.background is None:
        out.append(_item("Empty table", "warn", "not captured: objects are found against an estimated background",
                         "Clear the table, Setup: Capture empty table"))
    else:
        out.append(_item("Empty table", "ok", "captured"))
    if f is not None:
        boxes, occl = st.proposals(f)
        if occl:
            out.append(_item("Objects", "warn", f"{len(boxes)} found, but a hand or the arm is in view", "Clear the view"))
        elif not boxes:
            out.append(_item("Objects", "warn", "no objects found", "Put objects on the table or lower the object threshold"))
        else:
            out.append(_item("Objects", "ok", f"{len(boxes)} found"))
    return out


def _arm(st) -> list[dict]:
    out = []
    c = st.calib
    cal = f"{c.source}, error {c.as_dict()['error_mm']} mm" if c and c.ok else "none"
    if isinstance(st.arm, CartesianArm):
        if st.arm_blocked():
            out.append(_item("Calibration", "fail", cal + ": the real arm will refuse to move",
                             "Setup: Calibrate with arm (4-6 clicks on the tip), or ArUco with markers recorded by the arm"))
        elif c.error_mm() > 8:
            out.append(_item("Calibration", "warn", cal, "Error above 8 mm: redo with points spread over the whole table"))
        else:
            out.append(_item("Calibration", "ok", cal))
    else:
        out.append(_item("Calibration", "ok" if c and c.ok else "warn", cal))
    s = st.arm.status()
    if s.get("state") == "error" or s.get("error"):
        out.append(_item("Arm", "fail", f"{s['name']}: {s.get('error') or s['state']}",
                         "Pendant: clear the protective stop, enable Remote Control, then Arm home"))
    elif isinstance(st.arm, SimArm):
        out.append(_item("Arm", "warn", "simulated arm", "TROVA_ARM=ur:<ip> | xarm:<ip> | elite:<ip> for the real cobot"))
    else:
        out.append(_item("Arm", "ok", f"{s['name']}: {s['state']}, tool at {s.get('tcp_mm') or s.get('tip_mm')}"))
    if isinstance(st.arm, CartesianArm):
        out.append(_item("Safety box", "ok" if st.arm.limits else "fail",
                         "TROVA_ARM_LIMITS set" if st.arm.limits else "no workspace limits",
                         "" if st.arm.limits else "Set TROVA_ARM_LIMITS=xmin,xmax,ymin,ymax,zmin,zmax with staff"))
        home = st.arm.home_joints and f"joints {st.arm.home_joints}" or (st.arm.home_mm and f"{st.arm.home_mm}")
        out.append(_item("Home pose", "ok" if home else "fail", home or "none: the arm refuses to move without one",
                         "" if home else "Teach survey pose here (camera on the arm) or set TROVA_ARM_HOME_MM=x,y,z"))
    if st.camera_on_arm:
        from .arm import load_poses
        poses = load_poses(st.s.data_dir)
        if not isinstance(st.arm, CartesianArm):
            out.append(_item("Arm camera", "warn", "TROVA_CAMERA_ON_ARM=1 with a simulated arm", "Use the real arm or cartesian:fake"))
        elif not st.arm.home_joints:
            out.append(_item("Arm camera", "fail", "no survey pose", "Move the arm so the camera sees the whole table: Teach survey pose here"))
        else:
            out.append(_item("Arm camera", "ok" if st.at_survey() else "warn",
                             "at the survey pose" if st.at_survey() else "away from the survey pose",
                             "" if st.at_survey() else "Arm home"))
        out.append(_item("Room scan", "ok" if poses.get("look_joints") else "warn",
                         "look pose taught" if poses.get("look_joints") else "no look pose",
                         "" if poses.get("look_joints") else "Point the camera at the room by hand: Teach look pose here"))
    if st.first_run:
        problems = st.first_run_problems()
        out += [_item("First run", "fail", p, "Fix this before the first supervised run") for p in problems] or [
            _item("First run", "ok", f"ready: hover +{st.first_run_hover:.0f} mm, {st.first_run_speed:.0f} mm/s, "
                  "calibrated at this survey pose")]
    if lift_allowed():
        out.append(_item("Tray lift", "warn", "enabled", "Only with measured TROVA_TRAY_* values, agreed with staff"))
    if not os.getenv("TROVA_TOKEN", "").strip() and st.s.host not in ("127.0.0.1", "localhost"):
        out.append(_item("Wi-Fi access", "warn" if isinstance(st.arm, CartesianArm) else "ok",
                         "open to anyone on this network" if isinstance(st.arm, CartesianArm) else "open (simulated arm)",
                         "Set TROVA_TOKEN=<secret> and open the printed ?token= link on each device"
                         if isinstance(st.arm, CartesianArm) else ""))
    return out


def _grok(st, ping: bool) -> list[dict]:
    g = st.grok
    if g.mock:
        return [_item("Grok", "warn", "MOCK mode: answers are simulated", "Put XAI_API_KEY in .env and restart"),
                _item("Voice", "warn", "needs XAI_API_KEY", "")]
    out = []
    if ping:
        g.offline_until = 0.0                 # really try the network, even inside the offline hold window
        t0 = time.perf_counter()
        try:
            r = g.json(name="trova_ping", prompt='Reply with {"ok": true}.', schema=PING_SCHEMA, effort="")
            dt = time.perf_counter() - t0
            if r.offline:
                out.append(_item("Grok", "fail", f"unreachable: {g.last_error}", "Check Wi-Fi / hotspot; Trova falls back to colour matching"))
            else:
                out.append(_item("Grok", "warn" if dt > 6 else "ok", f"{g.model} answered in {dt:.1f} s",
                                 "Slow: try TROVA_EFFORT=low or the non-reasoning model" if dt > 6 else ""))
        except GrokError as e:
            out.append(_item("Grok", "fail", str(e)[:200], "401/403: check XAI_API_KEY; 400: check TROVA_MODEL"))
    else:
        out.append(_item("Grok", "ok", f"{g.model} (not pinged)"))
    try:
        voice_cfg.mint_token(st.s.xai_api_key, seconds=60)
        out.append(_item("Voice", "ok", f"token minted for {st.s.voice_model}"))
    except Exception as e:  # noqa: BLE001
        out.append(_item("Voice", "warn", str(e)[:200], "Typing and photos still work; the browser mic fallback works too"))
    return out


def _disk(st) -> list[dict]:
    d = st.s.data_dir
    try:
        p = d / ".preflight"
        p.write_text("ok")
        p.unlink()
        free = shutil.disk_usage(d).free / 1e9
        return [_item("Disk", "warn" if free < 1 else "ok", f"{d} writable, {free:.1f} GB free",
                      "Free some space: every find saves images" if free < 1 else "")]
    except OSError as e:
        return [_item("Disk", "fail", f"{d}: {e}", "Set TROVA_DATA to a writable folder")]


def run(st, ping_grok: bool = True) -> dict:
    t0 = time.perf_counter()
    cam, f = _camera(st)
    items = cam + _vision(st, f) + _arm(st) + _grok(st, ping_grok) + _disk(st)
    if st.reference:
        items.append(_item("Reference", "ok", f"saved at {st.reference['time']}, {st.reference['n_objects']} objects"))
    items.append(_item("Zones", "ok", ", ".join(z.name for z in st.zones) or "none drawn"))
    n = {k: sum(i["status"] == k for i in items) for k in ("ok", "warn", "fail")}
    return {"ok": n["fail"] == 0, "counts": n, "items": items, "time": time.strftime("%H:%M:%S"),
            "summary": f"{n['ok']} ok, {n['warn']} warnings, {n['fail']} failures",
            "seconds": round(time.perf_counter() - t0, 1)}


def register(app, st) -> None:
    @app.get("/api/preflight")
    def api_preflight(grok: bool = True):
        out = run(st, grok)
        st.audit.event("preflight", summary=out["summary"],
                       problems=[f"{i['name']}: {i['detail']}" for i in out["items"] if i["status"] != "ok"])
        return out


def main() -> None:
    """python -m trova.preflight [http://localhost:8000] : print the checklist of a running Trova; exit 1 on failure."""
    import requests
    base = (sys.argv[1] if len(sys.argv) > 1 else f"http://localhost:{os.getenv('TROVA_PORT', '8000')}").rstrip("/")
    d = requests.get(base + "/api/preflight", timeout=120, headers={"x-trova-token": os.getenv("TROVA_TOKEN", "")}).json()
    mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}
    for i in d["items"]:
        print(f"{mark[i['status']]}  {i['name']:<12} {i['detail']}" + (f"\n{'':18}-> {i['fix']}" if i["fix"] and i["status"] != "ok" else ""))
    print(f"\n{d['summary']} ({d['seconds']} s)")
    sys.exit(0 if d["ok"] else 1)


if __name__ == "__main__":
    main()
