"""Tests for the polish pass and the features added after the first seven (see AGENTS.md)."""

import re
import types
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "trova" / "static"

# Inside an inline handler (onclick="..."), names resolve on the element, then its form, then document,
# before the page's own functions. A helper called form() therefore calls button.form (null) and breaks.
SHADOWED = {"form", "submit", "reset", "click", "focus", "blur", "remove", "append", "prepend", "before", "after",
            "replaceWith", "animate", "closest", "matches", "contains", "select", "open", "close", "write",
            "toggle", "labels", "value", "name", "type", "title", "hidden", "style", "dataset", "checked"}


def test_inline_handlers_do_not_call_shadowed_names():
    for page in ("index.html", "jury.html", "history.html"):
        html = (STATIC / page).read_text()
        for handler in re.findall(r'\son\w+="([^"]*)"', html):
            for fn in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\(", handler):
                assert fn not in SHADOWED, f"{page}: inline handler calls {fn}(), which resolves to an element property"


# ---------------------------------------------------------------- feature 8: offline fallback
def test_offline_fallback_and_circuit_breaker(monkeypatch):
    import requests

    from trova import grok as grok_mod
    from trova.finder import find
    from trova.grok import GrokClient, GrokError, GrokUnavailable
    from trova.proposals import find_blobs
    from trova.synthetic import SyntheticTable

    t = SyntheticTable(seed=3)
    frame, empty = t.render(), t.empty()
    boxes = find_blobs(frame, empty)
    colors = [o.color for o in t.objects]
    unique = next(c for c in colors if colors.count(c) == 1)
    calls = []

    def down(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("no route to host")

    g = GrokClient("xai-test", offline_fallback=True)
    monkeypatch.setattr(g.session, "post", down)
    monkeypatch.setattr(grok_mod.time, "sleep", lambda s: None)
    r = find(g, frame, boxes, f"the {unique} thing")
    assert r.status == "found" and r.mode == "offline" and r.reason.startswith("Grok unreachable")
    assert g.offline and g.stats()["offline_calls"] == 1 and "no route" in g.stats()["last_error"]
    n = len(calls)
    find(g, frame, boxes, f"the {unique} thing")        # inside the hold window: no network attempt at all
    assert len(calls) == n
    g.offline_until = 0                                  # window over: live is tried again
    find(g, frame, boxes, f"the {unique} thing")
    assert len(calls) > n

    def unauthorized(*a, **k):
        return types.SimpleNamespace(status_code=401, text="invalid api key", json=lambda: {})
    g2 = GrokClient("xai-bad", offline_fallback=True)
    monkeypatch.setattr(g2.session, "post", unauthorized)
    with pytest.raises(GrokError, match="401"):          # configuration errors are never hidden
        find(g2, frame, boxes, "the pen")

    g3 = GrokClient("xai-test", offline_fallback=False)   # eval mode: never measure the stand-in
    monkeypatch.setattr(g3.session, "post", down)
    with pytest.raises(GrokUnavailable):
        find(g3, frame, boxes, "the pen")


# ---------------------------------------------------------------- arm safety (review fixes)
def _wait(cond, timeout=5.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _unique_color(c):
    colors = [o.split()[0] for o in c.get("/api/state").json()["source"]["objects"]]
    return next(col for col in colors if colors.count(col) == 1)


def test_stop_latches_in_every_driver():
    from trova.arm import CartesianArm, FakeCartesian, FakeJointBackend, SimArm, TaughtGridArm
    sim = SimArm(speed_mm_s=5000)
    sim.stop()
    assert not sim.point_at(100, 100) and sim.status()["state"] == "stopped"   # refused until ready()
    sim.ready()
    assert sim.point_at(100, 100)
    cart = CartesianArm(FakeCartesian(), 150, None, None, speed_mm_s=5000)
    cart.stop()
    assert not cart.point_at(300, 0) and cart.backend.history == []
    cart.ready()
    assert cart.point_at(300, 0)
    grid = TaughtGridArm(FakeJointBackend(), Path("/nonexistent/grid.json"))
    grid.stop()
    assert not grid._go([0.0, 0.0], None)


def test_stop_racing_a_send_is_sent_again_and_late_start_is_waited_for():
    import time

    from trova.arm import CartesianArm, FakeCartesian

    class RacingStop(FakeCartesian):           # STOP arrives while the move is still being sent (UR/Elite)
        def __init__(self):
            super().__init__()
            self.stops = 0
            self.arm = None

        def move_linear(self, x, y, z, speed_mm_s):
            super().move_linear(x, y, z, speed_mm_s)
            self.arm._stop.set()               # what stop() does from the other thread, mid-send

        def stop(self):
            self.stops += 1
            super().stop()
    be = RacingStop()
    arm = CartesianArm(be, 150, None, None, speed_mm_s=50)
    be.arm = arm
    assert not arm.point_at(400, 100) and arm.state == "stopped" and be.stops >= 1

    class LateStart(FakeCartesian):            # xArm: reports "not moving" for a moment after accepting a move
        def move_linear(self, x, y, z, speed_mm_s):
            self.t_go, self.go = time.time() + 0.25, (x, y, z, speed_mm_s)

        def is_moving(self):
            if getattr(self, "go", None) and time.time() >= self.t_go:
                super().move_linear(*self.go)
                self.go = None
            return False if getattr(self, "go", None) else super().is_moving()
    late = CartesianArm(LateStart(), 150, None, None, speed_mm_s=2000)
    assert late.point_at(400, 100), late.last_error


def test_stop_during_tour_pause_and_during_a_find(tmp_path, monkeypatch):
    import threading
    import time

    from test_pipeline import _client
    from trova import finder
    from trova.grok import register_mock
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    st.arm.speed = 5000
    c.post("/api/background", data={"mode": "capture"})
    r = c.post("/api/find_all", data={"text": "all the black things"}).json()
    assert r["moved"] and len(r["targets"]) >= 2
    assert _wait(lambda: st.tour and st.tour["index"] == 0 and st.arm.status()["state"] == "idle")
    time.sleep(0.2)                                       # inside the 1 s pause after the first target
    c.post("/api/arm/stop")
    st.arm_thread.join(timeout=3)
    assert st.tour["status"] == "stopped" and st.tour["index"] == 0

    unique = _unique_color(c)
    slow = lambda prompt, images, ctx: (time.sleep(0.8), finder._mock_select(prompt, images, ctx))[1]  # noqa: E731
    register_mock("trova_select", slow)
    try:
        out = {}
        th = threading.Thread(target=lambda: out.update(c.post("/api/find", data={"text": f"the {unique} one"}).json()))
        th.start()
        time.sleep(0.3)
        st.stop_arm()                                     # STOP while Grok is thinking
        th.join(timeout=5)
    finally:
        register_mock("trova_select", finder._mock_select)
    assert out["result"]["status"] == "found" and not out["moved"] and "STOP" in out["arm_blocked"]
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()   # a new request moves again
    assert r["moved"]


def test_one_motion_at_a_time_and_interrupted_lift(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, TROVA_ALLOW_LIFT="1")
    st = app.state.trova
    c.post("/api/background", data={"mode": "capture"})
    unique = _unique_color(c)
    c.post("/api/find", data={"text": f"the {unique} one", "move": "false"})
    st.arm.speed = 60
    assert c.post("/api/arm/lift_tray", json={"place_u": 900, "place_v": 500}).json()["ok"]
    assert _wait(lambda: st.arm.status()["state"] == "moving")
    r = c.post("/api/arm/point", json={"x_mm": 100, "y_mm": 100})
    assert r.status_code == 409 and "lifting" in str(r.json())          # a lift is never cut by a new move
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()       # finding works, moving does not
    assert not r["moved"] and "lifting" in r["arm_blocked"]
    c.post("/api/arm/stop")
    st.arm_thread.join(timeout=3)
    assert st.lift_interrupted and "lift was interrupted" in c.get("/api/state").json()["arm_blocked"]
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()
    assert not r["moved"] and "interrupted" in r["arm_blocked"]
    st.arm.speed = 5000
    c.post("/api/arm/home")                              # the operator's explicit way out
    assert not st.lift_interrupted and c.post("/api/find", data={"text": f"the {unique} one"}).json()["moved"]
    first = st.arm_thread
    c.post("/api/arm/point", json={"x_mm": 100, "y_mm": 100})
    assert not first.is_alive()                          # the previous move ended before the new one began


def test_calibration_safety(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from test_pipeline import _client
    from trova.calibration import Calibration
    good = [[0, 0, 0, 0], [1000, 0, 500, 0], [1000, 800, 500, 400], [0, 800, 0, 400]]
    assert Calibration(points=[list(map(float, p)) for p in good]).fit() < 0.01
    for bad, why in (([[0, 0, 0, 0], [5, 3, 0, 1], [1000, 800, 500, 400], [0, 800, 0, 400]], "same spot"),
                     ([[0, 0, 0, 0], [500, 400, 250, 200], [1000, 800, 500, 400], [0, 800, 0, 400]], "line"),
                     (good + [[500, 400, 400, 50]], "wrong")):
        with pytest.raises(ValueError, match=why):
            Calibration(points=[list(map(float, p)) for p in bad]).fit()
    (tmp_path / "bad.json").write_text('{"points": [[0,0,0,0],[1,1,1,1]]}')
    assert Calibration.load(tmp_path / "bad.json") is None              # a bad file never blocks start-up

    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="cartesian:fake", TROVA_ARM_LIMITS="100,700,-300,300,0,400",
                     TROVA_ARM_HOME_MM="300,250,250", TROVA_ARM_SPEED_MM_S="5000")
    st = app.state.trova
    fake = st.arm.backend
    for u, v, x, y in [(0, 0, 0, 0), (1280, 0, 600, 0), (1280, 800, 600, 400), (0, 800, 0, 400)]:
        c.post("/api/calibration/click", json={"u": u, "v": v, "x_mm": x, "y_mm": y})   # table-mm clicks
    fake.pos = [300.0, 0.0, 0.0]
    fake.target = list(fake.pos)
    out = c.post("/api/calibration/arm_click", json={"u": 640, "v": 400}).json()
    assert out["points"] == 1 and st.calib.source == "clicks" and st.arm_blocked()   # never mixed
    for (u, v), (x, y) in zip([(0, 0), (1280, 0), (1280, 800)], [(200, -200), (520, -200), (520, 0)]):
        fake.pos = [x, y, 0.0]
        fake.target = list(fake.pos)
        out = c.post("/api/calibration/arm_click", json={"u": u, "v": v})
    assert out.status_code == 400 and "not kept" in out.json()["error"]  # the first arm click was the wrong one
    c.post("/api/calibration/reset")
    for (u, v), (x, y) in zip([(0, 0), (1280, 0), (1280, 800), (0, 800)], [(200, -200), (520, -200), (520, 0), (200, 0)]):
        fake.pos = [x, y, 0.0]
        fake.target = list(fake.pos)
        out = c.post("/api/calibration/arm_click", json={"u": u, "v": v}).json()
    assert st.calib.source == "arm" and st.calib.size == st.frame_size and st.arm_blocked() is None
    ok, buf = cv2.imencode(".jpg", np.full((1500, 2000, 3), 128, np.uint8))
    c.post("/api/upload_scene", files={"photo": ("p.jpg", buf.tobytes(), "image/jpeg")})
    assert "uploaded photo" in st.arm_blocked()
    assert c.post("/api/arm/point", json={"u": 1900, "v": 1400}).status_code == 409
    c.post("/api/upload_scene/clear")
    st.frame_size = [1920, 1080]                          # the camera came back at another resolution
    assert "calibrate again" in st.arm_blocked()


def test_clarify_answer_uses_the_same_frame_and_numbers(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    c.post("/api/background", data={"mode": "capture"})
    r = c.post("/api/find", data={"text": "the black one", "move": "false"}).json()
    assert r["result"]["status"] == "clarify"
    asked = {b["id"]: b["box"] for b in r["boxes"]}
    pick = r["result"]["candidate_ids"][0]
    st.source.table.tamper("move")                        # the scene changes before the answer
    r2 = c.post("/api/clarify", data={"answer": f"number {pick}"}).json()
    assert r2["result"]["status"] == "found" and r2["result"]["box"]["box"] == asked[pick]
    assert c.post("/api/clarify", data={"answer": "1"}).status_code == 400   # a second click: nothing pending


# ---------------------------------------------------------------- robustness (review fixes)
def test_grok_field_dropping_is_exact_and_answers_are_validated(monkeypatch):
    from trova import grok as grok_mod
    from trova.grok import GrokClient, GrokError
    monkeypatch.setattr(grok_mod.time, "sleep", lambda s: None)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    sent = []

    def reply(status, text):
        return types.SimpleNamespace(status_code=status, text=text, json=lambda: {"output_text": text})

    def image_format_error(url, json=None, **k):
        sent.append(json)
        return reply(400, "Unsupported image format: image/heic")
    g = GrokClient("xai-test")
    monkeypatch.setattr(g.session, "post", image_format_error)
    with pytest.raises(GrokError, match="image format"):
        g.json(name="x", prompt="p", schema=schema)
    assert g.stats()["dropped_fields"] == [] and len(sent) == 1         # text.format was NOT dropped

    answers = iter([reply(400, "Argument not supported: reasoning_effort"), reply(200, '{"other": 1}')])
    g2 = GrokClient("xai-test")
    monkeypatch.setattr(g2.session, "post", lambda *a, **k: next(answers))
    with pytest.raises(GrokError, match="missing"):                    # never a KeyError deeper in the app
        g2.json(name="x", prompt="p", schema=schema)
    assert g2.stats()["dropped_fields"] == ["reasoning"]


def test_too_many_changes_skip_grok_and_stay_not_clear():
    import numpy as np

    from trova.boxes import Box
    from trova.changes import Change, verify_changes
    f = np.zeros((400, 600, 3), np.uint8)
    many = [Change(id=i + 1, kind="added", cur_box=Box(10 + 30 * i, 10, 30 + 30 * i, 40, id=i + 1)) for i in range(15)]

    class NoGrok:
        def json(self, **k):
            raise AssertionError("must not call Grok")
    real, filtered, clear, summary, _ = verify_changes(NoGrok(), f, f, many)
    assert len(real) == 15 and not clear and "camera" in summary and real[0].risk == "high"


def test_frozen_webcam_is_refused():
    import threading
    import time

    import numpy as np

    from trova.camera import WebcamSource
    cam = object.__new__(WebcamSource)
    cam.index, cam._lock = 0, threading.Lock()
    cam._frame, cam._ts = np.zeros((10, 10, 3), np.uint8), time.time()
    assert cam.read().shape == (10, 10, 3)
    cam._ts -= 5
    with pytest.raises(RuntimeError, match="no new frame"):
        cam.read()


def test_bad_requests_get_clean_answers(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    assert c.post("/api/find", data={"text": "x"}, files={"photo": ("e.jpg", b"", "image/jpeg")}).status_code == 400
    assert c.post("/api/upload_scene", files={"photo": ("e.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert c.post("/api/zones/point", json={"v": 3}).status_code == 422
    assert c.post("/api/judge", json={"kind": "change", "correct": True}).status_code == 400
    c.post("/api/calibration/reset")
    assert c.post("/api/arm/point", json={"u": 100, "v": 100}).status_code == 200   # default calibration rebuilt
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/find", data={"text": "the red one", "move": "false"})
    assert c.post("/api/judge", json={"kind": "find", "correct": True}).status_code == 200
    assert c.post("/api/judge", json={"kind": "find", "correct": True}).status_code == 409   # double click
    assert c.get("/api/state").json()["metrics"]["find_judged"] == 1

    def broken():
        raise RuntimeError("Camera 0 sent no new frame for 3 s")
    monkeypatch.setattr(st.source, "read", broken)
    s = c.get("/api/state")
    assert s.status_code == 200 and "no new frame" in s.json()["camera_error"]
    r = c.post("/api/find", data={"text": "the red one"})
    assert r.status_code == 503 and r.json()["camera"]


def test_access_token(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, TROVA_TOKEN="s3cret")
    assert c.get("/").status_code == 200                        # the page loads, the API does not
    assert c.get("/api/state").status_code == 401
    assert c.post("/api/arm/point", json={"x_mm": 10, "y_mm": 10}).status_code == 401
    assert c.get("/api/state", headers={"x-trova-token": "s3cret"}).status_code == 200
    assert c.get("/?token=s3cret").status_code == 200             # sets the cookie for this device
    assert c.get("/api/state").status_code == 200


# ---------------------------------------------------------------- feature 9: preflight
def test_preflight(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    d = c.get("/api/preflight").json()
    names = {i["name"]: i for i in d["items"]}
    assert {"Camera", "Light", "Focus", "Empty table", "Objects", "Calibration", "Arm", "Grok", "Disk"} <= set(names)
    assert names["Grok"]["status"] == "warn" and "MOCK" in names["Grok"]["detail"]
    assert names["Empty table"]["status"] == "warn" and d["ok"]           # warnings only: still OK to go
    c.post("/api/background", data={"mode": "capture"})
    assert {i["name"]: i for i in c.get("/api/preflight").json()["items"]}["Empty table"]["status"] == "ok"

    app2, c2 = _client(tmp_path / "real", monkeypatch, TROVA_ARM="cartesian:fake")
    d2 = c2.get("/api/preflight").json()
    n2 = {i["name"]: i for i in d2["items"]}
    assert not d2["ok"] and n2["Safety box"]["status"] == "fail" and n2["Calibration"]["status"] == "fail"
    assert "TROVA_ARM_LIMITS" in n2["Safety box"]["fix"] and n2["Home pose"]["status"] == "fail"

    from trova import grok as grok_mod
    import requests
    app3, c3 = _client(tmp_path / "live", monkeypatch, mock="0", XAI_API_KEY="xai-test")
    st3 = app3.state.trova
    monkeypatch.setattr(grok_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(st3.grok.session, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")))
    monkeypatch.setattr("trova.voice.mint_token", lambda key, seconds=300: {"token": "t"})
    n3 = {i["name"]: i for i in c3.get("/api/preflight").json()["items"]}
    assert n3["Grok"]["status"] == "fail" and "unreachable" in n3["Grok"]["detail"] and n3["Voice"]["status"] == "ok"


# ---------------------------------------------------------------- feature 10: watch the line
def test_watch_alarms_on_change_and_zone_pauses_on_hand(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    w = st.watch
    assert c.post("/api/watch", json={"on": True}).status_code == 400        # needs a reference
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    w.start(point=True)
    w.stop()                                     # drive it step by step instead of the thread
    w.point = True
    for _ in range(4):
        w.step()
    assert w.alert is None and "clear" in w.status
    st.source.hand = True
    w.step()
    assert "hand" in w.status and len(w._hist) == 0                           # no alarm on people working
    st.source.hand = False
    st.source.table.tamper("add")
    for _ in range(3):
        w.step()
    a = w.alert
    assert a and a["changes"][0]["kind"] == "added" and a["pointed"] and "added" in a["summary"]
    assert _wait(lambda: a["grok"] != "pending") and "explained" in a["grok"]    # mock Grok confirms it
    assert c.get("/api/state").json()["watch"]["alert"]["id"] == a["id"]
    for _ in range(3):
        w.step()
    assert w.alerts == 1                                                         # same situation: one alarm
    r = c.post("/api/watch/ack", data={"who": "Ana"}).json()
    assert r["acknowledged"] == a["id"] and w.alert is None
    for _ in range(3):
        w.step()
    assert w.alert is None and "acknowledged" in w.status                       # not again after the ack
    c.post("/api/reference", data={"describe": "false"})                        # new approved state
    for _ in range(4):
        w.step()
    assert w.alert is None and "clear" in w.status

    from trova.zones import Zone
    st.zones = [Zone(name="line", polygon=[[0, 0], [1280, 0], [1280, 800], [0, 800]], rule="empty")]
    for _ in range(3):
        w.step()
    assert w.alert and w.alert["zone_problems"]                                 # objects inside an empty zone


def test_watch_false_alarm_closes_itself_and_thread_runs(tmp_path, monkeypatch):
    from test_pipeline import _client
    from trova import changes as chg_mod
    from trova.grok import register_mock
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    st.source.table.tamper("add")
    no = lambda p, i, ctx: {"changes": [{"change_id": ch.id, "is_real": False, "label": "glare", "risk": "low",  # noqa: E731
                                         "explanation": "a reflection"} for ch in ctx["changes"]],
                            "line_clear": True, "summary": "nothing changed"}
    register_mock("trova_verify", no)
    try:
        w = st.watch
        for _ in range(3):
            w.step()
        assert _wait(lambda: w.alert is None) and "false alarm" in w.status
    finally:
        register_mock("trova_verify", chg_mod._mock_verify)
    r = c.post("/api/watch", json={"on": True}).json()
    assert r["on"] and _wait(lambda: st.watch.checks >= 3)
    assert not c.post("/api/watch", json={"on": False}).json()["on"]
    n = st.watch.checks
    import time
    time.sleep(0.8)
    assert st.watch.checks == n                                                 # the thread really stopped


# ---------------------------------------------------------------- feature 11: labels and batch
def test_lot_and_expiry_parsing_and_batch_rules():
    import datetime as dt

    from trova.labels import check_batch, parse_date, parse_expiry_text, parse_lot
    assert parse_lot("Paracetamol LOT A123 EXP 2027-06") == "A123" and parse_lot("Lotto: b777") == "B777"
    assert parse_lot("Batch 9-X1") == "9-X1" and parse_lot("no lot here") == ""
    assert parse_expiry_text("LOT A1 EXP 2027-06") == "2027-06" and parse_expiry_text("Scad. 30/06/2027") == "30/06/2027"
    assert parse_date("2027-06") == dt.date(2027, 6, 30) and parse_date("06/27") == dt.date(2027, 6, 30)
    assert parse_date("30.06.2027") == dt.date(2027, 6, 30) and parse_date("13/2027") is None and parse_date("soon") is None
    today = dt.date(2026, 11, 4)

    def it(i, lot, exp="", code_lot="", print_lot=None):
        return {"box_id": i, "lot": lot, "expiry": exp, "code_lot": code_lot, "print_lot": lot if print_lot is None else print_lot}
    probs = check_batch([it(1, "A123", "2027-06"), it(2, "B777"), it(3, "A123", "2025-01"), it(4, "A1?3"),
                         it(5, "B777", code_lot="B777", print_lot="A123"), it(6, "")], "A-123", today)
    kinds = {(p["box_id"], p["kind"]) for p in probs}
    assert (2, "wrong_batch") in kinds and (3, "expired") in kinds and (4, "unreadable") in kinds
    assert (5, "code_mismatch") in kinds and (6, "no_lot") in kinds and not any(b == 1 for b, _ in kinds)


def test_read_labels_on_the_table_and_the_printable_sheet(tmp_path):
    import cv2

    from trova.grok import GrokClient
    from trova.labels import decode_codes, label_sheet, read_labels
    from trova.proposals import find_blobs
    from trova.synthetic import SyntheticTable
    t = SyntheticTable(seed=11, n_objects=5)
    t.add_labelled("LOT A123 EXP 2028-06")
    t.add_labelled("LOT B777 EXP 2027-03")
    f, e = t.render(), t.empty()
    r = read_labels(GrokClient(mock=True), f, find_blobs(f, e), lot="A123")
    assert sorted(i["lot"] for i in r.items) == ["A123", "B777"]
    assert [p["kind"] for p in r.problems] == ["wrong_batch"] and "B777" in r.problems[0]["text"]
    from trova.boxes import Box
    sheet = cv2.resize(cv2.imread(str(label_sheet(tmp_path, lot="A123"))), None, fx=0.5, fy=0.5)
    cells = [Box((70 + (i % 2) * 600), (80 + (i // 2) * 560), (70 + (i % 2) * 600) + 550, (80 + (i // 2) * 560) + 500, id=i + 1)
             for i in range(6)]
    texts = [c["text"] for c in decode_codes(sheet, cells)]
    assert len(texts) == 6 and sum("A123" in x for x in texts) == 4      # 3 good + 1 expired; the mismatch's code says B777


def test_batch_check_inside_line_clearance(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    assert "batch" not in c.post("/api/changes").json()["result"]           # no batch set: no label step
    assert c.post("/api/batch", data={"lot": "a123"}).json()["lot"] == "A123"
    assert c.post("/api/demo/tamper", data={"kind": "label"}).json()["label"] == "carton LOT B777"
    r = c.post("/api/changes").json()["result"]
    assert not r["line_clear"] and r["batch"]["problems"][0]["kind"] == "wrong_batch" and "B777" in r["summary"]
    lab = c.post("/api/labels").json()
    assert lab["items"][0]["lot"] == "B777" and not lab["ok"] and lab["image"]
    assert "Labels (current batch LOT A123)" in c.get("/api/report").text
    pdf = c.get("/api/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert c.get("/api/state").json()["batch"]["lot"] == "A123"
    assert c.post("/api/labels/sheet").json()["file"].endswith(".png")


# ---------------------------------------------------------------- feature 12: browser speech fallback
def test_browser_speech_commands():
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    from trova.voice import STOP_WORDS
    said = ["Stop!", "fermo", "stop watching", "find my watch", "watch the line", "sorveglia la linea", "check",
            "what changed?", "check the lot", "leggi le etichette", "what's on the table", "trova la penna rossa",
            "all the pens", "tutte le forbici", "where is the red mug", "something to write with", "save the reference",
            "look around for the red umbrella", "guarda in giro per la borsa nera",
            "fai una foto e dimmi quali oggetti vedi", "what do you see", "seleziona la seconda penna blu",
            "oggetto 3", "number 2", "Move backward by 3 millimeters",
            "Sposta il braccio a sinistra di 2 cm",
            "Grok, trova la penna blu e posizionati sopra"]
    js = ("const S = require(process.argv[1]); const [said, sw] = JSON.parse(process.argv[2]);"
          "console.log(JSON.stringify(said.map(t => S.parse(t, sw))));")
    out = subprocess.run([node, "-e", js, str(STATIC / "speech.js"), json.dumps([said, STOP_WORDS])],
                         capture_output=True, text=True, check=True).stdout
    got = dict(zip(said, json.loads(out)))
    assert got["Stop!"]["cmd"] == got["fermo"]["cmd"] == "stop"
    assert got["stop watching"] == {"cmd": "stop", "also": "watch_off"}           # stops the arm too
    assert got["find my watch"] == {"cmd": "find", "text": "my watch"}           # not watch mode
    assert got["watch the line"]["cmd"] == got["sorveglia la linea"]["cmd"] == "watch_on"
    assert got["check"]["cmd"] == got["what changed?"]["cmd"] == "check"
    assert got["check the lot"]["cmd"] == got["leggi le etichette"]["cmd"] == "labels"
    assert got["what's on the table"]["cmd"] == "inventory" and got["save the reference"]["cmd"] == "reference"
    assert got["trova la penna rossa"] == {"cmd": "find", "text": "la penna rossa"}
    assert got["all the pens"]["cmd"] == got["tutte le forbici"]["cmd"] == "find_all"
    assert got["where is the red mug"]["text"] == "the red mug"
    assert got["look around for the red umbrella"] == {"cmd": "room", "text": "the red umbrella"}
    assert got["guarda in giro per la borsa nera"] == {"cmd": "room", "text": "la borsa nera"}
    assert got["fai una foto e dimmi quali oggetti vedi"]["cmd"] == "inventory"
    assert got["what do you see"]["cmd"] == "inventory"
    assert got["seleziona la seconda penna blu"] == {"cmd": "select", "text": "la seconda penna blu"}
    assert got["oggetto 3"] == {"cmd": "select", "text": "oggetto 3"}
    assert got["number 2"] == {"cmd": "find", "text": "number 2"}   # a find clarification, not a new selection
    assert got["Move backward by 3 millimeters"] == {"cmd": "jog", "direction": "backward", "distance_mm": 3}
    assert got["Sposta il braccio a sinistra di 2 cm"] == {"cmd": "jog", "direction": "left", "distance_mm": 20}
    assert got["Grok, trova la penna blu e posizionati sopra"]["cmd"] == "agent"


# ---------------------------------------------------------------- feature 13: audit trail page, tamper evidence
def test_history_rows_csv_and_tamper_evidence(tmp_path, monkeypatch):
    import json

    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/find", data={"text": "the red one", "move": "false"})
    c.post("/api/reference", data={"describe": "false"})
    st.source.table.tamper("add")
    c.post("/api/changes")
    c.post("/api/signoff", data={"operator": "Ana Rossi", "decision": "reject"})
    c.post("/api/arm/stop")
    assert c.get("/history").status_code == 200
    d = c.get("/api/history").json()
    kinds = [r["kind"] for r in d["rows"]]
    assert kinds[:3] == ["arm_stop", "signoff", "change_check"] and "grok_call" not in kinds and d["grok_calls"] >= 1
    check = next(r for r in d["rows"] if r["kind"] == "change_check")
    assert check["result"].startswith("NOT CLEAR") and check["image"]
    only = c.get("/api/history", params={"groups": "clearance"}).json()["rows"]
    assert {r["group"] for r in only} == {"clearance"}
    assert [r["kind"] for r in c.get("/api/history", params={"q": "ana rossi"}).json()["rows"]] == ["signoff"]
    csv_text = c.get("/api/history.csv").text
    assert csv_text.startswith("time,event id,kind") and "Ana Rossi" in csv_text

    v = c.get("/api/history/verify").json()
    assert v["ok"] and v["hashed"] == v["events"] > 5
    log = tmp_path / "events.jsonl"
    lines = log.read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if '"signoff"' in line)
    e = json.loads(lines[i])
    e["decision"] = "approve"                                  # someone edits the record afterwards
    log.write_text("\n".join(lines[:i] + [json.dumps(e)] + lines[i + 1:]) + "\n")
    v = c.get("/api/history/verify").json()
    assert not v["ok"] and v["broken_at"] == i + 1
    log.write_text("\n".join(lines[:i] + lines[i + 1:]) + "\n")   # or deletes the line
    v = c.get("/api/history/verify").json()
    assert not v["ok"] and v["broken_at"] == i + 1


def test_approving_a_not_clear_line_needs_a_reason(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    app.state.trova.source.table.tamper("add")
    assert not c.post("/api/changes").json()["result"]["line_clear"]
    assert c.post("/api/signoff", data={"operator": "Ana", "decision": "approve"}).status_code == 400
    assert c.post("/api/signoff", data={"operator": "Ana", "decision": "maybe"}).status_code == 400
    assert c.post("/api/signoff", data={"operator": "Ana", "decision": "reject"}).status_code == 200
    so = c.post("/api/signoff", data={"operator": "Ana", "decision": "approve", "note": "spare pen, QA informed"}).json()
    assert so["note"] == "spare pen, QA informed" and "spare pen" in c.get("/api/report").text


# ---------------------------------------------------------------- xArm 5 bring-up (the arm the team got)
class SimXArm:
    """Enough of xarm.wrapper.XArmAPI to rehearse the bring-up: moves over (fake) time, stops on set_state(4),
    goes soft in mode 2 so a 'hand' can move it."""

    def __init__(self, clock, err=0, ignore_stop=False):
        self.clock, self.err, self.ignore_stop = clock, err, ignore_stop
        self.pos, self.target, self.speed, self.t = [200.0, 0.0, 420.0, 90.0, 0.0, 0.0], None, 0.0, 0.0
        self.mode, self.state, self.tcp_offset, self.moves = 0, 2, [0.0] * 6, []

    def _adv(self):
        now = self.clock[0]
        if self.target:
            d = [b - a for a, b in zip(self.pos[:3], self.target[:3])]
            dist = sum(v * v for v in d) ** 0.5
            step = self.speed * (now - self.t)
            if dist <= step:
                self.pos, self.target = list(self.target), None
            else:
                self.pos[:3] = [a + v / dist * step for a, v in zip(self.pos[:3], d)]
        self.t = now

    def get_position(self):
        self._adv()
        return 0, list(self.pos)

    def set_position(self, x, y, z, roll, pitch, yaw, speed, mvacc, wait, timeout):
        assert self.mode == 0 and not self.err, "moved while soft or in error"
        self._adv()
        self.moves.append((x, y, z, speed))
        self.target, self.speed = [x, y, z, roll, pitch, yaw], speed
        if wait:
            self.pos, self.target = list(self.target), None
        return 0

    def set_state(self, s):
        self._adv()
        if s == 4 and not self.ignore_stop:
            self.target = None
        self.state = s
        return 0

    def set_mode(self, m):
        self.mode = m
        return 0

    def get_err_warn_code(self):
        return 0, [self.err, 0]

    def clean_error(self):
        self.err = 0
        return 0

    def get_version(self):
        return 0, "v2.6.0"

    def get_state(self):
        return 0, self.state

    def get_servo_angle(self):
        return 0, [0.0] * 5

    def get_reduced_states(self):
        return 0, [0, [0] * 6, 0, 0, 0, 0]

    def __getattr__(self, name):            # clean_warn, motion_enable, disconnect, ...
        return lambda *a, **k: 0


def _bringup(sim, clock, answers_no=()):
    from trova.arm.bringup import Bringup
    hand = {"15 cm above": [300, 0, 160, 180, 0, 0], "PARK pose": [150, -250, 300, 180, 0, 0],
            "far left": [450, 200, 10, 180, 0, 0], "far right": [450, -200, 10, 180, 0, 0],
            "near right": [250, -200, 10, 180, 0, 0], "near left": [250, 200, 10, 180, 0, 0],
            "Lift the arm": [350, 0, 150, 180, 0, 0], "again": [350, 0, 150, 180, 0, 0]}
    said = []

    def ask(q):
        for key, p in hand.items():
            if key in q:
                assert sim.mode == 2, "asked to move by hand while the arm is stiff"
                sim.pos = [float(v) for v in p]
                return ""
        return "n" if any(k in q for k in answers_no) else "y"

    def sleep(s):
        clock[0] += s
    return Bringup(sim, ask=ask, say=said.append, sleep=sleep), said


def test_xarm_bringup_full_run(tmp_path):
    import json
    clock = [0.0]
    sim = SimXArm(clock, err=24)                       # an old error left on the controller
    b, said = _bringup(sim, clock)
    res = b.run("192.168.1.185", tmp_path / "xarm_bringup.json")
    env = res["settings"]["env"]
    assert env["TROVA_ARM"] == "xarm:192.168.1.185" and env["TROVA_ARM_Z_MM"] == "110"
    assert env["TROVA_ARM_HOME_MM"] == "150,-250,300"
    assert env["TROVA_ARM_LIMITS"] == "130,490,-270,240,25,380"          # widened to include the park pose
    assert res["stop_test"]["ok"] and 5 < res["stop_test"]["moved_mm"] < 60
    assert all(m[3] <= 30 for m in sim.moves)                             # never faster than 30 mm/s
    assert json.loads((tmp_path / "xarm_bringup.json").read_text())["settings"]["env"] == env
    assert any("TROVA_ARM_LIMITS=" in line for line in said)


def test_xarm_bringup_refuses_without_yes_and_flags_a_bad_stop():
    from trova.arm.bringup import Abort
    clock = [0.0]
    sim = SimXArm(clock)
    b, _ = _bringup(sim, clock, answers_no=("Staff at the e-stop",))
    with pytest.raises(Abort, match="skipped"):
        b.run("ip", None)
    assert sim.moves == []                                                 # "no" means no motion at all
    clock = [0.0]
    bad = SimXArm(clock, ignore_stop=True)
    b, _ = _bringup(bad, clock)
    with pytest.raises(Abort, match="did not stop"):
        b.run("ip", None)


def test_xarm_tcp_offset_from_env_and_clear_errors_endpoint(tmp_path, monkeypatch):
    import sys

    calls = []

    class FakeXArm:
        def __init__(self, ip, is_radian=False):
            pass

        def get_position(self):
            return 0, [300.0, 0.0, 200.0, 180.0, 0.0, 0.0]

        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k)) or 0
    mod = types.ModuleType("xarm.wrapper")
    mod.XArmAPI = FakeXArm
    monkeypatch.setitem(sys.modules, "xarm", types.ModuleType("xarm"))
    monkeypatch.setitem(sys.modules, "xarm.wrapper", mod)
    monkeypatch.setenv("TROVA_ARM_TCP_MM", "0,0,120")
    from trova.arm.xarm import XArmBackend
    XArmBackend("192.168.1.185")
    assert ("set_tcp_offset", ([0.0, 0.0, 120.0, 0, 0, 0],), {"wait": True}) in calls

    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="cartesian:fake")
    app.state.trova.arm.last_error = "C31 collision"
    r = c.post("/api/arm/clear_errors").json()
    assert r["error"] is None and r["state"] == "idle"
    app2, c2 = _client(tmp_path / "sim", monkeypatch, TROVA_ARM="sim")
    assert c2.post("/api/arm/clear_errors").status_code == 400            # the simulated arm has no controller


# ---------------------------------------------------------------- camera on the arm (the event's xArm 5 sphere camera)
def _arm_cam_client(tmp_path, monkeypatch, **extra):
    from test_pipeline import _client
    env = dict(TROVA_ARM="cartesian:fake", TROVA_CAMERA="armsim", TROVA_CAMERA_ON_ARM="1", TROVA_CAMERA_SETTLE_S="0",
               TROVA_ARM_LIMITS="-2000,2000,-2000,2000,-2000,2000", TROVA_ARM_SPEED_MM_S="20000",
               TROVA_ARM_JOINT_SPEED_DEG_S="20000", TROVA_ARM_HOME_MM="")
    env.update(extra)
    return _client(tmp_path, monkeypatch, **env)


def test_camera_on_arm_survey_pose_flow(tmp_path, monkeypatch):
    from trova.calibration import Calibration
    app, c = _arm_cam_client(tmp_path, monkeypatch)
    st = app.state.trova
    assert "teach one" in st.arm_blocked()                                    # no survey pose yet
    r = c.post("/api/arm/teach_pose", data={"name": "home_joints"}).json()  # the arm is where the camera sees the table
    assert r["poses"]["home_joints"] == [0.0] * 5 and st.at_survey()
    st.calib = Calibration.default(1280, 800, 600, 400)
    st.calib.source, st.calib.size = "aruco_robot", [1280, 800]             # as if calibrated from ArUco at this pose
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    unique = _unique_color(c)
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()
    assert r["result"]["status"] == "found" and r["moved"], r["arm_blocked"]
    assert _wait(lambda: st.arm.status()["state"] == "idle" and not st.at_survey())
    s = c.get("/api/state").json()
    assert not s["at_survey"] and "survey pose" in s["arm_blocked"]          # the live picture is not the calibrated one
    assert c.post("/api/arm/point", json={"u": 100, "v": 100}).status_code == 409
    assert c.get("/api/frame.jpg").status_code == 200
    st.watch.step()
    assert "away from the survey pose" in st.watch.status                    # never compares a different view
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()      # goes back to the survey pose first
    assert r["result"]["status"] == "found" and r["moved"]
    c.post("/api/arm/home")
    assert _wait(lambda: st.at_survey())
    r = c.post("/api/arm/teach_pose", data={"name": "home_joints"}).json()   # a new survey pose invalidates the calibration
    assert "calibrate again" in r["note"] and "Calibrate" in st.arm_blocked()


def test_room_scan_finds_and_faces_the_object(tmp_path, monkeypatch):
    import numpy as np

    from trova.synthetic import COLORS, ObjSpec
    app, c = _arm_cam_client(tmp_path, monkeypatch)
    st = app.state.trova
    c.post("/api/arm/teach_pose", data={"name": "home_joints"})
    assert c.post("/api/room/find", data={"text": "the lamp"}).status_code == 400      # no look pose yet
    assert st.arm.go_joints([10.0, 50.0, 30.0, 0.0, 0.0])
    c.post("/api/arm/teach_pose", data={"name": "look_joints"})
    cam = st.source
    color = "purple"
    cam.room.objects = [o for o in cam.room.objects if o.color != color]     # exactly one purple thing in the room
    x = cam.room_img.shape[1] / 2 - 40 * cam.px_per_deg                     # the object sits at pan +40 degrees
    cam.room.objects.append(ObjSpec("lamp", color, "square", x, 420, 140))
    cam.room_img = cam.room.render(noise=0)
    assert c.post("/api/room/find", data={"text": f"the {color} lamp"}).json()["started"]
    assert _wait(lambda: st.room and st.room["status"] != "scanning", timeout=20)
    room = c.get("/api/state").json()["room"]
    assert room["status"] == "found" and room["facing"], room
    assert abs(room["angle"] - 40) < 4 and room["n_views"] == 7 and len(room["views"]) == 7
    facing = cam.read()                                                       # what the camera sees now
    m = (np.abs(facing.astype(int) - np.array(COLORS[color])).sum(axis=2) < 60)
    assert m.sum() > 500 and abs(np.nonzero(m)[1].mean() - 640) < 90           # the object is in the middle
    assert room["sheet"] and room["marked"]


def test_room_scan_stops_and_joint_poses_obey_the_safety_box(tmp_path, monkeypatch):
    app, c = _arm_cam_client(tmp_path, monkeypatch, TROVA_ARM_JOINT_SPEED_DEG_S="60")
    st = app.state.trova
    c.post("/api/arm/teach_pose", data={"name": "home_joints"})
    st.arm.joint_speed = 20000
    assert st.arm.go_joints([10.0, 50.0, 30.0, 0.0, 0.0])
    c.post("/api/arm/teach_pose", data={"name": "look_joints"})
    st.arm.joint_speed = 60
    c.post("/api/room/find", data={"text": "the red lamp"})
    assert _wait(lambda: st.arm.status()["state"] == "moving")
    c.post("/api/arm/stop")
    st.arm_thread.join(timeout=5)
    assert st.room["status"] == "stopped"
    from trova.arm import CartesianArm, FakeCartesian, Limits
    arm = CartesianArm(FakeCartesian(), 150, None, Limits.parse("100,600,-300,300,0,400"), speed_mm_s=5000)
    arm.joint_speed = 5000
    assert not arm.go_joints([0.0, 400.0, 0.0, 0.0, 0.0]) and "joint pose" in arm.last_error   # fk x = 700: outside
    assert arm.backend.history == []


def test_camera_list_and_bringup_for_the_arm_camera(tmp_path, monkeypatch):
    import numpy as np

    from trova import camera as cam_mod

    class FakeCap:
        def __init__(self, i):
            self.i = i

        def isOpened(self):
            return self.i in (0, 2)

        def set(self, *a):
            return True

        def read(self):
            return True, np.full((720, 1280, 3), 40 * (self.i + 1), np.uint8)

        def release(self):
            pass
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(cam_mod.time, "sleep", lambda s: None)
    found = cam_mod.list_cameras(tmp_path, max_index=4)
    assert [c["index"] for c in found] == [0, 2] and found[1]["size"] == [1280, 720]
    assert (tmp_path / "camera_2.jpg").exists()

    clock = [0.0]
    sim = SimXArm(clock)
    b, said = _bringup(sim, clock)
    b.camera_on_arm = True
    env = b.run("192.168.1.185", None)["settings"]["env"]
    assert env["TROVA_CAMERA_ON_ARM"] == "1" and env["TROVA_ARM_HOME_JOINTS"] == "0,0,0,0,0"
    assert any("ArUco" in line for line in said)


def test_led_ring_commands_and_never_breaking_trova(tmp_path, monkeypatch):
    from trova.light import NoLight, SerialRing, make_light
    sent = []

    class FakeSerialMod:
        class Serial:
            def __init__(self, port, baud, timeout, write_timeout):
                self.port = port

            def write(self, data):
                sent.append(data.decode().strip())
    ring = SerialRing("/dev/tty.fake", serial_mod=FakeSerialMod)
    assert sent == ["C 90 90 90"]                                        # white light for the camera at start
    ring.show("alarm")
    assert sent[-2:] == ["F 220 0 0 6", "C 220 0 0"]
    ring.show("found")
    assert sent[-2:] == ["F 0 180 0 2", "C 90 90 90"]
    ring.ser.write = lambda data: (_ for _ in ()).throw(OSError("unplugged"))
    ring.show("stop")                                                    # an unplugged ring never raises
    assert "unplugged" in ring.status()["error"]
    assert isinstance(make_light(""), NoLight)
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, TROVA_LED="serial:/dev/does-not-exist")
    assert c.get("/api/state").json()["light"]["name"] == "off"           # a bad port: Trova runs without the light


def test_state_stays_json_after_a_room_scan(tmp_path, monkeypatch):
    app, c = _arm_cam_client(tmp_path, monkeypatch)
    st = app.state.trova
    c.post("/api/arm/teach_pose", data={"name": "home_joints"})
    c.post("/api/arm/teach_pose", data={"name": "look_joints"})
    c.post("/api/room/find", data={"text": "the purple thing"})
    assert _wait(lambda: st.room and st.room["status"] != "scanning", timeout=20)
    s = c.get("/api/state")                                   # numpy numbers from the scan once broke this (500)
    assert s.status_code == 200 and s.json()["at_survey"] is False
    pre = {i["name"]: i for i in c.get("/api/preflight").json()["items"]}
    assert pre["Home pose"]["status"] == "ok" and "joints" in pre["Home pose"]["detail"]
