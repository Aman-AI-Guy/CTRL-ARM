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
    for page in ("index.html", "jury.html"):
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
