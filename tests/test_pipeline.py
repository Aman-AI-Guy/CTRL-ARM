"""Offline tests: no API key, no hardware. Run: .venv/bin/pytest -q"""

import json
import types

import numpy as np
import pytest

from trova import changes as chg
from trova.arm import FakeJointBackend, SimArm, TaughtGridArm
from trova.boxes import Box
from trova.calibration import Calibration, calibrate_aruco, make_marker_sheet
from trova.finder import find
from trova.grok import GrokClient
from trova.proposals import find_blobs, is_occluded
from trova.synthetic import SyntheticTable


def matched(found, truth, thr=0.5):
    return sum(any(b.iou(g) > thr for b in found) for g in truth)


@pytest.mark.parametrize("seed", range(4))
def test_blobs_find_every_object(seed):
    t = SyntheticTable(seed=seed)
    f = t.render()
    assert matched(find_blobs(f, t.empty()), t.truth()) == len(t.objects)
    assert matched(find_blobs(f, None), t.truth()) >= len(t.objects) - 1   # estimated background


def test_shadows_are_not_objects():
    t = SyntheticTable(seed=3)
    assert len(find_blobs(t.render(shadows=True), t.empty())) == len(t.objects)


def test_hand_is_occlusion():
    t = SyntheticTable(seed=1)
    f = t.render(hand=True)
    assert is_occluded(find_blobs(f, t.empty()), f.shape)


@pytest.mark.parametrize("kind", ["add", "remove", "move"])
def test_change_detection(kind):
    t = SyntheticTable(seed=5)
    empty = t.empty()
    ref = t.render()
    ref_boxes = find_blobs(ref, empty)
    did = t.tamper(kind)
    frames = [t.render() for _ in range(3)]
    res = chg.check(None, ref, ref_boxes, frames, [find_blobs(f, empty) for f in frames], verify=False)
    kinds = [c.kind for c in res.changes]
    assert kinds == [did["kind"]], (kinds, did)


def test_no_change_no_alarm_and_baseline_is_worse():
    t = SyntheticTable(seed=2)
    empty = t.empty()
    ref = t.render()
    frames = [t.render(noise=4) for _ in range(3)]
    res = chg.check(None, ref, find_blobs(ref, empty), frames, [find_blobs(f, empty) for f in frames], verify=False)
    assert res.changes == [] and res.line_clear
    # the room gets 20% brighter and warmer (exposure / white-balance drift):
    # the naive pixel diff fires, ours must report nothing
    drift = np.array([1.12, 1.2, 1.25])
    brighter = [np.clip(f.astype(np.float32) * drift, 0, 255).astype(np.uint8) for f in frames]
    ours = chg.check(None, ref, find_blobs(ref, empty), brighter,
                     [find_blobs(f, empty) for f in brighter], verify=False)
    assert ours.changes == []
    assert len(chg.naive_pixel_diff(ref, brighter[-1])) >= 1


def test_mock_verify_labels_changes():
    t = SyntheticTable(seed=4)
    empty = t.empty()
    ref = t.render()
    rb = find_blobs(ref, empty)
    t.tamper("add")
    cur = t.render()
    g = GrokClient(mock=True)
    res = chg.check(g, ref, rb, [cur, cur], [find_blobs(cur, empty)] * 2)
    assert len(res.changes) == 1 and res.changes[0].verified and not res.line_clear


def test_calibration_roundtrip_and_aruco(tmp_path):
    c = Calibration.default(1280, 800, 600, 400)
    x, y = c.pixel_to_table(640, 400)
    assert abs(x - 300) < 1e-6 and abs(y - 200) < 1e-6
    u, v = c.table_to_pixel(150, 100)
    assert abs(u - 320) < 1e-6 and abs(v - 200) < 1e-6
    import cv2
    paths = make_marker_sheet(tmp_path)
    img = np.full((900, 1400, 3), 255, np.uint8)
    spots = {0: (100, 100), 1: (1100, 100), 2: (1100, 600), 3: (100, 600)}
    for i, (px, py) in spots.items():
        m = cv2.imread(str(paths[i]))
        m = cv2.resize(m, (200, 220))
        img[py:py + 220, px:px + 200] = m
    cal = calibrate_aruco(img, {0: (0, 0), 1: (500, 0), 2: (500, 250), 3: (0, 250)})
    assert cal.error_mm() < 1.0


def test_sim_arm_moves_and_stops():
    import threading
    import time
    a = SimArm(speed_mm_s=2000)
    assert a.point_at(100, 100)
    assert a.status()["tip_mm"] == [100.0, 100.0]
    slow = SimArm(speed_mm_s=50)
    th = threading.Thread(target=slow.point_at, args=(0, 0))
    th.start()
    time.sleep(0.1)
    slow.stop()
    th.join(timeout=1)
    assert slow.status()["state"] == "stopped"


def test_taught_grid_interpolates(tmp_path):
    be = FakeJointBackend()
    arm = TaughtGridArm(be, tmp_path / "grid.json")
    for x, y in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        be.q = [x * 2.0, y * 3.0]   # pretend joints are linear in table position
        arm.record(x, y)
    q = arm.joints_for(50, 50)
    assert abs(q[0] - 100) < 1e-6 and abs(q[1] - 150) < 1e-6
    assert arm.point_at(100, 0) and be.q == [200.0, 0.0]


def test_mock_find_and_clarify():
    t = SyntheticTable(seed=0, n_objects=0)
    t.add_random(color="red", kind=("mug", "circle"))
    t.add_random(color="blue", kind=("pen", "bar"))
    t.add_random(color="blue", kind=("box", "rect"))
    f = t.render()
    boxes = find_blobs(f, t.empty())
    g = GrokClient(mock=True)
    r = find(g, f, boxes, "la tazza rossa")
    assert r.status == "found" and r.box.iou(t.truth()[0]) > 0.5
    r = find(g, f, boxes, "the blue one")
    assert r.status == "clarify" and len(r.candidate_ids) == 2
    r2 = find(g, f, boxes, "the blue one", history=[(r.clarifying_question, f"numero {r.candidate_ids[1]}")])
    assert r2.status == "found" and r2.box.id == r.candidate_ids[1]
    assert find(g, f, boxes, "una banana").status == "not_found"


def test_real_api_request_shape(monkeypatch):
    """The live path: request body format, retry without a rejected field, JSON parsing."""
    sent = []

    def fake_post(url, json=None, timeout=None, headers=None):
        sent.append(json)
        r = types.SimpleNamespace()
        if len(sent) == 1:
            r.status_code, r.text = 400, '{"error":"Argument not supported: reasoning"}'
            return r
        out = {"box_id": 2, "visible_but_unboxed": False, "candidate_ids": [2], "clarifying_question": None,
               "label": "penna", "reason": "È una penna.", "confidence": 0.9}
        r.status_code = 200
        r.json = lambda: {"output": [{"type": "message", "content": [{"type": "output_text",
                                                                      "text": __import__("json").dumps(out)}]}]}
        return r

    g = GrokClient(api_key="k", mock=False)
    monkeypatch.setattr(g.session, "post", fake_post)
    t = SyntheticTable(seed=1)
    f = t.render()
    boxes = find_blobs(f, t.empty())
    res = find(g, f, boxes, "qualcosa per scrivere")
    assert res.status == "found" and res.box.id == 2
    body = sent[1]
    assert "reasoning" not in body and body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    parts = body["input"][0]["content"]
    assert [p["type"] for p in parts] == ["input_image", "input_image", "input_text"]
    assert parts[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert g.stats()["dropped_fields"] == ["reasoning"]


def test_server_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("TROVA_DATA", str(tmp_path))
    monkeypatch.setenv("TROVA_MOCK", "1")
    monkeypatch.setenv("TROVA_CAMERA", "demo")
    from fastapi.testclient import TestClient

    from trova.config import load_settings
    from trova.server import create_app
    c = TestClient(create_app(load_settings()))
    assert c.get("/").status_code == 200
    assert c.get("/api/frame.jpg").headers["content-type"] == "image/jpeg"
    assert c.post("/api/background", data={"mode": "capture"}).json()["background"]
    st = c.get("/api/state").json()
    assert st["mock"] and st["source"]["kind"] == "demo"
    colors = [o.split()[0] for o in st["source"]["objects"]]
    unique = next(col for col in colors if colors.count(col) == 1)
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()
    assert r["result"]["status"] == "found" and r["table_mm"] is not None
    assert c.post("/api/judge", json={"kind": "find", "correct": True}).json()["find_ok"] == 1
    assert (tmp_path / "cases" / "finds.jsonl").exists()
    ref = c.post("/api/reference").json()
    assert ref["n_objects"] == len(colors)
    c.post("/api/demo/tamper", data={"kind": "add"})
    chk = c.post("/api/changes").json()
    assert not chk["result"]["line_clear"] and chk["result"]["changes"][0]["kind"] == "added"
    assert c.post("/api/signoff", data={"operator": "Test", "decision": "approve"}).status_code == 200
    rep = c.get("/api/report").text
    assert "Line clearance record" in rep and "Signed off by Test" in rep
    assert c.post("/api/arm/stop").status_code == 200
    events = [json.loads(line)["kind"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    for k in ("find", "grok_call", "reference", "change_check", "signoff"):
        assert k in events


def test_black_objects_survive_warm_light():
    """Regression: hue of near-black pixels flips under a light change; must not become removed+added."""
    t = SyntheticTable(seed=0, n_objects=0)
    for col, kind in [("black", ("mug", "circle")), ("black", ("box", "rect")), ("red", ("pen", "bar"))]:
        t.add_random(color=col, kind=kind)
    e = t.empty()
    ref = t.render()
    frames = [np.clip(t.render().astype(np.float32) * np.array([0.95, 1.0, 1.08]), 0, 255).astype(np.uint8)
              for _ in range(3)]
    res = chg.check(None, ref, find_blobs(ref, e), frames, [find_blobs(f, e) for f in frames], verify=False)
    assert res.changes == []


def test_cartesian_arm_limits_stop_and_guarded_server(tmp_path, monkeypatch):
    import threading
    import time
    from trova.arm import CartesianArm, FakeCartesian, Limits
    arm = CartesianArm(FakeCartesian(), z_hover_mm=120, home_mm=(300, 0, 200),
                       limits=Limits.parse("100,600,-300,300,50,400"), speed_mm_s=2000)
    assert arm.point_at(400, 100) and [round(v) for v in arm.tcp_mm()] == [400, 100, 120]
    assert not arm.point_at(900, 0) and arm.state == "refused" and "outside" in arm.last_error
    slow = CartesianArm(FakeCartesian(), 120, None, None, speed_mm_s=20)
    th = threading.Thread(target=slow.point_at, args=(500, 200))
    th.start()
    time.sleep(0.2)
    slow.stop()
    th.join(timeout=2)
    assert slow.state == "stopped" and slow.tcp_mm()[0] < 450

    monkeypatch.setenv("TROVA_DATA", str(tmp_path))
    monkeypatch.setenv("TROVA_MOCK", "1")
    monkeypatch.setenv("TROVA_CAMERA", "demo")
    monkeypatch.setenv("TROVA_ARM", "cartesian:fake")
    monkeypatch.setenv("TROVA_ARM_SPEED_MM_S", "5000")
    from fastapi.testclient import TestClient

    from trova.config import load_settings
    from trova.server import create_app
    bare = create_app(load_settings())                     # no safety box, no home pose: never moves
    assert "TROVA_ARM_LIMITS" in bare.state.trova.arm_blocked()
    assert TestClient(bare).post("/api/arm/point", json={"x_mm": 5000, "y_mm": -4000}).status_code == 409
    monkeypatch.setenv("TROVA_ARM_LIMITS", "100,700,-300,300,0,400")
    monkeypatch.setenv("TROVA_ARM_HOME_MM", "300,250,250")
    app = create_app(load_settings())
    c = TestClient(app)
    c.post("/api/background", data={"mode": "capture"})
    colors = [o.split()[0] for o in c.get("/api/state").json()["source"]["objects"]]
    unique = next(col for col in colors if colors.count(col) == 1)
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()
    assert r["result"]["status"] == "found" and r["arm_blocked"]          # not calibrated in robot frame yet
    assert c.post("/api/arm/point", json={"u": 100, "v": 100}).status_code == 409
    fake = app.state.trova.arm.backend
    for (u, v), (x, y) in zip([(0, 0), (1280, 0), (1280, 800), (0, 800)],
                              [(200, -200), (520, -200), (520, 0), (200, 0)]):
        fake.pos = [x, y, 0.0]
        fake.target = list(fake.pos)
        out = c.post("/api/calibration/arm_click", json={"u": u, "v": v}).json()
    assert out["points"] == 4 and out["error_mm"] < 0.5
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()
    assert not r["arm_blocked"]
    time.sleep(0.3)
    tip = app.state.trova.arm.tcp_mm()
    assert abs(tip[0] - r["table_mm"][0]) < 1 and abs(tip[1] - r["table_mm"][1]) < 1 and abs(tip[2] - 150) < 1
    assert c.post("/api/calibration/marker_from_arm", data={"marker_id": 0}).json()["markers"]["0"]


def _client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("TROVA_DATA", str(tmp_path))
    monkeypatch.setenv("TROVA_CAMERA", "demo")
    monkeypatch.setenv("TROVA_MOCK", env.pop("mock", "1"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from fastapi.testclient import TestClient

    from trova.config import load_settings
    from trova.server import create_app
    app = create_app(load_settings())
    return app, TestClient(app)


def test_voice_config_confirmed_move_and_token(tmp_path, monkeypatch):
    import time
    app, c = _client(tmp_path, monkeypatch, XAI_API_KEY="k", mock="1")
    cfg = c.get("/api/voice/config").json()
    names = [t["name"] for t in cfg["session"]["tools"]]
    assert {"find_object", "point_at_found", "check_changes", "stop_arm"} <= set(names)
    assert "search_other_rooms" not in names                     # no dog configured
    assert cfg["session"]["reasoning"] == {"effort": "none"} and "fermo" in cfg["stop_words"]
    assert "English or Italian" in cfg["session"]["instructions"]
    assert c.post("/api/voice/session").status_code == 400       # mock mode: no voice
    c.post("/api/background", data={"mode": "capture"})
    colors = [o.split()[0] for o in c.get("/api/state").json()["source"]["objects"]]
    unique = next(col for col in colors if colors.count(col) == 1)
    tip0 = app.state.trova.arm.status()["tip_mm"]
    r = c.post("/api/find", data={"text": f"the {unique} one", "move": "false"}).json()
    assert r["result"]["status"] == "found" and not r["moved"]
    time.sleep(0.1)
    assert app.state.trova.arm.status()["tip_mm"] == tip0       # waited for the yes
    assert c.post("/api/arm/point_last").json()["ok"]
    time.sleep(2.5)
    assert app.state.trova.arm.status()["tip_mm"] == r["table_mm"]

    import trova.voice as v
    monkeypatch.setattr(v.requests, "post", lambda *a, **k: types.SimpleNamespace(
        status_code=200, json=lambda: {"value": "eph-123", "expires_at": 99}))
    app2, c2 = _client(tmp_path / "live", monkeypatch, XAI_API_KEY="k", mock="0")
    assert c2.post("/api/voice/session").json()["token"] == "eph-123"


def test_english_and_italian_mock_answers():
    t = SyntheticTable(seed=0, n_objects=0)
    t.add_random(color="blue", kind=("pen", "bar"))
    t.add_random(color="blue", kind=("box", "rect"))
    f = t.render()
    boxes = find_blobs(f, t.empty())
    g = GrokClient(mock=True)
    assert find(g, f, boxes, "the blue one").clarifying_question.startswith("I see 2")
    assert find(g, f, boxes, "quello blu").clarifying_question.startswith("Ne vedo 2")


def test_optional_dog_search_sim(tmp_path, monkeypatch):
    from trova.dog import load_route
    route = load_route(tmp_path)
    room1 = SyntheticTable(seed=route[0].seed, n_objects=6)
    colours = [o.color for o in room1.objects]
    app, c = _client(tmp_path, monkeypatch, TROVA_DOG="sim")
    app.state.trova.dog.t = 0.01
    st = c.get("/api/state").json()
    assert st["dog"]["route"][0] == "Room 2, desk"
    assert "search_other_rooms" in [t["name"] for t in c.get("/api/voice/config").json()["session"]["tools"]]
    d = c.post("/api/dog/search", data={"description": f"the {colours[0]} thing"}).json()
    assert d["status"] == "found" and d["spot"] == "Room 2, desk" and d["photo"]
    d = c.post("/api/dog/search", data={"description": "a white unicorn"}).json()
    assert d["status"] == "not_found" and len(d["visited"]) == 3


@pytest.mark.parametrize("kind", ["ur", "elite"])
def test_socket_cobot_drivers_against_fake_controller(kind):
    import threading
    import time
    from fake_robot import FakeController
    from trova.arm import CartesianArm
    from trova.arm.elite import EliteBackend
    from trova.arm.ur import URBackend
    fc = FakeController()
    Backend = URBackend if kind == "ur" else EliteBackend
    be = Backend("127.0.0.1", script_port=fc.script, **({"rtde_port": fc.rtde} if kind == "ur" else {"rtsi_port": fc.rtde}))
    t0 = time.time()
    while time.time() - t0 < 3:
        try:
            be.tcp_mm()
            break
        except RuntimeError:
            time.sleep(0.05)
    assert [round(v) for v in be.tcp_mm()] == [300, 0, 200]
    assert fc.protocols[0] == (2 if kind == "ur" else 1)
    arm = CartesianArm(be, z_hover_mm=150, home_mm=None, limits=None, speed_mm_s=400)
    assert arm.point_at(400, 100), arm.last_error
    assert [round(v) for v in arm.tcp_mm()] == [400, 100, 150]
    assert fc.wait_for("movel")
    s = fc.scripts[-1]
    if kind == "ur":
        assert "movel(p[0.40000,0.10000,0.15000,3.14000," in s and s.startswith("def trova_move():\n  ")
    else:
        assert "movel([0.40000,0.10000,0.15000,3.14000," in s and "\tmovel" in s and "p[" not in s
    slow = CartesianArm(be, 150, None, None, speed_mm_s=30)
    th = threading.Thread(target=slow.point_at, args=(400, 300))
    th.start()
    time.sleep(1.0)
    slow.stop()
    th.join(timeout=3)
    assert slow.state == "stopped" and fc.wait_for("stopl(2.0)" if kind == "ur" else "stopj(2.0)")
    time.sleep(0.3)                    # let the stop program land and the pose stream catch up
    y_stop = arm.tcp_mm()[1]
    time.sleep(0.4)
    assert abs(arm.tcp_mm()[1] - y_stop) < 1 and y_stop < 290          # it holds where it stopped
    be.freedrive(True)
    assert fc.wait_for("freedrive_mode()") and "sync()" in fc.scripts[-1]
    be.close()
    fc.close()


def test_xarm_driver_against_sdk_stub(monkeypatch):
    import sys
    import types as t
    calls = []

    class FakeXArm:
        error_code = 0

        def __init__(self, ip, is_radian=False):
            self.pos, self.moving = [300.0, 0.0, 200.0, 180.0, 0.0, 15.0], False
            calls.append(("init", ip, is_radian))

        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k)) or 0

        def get_position(self):
            return 0, list(self.pos)

        def get_is_moving(self):
            return self.moving

        def set_position(self, **k):
            calls.append(("set_position", k))
            self.pos[:3] = [k["x"], k["y"], k["z"]]
            return 0

    mod = t.ModuleType("xarm.wrapper")
    mod.XArmAPI = FakeXArm
    monkeypatch.setitem(sys.modules, "xarm", t.ModuleType("xarm"))
    monkeypatch.setitem(sys.modules, "xarm.wrapper", mod)
    from trova.arm import CartesianArm
    from trova.arm.xarm import XArmBackend
    be = XArmBackend("192.168.1.185")
    arm = CartesianArm(be, z_hover_mm=150, home_mm=None, limits=None)
    assert arm.point_at(350, -40)
    sp = next(c for c in calls if c[0] == "set_position")[1]
    assert (sp["roll"], sp["pitch"], sp["yaw"], sp["wait"]) == (180, 0, 15.0, False) and sp["z"] == 150
    arm.stop()
    names = [c[0] for c in calls]
    i = len(names) - 1 - names[::-1].index("set_state")
    assert names[i - 2:i + 1] == ["set_state", "motion_enable", "set_state"]   # stop then re-enable


# ---------------------------------------------------------------- feature 1: inventory map
def test_inventory_endpoint_and_names_flow_into_changes(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch)
    c.post("/api/background", data={"mode": "capture"})
    n = len(c.get("/api/state").json()["source"]["objects"])
    inv = c.post("/api/inventory").json()
    assert len(inv["objects"]) == n and inv["summary"] and (tmp_path / inv["map"]).exists()
    ref = c.post("/api/reference").json()
    assert len(ref["names"]) == n                              # reference objects are named
    assert "describe_table" in [t["name"] for t in c.get("/api/voice/config").json()["session"]["tools"]]


def test_reference_names_reach_the_verify_prompt():
    from trova.grok import GrokResult
    prompts = []

    class Capture:
        mock = True

        def json(self, *, name, prompt, schema, images=(), mock_context=None, **k):
            prompts.append(prompt)
            return GrokResult(data={"changes": [], "line_clear": False, "summary": "x"}, latency_s=0, model="m", mock=True)

    t = SyntheticTable(seed=6)
    e = t.empty()
    ref = t.render()
    rb = find_blobs(ref, e)
    t.tamper("remove")
    cur = t.render()
    chg.check(Capture(), ref, rb, [cur, cur], [find_blobs(cur, e)] * 2, ref_names={b.id: f"thing {b.id}" for b in rb})
    assert "in the reference this was: thing" in prompts[0]


# ---------------------------------------------------------------- feature 2: zones
def test_zone_rules_unit():
    from trova.zones import Zone, evaluate
    b1, b2 = Box(10, 10, 30, 30, id=1), Box(200, 200, 220, 220, id=2)
    zone_empty = Zone("line area", [[0, 0], [100, 0], [100, 100], [0, 100]], "empty")
    zone_bin = Zone("reject bin", [[150, 150], [300, 150], [300, 300], [150, 300]], "allowed", ["blister pack"])
    r = evaluate([zone_empty, zone_bin], [b1, b2], {1: "blue pen", 2: "blister pack"})
    assert not r[0]["ok"] and "blue pen must not be in line area" in r[0]["problems"][0]
    assert r[1]["ok"]
    r = evaluate([zone_bin], [b2], {2: "usb key"})
    assert not r[0]["ok"]


def test_zones_endpoints_make_line_not_clear(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch)
    c.post("/api/background", data={"mode": "capture"})
    for u, v in [(0, 0), (1280, 0), (1280, 800), (0, 800)]:                 # whole table must be empty
        c.post("/api/zones/point", json={"u": u, "v": v})
    z = c.post("/api/zones/finish", data={"name": "line area", "rule": "empty"}).json()
    assert z["zones"][0]["name"] == "line area" and (tmp_path / "zones.json").exists()
    assert c.get("/api/frame.jpg").status_code == 200                     # overlay draws
    c.post("/api/reference")
    chk = c.post("/api/changes").json()["result"]
    assert chk["changes"] == [] and not chk["line_clear"]                  # nothing changed, but the zone is not empty
    assert chk["zones"][0]["problems"] and "Zones:" in chk["summary"]
    assert "Zones" in c.get("/api/report").text
    assert c.delete("/api/zones/line area").json()["zones"] == []


# ---------------------------------------------------------------- feature 3: PDF record
def test_pdf_clearance_record(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch)
    assert c.get("/api/report.pdf").status_code == 400                       # no check yet
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference")
    c.post("/api/demo/tamper", data={"kind": "add"})
    chk = c.post("/api/changes").json()
    c.post("/api/signoff", data={"operator": "M. Rossi", "decision": "reject"})
    r = c.get("/api/report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF" and len(r.content) > 5000                # images embedded
    assert (tmp_path / "records" / f"clearance_{chk['event_id']}.pdf").exists()


# ---------------------------------------------------------------- feature 4: jury scoreboard
def test_jury_scoreboard_and_reset(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch)
    assert "scoreboard" in c.get("/jury").text.lower()
    c.post("/api/background", data={"mode": "capture"})
    colors = [o.split()[0] for o in c.get("/api/state").json()["source"]["objects"]]
    unique = next(col for col in colors if colors.count(col) == 1)
    c.post("/api/find", data={"text": f"the {unique} one"})
    c.post("/api/judge", json={"kind": "find", "correct": True})
    m = c.get("/api/state").json()["metrics"]
    assert m["find_ok"] == 1 and m["median_find_s"] is not None
    c.post("/api/metrics/reset")
    m = c.get("/api/state").json()["metrics"]
    assert m["find_ok"] == 0 and m["median_find_s"] is None


# ---------------------------------------------------------------- feature 5: failure log
def test_failure_causes_and_summary(tmp_path, monkeypatch):
    import subprocess
    import sys
    app, c = _client(tmp_path, monkeypatch)
    c.post("/api/background", data={"mode": "capture"})
    st = c.get("/api/state").json()
    colors = [o.split()[0] for o in st["source"]["objects"]]
    unique = next(col for col in colors if colors.count(col) == 1)
    # 1) Grok picks a box, the judge points at a different boxed object -> wrong_pick
    r = c.post("/api/find", data={"text": f"the {unique} one"}).json()
    other = next(b for b in r["boxes"] if b["box"] != r["result"]["box"]["box"])
    c.post("/api/judge", json={"kind": "find", "correct": False})
    out = c.post("/api/label", json={"u": other["center"][0], "v": other["center"][1]}).json()
    assert out["cause"] == "wrong_pick" and out["failure"]["owner"].startswith("P3")
    # 2) the judge points at empty table -> the object had no box -> missed_box (P2)
    c.post("/api/find", data={"text": f"the {unique} one"})
    c.post("/api/judge", json={"kind": "find", "correct": False})
    assert c.post("/api/label", json={"u": 2, "v": 2}).json()["cause"] == "missed_box"
    # 3) found something, but the object is not on the table -> picked_absent_object
    c.post("/api/find", data={"text": f"the {unique} one"})
    c.post("/api/judge", json={"kind": "find", "correct": False})
    assert c.post("/api/label", json={"absent": True}).json()["cause"] == "picked_absent_object"
    # 4) right pick, arm off target -> arm_missed (P1), no label needed
    c.post("/api/find", data={"text": f"the {unique} one"})
    c.post("/api/judge", json={"kind": "find", "correct": False, "cause": "arm_missed"})
    sm = c.get("/api/failures").json()
    assert sm["total"] == 4 and {x["cause"] for x in sm["by_cause"]} >= {"wrong_pick", "missed_box",
                                                                          "picked_absent_object", "arm_missed"}
    env = {**__import__("os").environ, "TROVA_DATA": str(tmp_path), "TROVA_MOCK": "1", "TROVA_EVAL_RESULTS": str(tmp_path / "results")}
    rep = subprocess.run([sys.executable, "eval/run_eval.py", "failures"], capture_output=True, text=True, env=env)
    assert "wrong pick" in rep.stdout and "P1 arm" in rep.stdout


# ---------------------------------------------------------------- feature 6: point at all
def test_find_all_and_tour(tmp_path, monkeypatch):
    import time
    app, c = _client(tmp_path, monkeypatch)
    t = SyntheticTable(seed=0, n_objects=0)
    for kind in [("pen", "bar"), ("box", "rect"), ("cube", "square")]:
        t.add_random(color="blue", kind=kind)
    t.add_random(color="red", kind=("mug", "circle"))
    app.state.trova.source.table = t
    app.state.trova.arm.speed = 5000
    c.post("/api/background", data={"mode": "capture"})
    r = c.post("/api/find_all", data={"text": "all the blue things"}).json()
    assert r["result"]["count"] == 3 and len(r["targets"]) == 3 and r["moved"]
    t0 = time.time()
    while (app.state.trova.tour or {}).get("status") != "done" and time.time() - t0 < 10:
        time.sleep(0.1)
    assert app.state.trova.tour["status"] == "done"
    assert app.state.trova.arm.status()["tip_mm"] == r["targets"][-1]
    r = c.post("/api/find_all", data={"text": "all the blue things", "move": "false"}).json()
    assert not r["moved"] and c.post("/api/arm/point_last").json()["targets"] == r["targets"]
    assert "find_all" in [x["name"] for x in c.get("/api/voice/config").json()["session"]["tools"]]


def test_no_request_models_inside_create_app():
    """FastAPI cannot resolve pydantic models defined inside create_app (postponed annotations)."""
    import re
    src = open("trova/server.py").read()
    assert re.findall(r"\n    class \w+\(BaseModel\)", src) == []


def test_voice_log_endpoint(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch)
    assert c.post("/api/voice/log", json={"who": "you", "text": "trova la tazza rossa"}).json()["ok"]
    assert any(e["kind"] == "voice" for e in c.get("/api/log").json())


# ---------------------------------------------------------------- feature 7: tray lift (experimental)
def test_tray_lift_waypoints_safety_and_stop(tmp_path, monkeypatch):
    import threading
    import time
    from trova.arm import CartesianArm, FakeCartesian, Limits
    from trova.arm.skills import TrayParams, waypoints
    p = TrayParams(z_slot=30, insert=40, lift=40, standoff=60, approach=(0.0, 1.0))
    wps = waypoints((300, 0), (450, 100), 150, p)
    assert wps[3][1:] == (300, 0, 30)              # fork fully in under the pick centre, at slot height
    assert wps[2][1:] == (300, -40, 30)            # entered along +y
    assert max(w[3] for w in wps) == 150 and wps[6][1:3] == (450, 100)
    fake = FakeCartesian(start=(300, -100, 150))
    arm = CartesianArm(fake, 150, None, Limits.parse("100,600,-300,300,20,400"), speed_mm_s=4000)
    assert arm.run_waypoints(wps) and len(fake.history) == len(wps)
    low = CartesianArm(FakeCartesian(), 150, None, Limits.parse("100,600,-300,300,50,400"), speed_mm_s=4000)
    assert not low.run_waypoints(wps) and "outside" in low.last_error       # refused BEFORE moving
    assert low.backend.history == []
    slow = CartesianArm(FakeCartesian(start=(300, -100, 150)), 150, None, None, speed_mm_s=60)
    th = threading.Thread(target=slow.run_waypoints, args=(wps,))
    th.start()
    time.sleep(0.5)
    slow.stop()
    th.join(timeout=3)
    assert slow.state == "stopped" and len(slow.backend.history) < len(wps)

    app, c = _client(tmp_path, monkeypatch)
    c.post("/api/background", data={"mode": "capture"})
    assert c.post("/api/arm/lift_tray", json={"place_u": 600, "place_v": 400}).status_code == 403   # off by default
    monkeypatch.setenv("TROVA_ALLOW_LIFT", "1")
    colors = [o.split()[0] for o in c.get("/api/state").json()["source"]["objects"]]
    unique = next(col for col in colors if colors.count(col) == 1)
    c.post("/api/find", data={"text": f"the {unique} one", "move": "false"})
    r = c.post("/api/arm/lift_tray", json={"place_u": 600, "place_v": 400}).json()
    assert r["ok"] and r["waypoints"] == 12
