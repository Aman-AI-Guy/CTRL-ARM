"""Voice table scan with the V2 camera model: look poses, object map in robot mm, duplicates, camera above an object.

Geometry tests use the real pinned V2 file. The end-to-end test drives the simulated xArm 5 (real kinematics) with
the demo camera and mock Grok, so labels and positions are not physical; it checks the flow, the gates and the
motion path through CartesianArm.go_joints.
"""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from test_features import _wait  # noqa: E402
from test_pipeline import _client  # noqa: E402
from trova import table_scan as ts  # noqa: E402
from trova.arm.cartesian import Limits  # noqa: E402
from trova.scan3d.flange import flange_pose  # noqa: E402

LIMITS = "-500,500,-500,500,100,1000"


@pytest.fixture(scope="module")
def cal():
    c, meta = ts.load_v2()
    assert meta["sha256"].startswith("59c1afad") and meta["physical_qualification"] is False
    return c


def test_scan_plan_holds_height_tilt_and_keeps_the_room_out_of_view(cal):
    s = ts.settings({})
    plan = ts.plan_scan(cal, s, [0, 0, 0, 0, 0], Limits.parse(LIMITS))
    assert plan["problems"] == [], plan["problems"]
    views = [st for st in plan["steps"] if st["capture"]]
    assert len(views) == 5 and not plan["steps"][0]["capture"]
    assert plan["center_deg"] == 0.0 and plan["center_from"] == "start pose"   # the table is under the rest pose
    assert plan["steps"][0]["q"][0] == 0.0                              # rises and tilts before turning
    for v in views:
        tcp = flange_pose(v["q"])[:3, 3]
        assert abs(tcp[2] - 600.0) < 2.0
        view = ts.view_on_table(cal, v["q"])
        assert view["table_only"] and view["far_mm"] <= s.far_mm      # top edge, corners included, on the table
        assert np.allclose(v["q"][1:], views[0]["q"][1:])              # only J1 turns between views
        hit = cal.pixel_to_table(cal.width / 2, cal.height / 2, v["q"])   # the real centre ray, not a projection
        assert abs(math.hypot(hit[0], hit[1]) - plan["aim_mm"]) < 5
    assert sorted(v["offset_deg"] for v in views) == [-60.0, -30.0, 0.0, 30.0, 60.0]


def test_settings_from_environment():
    s = ts.settings({"TROVA_TABLE_SCAN_Z_MM": "550", "TROVA_TABLE_SCAN_PAN_DEG": "-40,40,20",
                     "TROVA_TABLE_SCAN_SPEED_DEG_S": "90"})
    assert s.z_mm == 550 and s.pan_deg == [-40, -20, 0, 20, 40] and s.joint_speed_deg_s == 20.0   # speed capped
    with pytest.raises(ValueError):
        ts.settings({"TROVA_TABLE_SCAN_HOVER_MM": "50"})


def test_boxes_map_back_to_their_table_position_and_sightings_merge(cal):
    s = ts.settings({})
    plan = ts.plan_scan(cal, s, [0, 0, 0, 0, 0])
    truth = {"cup": (430.0, 60.0), "pen": (560.0, 0.0), "tape": (380.0, -170.0)}   # on the table under the rest pose
    sightings = []
    for view in (st for st in plan["steps"] if st["capture"]):
        for name, (x, y) in truth.items():
            try:
                u, v = cal.robot_to_pixel([x, y, cal.table_z_at(x, y)], view["q"])
            except ValueError:
                continue
            if not (40 <= u < cal.width - 40 and 100 <= v < cal.height):
                continue
            loc = ts.locate_box(cal, view["q"], [u - 30, v - 80, u + 30, v])
            assert math.dist(loc["foot_mm"][:2], (x, y)) < 1.0           # the box foot is the table contact point
            sightings.append({"view": view["label"], "name": name, "box": [u - 30, v - 80, u + 30, v], "image": "",
                              **loc})
    objects = ts.merge(sightings, s.merge_mm)
    assert sorted(o["name"] for o in objects) == ["cup", "pen", "tape"]
    assert all(o["sightings"] >= 2 for o in objects)


def test_two_identical_objects_ask_which_one(cal):
    from trova.grok import GrokClient
    objects = [{"id": 1, "name": "paper cup", "aliases": [], "category": "container", "note": "green",
                "distance_mm": 450, "bearing_deg": -60.0, "centre_mm": [225, -390, -41]},
               {"id": 2, "name": "paper cup", "aliases": [], "category": "container", "note": "green",
                "distance_mm": 450, "bearing_deg": -120.0, "centre_mm": [-225, -390, -41]},
               {"id": 3, "name": "blue pen", "aliases": [], "category": "stationery", "note": "",
                "distance_mm": 300, "bearing_deg": -90.0, "centre_mm": [0, -300, -41]}]
    grok = GrokClient(api_key="")                                       # mock: word matching
    res = ts.find(grok, "il bicchiere, paper cup", objects, None, -90.0, "it")
    assert res["status"] == "clarify" and res["candidates"] == [1, 2]
    assert "il numero 1" in res["question"] and "il numero 2" in res["question"]
    assert "sinistra del robot" in res["question"] and "destra del robot" in res["question"]
    assert ts.resolve_choice("il numero 2", [1, 2], objects, -90.0) == 2
    assert ts.resolve_choice("quello a sinistra", [1, 2], objects, -90.0) == 1    # bearing -60 is the robot's left
    assert ts.resolve_choice("non so", [1, 2], objects, -90.0) is None
    assert ts.find(grok, "blue pen", objects, None, -90.0, "it")["status"] == "found"
    assert ts.find(grok, "scissors", objects, None, -90.0, "it")["status"] == "not_found"


def test_camera_over_object_centres_it_and_respects_the_workspace(cal):
    s = ts.settings({})
    start = ts.plan_scan(cal, s, [0, 0, 0, 0, 0])["steps"][-1]["q"]
    target = [430.0, 60.0, cal.table_z_at(430, 60)]
    plan = ts.plan_camera_over(cal, s, start, target, Limits.parse(LIMITS))
    assert plan["problems"] == [], plan["problems"]
    final = plan["steps"][-1]["q"]
    assert ts.aim_error_deg(cal, target, final) < 0.5                  # the object is ON the image-centre ray
    hit = cal.pixel_to_table(cal.width / 2, cal.height / 2, final)
    assert math.dist(hit[:2], target[:2]) < 5
    assert abs(math.dist(plan["camera_mm"], target) - s.hover_mm) < 10 and plan["look_down_deg"] > 70
    horizontal = math.dist(plan["camera_mm"][:2], target[:2])
    assert abs(horizontal - s.hover_mm * math.cos(math.radians(plan["look_down_deg"]))) < 15   # geometry agrees
    far = ts.plan_camera_over(cal, s, start, [650.0, 0.0, cal.table_z_at(650, 0)], Limits.parse(LIMITS))
    assert any("x=" in p.lower() or "outside" in p.lower() for p in far["problems"]), far["problems"]


def test_camera_over_is_never_a_distortion_fold(cal):
    """Regression: a far off-axis point can project to the image centre through the distortion polynomial. The
    solver must aim the real centre ray at the object or refuse."""
    s = ts.settings({})
    start = ts.plan_scan(cal, s, [0, 0, 0, 0, 0])["steps"][-1]["q"]
    for xy in ((207.0, 0.0), (263.9, 330.7), (430.0, 60.0), (300.0, -200.0), (150.0, 250.0)):
        target = [*xy, cal.table_z_at(*xy)]
        try:
            q = ts.camera_over_pose(cal, target, s.hover_mm, start)
        except ValueError:
            continue                                                     # refusing is acceptable; folding is not
        assert ts.aim_error_deg(cal, target, q) < 0.5, xy
        assert math.dist(cal.pixel_to_table(cal.width / 2, cal.height / 2, q)[:2], xy) < 5, xy


def _sim_app(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="xarm5sim", TROVA_ARM_LIMITS=LIMITS,
                     TROVA_ARM_HOME_JOINTS="0,0,0,0,0", TROVA_SIM_FAST="400", TROVA_TABLE_SCAN_SETTLE_S="0",
                     TROVA_TABLE_SCAN_LANG="it", TROVA_ARM_READ="", TROVA_TOKEN="")
    return app.state.trova, c


def test_scan_find_choose_and_camera_above_on_the_simulated_xarm(tmp_path, monkeypatch):
    st, c = _sim_app(tmp_path, monkeypatch)
    assert not st.physical and st.arm.gateway.mode == "simulation"
    plan = c.post("/api/table_scan/plan").json()
    assert plan["executable"] and not plan["moved"] and not plan["physical_pose"]
    started = c.post("/api/table_scan/start", json={"move": True})
    assert started.status_code == 200, started.text
    assert _wait(lambda: c.get("/api/table_scan").json()["status"] not in ("moving", "planned", "labelling"), 60)
    t = c.get("/api/table_scan").json()
    assert t["status"] == "done", t
    assert t["physical_positions"] is False and "simulate" in t["summary"]
    assert len(t["views"]) == 5 and t["objects"], t
    q = st.arm.backend.joints()
    assert abs(flange_pose(q)[2, 3] - 600) < 2                         # it ended at the scan height
    history = [h for h in st.arm.backend.history if h[0] == "joints"]
    assert len(history) == 6                                            # rise+tilt, then five views
    name = t["objects"][0]["name"]
    found = c.post("/api/table_scan/find", json={"text": name}).json()
    assert found["status"] in ("found", "clarify"), found
    oid = found.get("object_id") or found["candidates"][0]
    chosen = c.post("/api/table_scan/choose", json={"object_id": oid}).json()
    assert chosen["status"] == "selected" and chosen["object_id"] == oid
    reachable = [o["id"] for o in t["objects"]
                 if c.post("/api/table_scan/approach", json={"object_id": o["id"]}).status_code == 200]
    assert reachable, "no mapped object has a camera pose inside the workspace"
    oid = reachable[0]
    prop = c.post("/api/table_scan/approach", json={"object_id": oid}).json()
    assert prop["status"] == "awaiting_confirmation" and prop["proposal_id"]
    assert c.post("/api/table_scan/confirm", json={"proposal_id": "wrong"}).status_code == 409   # consumes nothing
    done = c.post("/api/table_scan/confirm", json={"proposal_id": prop["proposal_id"]})
    assert done.status_code == 200, done.text
    assert _wait(lambda: c.get("/api/table_scan").json()["status"] in ("over_object", "error", "stopped"), 30)
    t = c.get("/api/table_scan").json()
    assert t["status"] == "over_object", t
    final = prop["plan"]["steps"][-1]["q"]
    assert max(abs(a - b) for a, b in zip(st.arm.backend.joints(), final)) < 1.0
    again = c.post("/api/table_scan/confirm", json={"proposal_id": prop["proposal_id"]})
    assert again.status_code == 409                                     # single use


def test_stop_ends_the_scan_and_voids_the_proposal(tmp_path, monkeypatch):
    st, c = _sim_app(tmp_path, monkeypatch)
    monkeypatch.setenv("TROVA_SIM_FAST", "1")
    st.arm.backend.fast = 1.0                                           # slow enough to stop mid-move
    assert c.post("/api/table_scan/start", json={"move": True}).status_code == 200
    assert _wait(lambda: st.arm.backend.is_moving(), 5)
    c.post("/api/arm/stop")
    assert _wait(lambda: c.get("/api/table_scan").json()["status"] in ("stopped", "error"), 10)
    assert not st.arm.backend.is_moving()
    assert st.table_pending is None


def test_hardware_needs_the_operator_v2_acknowledgement(tmp_path, monkeypatch):
    st, _c = _sim_app(tmp_path, monkeypatch)
    st.arm.gateway.mode = "hardware"                                   # only to exercise the check itself
    monkeypatch.delenv("TROVA_TABLE_SCAN_V2_HARDWARE", raising=False)
    from trova.server import ArmBusy
    with pytest.raises(ArmBusy, match="accept-unqualified-v2"):
        st.table_motion_check()
    monkeypatch.setenv("TROVA_TABLE_SCAN_V2_HARDWARE", ts.HARDWARE_ACK)
    st.table_motion_check()
    st.arm.gateway.mode = "dry_run"
    with pytest.raises(ArmBusy, match="dry_run"):
        st.table_motion_check()


def test_voice_session_offers_the_table_tools(tmp_path, monkeypatch):
    _st, c = _sim_app(tmp_path, monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "k")
    cfg = c.get("/api/voice/config").json()
    names = {t["name"] for t in cfg["session"]["tools"]}
    assert {"scan_table", "find_on_table", "choose_table_object", "move_camera_above", "confirm_camera_move"} <= names
    assert "scan_table" in cfg["session"]["instructions"]


def test_live_inventory_of_an_empty_view_reports_its_mode():
    """inventory._grok_mode used to call the GrokClient.offline property: live + no boxes raised TypeError."""
    from trova.grok import GrokClient
    from trova.inventory import describe
    grok = GrokClient(api_key="not-used-no-call-is-made")
    assert not grok.mock
    inv = describe(grok, np.zeros((90, 160, 3), np.uint8), [], "it")
    assert inv.grok_mode == "live" and inv.items == {}


def test_find_returns_to_the_photo_pose_then_over_it_never_lower_z(cal):
    """Operator's P1 (2026-11-05): 'trovami X' goes back to the joints of X's photo (J1 the short way round), then
    leans over X; the sampled TCP never drops below the scan height, and the stored photo pose is used as is."""
    p1 = np.array([2.45, -30.22, -49.65, 119.24, 221.76])
    photo_q = p1.copy()
    photo_q[0] -= 270.0                                   # the 360 photo at -270 deg: same pose as J1 = +92.4
    target = cal.pixel_to_table(cal.width * 0.4, cal.height * 0.7, photo_q)
    obj = {"centre_mm": list(target), "photo": {"view": "view -270 deg", "joints": list(photo_q), "tcp_mm": None}}
    plan = ts.plan_photo_over(cal, ts.settings({}), p1, obj, Limits.parse(LIMITS))
    assert not plan["problems"] and plan["over"]
    assert abs(plan["steps"][0]["q"][0] - 92.45) < 0.01 and plan["steps"][0]["capture"] == "photo_pose"
    assert np.allclose(plan["steps"][0]["q"][1:], p1[1:])           # only the base turns back to the photo
    z = ts.tcp_z(p1)
    assert plan["min_z_mm"] >= z - ts.Z_TOL_MM and ts.path_low_z(plan["steps"], p1) >= z - ts.Z_TOL_MM
    assert all(abs(b["q"][0] - a["q"][0]) < 30 for a, b in zip(plan["steps"], plan["steps"][1:]))
    assert all(s.get("fast") for s in plan["steps"])                 # post-scan speed
    with pytest.raises(ValueError):
        ts.level_path(p1, np.r_[p1[0], 0, 0, 0, 0], z)               # rest pose is far below: refused
