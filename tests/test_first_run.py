"""First supervised run (TROVA_FIRST_RUN=1): find -> confirm -> hover -> confirm -> descend -> measure.
Simulated Cartesian arm + simulated arm camera + mock Grok only. Nothing here can reach hardware."""

import json
import math

import pytest

import trova.server as server
from test_features import _unique_color, _wait
from test_motion_safety import confirmation
from test_pipeline import _client
from trova.calibration import Calibration

WARNING = "Collision avoidance: NOT PRESENT. Hold the physical E-stop."


def first_run_client(tmp_path, monkeypatch, **extra):
    def fake_aruco(frame, pos):        # the simulated table has no printed markers: a known pixel -> robot mapping
        c = Calibration.default(frame.shape[1], frame.shape[0], 600, 400)
        c.size = [frame.shape[1], frame.shape[0]]
        return c
    monkeypatch.setattr(server, "calibrate_aruco", fake_aruco)
    env = dict(TROVA_ARM="cartesian:fake", TROVA_CAMERA="armsim", TROVA_CAMERA_ON_ARM="1", TROVA_CAMERA_SETTLE_S="0",
               TROVA_ARM_LIMITS="-2000,2000,-2000,2000,-2000,2000", TROVA_ARM_SPEED_MM_S="20000",
               TROVA_ARM_JOINT_SPEED_DEG_S="20000", TROVA_ARM_HOME_MM="", TROVA_ARM_HOME_JOINTS="", TROVA_TOKEN="",
               TROVA_FIRST_RUN="1", TROVA_OPERATOR_NAME="staff-ana", TROVA_OPERATOR_TOKEN="op-secret")
    env.update(extra)
    app, c = _client(tmp_path, monkeypatch, **env)
    st = app.state.trova
    st.speeds, st.fast = [], 1000
    move = st.arm.backend.move_linear

    def recorded(x, y, z, speed):      # keep the commanded speed; the fake then runs `fast` times quicker
        st.speeds.append(speed)
        move(x, y, z, speed * st.fast)
    st.arm.backend.move_linear = recorded
    assert c.post("/api/arm/teach_pose", data={"name": "home_joints"}).status_code == 200   # survey pose
    (tmp_path / "markers.json").write_text(json.dumps({"_frame": "robot", "0": [0, 0], "1": [600, 0],
                                                       "2": [600, 400], "3": [0, 400]}))
    assert c.post("/api/calibration/aruco").json()["pose"] == [0.0] * 5        # recorded at this survey pose
    c.post("/api/background", data={"mode": "capture"})
    return st, c


def find(c):
    r = c.post("/api/find", data={"text": f"the {_unique_color(c)} one", "move": "false"}).json()
    assert r["result"]["status"] == "found" and not r["moved"], r
    return r["approach"]


def confirm(c, ap, **over):
    return c.post("/api/arm/point_last", json={"stage": ap["stage"], "confirm": ap["confirm"],
                                               "x_mm": ap["target_mm"][0], "y_mm": ap["target_mm"][1], **over})


def approach(c):
    return c.get("/api/state").json()["approach"]


def stage_is(c, stage):
    return _wait(lambda: (approach(c) or {}).get("stage") == stage)


def events(tmp_path, kind):
    return [e for e in map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines()) if e["kind"] == kind]


def test_full_flow_find_hover_descend_record_and_measure(tmp_path, monkeypatch):
    st, c = first_run_client(tmp_path, monkeypatch, TROVA_FIRST_RUN_SPEED_MM_S="50")   # env cannot raise the cap
    assert st.first_run_speed == 20 and st.arm.speed == 20
    items = [i for i in c.get("/api/preflight?grok=0").json()["items"] if i["name"] == "First run"]
    assert [i["status"] for i in items] == ["ok"], items
    ap = find(c)
    x, y = ap["target_mm"]
    assert ap["stage"] == 1 and ap["problems"] == [] and ap["warning"] == WARNING and ap["marked"]
    assert ap["z_mm"] == 250 and ap["speed_mm_s"] == 20 and ap["distance_mm"] > 0
    assert st.arm.backend.history == []                          # showing the target never moves the arm
    r = confirm(c, ap).json()
    assert r["stage"] == 1 and r["target_mm"] == [x, y, 250] and r["warning"] == WARNING
    assert stage_is(c, 2)
    tx, ty, tz = st.arm.tcp_mm()
    assert math.hypot(tx - x, ty - y) < 1 and abs(tz - 250) < 1    # hovering 100 mm above the object
    ap2 = approach(c)
    assert ap2["z_mm"] == 150 and ap2["confirm"] != ap["confirm"] and abs(ap2["distance_mm"] - 100) < 1
    assert ap2["problems"] == []
    assert confirm(c, ap2).status_code == 200
    assert stage_is(c, "measure") and abs(st.arm.tcp_mm()[2] - 150) < 1
    assert st.speeds and max(st.speeds) <= 20
    arrived = events(tmp_path, "arm_approach")
    assert [(e["stage"], e["reached"], e["target_mm"]) for e in arrived] == [(1, True, [x, y, 250]), (2, True, [x, y, 150])]
    assert all(e["tcp_mm"] and e["offset_mm"] < 1 for e in arrived)
    r = c.post("/api/arm/measured_error", data={"error_mm": 4}).json()
    assert r == {"error_mm": 4.0, "missed": False, "threshold_mm": 10.0}
    assert c.post("/api/arm/measured_error", data={"error_mm": 4}).status_code == 400   # once per approach
    # next object: home first (a find with move=false never repositions), then a miss above the threshold
    assert c.post("/api/arm/home").status_code == 200 and _wait(st.at_survey)
    ap = find(c)
    assert confirm(c, ap).status_code == 200 and stage_is(c, 2)
    assert confirm(c, approach(c)).status_code == 200 and stage_is(c, "measure")
    assert c.post("/api/arm/measured_error", data={"error_mm": 25}).json()["missed"]
    fails = c.get("/api/failures").json()
    assert fails["by_cause"][0]["cause"] == "arm_missed" and fails["recent"][0]["error_mm"] == 25
    acc = events(tmp_path, "arm_accuracy")
    assert [(e["measured_error_mm"], e["missed"]) for e in acc] == [(4.0, False), (25.0, True)]
    hist = json.dumps(c.get("/api/history", params={"groups": "arm"}).json())
    assert "ruler error 25" in hist and "stage 2: target" in hist


@pytest.mark.parametrize("problem", ["no_calibration", "other_survey_pose", "other_image_size", "no_operator",
                                     "stop_latched"])
def test_refuses_when_a_first_run_check_fails(tmp_path, monkeypatch, problem):
    extra = {"TROVA_OPERATOR_TOKEN": ""} if problem == "no_operator" else {}
    st, c = first_run_client(tmp_path, monkeypatch, **extra)
    ap = find(c)
    if problem == "no_calibration":
        c.post("/api/calibration/reset")
    elif problem == "other_survey_pose":
        st.calib.pose = [5.0, 0.0, 0.0, 0.0, 0.0]       # e.g. a calibration file left from an older survey pose
    elif problem == "other_image_size":
        st.calib.size = [640, 480]
    elif problem == "stop_latched":
        c.post("/api/arm/stop")
    r = confirm(c, ap)
    assert r.status_code == 409 and "refused" in r.json()["error"], r.json()
    assert approach(c)["problems"]
    assert [i for i in c.get("/api/preflight?grok=0").json()["items"] if i["name"] == "First run" and i["status"] == "fail"]
    assert st.arm.backend.history == []


def test_target_outside_the_safety_box_is_refused_before_moving(tmp_path, monkeypatch):
    st, c = first_run_client(tmp_path, monkeypatch, TROVA_ARM_LIMITS="-2000,2000,-2000,2000,-2000,220")
    ap = find(c)                                         # hover at 250 mm is above the box's zmax 220
    assert any("Safety box" in p for p in ap["problems"])
    r = confirm(c, ap)
    assert r.status_code == 409 and "safe box" in r.json()["error"]
    assert st.arm.backend.history == []


def test_confirmations_are_single_use_and_bound_to_stage_and_target(tmp_path, monkeypatch):
    st, c = first_run_client(tmp_path, monkeypatch)
    ap = find(c)
    assert c.post("/api/arm/point_last").status_code == 409                 # a bare 'yes' (voice) is not enough
    assert confirm(c, ap, confirm="0" * 32).status_code == 409              # a made-up confirmation burns nothing
    assert confirm(c, ap, stage=2).status_code == 409                       # stage-1 confirmation cannot descend...
    assert confirm(c, ap).status_code == 409                                # ...and is burned by trying
    ap = find(c)
    r = confirm(c, ap, x_mm=ap["target_mm"][0] + 10)                        # applied to a different target...
    assert r.status_code == 409 and "different target" in r.json()["error"]
    assert confirm(c, ap).status_code == 409                                # ...burns it: find again
    assert st.arm.backend.history == []
    old = ap
    ap = find(c)
    assert confirm(c, old, x_mm=ap["target_mm"][0], y_mm=ap["target_mm"][1]).status_code == 409   # older find
    assert confirm(c, ap, speed_mm_s=21).status_code == 409                 # above the cap; burns nothing yet
    assert confirm(c, ap, speed_mm_s=5).status_code == 200 and stage_is(c, 2)
    assert st.speeds == [5]
    assert confirm(c, ap).status_code == 409                                # reused stage-1 confirmation
    ap2 = approach(c)
    assert confirm(c, ap2, stage=1).status_code == 409                      # stage-2 confirmation cannot be stage 1
    assert approach(c)["stage"] == 2 and approach(c)["confirm"] is None     # and it is burned
    assert c.post("/api/arm/measured_error", data={"error_mm": 1}).status_code == 400   # no stage 2 happened
    assert len(st.speeds) == 1


def test_stage_two_needs_a_fresh_confirmation_at_the_hover_point(tmp_path, monkeypatch):
    st, c = first_run_client(tmp_path, monkeypatch)
    ap = find(c)
    assert confirm(c, ap).status_code == 200 and stage_is(c, 2)
    ap2 = approach(c)
    c.post("/api/arm/home")                                                 # left the hover point
    assert _wait(st.at_survey)
    r = confirm(c, ap2)
    assert r.status_code == 409 and "hover point" in r.json()["error"]
    # a stage-2 confirmation issued before a STOP is dead after the operator reset
    ap = find(c)
    assert confirm(c, ap).status_code == 200 and stage_is(c, 2)
    ap2 = approach(c)
    c.post("/api/arm/stop")
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    n = len(st.speeds)
    assert confirm(c, ap2).status_code == 409
    assert len(st.speeds) == n and abs(st.arm.tcp_mm()[2] - 250) < 1        # still hovering, never descended


def test_stale_confirmation_expires(tmp_path, monkeypatch):
    st, c = first_run_client(tmp_path, monkeypatch)
    ap = find(c)
    st.pending_move["approach"]["issued"] -= server.CONFIRM_TTL_S + 1
    assert "Stale" in confirm(c, ap).json()["error"] and st.arm.backend.history == []


@pytest.mark.parametrize("stage", [1, 2])
def test_stop_during_either_stage_holds_the_arm(tmp_path, monkeypatch, stage):
    st, c = first_run_client(tmp_path, monkeypatch)
    ap = find(c)
    if stage == 2:
        assert confirm(c, ap).status_code == 200 and stage_is(c, 2)
        ap = approach(c)
    st.fast = 1                                                             # real 20 mm/s from here on
    start = st.arm.tcp_mm()
    assert confirm(c, ap).status_code == 200
    assert _wait(lambda: math.dist(st.arm.tcp_mm(), start) > 2)
    c.post("/api/arm/stop")
    st.arm_thread.join(3)
    held = st.arm.tcp_mm()
    assert st.arm.state == "stopped" and st.arm.backend.target == list(held)   # holds where it was
    assert abs(held[2] - ap["z_mm"]) > 5                                    # never reached the stage target
    assert approach(c) is None and st.stop_latched.is_set()
    assert confirm(c, ap).status_code == 409
    assert c.post("/api/arm/measured_error", data={"error_mm": 1}).status_code == 400
    assert math.dist(st.arm.tcp_mm(), held) < 0.5


def test_other_motion_is_refused_in_first_run_mode(tmp_path, monkeypatch):
    st, c = first_run_client(tmp_path, monkeypatch)
    r = c.post("/api/find", data={"text": f"the {_unique_color(c)} one", "move": "true"}).json()
    assert not r["moved"] and r["approach"]["stage"] == 1                   # move=true only prepares stage 1
    assert c.post("/api/arm/point", json={"x_mm": 300, "y_mm": 100}).status_code == 409
    assert c.post("/api/find_all", data={"text": "all the things", "move": "true"}).json().get("moved") is not True
    assert st.arm.backend.history == []


def test_stage_two_worker_rechecks_the_hover_point(tmp_path, monkeypatch):
    """A home that finishes between the endpoint check and dispatch must not turn stage 2 into a diagonal move."""
    st, c = first_run_client(tmp_path, monkeypatch)
    ap = find(c)
    assert confirm(c, ap).status_code == 200 and stage_is(c, 2)
    ap2 = st.pending_move["approach"]
    st.arm.home()                                              # the arm left the hover point after admission
    n = len(st.speeds)
    st.run_approach(st.pending_move, dict(ap2, confirm=None), 20)
    assert len(st.speeds) == n and st.arm.state == "refused" and approach(c) is None
