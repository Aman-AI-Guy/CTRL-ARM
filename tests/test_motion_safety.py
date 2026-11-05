"""P0 regressions: only synthetic images, mock Grok and simulated controllers."""

import threading

import pytest

from test_features import _unique_color, _wait
from test_pipeline import _client
from trova.arm import save_pose
from trova.calibration import Calibration
from trova.server import ArmBusy, StopReset


RESET = {"operator": "Ana", "area_clear": True, "controller_ready": True}


def confirmation(c, body=None):
    return {**(RESET if body is None else body), "stop_nonce": c.get("/api/state").json().get("stop_nonce", "missing")}


def cart_client(tmp_path, monkeypatch, **extra):
    env = dict(TROVA_ARM="cartesian:fake", TROVA_ARM_LIMITS="-1000,1000,-1000,1000,-1000,1000",
               TROVA_ARM_HOME_MM="300,0,200", TROVA_ARM_HOME_JOINTS="", TROVA_CAMERA_ON_ARM="0",
               TROVA_CAMERA_SETTLE_S="0", TROVA_ARM_SPEED_MM_S="20000", TROVA_ALLOW_LIFT="1")
    env.update(extra)
    app, c = _client(tmp_path, monkeypatch, **env)
    st = app.state.trova
    st.frame()
    st.calib = Calibration.default(1280, 800, 600, 400)
    st.calib.source, st.calib.camera = "arm", st.s.camera
    save_pose(tmp_path, "look_joints", [0.0] * 5)
    return st, c


def test_stop_active_move_blocks_every_motion_until_explicit_reset(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    st.arm.speed = 30
    assert c.post("/api/arm/point", json={"x_mm": 600, "y_mm": 100}).status_code == 200
    assert _wait(lambda: len(st.arm.backend.history) == 1)
    assert _wait(lambda: st.arm.tcp_mm()[0] > 305)  # outside the parked-pose tolerance before STOP
    c.post("/api/arm/stop")
    st.arm_thread.join(3)
    before = list(st.arm.backend.history)
    old_cancel = st.cancel
    for path, kwargs in [
        ("/api/arm/home", {}),
        ("/api/arm/point", {"json": {"x_mm": 100, "y_mm": 100}}),
        ("/api/find", {"data": {"text": "the red one"}}),
        ("/api/find_all", {"data": {"text": "all the black things"}}),
        ("/api/arm/lift_tray", {"json": {"pick_u": 100, "pick_v": 100, "place_u": 200, "place_v": 200}}),
        ("/api/room/find", {"data": {"text": "lamp"}}),
        ("/api/background", {}), ("/api/reference", {}), ("/api/inventory", {}),
        ("/api/labels", {}), ("/api/calibration/aruco", {}),
    ]:
        r = c.post(path, **kwargs)
        assert r.status_code == 409, (path, r.json())
        assert "STOP" in str(r.json())
    assert st.arm.backend.history == before and old_cancel.is_set() and st.arm._stop.is_set()
    s = c.get("/api/state").json()
    assert s["stop_latched"] and "STOP" in s["arm_blocked"]
    # Controller recovery does not reset either software latch.
    c.post("/api/arm/clear_errors")
    assert st.stop_latched.is_set() and st.arm._stop.is_set()
    assert c.post("/api/arm/home").status_code == 409
    assert c.post("/api/arm/reset", json=confirmation(c)).json() == {
        "stop_latched": False, "lift_interrupted": False, "moved": False}
    assert old_cancel.is_set() and st.arm.backend.history == before
    st.arm.speed = 20000
    assert c.post("/api/arm/home").status_code == 200
    st.arm_thread.join(3)
    assert len(st.arm.backend.history) == len(before) + 1


@pytest.mark.parametrize("route", ["/api/find", "/api/find_all"])
def test_delayed_grok_cannot_move_after_stop_and_reset(tmp_path, monkeypatch, route):
    import trova.server as server
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/background")
    unique = _unique_color(c)
    entered, release = threading.Event(), threading.Event()
    name = "grok_find" if route.endswith("/find") else "grok_find_all"
    original = getattr(server, name)

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, name, delayed)
    out = []
    worker = threading.Thread(target=lambda: out.append(c.post(route, data={"text": f"the {unique} one"})))
    worker.start()
    assert entered.wait(3)
    c.post("/api/arm/stop")
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    release.set()
    worker.join(5)
    assert not worker.is_alive() and out[0].status_code == 200
    result = out[0].json()
    assert not result["moved"] and "STOP" in result["arm_blocked"]
    assert st.arm.backend.history == []
    assert c.post("/api/arm/point_last").status_code == 409  # stale confirmation is also cancelled
    monkeypatch.setattr(server, name, original)
    assert c.post(route, data={"text": f"the {unique} one"}).json()["moved"]
    c.post("/api/arm/stop")
    st.arm_thread.join(3)


def test_request_waiting_for_find_lock_stays_cancelled_across_reset(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    unique = _unique_color(c)
    entered = threading.Event()
    original = st.motion_ticket

    def capture(ticket=None):
        ticket = original(ticket)
        entered.set()
        return ticket

    monkeypatch.setattr(st, "motion_ticket", capture)
    out = []
    with st.lock:
        worker = threading.Thread(target=lambda: out.append(c.post("/api/find", data={"text": f"the {unique} one"})))
        worker.start()
        assert entered.wait(3)
        st.stop_arm()
        assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    worker.join(5)
    assert not worker.is_alive() and not out[0].json()["moved"]
    assert "STOP" in out[0].json()["arm_blocked"] and st.arm.backend.history == []


def test_stop_does_not_wait_for_arm_lock_and_waiting_command_cannot_restart(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    out = []
    with st.arm_lock:
        request = threading.Thread(target=lambda: out.append(c.post("/api/arm/home")))
        request.start()
        stopped = threading.Event()
        stopper = threading.Thread(target=lambda: (st.stop_arm(), stopped.set()))
        stopper.start()
        assert stopped.wait(1), "STOP waited on the motion lock"
        assert st.cancel.is_set() and st.arm._stop.is_set()
        assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    request.join(5)
    stopper.join(3)
    assert not request.is_alive() and out[0].status_code == 409
    assert st.arm.backend.history == []


def test_reset_requires_operator_checks_idle_driver_and_existing_api_access(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch, TROVA_TOKEN="safety-test")
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 401
    c.get("/?token=safety-test")
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409  # no latched STOP
    st.stop_arm()
    for bad in ({"operator": "Ana"}, {**confirmation(c), "operator": " "}, {**confirmation(c), "area_clear": False},
                {**confirmation(c), "controller_ready": False}):
        assert c.post("/api/arm/reset", json=confirmation(c, bad)).status_code == 409
        assert st.stop_latched.is_set() and st.arm._stop.is_set()
    st.manual_mode = True
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    st.manual_mode = False
    st.arm.last_error = "controller fault"
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    st.arm.last_error = None
    monkeypatch.setattr(st.arm.backend, "is_moving", lambda: True)
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    monkeypatch.setattr(st.arm.backend, "is_moving", lambda: False)
    monkeypatch.setattr(st.arm.backend, "clear_errors", lambda: pytest.fail("software reset recovered the controller"))
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    assert st.arm.backend.history == []


def test_reset_cannot_overtake_inflight_driver_stop_or_new_stop(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = st.arm.stop

    def delayed():
        original()
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(st.arm, "stop", delayed)
    worker = threading.Thread(target=st.stop_arm)
    worker.start()
    assert entered.wait(3) and st.stop_latched.is_set() and st.cancel.is_set()
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    release.set()
    worker.join(3)
    monkeypatch.setattr(st.arm, "stop", original)
    # A new STOP during the reset's read-only prerequisites wins, even when its driver call already finished.
    monkeypatch.setattr(st.arm.backend, "is_moving", lambda: (st.stop_arm(), False)[1])
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    assert st.stop_latched.is_set() and st.arm._stop.is_set() and st.arm.backend.history == []


@pytest.mark.parametrize("missing", ["limits", "home"])
def test_missing_cartesian_prerequisites_block_homing_and_camera_preparation(tmp_path, monkeypatch, missing):
    st, c = cart_client(tmp_path, monkeypatch,
                        **({"TROVA_ARM_LIMITS": ""} if missing == "limits" else {"TROVA_ARM_HOME_MM": ""}))
    st.arm.backend.pos = st.arm.backend.target = [400.0, 100.0, 200.0]
    st.reference = {"frame": st.frame(), "boxes": [], "time": "test", "path": "test", "n_objects": 0}
    for path, kwargs in [
        ("/api/arm/home", {}), ("/api/background", {}), ("/api/reference", {}),
        ("/api/inventory", {}), ("/api/labels", {}), ("/api/calibration/aruco", {}),
        ("/api/changes", {}),
        ("/api/find", {"data": {"text": "red"}}),
        ("/api/find_all", {"data": {"text": "black"}}),
        ("/api/room/find", {"data": {"text": "lamp"}}),
    ]:
        r = c.post(path, **kwargs)
        assert r.status_code == 409, (path, r.json())
        assert ("safety box" if missing == "limits" else "home pose") in str(r.json())
    assert st.arm.backend.history == []
    alert = {"id": 1, "signature": ("added",), "pointed": [300, 100]}
    st.watch.alert, st.watch._alert_motion = alert, None
    r = c.post("/api/watch/ack").json()
    assert r["motion_blocked"] and st.arm.backend.history == []
    st.stop_arm()
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    assert st.stop_latched.is_set()


@pytest.mark.parametrize("on_arm", ["0", "1"])
@pytest.mark.parametrize("route", ["/api/find", "/api/find_all"])
def test_move_false_never_homes_fake_cartesian(tmp_path, monkeypatch, on_arm, route):
    st, c = cart_client(tmp_path, monkeypatch, TROVA_CAMERA_ON_ARM=on_arm)
    st.arm.backend.pos = st.arm.backend.target = [400.0, 100.0, 200.0]
    r = c.post(route, data={"text": "red", "move": "false"})
    assert r.status_code == 409 and "move=false" in str(r.json())
    assert st.arm.backend.history == []
    st.arm.backend.pos = st.arm.backend.target = [300.0, 0.0, 200.0]
    st.stop_arm()                 # already-safe frames remain usable, even while software STOP is latched
    r = c.post(route, data={"text": "red", "move": "false"})
    assert r.status_code == 200 and not r.json()["moved"]
    assert st.arm.backend.history == [] and st.stop_latched.is_set() and st.arm._stop.is_set()


def test_interrupted_lift_requires_separate_manual_recovery_and_no_automatic_home(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    st.arm.speed = 30
    assert c.post("/api/arm/lift_tray", json={"pick_u": 640, "pick_v": 400,
                                             "place_u": 900, "place_v": 500}).status_code == 200
    assert _wait(lambda: st.lift_interrupted and len(st.arm.backend.history) > 0)
    assert _wait(lambda: sum(abs(p - h) for p, h in zip(st.arm.tcp_mm(), st.arm.home_mm)) > 8)
    assert c.post("/api/arm/home").status_code == 409  # cannot cut a running lift or clear its interlock
    assert st.lift_interrupted
    st.stop_arm()
    st.arm_thread.join(3)
    before = list(st.arm.backend.history)
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    assert st.lift_interrupted and not st.stop_latched.is_set()
    assert c.post("/api/arm/home").status_code == 409
    with pytest.raises(ArmBusy, match="interrupted"):
        st.clear_view()
    assert c.post("/api/arm/recover_lift", json={**confirmation(c), "fork_freed": True, "note": "Staff freed fork"}).status_code == 409
    st.stop_arm()
    for bad in (RESET, {**confirmation(c), "fork_freed": True}, {**confirmation(c), "fork_freed": False, "note": "checked"}):
        assert c.post("/api/arm/recover_lift", json=confirmation(c, bad)).status_code == 409
        assert st.lift_interrupted
    r = c.post("/api/arm/recover_lift", json={**confirmation(c), "fork_freed": True, "note": "Staff manually freed fork"})
    assert r.json() == {"stop_latched": True, "lift_interrupted": False, "moved": False}
    assert st.arm.backend.history == before and c.post("/api/arm/home").status_code == 409
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    assert st.arm.backend.history == before


def test_reset_refuses_live_worker_including_room_model_processing(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def processing():
        entered.set()
        assert release.wait(5)

    st.start_arm(processing, pixels=False, job="room")
    assert entered.wait(3)
    st.stop_arm()
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    release.set()
    st.arm_thread.join(3)
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200


def test_watch_motion_and_ack_home_remain_blocked_after_stop(tmp_path, monkeypatch):
    from trova import changes
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/background")
    c.post("/api/reference", data={"describe": "false"})
    w = st.watch
    w.point = True
    w.motion_ticket = st.motion_ticket()
    st.source.table.tamper("add")
    st.stop_arm()
    for _ in range(3):
        w.step()
    assert w.alert and not w.alert["pointed"] and st.arm.backend.history == []
    assert _wait(lambda: w.alert["grok"] != "pending")
    old_alert = w.alert
    # A pointed alarm that predates STOP must still be acknowledged normally, without homing.
    w.alert["pointed"] = [300, 100]
    r = c.post("/api/watch/ack", data={"who": "Ana"})
    assert r.status_code == 200 and r.json()["acknowledged"] and "STOP" in r.json()["motion_blocked"]
    assert w.alert is None and st.arm.backend.history == []
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    # A delayed background alert from the old watch must not move after reset either.
    f = st.frame()
    boxes, _ = st.proposals(f)
    cands = changes.diff_objects(st.reference["frame"], st.reference["boxes"], f, boxes)
    w.alert = old_alert
    w.alert["pointed"] = False
    w._point_at(f, cands, [], [], alert=old_alert, run=w._run)
    assert not w.alert["pointed"] and st.arm.backend.history == []


def test_home_target_still_obeys_safety_box(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch, TROVA_ARM_HOME_MM="5000,0,200")
    assert c.post("/api/arm/home").status_code == 200
    st.arm_thread.join(3)
    assert st.arm.state == "refused" and "safe box" in st.arm.last_error
    assert st.arm.backend.history == []


def test_old_clarification_cannot_move_after_stop_reset(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/background")
    r = c.post("/api/find", data={"text": "the black one"}).json()
    assert r["result"]["status"] == "clarify"
    choice = r["result"]["candidate_ids"][0]
    st.stop_arm()
    st.operator_reset(StopReset(**confirmation(c)))
    r = c.post("/api/clarify", data={"answer": f"number {choice}"}).json()
    assert r["result"]["status"] == "found" and not r["moved"] and "STOP" in r["arm_blocked"]
    assert st.arm.backend.history == []


def test_newer_result_ticket_cannot_override_a_cancelled_request(tmp_path, monkeypatch):
    from trova.server import MOTION_TICKET
    st, c = cart_client(tmp_path, monkeypatch)
    old = st.motion_ticket()
    st.stop_arm()
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    token = MOTION_TICKET.set(old)  # an older request now happens to read a newer pending result/alarm
    try:
        with pytest.raises(ArmBusy, match="STOP"):
            st.start_arm(st.arm.home, pixels=False, ticket=st.stop_epoch)
    finally:
        MOTION_TICKET.reset(token)
    assert st.arm.backend.history == []


def test_failed_driver_stop_stays_latched_and_blocks_reset(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    original = st.arm.backend.stop

    def down():
        raise RuntimeError("controller disconnected")

    monkeypatch.setattr(st.arm.backend, "stop", down)
    r = c.post("/api/arm/stop")
    assert r.status_code == 400 and "disconnected" in str(r.json())
    assert st.stop_latched.is_set() and st.cancel.is_set() and st.arm._stop.is_set()
    assert "disconnected" in c.get("/api/state").json()["stop_error"]
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409
    monkeypatch.setattr(st.arm.backend, "stop", original)
    assert c.post("/api/arm/stop").status_code == 200
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    assert st.arm.backend.history == []


def test_stop_during_preparatory_homing_cancels_the_find(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    st.arm.backend.pos = st.arm.backend.target = [500.0, 100.0, 200.0]
    st.arm.speed = 30
    out = []
    worker = threading.Thread(target=lambda: out.append(c.post("/api/find", data={"text": "the red one"})))
    worker.start()
    assert _wait(lambda: len(st.arm.backend.history) == 1)
    c.post("/api/arm/stop")
    worker.join(5)
    assert not worker.is_alive() and out[0].status_code == 409
    assert st.stop_latched.is_set() and st.cancel.is_set()
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
    assert len(st.arm.backend.history) == 1
