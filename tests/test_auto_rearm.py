"""TROVA_AUTO_REARM=1: the next new command re-arms; commands admitted before the STOP stay cancelled."""

import threading

from test_features import _unique_color, _wait
from test_motion_safety import cart_client


def test_next_command_rearms_after_stop(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch, TROVA_AUTO_REARM="1")
    st.arm.speed = 30
    assert c.post("/api/arm/point", json={"x_mm": 600, "y_mm": 100}).status_code == 200
    assert _wait(lambda: len(st.arm.backend.history) == 1)
    c.post("/api/arm/stop")
    old_cancel, old_epoch = st.cancel, st.stop_epoch
    assert st.stop_latched.is_set() and c.get("/api/state").json()["auto_rearm"]
    st.arm.speed = 20000
    assert c.post("/api/arm/home").status_code == 200
    st.arm_thread.join(3)
    assert not st.stop_latched.is_set() and not st.arm._stop.is_set()
    assert old_cancel.is_set() and old_epoch.is_set()
    assert len(st.arm.backend.history) == 2


def test_command_admitted_before_stop_stays_cancelled(tmp_path, monkeypatch):
    import trova.server as server
    st, c = cart_client(tmp_path, monkeypatch, TROVA_AUTO_REARM="1")
    c.post("/api/background")
    unique = _unique_color(c)
    entered, release = threading.Event(), threading.Event()
    original = server.grok_find

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "grok_find", delayed)
    out = []
    worker = threading.Thread(target=lambda: out.append(c.post("/api/find", data={"text": f"the {unique} one"})))
    worker.start()
    assert entered.wait(3)
    c.post("/api/arm/stop")
    assert c.post("/api/arm/home").status_code == 200     # a new command: re-armed
    st.arm_thread.join(3)
    moves = len(st.arm.backend.history)
    release.set()
    worker.join(5)
    assert not worker.is_alive() and out[0].status_code == 200
    assert not out[0].json()["moved"]
    if st.arm_thread:
        st.arm_thread.join(3)
    assert len(st.arm.backend.history) == moves


def test_disabled_controller_is_enabled_but_a_fault_is_left_to_the_operator(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch, TROVA_AUTO_REARM="1")
    status = {"state_code": 5, "error_code": 0, "moving": False}
    enabled = []
    st.arm.backend.read_status = lambda: dict(status)
    st.arm.backend.clear_errors = lambda: enabled.append(1)
    assert c.post("/api/arm/home").status_code == 200
    st.arm_thread.join(3)
    assert enabled == [1]

    c.post("/api/arm/stop")
    status.update(state_code=4, error_code=22)
    r = c.post("/api/arm/home")
    assert r.status_code == 409 and "Controller error 22" in str(r.json())
    assert enabled == [1] and st.stop_latched.is_set()
    assert "Controller error 22" in c.get("/api/state").json()["arm_blocked"]

    status.update(error_code=0)
    assert c.post("/api/arm/home").status_code == 200
    assert enabled == [1, 1] and not st.stop_latched.is_set()
