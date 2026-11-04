"""Independent review regressions, promoted from /private/tmp. All controllers are test doubles."""

import sys
import threading
import types

import pytest
from fastapi.testclient import TestClient

from test_features import _wait
from test_motion_safety import RESET, cart_client
from trova.server import create_app

AUTH = {"x-trova-operator-token": "test-operator-secret"}


def physical_client(tmp_path, monkeypatch, **extra):
    """Exercise physical-session policy without importing/connecting any hardware backend."""
    import trova.server as server
    from trova.arm import make_arm
    monkeypatch.setattr(server, "make_arm", lambda spec, *args: make_arm("cartesian:fake", *args))
    env = dict(TROVA_ARM="xarm:test-double-only", TROVA_OPERATOR_NAME="staff-ana",
               TROVA_OPERATOR_TOKEN="test-operator-secret", TROVA_TOKEN="")
    env.update(extra)
    return cart_client(tmp_path, monkeypatch, **env)


def confirmation(c, **extra):
    return {**RESET, "stop_nonce": c.get("/api/state").json().get("stop_nonce", "not-yet-implemented"), **extra}


@pytest.mark.parametrize("previous_state", ["STOP", "crash_during_lift"])
def test_restart_requires_rearming_and_preserves_lift_recovery(tmp_path, monkeypatch, previous_state):
    st, _ = physical_client(tmp_path, monkeypatch)
    if previous_state == "STOP":
        st.stop_arm()
    else:
        st.lift_store.write("in_progress")
    app2 = create_app(st.s)         # new server state, same data directory/configuration
    second = app2.state.trova
    response = TestClient(app2).post("/api/arm/home")
    if second.arm_thread:
        second.arm_thread.join(3)
    observed = (second.stop_latched.is_set(), second.lift_interrupted,
                response.status_code, second.arm.backend.history)
    assert second.stop_latched.is_set() and response.status_code == 409, observed
    if previous_state == "crash_during_lift":
        assert second.lift_interrupted, observed


@pytest.mark.parametrize("endpoint", ["reset", "recover_lift"])
def test_safety_recovery_rejects_unauthenticated_caller(tmp_path, monkeypatch, endpoint):
    st, c = physical_client(tmp_path, monkeypatch, TROVA_OPERATOR_NAME="", TROVA_OPERATOR_TOKEN="")
    st.stop_arm()
    st.lift_interrupted = endpoint == "recover_lift"
    body = {**confirmation(c), "operator": "any unauthenticated caller", "fork_freed": True, "note": "claimed recovery"}
    response = c.post("/api/arm/" + endpoint, json=body)
    assert response.status_code in (401, 403), (response.status_code, response.json())


def test_old_watch_callback_cannot_borrow_restarted_watch_authority(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/background")
    c.post("/api/reference", data={"describe": "false"})
    st.source.table.tamper("add")
    w = st.watch
    w.interval = 0.01
    entered, release = threading.Event(), threading.Event()
    original = w._point_at

    def delayed_point(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
        original(*args, **kwargs)

    monkeypatch.setattr(w, "_point_at", delayed_point)
    monkeypatch.setattr(w, "_explain", lambda *args: None)
    w.start(point=True)
    old_thread = w._thread
    try:
        assert entered.wait(3)
        old_alert = w.alert
        st.stop_arm()
        assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
        assert c.post("/api/watch/ack").status_code == 200
        st.source.table.tamper("move")  # a different signature can raise the replacement alert
        # stop() times out joining the paused old callback; start() launches a second loop.
        assert c.post("/api/watch", json={"on": True, "point": False}).status_code == 200
        assert old_thread.is_alive()
        assert _wait(lambda: w.alert is not None and w.alert is not old_alert)
        assert not w.point and st.arm.backend.history == []
        release.set()
        old_thread.join(3)
        assert not old_thread.is_alive()
        if st.arm_thread:
            st.arm_thread.join(3)
        assert st.arm.backend.history == [], st.arm.backend.history
    finally:
        release.set()
        w.stop()
        old_thread.join(3)
        st.stop_arm()
        if st.arm_thread:
            st.arm_thread.join(3)


def test_xarm_connection_does_not_recover_or_enable_controller(tmp_path, monkeypatch):
    calls = []

    class FakeSDK:
        def __init__(self, *args, **kwargs):
            pass

        def get_position(self):
            return 0, [300, 0, 200, 180, 0, 0]

        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k)) or 0

    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = FakeSDK
    monkeypatch.setitem(sys.modules, "xarm", types.ModuleType("xarm"))
    monkeypatch.setitem(sys.modules, "xarm.wrapper", wrapper)
    monkeypatch.setenv("TROVA_ARM_TCP_MM", "")
    from trova.arm.xarm import XArmBackend
    XArmBackend("fake-sdk-only")
    assert not any(name in {"clean_error", "clean_warn", "motion_enable", "set_mode", "set_state"}
                   for name, _, _ in calls), calls


def test_xarm_stop_reports_nonzero_sdk_result():
    from trova.arm.xarm import XArmBackend

    class RejectedSDK:
        def set_state(self, state):
            return 9

        def motion_enable(self, on):
            return 9

    backend = XArmBackend.__new__(XArmBackend)
    backend.lock = threading.Lock()
    backend.arm = RejectedSDK()
    with pytest.raises(RuntimeError):
        backend.stop()


def test_software_reset_does_not_release_controller_protection(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    st.stop_arm()
    st.arm.backend.protective_stopped = True  # controller remains stopped but not moving
    monkeypatch.setattr(st.arm.backend, "is_moving", lambda: False)
    monkeypatch.setattr(st.arm.backend, "clear_errors", lambda: pytest.fail("reset invoked controller recovery"))
    response = c.post("/api/arm/reset", json=confirmation(c))
    assert response.status_code == 200
    assert st.arm.backend.protective_stopped and st.arm.backend.history == []


def test_joint_move_with_unavailable_fk_fails_closed(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)

    def unavailable(q):
        raise NotImplementedError

    monkeypatch.setattr(st.arm.backend, "fk_mm", unavailable)
    st.arm.home_joints = [0.0, 2000.0, 0.0, 0.0, 0.0]  # toy FK x=2300, outside configured +/-1000 box
    st.arm.joint_speed = 20000
    c.post("/api/arm/home")
    st.arm_thread.join(3)
    assert st.arm.backend.history == [], st.arm.backend.history


def test_reset_confirmation_cannot_clear_a_later_stop(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    st.stop_arm()  # operator sees and confirms this STOP
    entered, release = threading.Event(), threading.Event()
    original = st.operator_reset

    def queued(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(st, "operator_reset", queued)
    responses = []
    body = confirmation(c)
    worker = threading.Thread(target=lambda: responses.append(c.post("/api/arm/reset", json=body)))
    worker.start()
    try:
        assert entered.wait(3)
        st.stop_arm()  # a NEW STOP after the reset request/confirmations were submitted
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert responses[0].status_code == 409 and st.stop_latched.is_set(), (
            responses[0].status_code, st.stop_latched.is_set(), st.stop_sequence)
    finally:
        release.set()
        worker.join(3)


@pytest.mark.parametrize("content", [None, "{", '{}', '{"version":true,"lift":"clear"}',
                                    '{"version":1,"lift":"clear","extra":true}',
                                    '{"version":1,"lift":"in_progress","lift":"clear"}',
                                    '{"version":1,"lift":"in_progress"}'])
def test_missing_corrupt_ambiguous_lift_record_needs_recovery(tmp_path, monkeypatch, content):
    tmp_path.mkdir(exist_ok=True)
    if content is not None:
        (tmp_path / "arm_safety.json").write_text(content)
    st, c = physical_client(tmp_path, monkeypatch)
    assert st.physical and st.stop_latched.is_set() and st.lift_interrupted
    for path, args in [("/api/arm/home", {}), ("/api/arm/point", {"json": {"x_mm": 100, "y_mm": 100}}),
                       ("/api/room/find", {"data": {"text": "lamp"}}),
                       ("/api/arm/lift_tray", {"json": {"pick_u": 1, "pick_v": 1, "place_u": 2, "place_v": 2}})]:
        assert c.post(path, **args).status_code == 409
    assert st.arm.backend.history == []


def test_authenticated_boot_recovery_and_reset_never_move(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 401
    assert c.post("/api/arm/recover_lift", headers=AUTH, json=confirmation(
        c, fork_freed=True, note="Staff verified no load/fork remains engaged", operator="spoofed")).status_code == 200
    assert st.stop_latched.is_set() and not st.lift_interrupted
    assert c.post("/api/arm/reset", headers=AUTH, json=confirmation(c, operator="spoofed")).status_code == 200
    assert not st.stop_latched.is_set() and st.arm.backend.history == []
    assert st.audit.recent(1)[0]["operator"] == "staff-ana"


def test_lift_marker_is_durable_before_first_waypoint_and_survives_failure(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    c.post("/api/arm/recover_lift", headers=AUTH, json=confirmation(c, fork_freed=True, note="Fork free"))
    c.post("/api/arm/reset", headers=AUTH, json=confirmation(c))
    observed = []

    def interrupted(wps):
        from trova.safety import LiftStore
        observed.append(LiftStore(tmp_path).read())
        return False

    monkeypatch.setattr(st.arm, "run_waypoints", interrupted)
    st.start_arm(st.run_lift, [("test", 300, 0, 150)], job="lift")
    st.arm_thread.join(3)
    assert observed == ["in_progress"] and st.lift_store.read() == "in_progress" and st.lift_interrupted
    second = create_app(st.s).state.trova
    assert second.stop_latched.is_set() and second.lift_interrupted


def test_failed_marker_write_dispatches_no_waypoint(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    c.post("/api/arm/recover_lift", headers=AUTH, json=confirmation(c, fork_freed=True, note="Fork free"))
    c.post("/api/arm/reset", headers=AUTH, json=confirmation(c))

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(st.lift_store, "write", broken)
    monkeypatch.setattr(st.arm, "run_waypoints", lambda *a: pytest.fail("waypoint before durable marker"))
    st.start_arm(st.run_lift, [("test", 300, 0, 150)], job="lift")
    st.arm_thread.join(3)
    assert st.stop_latched.is_set() and st.lift_interrupted and "disk full" in st.persistence_error


@pytest.mark.parametrize("endpoint", ["reset", "recover_lift"])
def test_operator_credential_is_required_even_with_venue_access(tmp_path, monkeypatch, endpoint):
    st, c = physical_client(tmp_path, monkeypatch, TROVA_TOKEN="venue-token")
    c.get("/?token=venue-token")
    body = confirmation(c, fork_freed=True, note="Fork checked", operator="staff-ana")
    for headers in ({}, {"x-trova-operator-token": "wrong"}):
        assert c.post("/api/arm/" + endpoint, json=body, headers=headers).status_code == 401
        assert st.stop_latched.is_set() and st.lift_interrupted and st.lift_store.read() == "unknown"


@pytest.mark.parametrize("endpoint", ["reset", "recover_lift"])
def test_stale_nonce_is_rejected_without_changing_interlocks(tmp_path, monkeypatch, endpoint):
    st, c = physical_client(tmp_path, monkeypatch)
    body = confirmation(c, fork_freed=True, note="Checked fork")
    st.stop_arm()
    assert c.post("/api/arm/" + endpoint, json=body, headers=AUTH).status_code == 409
    assert st.stop_latched.is_set() and st.lift_interrupted and st.arm.backend.history == []


def test_stop_during_recovery_persistence_invalidates_confirmation(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    original = st.lift_store.write

    def concurrent(lift):
        original(lift)
        if lift == "clear":
            st.stop_arm()

    monkeypatch.setattr(st.lift_store, "write", concurrent)
    response = c.post("/api/arm/recover_lift", headers=AUTH,
                      json=confirmation(c, fork_freed=True, note="Staff freed fork"))
    assert response.status_code == 409
    assert st.lift_interrupted and st.stop_latched.is_set() and st.lift_store.read() == "in_progress"


@pytest.mark.parametrize("code", [1, -1, 9, None])
def test_xarm_stop_is_hold_only_and_propagates_errors(code):
    from trova.arm.xarm import XArmBackend
    calls = []

    class SDK:
        def set_state(self, state):
            calls.append(("set_state", state))
            return code

        def motion_enable(self, value):
            pytest.fail("STOP must not change motor enable")

    backend = XArmBackend.__new__(XArmBackend)
    backend.lock, backend.arm = threading.Lock(), SDK()
    with pytest.raises(RuntimeError, match="STOP"):
        backend.stop()
    assert calls == [("set_state", 4)]


def test_xarm_state_read_errors_are_not_cached_as_idle():
    from trova.arm.xarm import XArmBackend
    backend = XArmBackend.__new__(XArmBackend)
    backend.lock = threading.Lock()
    backend.arm = types.SimpleNamespace(get_state=lambda: (9, 2))
    with pytest.raises(RuntimeError, match="get_state"):
        backend.is_moving()


def test_watch_generation_cannot_dispatch_from_a_different_worker(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/reference", data={"describe": "false"})
    w = st.watch
    w.point = True
    w.step()
    run = w._run
    other = threading.Thread()
    assert not w.valid_run(run, other)
    w.stop()
    assert run.cancel.is_set() and not w.valid_run(run, run.worker)


def test_complete_lift_persists_clear_but_next_boot_still_requires_rearm(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    c.post("/api/arm/recover_lift", headers=AUTH, json=confirmation(c, fork_freed=True, note="Fork free"))
    c.post("/api/arm/reset", headers=AUTH, json=confirmation(c))
    monkeypatch.setattr(st.arm, "run_waypoints", lambda wps: st.lift_store.read() == "in_progress")
    st.start_arm(st.run_lift, [("test", 300, 0, 150)], job="lift")
    st.arm_thread.join(3)
    assert not st.lift_interrupted and st.lift_store.read() == "clear"
    second = create_app(st.s).state.trova
    assert second.stop_latched.is_set() and not second.lift_interrupted
    assert second.stop_nonce != st.stop_nonce


@pytest.mark.parametrize("failure", ["clean_error", "clean_warn", "motion_enable", "set_mode", "set_state"])
def test_xarm_explicit_recovery_checks_each_sdk_result(failure):
    from trova.arm.xarm import XArmBackend
    calls = []

    class SDK:
        def __getattr__(self, name):
            def call(*a, **k):
                calls.append(name)
                return 9 if name == failure else 0
            return call

    backend = XArmBackend.__new__(XArmBackend)
    backend.lock, backend.arm = threading.Lock(), SDK()
    with pytest.raises(RuntimeError, match=failure):
        backend.clear_errors()
    assert calls[-1] == failure  # no enabling/recovery steps follow a failed SDK operation


def test_controller_idle_read_failure_blocks_reset_with_clear_response(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)

    def failed():
        raise RuntimeError("controller state unavailable")

    monkeypatch.setattr(st.arm.backend, "is_moving", failed)
    response = c.post("/api/arm/reset", headers=AUTH, json=confirmation(c))
    assert response.status_code == 409 and "Cannot verify controller idle" in str(response.json())
    assert st.stop_latched.is_set()


def test_acknowledgement_permission_dies_with_watch_generation(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/reference", data={"describe": "false"})
    w = st.watch
    w.point = True
    w.step()
    old_run = w._run
    alert = {"id": 1, "signature": ("test",), "pointed": [100, 100]}
    w.alert, w._alert_motion = alert, (alert, old_run.ticket, old_run)
    old_guard = w.motion_guard_for(alert)
    assert old_guard()
    w.stop()
    w.step()  # manually start a replacement generation for deterministic testing
    assert w._run.history is not old_run.history or not old_guard()
    assert not old_guard()
    from trova.server import ArmBusy
    with pytest.raises(ArmBusy):
        st.start_arm(st.arm.home, pixels=False, ticket=old_run.ticket, guard=old_guard)
    assert st.arm.backend.history == []


def test_concurrent_recovery_confirmations_can_be_used_only_once(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    body = confirmation(c, fork_freed=True, note="Staff freed the fork")
    barrier = threading.Barrier(2)
    responses = []

    def recover():
        barrier.wait(3)
        responses.append(c.post("/api/arm/recover_lift", headers=AUTH, json=body).status_code)

    workers = [threading.Thread(target=recover) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert sorted(responses) == [200, 409]
    assert st.stop_latched.is_set() and not st.lift_interrupted and st.arm.backend.history == []


def test_joint_dispatch_requires_workspace_limits_even_at_driver_boundary():
    from trova.arm import CartesianArm, FakeCartesian
    arm = CartesianArm(FakeCartesian(), 150, (300, 0, 200), None)
    assert not arm.go_joints([0.0] * 5)
    assert arm.state == "refused" and "safety box" in arm.last_error and arm.backend.history == []


def test_failed_stop_resend_is_visible_and_prevents_reset(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    st.stop_arm()

    def rejected():
        raise RuntimeError("controller rejected resend")

    monkeypatch.setattr(st.arm.backend, "stop", rejected)
    assert not st.arm._halted()
    assert st.arm.state == "error" and "STOP resend failed" in st.arm.last_error
    assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 409


def test_overlapping_watch_stop_does_not_clobber_replacement_worker(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/reference", data={"describe": "false"})
    w = st.watch
    entered, release = threading.Event(), threading.Event()
    old = types.SimpleNamespace(is_alive=lambda: True)

    def joining(timeout):
        entered.set()
        assert release.wait(5)

    old.join = joining
    w._thread = old
    stopping = threading.Thread(target=w.stop)
    stopping.start()
    try:
        assert entered.wait(3)
        # The replacement is published after stop invalidated the old run but before its join completed.
        with w._control:
            replacement = threading.Thread()
            w._thread = replacement
            w.on, w.status = True, "replacement watching"
        release.set()
        stopping.join(3)
        assert not stopping.is_alive() and w.on and w.status == "replacement watching"
    finally:
        release.set()
        stopping.join(3)
        w._thread = None
        w.stop()
