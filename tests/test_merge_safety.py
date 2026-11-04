"""M1/M2/M3 merge regressions: synthetic images, fake controllers and model responses only."""

import threading

import pytest

from test_motion_safety import cart_client
from test_safety_review import AUTH, confirmation, physical_client
from trova.motion import KeepOut
from trova.server import AgentHost


@pytest.mark.parametrize("blocked", ["startup", "STOP", "lift", "limits", "home"])
def test_m2_jog_cannot_bypass_existing_interlocks(tmp_path, monkeypatch, blocked):
    if blocked == "startup":
        st, client = physical_client(tmp_path, monkeypatch)
    else:
        config = {"TROVA_ARM_LIMITS": ""} if blocked == "limits" else {}
        if blocked == "home":
            config["TROVA_ARM_HOME_MM"] = ""
        st, client = cart_client(tmp_path, monkeypatch, **config)
        if blocked == "STOP":
            st.stop_arm()
        if blocked == "lift":
            st.lift_interrupted = True
    response = client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 3})
    assert response.status_code == 409, response.json()
    assert st.arm.backend.history == []
    if blocked == "startup":
        assert client.post("/api/arm/reset", json=confirmation(client)).status_code == 401
        assert st.stop_latched.is_set()


def test_m2_metric_jog_uses_central_guard_and_gateway(tmp_path, monkeypatch):
    st, client = cart_client(tmp_path, monkeypatch)
    st.calib = None  # a base-frame metric jog does not use pixel calibration
    result = client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 3}).json()
    assert result["status"] == "executed" and result["requested"]["distance_mm"] == 3
    assert st.arm.backend.history == [(297.0, 0.0, 200.0)]
    st.arm.gateway.keepouts = [KeepOut((292, 294), (-5, 5), (190, 210))]
    # Endpoint 287 is outside the keep-out; the intermediate sample at 292 is inside it.
    before = list(st.arm.backend.history)
    result = client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 10}).json()
    assert result["status"] == "rejected" and result["requested"]["distance_mm"] == 10
    assert "keep-out" in result["reason"] and st.arm.backend.history == before


def test_delayed_jog_permission_does_not_survive_stop_and_reset(tmp_path, monkeypatch):
    st, client = cart_client(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = st.arm.gateway.plan_relative

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(st.arm.gateway, "plan_relative", delayed)
    responses = []
    worker = threading.Thread(target=lambda: responses.append(client.post(
        "/api/arm/jog", json={"direction": "backward", "distance_mm": 3})))
    worker.start()
    try:
        assert entered.wait(3)
        st.stop_arm()
        assert client.post("/api/arm/reset", json=confirmation(client)).status_code == 200
        release.set()
        worker.join(5)
        assert not worker.is_alive() and responses[0].status_code == 409
        assert st.arm.backend.history == []
    finally:
        release.set()
        worker.join(5)


def test_delayed_agent_retains_cancel_ticket_and_cannot_overlap(tmp_path, monkeypatch):
    import trova.server as server
    st, client = cart_client(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    hosts = []

    def delayed(host, text):
        hosts.append(host)
        entered.set()
        assert release.wait(5)
        # Synthetic live metadata isolates cancellation from unrelated recognition prerequisites.
        host.snapshot_mode = "live"
        host.grok.modes.append("live")
        outcome = host.execute([297.0, 0.0, 200.0])
        assert not outcome["ok"]
        return {"status": "stopped", "authoritative": True, "reason": outcome["reason"]}

    monkeypatch.setattr(server.agent_mod, "run", delayed)
    responses = []
    worker = threading.Thread(target=lambda: responses.append(client.post(
        "/api/agent/task", json={"text": "inspect and approach the tool"})))
    worker.start()
    try:
        assert entered.wait(3)
        assert client.post("/api/agent/task", json={"text": "replacement"}).status_code == 409
        st.stop_arm()
        assert hosts[0].cancel.is_set() and hosts[0].ticket.is_set()
        assert client.post("/api/arm/reset", json=confirmation(client)).status_code == 200
        release.set()
        worker.join(5)
        assert not worker.is_alive() and responses[0].status_code == 200
        assert st.arm.backend.history == []
    finally:
        release.set()
        worker.join(5)


@pytest.mark.parametrize("invalidated", ["cancel", "view", "calibration", "offline"])
def test_agent_dispatch_revalidates_immutable_context(tmp_path, monkeypatch, invalidated):
    st, client = cart_client(tmp_path, monkeypatch)
    host = AgentHost(st)
    host.snapshot_mode = "live"
    host.grok.modes.append("live")
    if invalidated == "cancel":
        host.cancel.set()
    elif invalidated == "view":
        assert client.post("/api/source", data={"spec": "demo"}).status_code == 200
    elif invalidated == "calibration":
        st.vision.version += 1
    else:
        host.grok.modes.append("offline")
    outcome = host.execute([297.0, 0.0, 200.0])
    assert not outcome["ok"] and st.arm.backend.history == []


def test_new_motion_routes_do_not_consume_a_scene_clarification(tmp_path, monkeypatch):
    st, client = cart_client(tmp_path, monkeypatch)
    client.post("/api/inventory")
    question = client.post("/api/select", data={"text": "black object"}).json()
    assert question["status"] == "clarify"
    for path, body in [("/api/arm/jog", {"direction": "backward", "distance_mm": 3}),
                       ("/api/agent/task", {"text": "move above that one"})]:
        assert client.post(path, json=body).status_code == 409
    assert st.pending_select == question["selection_context"]
    assert st.arm.backend.history == []


def test_agent_selection_fallback_cannot_launder_live_snapshot_authority(tmp_path, monkeypatch):
    import trova.server as server
    from trova.grok import GrokResult
    st, client = cart_client(tmp_path, monkeypatch)
    monkeypatch.setattr(st.grok, "json", lambda **kwargs: GrokResult(
        data={}, latency_s=0, model="synthetic", mock=False, offline=True))

    def fallback(host, text):
        host.snapshot_mode = "live"
        host.grok.json(name="synthetic-selection")
        assert not host.execute([297.0, 0.0, 200.0])["ok"]
        return {"status": "held", "authoritative": True, "grok_mode": "live"}

    monkeypatch.setattr(server.agent_mod, "run", fallback)
    result = client.post("/api/agent/task", json={"text": "my tool"}).json()
    assert result["authoritative"] is False and result["grok_mode"] == "offline"
    assert result["provenance"] == {"snapshot": "live", "decisions": ["offline"]}
    assert st.arm.backend.history == []


def test_readonly_telemetry_does_not_rearm_physical_policy_session(tmp_path, monkeypatch):
    st, client = physical_client(tmp_path, monkeypatch)
    before = (st.stop_nonce, st.lift_interrupted)
    robot = client.get("/api/state").json()["robot"]
    assert robot["process_can_move"] is False
    assert st.stop_latched.is_set() and (st.stop_nonce, st.lift_interrupted) == before
    assert st.arm.backend.history == []
    # Authentication is still required, even when telemetry and a motion gateway are available.
    assert client.post("/api/arm/reset", json=confirmation(client)).status_code == 401
    assert client.post("/api/arm/reset", headers=AUTH, json=confirmation(client)).status_code == 200
    assert st.lift_interrupted  # resetting never recovers the load


def test_a_gateway_ticket_alone_cannot_authorize_a_physical_session(tmp_path, monkeypatch):
    """A fake represents the hardware boundary; this test never selects hardware mode."""
    st, client = cart_client(tmp_path, monkeypatch)
    class Controller:
        def __init__(self, fake):
            self.fake = fake

        def __getattr__(self, name):
            return getattr(self.fake, name)

    st.arm.backend = Controller(st.arm.backend)
    st.arm.gateway.simulated = False
    assert not st.arm.point_at(297, 0)
    assert "authorized server session" in st.arm.last_error
    assert st.arm.backend.history == []
    st.arm.state, st.arm.last_error = "idle", None  # clear only the test refusal
    result = client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 3})
    assert result.status_code == 200 and result.json()["status"] == "executed"
    assert st.arm.backend.history == [(297.0, 0.0, 200.0)]


def test_gateway_simulator_heuristic_cannot_override_physical_startup_policy(tmp_path, monkeypatch):
    import trova.server as server
    from test_pipeline import _client
    from trova.arm import CartesianArm, FakeCartesian, Limits

    class WrappedController:
        """Synthetic controller wrapper, deliberately not a built-in simulator type."""
        name = "xarm"

        def __init__(self):
            self.fake = FakeCartesian()

        def __getattr__(self, name):
            return getattr(self.fake, name)

    arm = CartesianArm(WrappedController(), 150, (300, 0, 200), Limits.parse("-1000,1000,-1000,1000,-1000,1000"))
    arm.gateway.simulated, arm.gateway.mode = True, "simulation"  # a mistaken SDK/loopback heuristic
    monkeypatch.setattr(server, "make_arm", lambda *args: arm)
    app, client = _client(tmp_path, monkeypatch, TROVA_ARM="xarm:synthetic-wrapper-only")
    st = app.state.trova
    assert st.physical and st.stop_latched.is_set()
    assert not arm.gateway.simulated and arm.gateway.mode == "dry_run"
    assert arm.gateway.joint_limits is not None
    assert client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 3}).status_code == 409
    assert arm.backend.fake.history == []


def test_cancelling_active_agent_motion_holds_the_arm(tmp_path, monkeypatch):
    import trova.server as server
    from test_features import _wait
    st, client = cart_client(tmp_path, monkeypatch)
    st.arm.speed = 30

    def approach(host, text):
        host.snapshot_mode = "live"
        host.grok.modes.append("live")
        outcome = host.execute([500.0, 0.0, 200.0])
        return {"status": "stopped", "authoritative": True, "reason": outcome["reason"]}

    monkeypatch.setattr(server.agent_mod, "run", approach)
    responses = []
    worker = threading.Thread(target=lambda: responses.append(client.post("/api/agent/task", json={"text": "approach"})))
    worker.start()
    try:
        assert _wait(lambda: len(st.arm.backend.history) == 1)
        assert _wait(lambda: st.arm.tcp_mm()[0] > 305)
        assert client.post("/api/agent/cancel").status_code == 200
        worker.join(3)
        assert not worker.is_alive() and responses[0].status_code == 200
        assert st.stop_latched.is_set() and st.task_cancel.is_set()
        assert st.arm.tcp_mm()[0] < 500 and not st.arm.backend.is_moving()
    finally:
        st.stop_arm()
        worker.join(3)
