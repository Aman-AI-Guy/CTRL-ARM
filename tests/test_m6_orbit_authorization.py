"""Bounded autonomy still consumes per-view proposals and operator-owned prerequisites."""

import time
import numpy as np

from trova.scan3d.authorization import OrbitPermit


def test_scope_cannot_unlock_or_skip_supervised_qualification(tmp_path, monkeypatch):
    from test_m5_agent import orbit_scene
    from trova.server import AgentHost

    st, client, backend = orbit_scene(tmp_path, monkeypatch)
    plan = client.post("/api/agent/task", json={"text": "Scansionalo."}).json()
    assert plan["confirmation"]
    denied = client.post(
        "/api/scan/orbit/authorize",
        json={"operator": "test", "max_views": 2, "confirm": True},
    )
    assert denied.status_code == 400 and not backend.calls
    assert "supervised" in denied.json()["detail"]
    scope_tools = [
        tool["name"]
        for tool in client.get("/api/voice/config").json()["session"]["tools"]
    ]
    assert not any("authorize" in name for name in scope_tools)
    st.orbits.settings.motion_qualification = {
        "supervised_orbit_verified": True,
        "verification_note": "synthetic scope unit test",
    }
    permit = OrbitPermit.approve(
        AgentHost(st),
        st.pending_motion,
        st.scans.active,
        operator="test",
        max_views=2,
        ttl_s=30,
        confirm=True,
    )
    assert permit.remaining == 2 and not permit.started and not backend.calls
    st.arm.gateway.max_speed_mm_s -= 1
    assert "changed" in permit.invalid_reason(
        AgentHost(st), st.pending_motion, st.scans.active
    )
    permit.expires_at = time.time() - 1
    assert "exhausted" in permit.invalid_reason(
        AgentHost(st), st.pending_motion, st.scans.active
    )


def test_bounded_worker_executes_only_reviewed_move_budget_in_simulation(
    tmp_path, monkeypatch
):
    from test_m5_agent import orbit_scene
    from trova.server import AgentHost
    import trova.arm.cartesian

    monkeypatch.setattr(trova.arm.cartesian, "START_GRACE_S", 0)
    st, client, backend = orbit_scene(tmp_path, monkeypatch)
    task = dict(st.selected_context)
    from trova.agent import skill_locate

    host = AgentHost(st)
    skill_locate(host, task, "")
    context = st.orbits.context(host, task)
    settings = st.orbits.settings
    settings.motion_qualification = {
        "operator": "simulated-test-only",
        "measurement_note": "Synthetic motion-envelope regression; not physical evidence",
        "model_sha256": context.model_sha256,
        "mount_sha256": context.mount_signature(),
        "head_link4_contact_reviewed": True,
        "tool_body_assessed": True,
        "static_object_assessed": True,
        "controller_segment_box_verified": True,
        "tracking_error_mm": 0,
        "joint_min_deg": [-360, -60, -225, -97, -360],
        "joint_max_deg": [360, 120, 11, 180, 360],
        "object_foot_min_mm": (context.foot - 100).tolist(),
        "object_foot_max_mm": (context.foot + 100).tolist(),
        "max_object_height_mm": context.settings.object_height_mm+10,
        "max_object_radius_mm": context.scene.obj_radius+10,
        "supervised_orbit_verified": True,
        "verification_note": "simulated regression only",
    }
    st.arm.gateway.keepout_assessed = True
    frame = np.random.default_rng(12).integers(30, 220, (800, 1280, 3), dtype=np.uint8)

    def read(host):
        st.inventory["snapshot"]["pose"] = st.robot_telemetry()
        return frame, {
            "kind": "pi",
            "requested_at": time.time(),
            "width": 1280,
            "height": 800,
        }

    monkeypatch.setattr(AgentHost, "read_frame", read)
    plan = client.post("/api/agent/task", json={"text": "Scansionalo."}).json()
    assert plan["confirmation"], '\n'.join(c.get('reason','') for c in (plan.get('scan_plan') or {}).get('candidates',[])[:3])
    scope = client.post(
        "/api/scan/orbit/authorize",
        json={
            "operator": "simulated test",
            "max_views": 2,
            "ttl_s": 120,
            "confirm": True,
        },
    )
    assert scope.status_code == 200, scope.text
    assert not backend.calls
    first = client.post(
        "/api/agent/confirm", json={"proposal_id": plan["confirmation"]["id"]}
    ).json()
    assert not first["hardware_moved"]
    deadline = time.monotonic() + 30
    while (
        st.orbit_worker and st.orbit_worker.is_alive() and time.monotonic() < deadline
    ):
        st.orbit_worker.join(0.1)
    assert not (st.orbit_worker and st.orbit_worker.is_alive())
    assert st.orbit_permit.remaining == 0, st.orbit_permit.public()
    good = [v for v in st.scans._session(None)["views"] if v.get("accepted")]
    assert len(good) == 2, (st.last_agent, st.orbit_permit.public())
    count = len(backend.calls)
    time.sleep(0.1)
    assert len(backend.calls) == count
    assert not st.last_agent["hardware_moved"]
