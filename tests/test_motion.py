"""Motion gateway. Simulated backends and SDK stubs only: this file never enables the motors."""

import math
import sys
from pathlib import Path

import pytest

from trova.arm import CartesianArm, FakeCartesian, Limits, SimArm
from trova.motion import (
    BRINGUP_MAX_SPEED_MM_S,
    CameraBody,
    KeepOut,
    MotionGateway,
    approve_bringup_speed,
    resolve_mode,
)

ROOT = Path(__file__).resolve().parent.parent


def test_pytest_cannot_select_hardware_mode():
    env = {"TROVA_MOTION_MODE": "hardware", "TROVA_HARDWARE_UNLOCK": "MOVE"}
    mode, note = resolve_mode(False, environ=env)
    assert mode != "hardware" and "pytest" in note
    mode, note = resolve_mode(False, environ=env, under_pytest=False)
    assert mode == "hardware" and note == ""
    assert resolve_mode(True, environ=env, under_pytest=False)[0] == "dry_run"
    assert resolve_mode(False, environ={"TROVA_MOTION_MODE": "hardware"}, under_pytest=False)[0] == "dry_run"
    blocked = MotionGateway(None, mode="hardware", simulated=False, limits=None, jog_max_mm=300,
                            max_speed_mm_s=100, max_accel_mm_s2=500, max_joint_speed_deg_s=20,
                            joint_limits=None, keepouts=[], camera=None)
    assert blocked.mode != "hardware"


def test_requested_distance_is_kept_and_rejected_in_the_open():
    speech = str(ROOT / "speech_to_text")
    if speech not in sys.path:
        sys.path.insert(0, speech)
    from grok_to_xarm_json import RobotCommand, offline_robot_command, preserve_robot_command

    assert offline_robot_command("Move backward by 3 millimeters").model_dump() == {
        "action": "move", "direction": "backward", "distance_mm": 3}
    assert offline_robot_command("Move backward by three millimeters").distance_mm == 3
    left = offline_robot_command("Sposta il braccio a sinistra di 2 cm")
    assert left.direction == "left" and left.distance_mm == 20
    far = offline_robot_command("Sposta indietro di 40 centimetri")
    assert far.direction == "backward" and far.distance_mm == 400
    assert offline_robot_command("Move forward by 50 centimeters").distance_mm == 500
    assert preserve_robot_command(RobotCommand(action="move", direction="backward", distance_mm=500)).distance_mm == 500

    gateway = MotionGateway.standalone(environ={})
    rejected = gateway.plan_relative("backward", far.distance_mm, pose=(200, 0, 150))
    assert rejected.status == "rejected" and rejected.requested["distance_mm"] == 400
    assert "400" in rejected.reason and rejected.collision_avoidance is False
    proposal = gateway.plan_relative("backward", 3, pose=(200, 0, 150))
    assert proposal.status == "dry_run" and proposal.proposal["delta_mm"] == [-3.0, 0.0, 0.0]
    assert proposal.safety["clearance"] == "not_configured"
    assert proposal.allow_motion is False
    capped = MotionGateway.standalone(environ={"TROVA_JOG_MAX_MM": "9999"})
    assert capped.jog_max_mm == 300
    small = MotionGateway.standalone(environ={"TROVA_JOG_MAX_MM": "3"})
    assert small.jog_max_mm == 3
    over = small.plan_relative("backward", 10, pose=(200, 0, 150))
    assert over.status == "rejected" and over.requested["distance_mm"] == 10
    locked = gateway.plan_relative("backward", 3, overrides={"mode": "hardware"})
    assert locked.status == "rejected" and "safety parameters" in locked.reason


def test_voice_cli_execute_does_not_call_the_sdk():
    import xarm_voice_execute as exe

    source = (ROOT / "xarm_voice_execute.py").read_text()
    preview = (ROOT / "xarm_voice_preview.py").read_text()
    assert "set_position" not in source and "motion_enable" not in source and "XArmAPI" not in source
    assert "set_position" not in preview and "motion_enable" not in preview
    before = {name for name in sys.modules if name == "xarm" or name.startswith("xarm.")}
    out = exe.run_text("Move backward by 3 millimeters", execute=True, confirm="MOVE",
                       environ={"TROVA_MOTION_MODE": "hardware", "TROVA_HARDWARE_UNLOCK": "MOVE"})
    after = {name for name in sys.modules if name == "xarm" or name.startswith("xarm.")}
    assert after == before and out["sdk_called"] is False
    assert out["motion"]["status"] == "rejected" and out["motion"]["mode"] != "hardware"
    assert out["motion"]["requested"]["distance_mm"] == 3
    assert out["motion"]["proposal"]["delta_mm"][0] == -3.0
    dry = exe.run_text("Move backward by 3 millimeters", environ={})
    assert dry["motion"]["status"] == "dry_run" and dry["sdk_called"] is False
    left = exe.run_text("Move left by 2 centimeters", environ={})
    assert left["command"]["distance_mm"] == 20 and left["motion"]["status"] == "rejected"
    assert "backward" in left["motion"]["reason"]
    opened = exe.run_text("Move left by 2 centimeters", allow_all_directions=True, environ={})
    assert opened["motion"]["status"] == "dry_run" and opened["motion"]["proposal"]["delta_mm"][1] == 20.0


def test_dry_run_does_not_move_and_keepout_uses_the_midpoint():
    arm = CartesianArm(FakeCartesian(), 150, None, Limits.parse("0,500,-200,200,0,400"), speed_mm_s=5000)
    arm.gateway = MotionGateway.for_arm(arm, mode="dry_run", environ={})
    assert not arm.point_at(320, 0)
    assert arm.backend.history == [] and arm.state == "idle" and arm.last_error is None
    assert arm.last_motion.status == "dry_run"

    fake = FakeCartesian(start=(50.0, 0.0, 150.0))
    guarded = CartesianArm(fake, 150, None, Limits.parse("0,400,-100,100,0,300"), speed_mm_s=5000)
    # Endpoints x=50 and x=350 are outside this slab. A sample at x=200 is inside it.
    guarded.gateway.keepouts = [KeepOut((198, 202), (-50, 50), (0, 300))]
    assert not guarded.point_at(350, 0)
    assert fake.history == [] and "keep-out" in guarded.last_error and "x=200" in guarded.last_error


def test_joint_path_that_bows_outside_the_box_is_refused():
    class Bent(FakeCartesian):
        def fk_mm(self, q):
            bow = 200 * math.sin(math.pi * q[1] / 100) if abs(q[1]) <= 100 else 0.0
            return (300 + q[1] + bow, q[0], 200 + q[2])

    arm = CartesianArm(Bent(), 150, None, Limits.parse("0,500,-100,100,0,400"), speed_mm_s=5000)
    assert not arm.go_joints([0.0, 100.0, 0.0, 0.0, 0.0])
    assert "joint pose" in arm.last_error and arm.backend.history == []


def test_hardware_checks_reject_without_sending_motion():
    gateway = MotionGateway.standalone(environ={})
    gateway.mode = "hardware"
    gateway.limits = None
    missing = gateway.authorize_cartesian((10, 0, 100), 20, start=(0, 0, 100))
    assert missing.status == "rejected" and "not configured" in missing.reason and not missing.allow_motion
    gateway.limits = Limits.parse("-2000,2000,-2000,2000,-2000,2000")
    long = gateway.authorize_cartesian((800, 0, 0), 20, start=(0, 0, 0))
    assert long.status == "rejected" and "700" in long.reason
    fast = gateway.authorize_cartesian((10, 0, 0), 400, start=(0, 0, 0))
    assert fast.status == "rejected" and "speed" in fast.reason
    gateway.limits = Limits.parse("0,500,-200,200,0,400")
    with pytest.raises(RuntimeError, match="refusing hardware"):
        gateway.plan_relative("backward", 3, pose=(200, 0, 150), execute=True)


def test_camera_sphere_is_optional_and_a_fault_does_not_clear_itself():
    plain = CartesianArm(FakeCartesian(start=(100.0, 0.0, 180.0)), 180, None,
                         Limits.parse("0,400,-100,100,0,200"), speed_mm_s=5000)
    assert plain.point_at(150, 0) and plain.last_motion.safety["clearance"] == "not_configured"
    assert plain.last_motion.collision_avoidance is False

    watched = CartesianArm(FakeCartesian(start=(100.0, 0.0, 180.0)), 180, None,
                           Limits.parse("0,400,-100,100,0,200"), speed_mm_s=5000)
    watched.gateway.camera = CameraBody((0, 0, 50), 30)
    assert not watched.point_at(150, 0) and "camera" in watched.last_error
    assert watched.backend.history == []

    faulted = CartesianArm(FakeCartesian(), 150, None, Limits.parse("0,600,-300,300,0,400"), speed_mm_s=5000)
    faulted.state, faulted.last_error = "error", "protective stop"
    assert not faulted.point_at(350, 0)
    assert faulted.last_motion.status == "fault" and faulted.backend.history == []
    assert "protective stop" in faulted.last_motion.reason


def test_xarm_connect_does_not_enable_motion_and_a_direct_move_is_refused(monkeypatch):
    import types
    calls = []

    class FakeXArm:
        error_code = 0

        def __init__(self, ip, is_radian=False):
            self.pos = [300.0, 0.0, 200.0, 180.0, 0.0, 15.0]
            calls.append(("init", ip))

        def __getattr__(self, name):
            def _record(*args, **kwargs):
                calls.append((name, args, kwargs))
                return 0
            return _record

        def get_position(self):
            calls.append(("get_position",))
            return 0, list(self.pos)

    mod = types.ModuleType("xarm.wrapper")
    mod.XArmAPI = FakeXArm
    monkeypatch.setitem(sys.modules, "xarm", types.ModuleType("xarm"))
    monkeypatch.setitem(sys.modules, "xarm.wrapper", mod)
    from trova.arm.xarm import XArmBackend
    backend = XArmBackend("192.168.1.185")
    names = [item[0] for item in calls]
    assert "get_position" in names
    assert "motion_enable" not in names and "clean_error" not in names and "set_state" not in names
    with pytest.raises(RuntimeError, match="gateway"):
        backend.move_linear(310, 0, 200, 20)
    assert not any(item[0] == "set_position" for item in calls)
    approve_bringup_speed(BRINGUP_MAX_SPEED_MM_S)
    with pytest.raises(RuntimeError):
        approve_bringup_speed(BRINGUP_MAX_SPEED_MM_S + 1)


def test_sim_jog_and_api_feedback(tmp_path, monkeypatch):
    arm = SimArm()
    origin = arm.tip[0]
    moved = arm.gateway.plan_relative("backward", 3, execute=True)
    assert moved.status == "executed" and abs(arm.tip[0] - (origin - 3)) < 1e-6
    refused = arm.gateway.plan_relative("up", 3, execute=True)
    assert refused.status == "rejected" and "vertical" in refused.reason
    assert abs(arm.tip[0] - (origin - 3)) < 1e-6

    from test_pipeline import _client
    from trova.history import GROUP_OF
    app, client = _client(tmp_path, monkeypatch, TROVA_ARM="sim")
    state = app.state.trova
    start = state.arm.tip[0]
    body = client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 3}).json()
    assert body["status"] == "executed" and body["requested"]["distance_mm"] == 3
    assert body["software_stop_is_not_safety_rated"] is True and body["collision_avoidance"] is False
    assert abs(state.arm.tip[0] - (start - 3)) < 0.2
    far = client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 500}).json()
    assert far["status"] == "rejected" and far["requested"]["distance_mm"] == 500
    assert abs(state.arm.tip[0] - (start - 3)) < 0.2
    assert client.post("/api/arm/jog", json={"direction": "backward", "distance_mm": 3, "mode": "hardware"}).status_code == 422
    tools = client.get("/api/voice/config").json()["session"]["tools"]
    jog = next(tool for tool in tools if tool["name"] == "jog_relative")
    assert "mode" not in jog["parameters"]["properties"]
    assert "speed" not in jog["parameters"]["properties"]
    snapshot = client.get("/api/state").json()
    assert snapshot["motion"]["mode"] == "simulation" and snapshot["motion"]["collision_avoidance"] is False
    assert snapshot["arm"]["motion_mode"] == "simulation"
    assert any(event["kind"] == "arm_jog" for event in client.get("/api/log").json())
    assert GROUP_OF["arm_jog"] == "arm"
