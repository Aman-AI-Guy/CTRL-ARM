import math

import pytest

from trova.arm.cartesian import Limits
from trova.mirror_demo import build_path, build_y_shuttle, execute


def test_y_shuttle_is_two_200_mm_cycles_with_fixed_x_and_z():
    limits = Limits.parse("-500,500,-500,500,200,1000")
    start = (17.613, -485.166, 300)
    path = build_y_shuttle(start, 200, 2, limits)
    assert len(path) == 9 and path[0] == path[4] == path[8] == start
    assert path[2] == path[6] == (17.613, -285.166, 300)
    assert all(p[0] == start[0] and p[2] == start[2] for p in path)
    assert sum(math.dist(a,b) for a,b in zip(path,path[1:])) == 800
    assert all(math.dist(a,b) <= 100 for a,b in zip(path,path[1:]))


def test_y_shuttle_refuses_outside_workspace_and_invalid_cycles():
    limits = Limits.parse("-500,500,-500,500,200,1000")
    for stroke, cycles in ((-200,2), (float("nan"),2), (301,2), (200,0), (200,2.5)):
        with pytest.raises(ValueError):
            build_y_shuttle((17,-485,300), stroke, cycles, limits)


def test_positive_x_mirror_path_keeps_corners_inside_operator_cylinder():
    start = (17.613, -485.166, 252.098)
    limits = Limits.parse("cylinder:500,200,1000")
    path = build_path(start, 300, limits)
    assert path[0] == start
    assert path[1] == (start[0], start[1], 300)
    assert path[-1] == (start[0], -start[1], start[2])
    assert max(p[0] for p in path) > 480
    for point in path:
        assert limits.check(*point) is None and point[0] >= 0
    for a, b in zip(path, path[1:]):
        assert sum(abs(x-y) > 1e-8 for x, y in zip(a, b)) == 1
        assert math.dist(a, b) < 50


def test_path_rejects_bad_start_height_and_radius():
    limits = Limits.parse("cylinder:500,200,1000")
    for start, transit in (((0, -600, 250), 300), ((0, -480, 199), 300),
                           ((0, -480, 250), 1001), ((0, -480, 250), 200),
                           ((0, 480, 250), 300)):
        with pytest.raises(ValueError):
            build_path(start, transit, limits)


def test_execute_aborts_remaining_steps_on_rejection(monkeypatch):
    import trova.mirror_demo as demo
    pose = [10, -400, 250, 180, 0, 0]
    monkeypatch.setattr(demo, "stationary_pose", lambda base: {"pose6": pose, "cartesian_target_orientation_deg": pose[3:]})
    calls = []
    def fake_api(base, path, body=None):
        calls.append((path, body))
        if body is None:
            return {"mode": "hardware", "workspace_limits": "cylinder:500,200,1000"}
        if path == "/api/safety":
            return {"ok": True}
        if path == "/api/arm/jog":
            return {"ok": False, "status": "rejected", "reason": "STOP latched"}
        return {"ok": True}
    monkeypatch.setattr(demo, "api", fake_api)
    with pytest.raises(RuntimeError, match="STOP latched"):
        execute("mock", {"start_pose": pose, "path": [pose[:3], [10,-400,300], [20,-400,300]],
                         "workspace_limits": "cylinder:500,200,1000", "speed_cap_mm_s": 5})
    assert sum(path == "/api/arm/jog" for path, _ in calls) == 1
    assert calls[-1][0] == "/api/arm/stop"


def test_execute_refuses_changed_start_before_setting_or_moving(monkeypatch):
    import trova.mirror_demo as demo
    monkeypatch.setattr(demo, "stationary_pose", lambda base: {"pose6": [20,-400,250,180,0,0], "cartesian_target_orientation_deg": [180,0,0]})
    monkeypatch.setattr(demo, "api", lambda *a, **k: pytest.fail("No requests allowed after start changed"))
    with pytest.raises(RuntimeError, match="changed position"):
        execute("mock", {"start_pose": [10,-400,250,180,0,0]})


def test_execute_verifies_each_checkpoint_and_final_pose(monkeypatch, capsys):
    import trova.mirror_demo as demo
    from trova.motion import DIRECTION_AXIS_SIGN
    pose = [10., -400., 250., 180., 0., 0.]
    monkeypatch.setattr(demo, "stationary_pose", lambda base: {"pose6": pose.copy(), "cartesian_target_orientation_deg": pose[3:].copy()})
    def fake_api(base, path, body=None):
        if body is None:
            return {"mode": "hardware", "workspace_limits": "cylinder:500,200,1000"}
        if path == "/api/safety":
            return {"ok": True}
        assert path == "/api/arm/jog"
        axis, sign = DIRECTION_AXIS_SIGN[body["direction"]]
        pose[axis] += sign * body["distance_mm"]
        return {"ok": True, "status": "executed"}
    monkeypatch.setattr(demo, "api", fake_api)
    execute("mock", {"start_pose": pose.copy(), "path": [[10,-400,250],[10,-400,300],[20,-400,300]],
                     "final_xyz": [20,-400,300], "workspace_limits": "cylinder:500,200,1000", "speed_cap_mm_s": 5})
    assert pose[:3] == [20,-400,300]
    assert '"status": "completed"' in capsys.readouterr().out


def test_stale_driver_yaw_is_rejected_before_motion():
    from trova.mirror_demo import check_driver_orientation
    with pytest.raises(RuntimeError, match="differs"):
        check_driver_orientation({"pose6": [17,-485,252,180,0,34.347],
                                  "cartesian_target_orientation_deg": [180,0,36.02]})
    check_driver_orientation({"pose6": [17,-485,300,-180,0,36.02],
                              "cartesian_target_orientation_deg": [180,0,36.02]})


def test_decimal_100_mm_return_completes_without_rounding_abort(monkeypatch):
    import trova.mirror_demo as demo
    from trova.motion import DIRECTION_AXIS_SIGN
    pose = [17.613, -386.593, 300., 180., 0., 36.02]
    initial = pose.copy()
    path = build_y_shuttle(pose[:3],200,2,Limits.parse("-500,500,-500,500,200,1000"))
    assert abs(path[3][1]-round(path[2][1],3)) > 100  # Reported telemetry is rounded to 3 decimals.
    monkeypatch.setattr(demo,"stationary_pose",lambda base: {"pose6":pose.copy(),"cartesian_target_orientation_deg":pose[3:].copy()})
    moves=[]
    def fake_api(base, route, body=None):
        if body is None:
            return {"mode":"hardware","workspace_limits":"-500,500,-500,500,200,1000"}
        if route == "/api/safety":
            return {"ok":True}
        assert route == "/api/arm/jog"
        axis,sign=DIRECTION_AXIS_SIGN[body["direction"]]
        pose[axis]=round(pose[axis]+sign*body["distance_mm"],3)
        moves.append(body)
        return {"ok":True,"status":"executed"}
    monkeypatch.setattr(demo,"api",fake_api)
    execute("mock", {"start_pose":initial,"path":path,"final_xyz":path[-1],
                     "workspace_limits":"-500,500,-500,500,200,1000","speed_cap_mm_s":20})
    assert len(moves)==8 and pose[:3]==pytest.approx(initial[:3])


def test_checkpoint_continuation_preserves_original_goal_and_remaining_cycles(monkeypatch):
    import trova.mirror_demo as demo
    original = [17.613,-386.593,300.,180.,0.,36.02]
    path=build_y_shuttle(original[:3],200,2,Limits.parse("-500,500,-500,500,200,1000"))
    robot={"pose6":list(path[2])+original[3:], "joints":[0]*5,
           "cartesian_target_orientation_deg":original[3:]}
    monkeypatch.setattr(demo,"stationary_pose",lambda base:robot)
    monkeypatch.setattr(demo,"api",lambda *a,**k:{"workspace_limits":"-500,500,-500,500,200,1000","max_speed_mm_s":20})
    monkeypatch.setattr(demo,"validate_ik_path",lambda p,pose,joints:120)
    source={"start_pose":original,"path":path,"final_xyz":path[-1],
            "workspace_limits":"-500,500,-500,500,200,1000","speed_cap_mm_s":20}
    remaining=demo.resume_plan("mock",source,2)
    assert len(remaining["path"])==7 and remaining["distance_mm"]==pytest.approx(600)
    assert remaining["final_xyz"]==path[-1] and remaining["original_start_pose"]==original
    robot["pose6"][1] += 10
    with pytest.raises(RuntimeError,match="checkpoint"):
        demo.resume_plan("mock",source,2)
