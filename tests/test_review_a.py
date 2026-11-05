"""Review-A regressions. All controllers, cameras and model calls are offline fakes."""
import sys
import threading
import types

import pytest

from trova.arm.cartesian import CartesianArm, FakeCartesian, Limits
from trova.arm.xarm import XArmBackend
from trova.arm import xarm_config


class ConfigController:
    tcp_offset, tcp_load, collision_sensitivity = [0] * 6, [0, [0, 0, 0]], 3
    is_collision_rebound, self_collision_params, is_fence_mode = True, [False], False

    def __init__(self, ip='fake'):
        self.calls, self.pos = [], [300, 0, 300, 180, 0, 0]
        self.core = self

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args))
            return [0] if name in ('set_collis_sens', 'set_collision_rebound', 'set_fence_mode', 'save_conf') else 0
        return call

    def get_position(self):
        return 0, self.pos

    def get_reduced_states(self):
        return 0, [False, [0] * 6, 50, 20, [[-360, 360]] * 5, False, False]

    def get_mount_direction(self):
        return 0, [0, 0]

    def get_state(self):
        return 0, 4


def test_issue_15_bringup_reads_fence_flag():
    from trova.arm.bringup import Bringup
    from test_features import SimXArm
    api = SimXArm([0])
    api.get_reduced_states = lambda: (0, [False, [0]*6, 50, 20, [[-360, 360]]*5, False, False])
    said = []
    Bringup(api, say=said.append).step_connect()
    assert any('safety box on: False' in s for s in said)


def test_issue_16_config_handles_lists_without_restart():
    api = ConfigController()
    xarm_config.main(['fake', '--apply', '--limits', '0,600,-300,300,50,500'],
                     api_factory=lambda ip: api, ask=lambda q: 'YES')
    names = [n for n, _ in api.calls]
    assert 'save_conf' in names and 'set_collis_sens' in names
    assert not {'set_state', 'set_collision_sensitivity', 'motion_enable'} & set(names)


def test_issue_18_bringup_interrupt_stops(monkeypatch, capsys):
    from trova.arm import bringup
    api = ConfigController()
    monkeypatch.setitem(sys.modules, 'xarm.wrapper', types.SimpleNamespace(XArmAPI=lambda *a, **k: api))
    monkeypatch.setattr(sys, 'argv', ['bringup', 'fake'])
    def interrupted(*a):
        raise KeyboardInterrupt
    monkeypatch.setattr(bringup.Bringup, 'run', interrupted)
    bringup.main()
    assert api.calls == [('set_state', (4,)), ('disconnect', ())]
    assert 'STOP accepted' in capsys.readouterr().out


def test_issue_19_timeout_tracks_slow_cartesian_and_joint_moves(monkeypatch):
    import trova.arm.cartesian as cart
    clock = [0.0]
    monkeypatch.setattr(cart.time, 'time', lambda: clock[0])
    monkeypatch.setattr(cart.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    class Slow(FakeCartesian):
        def move_linear(self, x, y, z, speed):
            self.end = clock[0] + 50
            self.p = [x, y, z]
        def is_moving(self):
            return clock[0] < self.end
        def tcp_mm(self):
            return tuple(getattr(self, 'p', [0, 0, 0]))
        def joints(self):
            return getattr(self, 'q_slow', [0]*5)
        def move_joints(self, q, speed):
            self.end = clock[0] + 50
            self.q_slow = q
    b = Slow()
    arm = CartesianArm(b, 0, (0,0,0), Limits.parse('-2000,2000,-2000,2000,-2000,2000'), speed_mm_s=10,
                       joint_speed_deg_s=5)
    assert arm.point_at(500, 0), arm.last_error
    assert arm.go_joints([250, 0, 0, 0, 0]), arm.last_error


def test_issue_44_rejects_bad_tray_approach(monkeypatch):
    from trova.arm.skills import TrayParams
    for value in ('0,0', 'nan,1', 'inf,0'):
        monkeypatch.setenv('TROVA_TRAY_APPROACH', value)
        with pytest.raises(ValueError, match='finite non-zero'):
            TrayParams.from_env()


def test_issue_46_unreadable_arrival_fails_closed(monkeypatch):
    import trova.arm.cartesian as cart
    monkeypatch.setattr(cart, 'START_GRACE_S', -1)
    class LostPose(FakeCartesian):
        lost = False
        def move_linear(self, *args):
            self.lost = True
        def is_moving(self):
            return False
        def tcp_mm(self):
            if self.lost:
                raise RuntimeError('position timeout')
            return (0, 0, 0)
    arm = CartesianArm(LostPose(), 0, (0,0,0), None)
    assert not arm.point_at(100, 100)
    assert arm.state == 'error' and 'verify arrival' in arm.last_error


def test_issue_47_config_rechecks_new_tcp_and_payload():
    class Offset(ConfigController):
        def set_tcp_offset(self, *args):
            self.pos[2] = -100
            return 0
    api = Offset()
    with pytest.raises(SystemExit, match='after TCP'):
        xarm_config.main(['fake', '--apply', '--tcp', '0,0,400', '--limits', '0,600,-300,300,50,500'],
                         api_factory=lambda ip: api, ask=lambda q: 'YES')
    assert 'set_fence_mode' not in [n for n, _ in api.calls]
    assert 'save_conf' not in [n for n, _ in api.calls]
    for weight in ('25', 'nan', '-1'):
        with pytest.raises(SystemExit, match='payload'):
            xarm_config.main(['fake', '--apply', '--payload-kg', weight, '--limits', '0,600,-300,300,50,500'],
                             api_factory=ConfigController, ask=lambda q: 'YES')


def test_issue_48_stop_does_not_wait_for_sdk_lock():
    b = XArmBackend.__new__(XArmBackend)
    b.lock = threading.Lock()
    stopped = threading.Event()
    b.arm = types.SimpleNamespace(set_state=lambda state: stopped.set() or 0)
    b.lock.acquire()  # a motion SDK call is paused with the driver lock held
    th = threading.Thread(target=b.stop, daemon=True)
    th.start()
    try:
        assert stopped.wait(0.5)
    finally:
        b.lock.release()
        th.join(1)


def test_issue_49_each_move_uses_current_yaw():
    b = XArmBackend.__new__(XArmBackend)
    b.lock, b.accel, b.yaw = threading.Lock(), 500, -40
    sent = []
    b.arm = types.SimpleNamespace(get_state=lambda: (0, 0), get_position=lambda: (0, [300,0,200,180,0,17]),
                                 set_position=lambda **kw: sent.append(kw) or 0)
    b.move_linear(300, 100, 250, 10)
    assert sent[0]['yaw'] == 17 and 'motion_type' not in sent[0]


def test_issue_50_bringup_stop_matches_driver():
    from trova.arm.bringup import Abort, Bringup
    api = ConfigController()
    Bringup(api).stop()
    assert api.calls == [('set_state', (4,))]
    api.set_state = lambda state: [7]
    with pytest.raises(Abort, match='STOP failed'):
        Bringup(api).stop()


def test_issue_11_pending_targets_die_with_their_view(tmp_path, monkeypatch):
    from test_motion_safety import cart_client
    from test_features import _unique_color
    st, c = cart_client(tmp_path, monkeypatch)
    query = f'the {_unique_color(c)} one'
    def find():
        r = c.post('/api/find', data={'text': query, 'move': 'false'})
        assert r.status_code == 200 and r.json()['result']['status'] == 'found'
    find()
    st.calib.points[0][2] += 1  # a mapping change, even without the endpoint
    assert c.post('/api/arm/point_last').status_code == 409
    find()
    assert c.post('/api/upload_scene/clear').status_code == 200
    assert c.post('/api/arm/point_last').status_code in (400, 409)
    find()
    st.override = st.frame()
    find()  # blocked uploaded photo must never become a move after Back to live
    assert st.pending_move is None
    c.post('/api/upload_scene/clear')
    assert c.post('/api/arm/point_last').status_code in (400, 409)
    assert c.post('/api/arm/lift_tray', json={'place_u': 400, 'place_v': 400}).status_code in (400, 409)
    find()
    st.pending_move['_issued'] = 0
    assert c.post('/api/arm/point_last').status_code == 409
    assert st.arm.backend.history == []


def test_issue_12_first_run_find_never_homes(tmp_path, monkeypatch):
    from test_first_run import first_run_client
    st, c = first_run_client(tmp_path, monkeypatch)
    st.arm.backend.pos = [400, 0, 150]
    st.arm.backend.target = [400, 0, 150]
    before = list(st.arm.backend.history)
    r = c.post('/api/find', data={'text': 'red', 'move': 'true'})
    assert r.status_code == 409 and 'Arm home first' in r.json()['error']
    assert st.arm.backend.history == before


def test_issue_17_arm_calibration_needs_redundancy_and_reports_max():
    import numpy as np
    from trova.calibration import Calibration
    pts = [[0,0,0,0], [1000,0,500,0], [1000,800,500,400], [0,800,0,400]]
    with pytest.raises(ValueError, match='at least 5'):
        Calibration(points=pts, source='arm').fit()
    with pytest.raises(ValueError, match='wrong'):
        Calibration(points=pts + [[500,400,400,50]], source='arm').fit()
    c = Calibration(H=np.eye(3), points=[[0,0,0,0], [10,0,30,0]])
    assert c.error_mm() == 20


def test_issue_23_aruco_refuses_uploaded_scene(tmp_path, monkeypatch):
    from test_motion_safety import cart_client
    import trova.server as server
    st, c = cart_client(tmp_path, monkeypatch)
    st.override = st.frame()
    monkeypatch.setattr(server, 'calibrate_aruco', lambda *a: pytest.fail('used uploaded photo'))
    r = c.post('/api/calibration/aruco')
    assert r.status_code == 409 and 'live camera' in r.json()['detail']
    assert st.arm.backend.history == []


def test_issue_24_source_and_survey_pose_bind_calibration(tmp_path, monkeypatch):
    from test_first_run import first_run_client
    from trova.arm import load_poses
    st, c = first_run_client(tmp_path, monkeypatch, TROVA_FIRST_RUN='0')
    st.arm.home_joints = [5,0,0,0,0]
    assert 'survey pose' in st.arm_blocked(interlocks=False)
    st.arm.home_joints = [0]*5
    st.source.info = lambda: {'kind': 'video'}
    assert 'recorded' in st.arm_blocked()
    assert c.post('/api/source', data={'spec': 'demo'}).status_code == 200
    assert st.calib is None and st.pending_move is None and not st.calib_path.exists()
    (tmp_path/'arm_poses.json').write_text('{')
    with pytest.raises(RuntimeError, match='staff must restore'):
        load_poses(tmp_path)


def test_issue_43_refused_lift_does_not_dirty_store(tmp_path, monkeypatch):
    from test_motion_safety import cart_client
    st, c = cart_client(tmp_path, monkeypatch)
    writes = []
    st.lift_store = types.SimpleNamespace(write=writes.append)
    before = list(st.arm.backend.history)
    r = c.post('/api/arm/lift_tray', json={'pick_u': 300, 'pick_v': 300, 'place_u': 20000, 'place_v': 300})
    if st.arm_thread:
        st.arm_thread.join(2)
    assert r.status_code == 409 and not st.lift_interrupted and writes == []
    assert st.arm.backend.history == before


def test_issue_45_grid_targets_obey_limits_and_calibration(tmp_path, monkeypatch):
    from test_pipeline import _client
    from trova.calibration import Calibration
    from trova.arm import TaughtGridArm, FakeJointBackend
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM='grid:fake', TROVA_ARM_LIMITS='0,600,0,400,0,300')
    st = app.state.trova
    st.arm.record_home()
    st.arm.record(0, 0)
    st.frame()
    assert c.post('/api/arm/point', json={'u': 10, 'v': 10}).status_code == 409
    st.calib.source = 'arm'
    assert c.post('/api/arm/point', json={'x_mm': 5000, 'y_mm': 100}).status_code == 409
    assert not st.arm.point_at(float('nan'), 10) and st.arm.backend.q == [0,0]
    class Physical(FakeJointBackend):
        pass
    grid = TaughtGridArm(Physical(), tmp_path/'other.json', limits=st.arm.limits)
    grid.home_q = [0,0]
    grid.home()
    assert grid.state == 'refused' and 'kinematics' in grid.last_error


def test_issue_21_stop_has_capacity_when_default_pool_is_full(tmp_path, monkeypatch):
    import anyio
    import httpx
    from test_pipeline import _client
    app, _ = _client(tmp_path, monkeypatch)
    async def saturated():
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 1
        async with limiter:  # every shared worker token is occupied by a model-bound request
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as c:
                with anyio.fail_after(1):
                    r = await c.post('/api/arm/stop')
                assert r.status_code == 200 and app.state.trova.stop_latched.is_set()
    anyio.run(saturated)


def test_issue_22_nonfinite_http_coordinates_never_start_motion(tmp_path, monkeypatch):
    from test_pipeline import _client
    from trova.server import ArmBusy
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    for value in ('NaN', 'Infinity', '-Infinity'):
        for path, data in [('/api/arm/point', '{"x_mm":VALUE,"y_mm":100}'),
                           ('/api/calibration/click', '{"u":VALUE,"v":1,"x_mm":2,"y_mm":3}'),
                           ('/api/arm/lift_tray', '{"place_u":1,"place_v":VALUE}')]:
            r = c.post(path, content=data.replace('VALUE', value), headers={'content-type': 'application/json'})
            assert r.status_code == 422, (path, r.text)
    with pytest.raises(ArmBusy, match='finite'):
        st.start_arm(st.arm.point_at, float('nan'), 10)
    assert st.arm_thread is None and c.get('/api/state').status_code == 200


def test_issue_32_signoff_is_bound_and_single_use(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    st.last_check = {'event_id': 'B', 'result': {'line_clear': True}}
    assert c.post('/api/signoff', data={'check_id': 'A', 'operator': 'Ana'}).status_code == 409
    assert not st.last_check.get('signoff')
    assert c.post('/api/signoff', data={'check_id': 'B', 'operator': 'Ana', 'decision': 'reject'}).status_code == 200
    assert c.post('/api/signoff', data={'check_id': 'B', 'operator': 'Ana', 'decision': 'approve'}).status_code == 409
    assert st.last_check['signoff']['decision'] == 'reject'
    assert c.post('/api/signoff', data={'operator': 'Ana'}).status_code == 422


def test_issue_38_http_bad_inputs_have_client_errors(tmp_path, monkeypatch):
    from test_pipeline import _client
    import trova.server as server
    app, c = _client(tmp_path, monkeypatch)
    for width in (0, -1):
        assert c.get('/api/frame.jpg', params={'max_w': width}).status_code == 422
    st = app.state.trova
    st.label_wait = {'frame': 'example.jpg'}
    assert c.post('/api/label', json={}).status_code == 400
    assert c.post('/api/judge', json={'false_alarms': -1}).status_code == 422
    assert c.post('/api/judge', json={'missed': -1}).status_code == 422
    monkeypatch.setattr(server, 'MAX_UPLOAD_BYTES', 10)
    assert c.post('/api/upload_scene', files={'photo': ('large.jpg', b'x'*11)}).status_code == 413
    assert c.post('/api/find', files={'photo': ('large.jpg', b'x'*11)}).status_code == 413
    _, secured = _client(tmp_path/'secured', monkeypatch, TROVA_TOKEN='secret')
    assert secured.get('/api/state', params={'token': 'é'}).status_code == 401


def test_issue_39_foreign_browser_requests_are_refused(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, TROVA_TOKEN='')
    before = app.state.trova.stop_sequence
    for headers in ({'origin': 'https://attacker.example'}, {'origin': 'null'},
                    {'referer': 'http://attacker.example/page'}, {'sec-fetch-site': 'cross-site'}):
        assert c.post('/api/arm/stop', headers=headers).status_code == 403
    assert app.state.trova.stop_sequence == before
    assert c.post('/api/arm/stop', headers={'origin': 'http://testserver'}).status_code == 200
    assert c.post('/api/arm/stop').status_code == 200  # non-browser CLI


def test_issue_55_clarification_is_not_an_absent_eval_case(tmp_path, monkeypatch):
    from test_pipeline import _client
    from trova.finder import FindResult
    import trova.server as server
    app, c = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(server, 'grok_find', lambda *a: FindResult(status='clarify', clarifying_question='Which pen?'))
    assert c.post('/api/find', data={'text': 'a pen'}).json()['result']['status'] == 'clarify'
    assert c.post('/api/judge', json={'correct': True}).status_code == 409
    assert not (tmp_path/'cases'/'finds.jsonl').exists()
    assert app.state.trova.metrics['find_ok'] == 0 and app.state.trova.metrics['find_wrong'] == 0
