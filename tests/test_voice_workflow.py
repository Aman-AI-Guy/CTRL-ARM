"""Speech adapters -> real task endpoint -> simulated gateway. No controller connections."""

import subprocess
import json
import time
from pathlib import Path

import numpy as np
import pytest

from trova.workflow import feedback


def node_binary():
    import shutil
    import importlib.util
    node = shutil.which('node')
    playwright = importlib.util.find_spec('playwright')
    if not node and playwright:
        bundled = Path(playwright.origin).parent / 'driver' / 'node'
        node = str(bundled) if bundled.exists() else None
    if not node:
        pytest.skip('node unavailable')
    return node


def setup_scene(tmp_path, monkeypatch, calibrated=False):
    from test_pipeline import _client
    from test_agent import _arm
    from test_geom import _fixture_store
    from trova.boxes import Box
    app, client = _client(tmp_path, monkeypatch)
    st = app.state.trova
    st.arm = _arm()
    if calibrated:
        st.vision = _fixture_store()
    st.observe_clearance_mm = 150.0
    st.source.info = lambda: {"kind": "pi", "size": [1280, 800]}
    st.frame_size = [1280, 800]
    st.settle_s = 0

    def robot():
        pose = list(st.arm.backend.tcp_pose())
        return {"xyz_mm": pose[:3], "pose6": pose, "source": "telemetry", "moving": False,
                "joints": list(st.arm.backend.joints()), "error_code": 0, "state_code": 2}

    st.robot_telemetry = robot
    obj = {"track_id": 7, "box_id": 1, "name": "mouse rosso", "color": "red",
           "box": [800, 260, 880, 340], "center": [840, 300], "confidence": .95,
           "association": "same_view", "identity_uncertain": False, "ambiguous": False,
           "label_authoritative": True}
    snap = {"id": "fixture", "objects": [obj], "pose": robot(), "width": 1280, "height": 800,
            "grok_mode": "live", "camera_kind": "pi", "authoritative": True}
    st.scene_frame = np.zeros((800, 1280, 3), np.uint8)
    st.scene_boxes = [Box(800, 260, 880, 340, id=1)]
    st.inventory = {"snapshot": snap}
    from trova.server import AgentHost
    monkeypatch.setattr(AgentHost, "redetect", lambda _host, frame: [obj])
    monkeypatch.setattr(AgentHost, "snapshot", lambda _host: snap)
    monkeypatch.setattr(AgentHost, "read_frame", lambda host: (st.scene_frame, {"requested_at": time.time()}))
    # Synthetic mathematical calibration, never persisted or represented as physical qualification.
    if calibrated:
        st.vision.localize = lambda *a, **kw: {"ok": True, "point_mm": [100, 50, 0], "reason": "fixture", "uncertainty_mm": 1}
    return st, client


def select_and_plan(st, client):
    selected = client.post('/api/select', data={"text": "oggetto 7"}).json()
    assert selected["object"]["name"] == "mouse rosso"
    assert st.selected_context["selected"]["track_id"] == 7
    before = list(st.arm.tcp_mm())
    planned = client.post('/api/agent/task', json={"text": "Ora posizionati sopra."}).json()
    assert planned["selected"]["track_id"] == 7
    assert np.allclose(st.arm.tcp_mm(), before)
    return planned


def test_selection_context_and_calibration_refusal(tmp_path, monkeypatch):
    st, c = setup_scene(tmp_path, monkeypatch)
    result = select_and_plan(st, c)
    assert result["workflow_state"] == "localization_blocked"
    assert not result["hardware_moved"] and not result["confirmation"]
    assert "mouse rosso" in result["spoken_message"] and "calibrazione" in result["spoken_message"]
    assert result["blocker"] and not result["motion_proposed"]
    denied = c.post('/api/agent/confirm', json={"proposal_id": "invented"}).json()
    assert denied["status"] == "refused" and not denied["hardware_moved"]


def test_valid_simulation_requires_specific_confirmation(tmp_path, monkeypatch):
    st, c = setup_scene(tmp_path, monkeypatch, True)
    result = select_and_plan(st, c)
    assert result["workflow_state"] == "awaiting_operator_confirmation"
    receipt = result["confirmation"]
    assert receipt["object_id"] == 7 and receipt["target_mm"] == [100, 50, 150]
    assert result["proposal"]["allow_motion"] and not result["proposal"]["execution_authorized"]
    moved = c.post('/api/agent/confirm', json={"proposal_id": receipt["id"]}).json()
    assert moved["simulation_moved"] and not moved["hardware_moved"]
    assert "simulazione" in moved["spoken_message"]
    assert np.allclose(st.arm.tcp_mm(), [100, 50, 150], atol=1)
    assert moved["evaluation"]["robot_reached_planned_position"]
    assert not c.post('/api/agent/confirm', json={"proposal_id": receipt["id"]}).json()["ok"]


@pytest.mark.parametrize('unknown', [False, True])
def test_failed_execution_reports_partial_or_unknown_movement(tmp_path, monkeypatch, unknown):
    from trova.server import AgentHost
    st, c = setup_scene(tmp_path, monkeypatch, True)
    planned = select_and_plan(st, c)
    monkeypatch.setattr(AgentHost, 'execute', lambda host, target: {
        'ok': False, 'motion_observed': not unknown, 'movement_status_unknown': unknown,
        'reason': 'Execution interrupted; final position unavailable' if unknown else 'Execution interrupted after partial travel'})
    result = c.post('/api/agent/confirm', json={'proposal_id': planned['confirmation']['id']}).json()
    assert not result['goal_achieved'] and not result['hardware_moved']
    assert result['movement_status_unknown'] is unknown
    if unknown:
        assert 'Non posso verificare' in result['spoken_message']
    else:
        assert result['simulation_moved'] and 'partial travel' in result['spoken_message']


@pytest.mark.parametrize('change', ['expired', 'pose', 'rotation', 'joints', 'controller_state', 'invalid_pose', 'calibration', 'safety', 'stop', 'cancel', 'selection', 'scene'])
def test_stale_confirmation_never_moves(tmp_path, monkeypatch, change):
    st, c = setup_scene(tmp_path, monkeypatch, True)
    result = select_and_plan(st, c)
    receipt = result["confirmation"]
    if change == 'expired':
        st.pending_motion["confirmation"]["expires_at"] = time.time() - 1
    elif change == 'pose':
        original = st.robot_telemetry
        def changed():
            robot = original()
            robot["pose6"][0] += 10
            return robot
        st.robot_telemetry = changed
    elif change == 'rotation':
        original = st.robot_telemetry
        def changed():
            robot = original()
            robot["pose6"][5] += 1
            return robot
        st.robot_telemetry = changed
    elif change in {'joints', 'controller_state', 'invalid_pose'}:
        original = st.robot_telemetry
        def changed():
            robot = original()
            if change == 'joints':
                robot['joints'][0] += 1
            elif change == 'controller_state':
                robot['state_code'] = 3
            else:
                robot['pose6'][0] = float('nan')
            return robot
        st.robot_telemetry = changed
    elif change == 'calibration':
        st.vision.intrinsics = None
    elif change == 'safety':
        st.arm.gateway.max_speed_mm_s = 5
    elif change == 'stop':
        c.post('/api/arm/stop')
    elif change == 'cancel':
        c.post('/api/agent/cancel')
    elif change == 'selection':
        c.post('/api/select', data={"text": "oggetto 7"})
    elif change == 'scene':
        st.inventory["snapshot"]["id"] = "new scene"
    before = list(st.arm.tcp_mm())
    denied = c.post('/api/agent/confirm', json={"proposal_id": receipt["id"]}).json()
    assert denied["status"] == "refused" and not denied["hardware_moved"]
    assert np.allclose(st.arm.tcp_mm(), before)


def test_dry_run_has_no_executable_confirmation(tmp_path, monkeypatch):
    st, c = setup_scene(tmp_path, monkeypatch, True)
    st.arm.gateway.mode = 'dry_run'
    result = select_and_plan(st, c)
    assert result["workflow_state"] == 'hardware_authorization_missing'
    assert result["motion_proposed"] and not result["confirmation"]
    assert "autorizzazione hardware" in result["spoken_message"]


def test_plain_yes_cannot_generate_a_motion(tmp_path, monkeypatch):
    st, c = setup_scene(tmp_path, monkeypatch, True)
    before = list(st.arm.tcp_mm())
    for text in ('Sì, procedi.', 'yes', 'ok'):
        assert c.post('/api/agent/task', json={"text": text}).json()["status"] == 'refused'
    assert np.allclose(st.arm.tcp_mm(), before)


def test_scan_reference_preserves_selection_without_approaching(tmp_path, monkeypatch):
    st, c = setup_scene(tmp_path, monkeypatch)
    c.post('/api/select', data={"text": "oggetto 7"})
    before = list(st.arm.tcp_mm())
    result = c.post('/api/agent/task', json={"text": "Scansionalo."}).json()
    assert result["selected"]["track_id"] == 7
    assert not result["motion_proposed"] and not result["hardware_moved"]
    assert np.allclose(st.arm.tcp_mm(), before)


def test_legacy_find_retains_its_target_for_a_followup_approach(tmp_path, monkeypatch):
    from trova.server import AgentHost
    st, c = setup_scene(tmp_path, monkeypatch)
    st.frame = lambda: st.scene_frame
    st.proposals = lambda frame: (st.scene_boxes, False)
    monkeypatch.setattr(AgentHost, 'snapshot', lambda host: st.inventory['snapshot'])
    found = c.post('/api/find', data={'text': 'the black one', 'move': 'false'}).json()
    assert found['result']['status'] == 'found'
    chosen = st.selected_context['selected']
    result = c.post('/api/agent/task', json={'text': 'Ora posizionati sopra.'}).json()
    assert result['selected']['track_id'] == chosen['track_id']
    assert result['workflow_state'] == 'localization_blocked'
    assert not result['hardware_moved']


def test_spoken_reporting_never_promotes_unverified_movement():
    unverified = feedback({"status": "completed", "hardware_moved": True, "workflow_state": "movement_unverified"}, 'posizionati')
    assert "non è ancora verificato" in unverified["spoken_message"]
    selected = feedback({"selected": {"name": "mouse rosso"}, "workflow_state": "object_selected"}, 'seleziona')
    assert selected["spoken_message"] == 'Ho selezionato mouse rosso.'


def test_browser_voice_adapter_and_parser(tmp_path):
    node = node_binary()
    static = Path(__file__).resolve().parents[1] / 'trova' / 'static'
    script = r'''
const assert = require('assert');
const speech = require(process.argv[1] + '/speech.js');
const voice = require(process.argv[1] + '/voice.js');
let calls = [], messages = [];
global.fetch = async (url, opts) => {
  calls.push([url, opts && opts.body]);
  return {ok: true, json: async () => url === '/api/agent/task' ?
    {workflow_state: 'localization_blocked', spoken_message: 'Manca la calibrazione.', confirmation: null, hardware_moved: false} : {cancelled: true}};
};
global.TrovaSpeech = {...speech, say: m => messages.push(m)};
voice.configure({stop_words: ['stop']});
(async () => {
  assert.equal(speech.parse('Ora posizionati sopra.', []).cmd, 'agent');
  assert.equal(speech.parse('Sì, procedi.', []).cmd, 'confirm');
  assert.equal(speech.parse('Scansionalo.', []).cmd, 'agent');
  assert.equal(speech.parse('annulla', []).cmd, 'cancel');
  await voice.onEvent({type:'response.function_call_arguments.done', name:'run_vision_task', arguments:'{"text":"Ora posizionati sopra."}'});
  await voice.onEvent({type:'response.done'});
  assert(calls.some(c => c[0] === '/api/agent/task' && JSON.parse(c[1]).text === 'Ora posizionati sopra.'));
  assert(messages.includes('Manca la calibrazione.'));
  const denied = await voice.tools.confirm_motion({proposal_id:'fake'});
  assert.equal(denied.status, 'refused');
  assert(!calls.some(c => c[0] === '/api/agent/confirm'));
  // Speculative audio cannot reach the speaker.
  await voice.onEvent({type:'response.output_audio.delta', delta:'invented'});
  await voice.tools.stop_arm();
  assert(calls.some(c => c[0] === '/api/arm/stop'));
})().catch(e => {console.error(e); process.exit(1);});
'''
    subprocess.run([node, '-e', script, str(static)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('calibrated', [False, True])
def test_spoken_sequence_through_browser_tools_and_real_backend(tmp_path, monkeypatch, calibrated):
    st, client = setup_scene(tmp_path, monkeypatch, calibrated)
    static = Path(__file__).resolve().parents[1] / 'trova' / 'static'
    script = r'''
const assert = require('assert');
const readline = require('readline');
const input = readline.createInterface({input: process.stdin});
const pending = [];
input.on('line', line => pending.shift()(JSON.parse(line)));
global.fetch = (url, opts) => new Promise(resolve => {
  pending.push(d => resolve({ok: d.code < 400, json: async () => d.body}));
  let body = opts && opts.body;
  const isJson = opts && opts.headers && opts.headers['Content-Type'] === 'application/json';
  process.stdout.write(JSON.stringify({url, body: isJson ? JSON.parse(body) : body ? Object.fromEntries(body.entries()) : {}, isJson}) + '\n');
});
const speech = require(process.argv[1] + '/speech.js');
const voice = require(process.argv[1] + '/voice.js');
voice.configure({stop_words: ['stop']});
(async () => {
  const selection = speech.parse('Seleziona oggetto 7', []);
  const selected = await voice.tools.select_object({description: selection.text});
  assert.equal(selected.track_id, 7);
  const movement = speech.parse('Ora posizionati sopra.', []);
  const plan = await voice.tools.run_vision_task({text: movement.text});
  assert.equal(plan.selected.track_id, 7);
  if (plan.confirmation) {
    const no = await voice.tools.confirm_motion({proposal_id: plan.confirmation.id});
    assert.equal(no.status, 'refused'); // The model cannot confirm without a user yes.
    await voice.onEvent({type:'conversation.item.input_audio_transcription.completed', transcript:'Sì, procedi.'});
    const result = await voice.tools.confirm_motion({proposal_id: plan.confirmation.id});
    assert(result.simulation_moved && !result.hardware_moved);
  } else {
    assert.equal(plan.workflow_state, 'localization_blocked');
    assert(plan.spoken_message.includes('calibrazione'));
    const no = await voice.tools.confirm_motion({proposal_id:'invented'});
    assert.equal(no.status, 'refused');
  }
  await voice.tools.stop_arm();
  process.stdout.write(JSON.stringify({done:true}) + '\n');
  input.close();
})().catch(e => {console.error(e); process.exit(1);});
'''
    process = subprocess.Popen([node_binary(), '-e', script, str(static)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in process.stdout:
            request = json.loads(line)
            if request.get('done'):
                break
            kwargs = {'json' if request.get('isJson') else 'data': request['body']}
            response = client.post(request['url'], **kwargs)
            process.stdin.write(json.dumps({'code': response.status_code, 'body': response.json()}) + '\n')
            process.stdin.flush()
        process.wait(timeout=10)
        assert process.returncode == 0, process.stderr.read()
        assert not st._process_can_move()
        assert np.allclose(st.arm.tcp_mm(), [100, 50, 150] if calibrated else [300, 0, 200], atol=1)
    finally:
        if process.poll() is None:
            process.kill()
