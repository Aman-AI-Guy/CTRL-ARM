"""Actual numerical orbit planner + speech endpoint + simulated driver. No SDK."""
import json
import time
from pathlib import Path
import numpy as np
import pytest
from trova.scan3d.evidence import workspace
from trova.agent import SKILLS,SKILL_FN


def test_registered_skills_and_compound_speech_use_the_agent():
    from test_voice_workflow import node_binary
    import subprocess
    expected={'start_scan','plan_orbit','observe_from_viewpoint','capture_viewpoint','evaluate_coverage',
              'choose_next_view','reconstruct_object','inspect_reconstruction','request_object_reorientation','finish_task'}
    assert expected<=set(SKILLS) and expected<=set(SKILL_FN)
    code="const s=require('./trova/static/speech.js'); console.log(JSON.stringify(s.parse('Grok, seleziona il mouse rosso e crea un modello 3D', ['stop'])));"
    parsed=json.loads(subprocess.check_output([node_binary(),'-e',code],text=True))
    assert parsed['cmd']=='agent'


def test_speech_scan_missing_measurements_and_stop_are_truthful(tmp_path,monkeypatch):
    from test_voice_workflow import setup_scene
    st,client=setup_scene(tmp_path,monkeypatch,True)
    client.post('/api/select',data={'text':'oggetto 7'})
    result=client.post('/api/agent/task',json={'text':'Scansionalo.'}).json()
    assert result['selected']['track_id']==7 and result['scan_session']
    assert not result['confirmation'] and not result['hardware_moved']
    assert 'Measure object_height_mm' in result['blocker']
    assert client.post('/api/agent/cancel').json()['cancelled']
    assert not client.post('/api/agent/confirm',json={'proposal_id':'not-valid'}).json()['ok']
    assert client.post('/api/orbit/settings',json={'settings':{},'confirm':False}).status_code==400
    assert client.post('/api/orbit/plan').json()['next'] is None


def orbit_scene(tmp_path,monkeypatch):
    from test_voice_workflow import setup_scene
    from trova.scan3d.orbit_planner import recorded_context
    from trova.scan3d.flange import flange_pose
    from trova.arm import CartesianArm,FakeCartesian
    Rotation=pytest.importorskip('scipy.spatial.transform').Rotation
    try: original,context=recorded_context()
    except FileNotFoundError:pytest.skip('Retained orbit absent')
    st,client=setup_scene(tmp_path,monkeypatch,True)
    class SimulatedJoints(FakeCartesian):
        def __init__(self):
            super().__init__();self.q=list(original['q0']);self.calls=[]
        def joints(self):return list(self.q)
        def tcp_mm(self):return tuple(flange_pose(self.q)[:3,3])
        def tcp_pose(self):
            transform=flange_pose(self.q)
            return [*transform[:3,3],*Rotation.from_matrix(transform[:3,:3]).as_euler('xyz',degrees=True)]
        def fk_mm(self,q):return flange_pose(q)[:3,3]
        def move_joints(self,q,speed_deg_s):self.calls.append((list(q),speed_deg_s));self.q=list(q)
        def is_moving(self):return False
        def stop(self):pass
    backend=SimulatedJoints()
    st.arm=CartesianArm(backend,z_hover_mm=300,home_mm=None,limits=context.gateway.limits,speed_mm_s=20)
    st.arm.gateway.camera=context.gateway.camera
    # The retained replay doesn't use a fixed-orientation sphere; configure a fixture prerequisite.
    from trova.motion import CameraBody
    st.arm.gateway.camera=CameraBody((0,0,0),20)
    st.arm.gateway.max_speed_mm_s=20
    st.arm.gateway.joint_limits=context.gateway.joint_limits
    st.vision.hand_eye['matrix']=context.flange_camera.tolist()
    st.vision.table.update(point_mm=[0,0,context.scene.table_z],normal=[0,0,1])
    st.vision.localize=lambda *a,**k:{'ok':True,'point_mm':context.foot.tolist(),'uncertainty_mm':1,'reason':'mathematical test fixture'}
    config=st.orbits.public()|{'object_height_mm':context.settings.object_height_mm,
        'object_radius_mm':context.settings.object_radius_mm,'tcp_offset_mm':[0]*6,'geometry_confirmed':True,
        'start_azimuth_deg':context.settings.start_azimuth_deg}
    st.orbits.configure(config,True)
    st.inventory['snapshot']['pose']=st.robot_telemetry()
    client.post('/api/select',data={'text':'oggetto 7'})
    return st,client,backend


def test_joint_orbit_specific_confirmation_executes_only_simulation(tmp_path,monkeypatch):
    st,client,backend=orbit_scene(tmp_path,monkeypatch)
    planned=client.post('/api/agent/task',json={'text':'Scansionalo.'}).json()
    assert planned['confirmation'],planned
    assert not backend.calls and planned['scan_plan']['next']['validated']
    proposal=planned['confirmation']['id']
    moved=client.post('/api/agent/confirm',json={'proposal_id':proposal}).json()
    assert backend.calls,moved
    assert moved['simulation_moved'] and not moved['hardware_moved']
    assert len(backend.calls)==1
    assert 'simulazione' in moved['spoken_message']
    assert not client.post('/api/agent/confirm',json={'proposal_id':proposal}).json()['ok']


def test_orbit_confirmation_is_invalidated_by_settings_and_stop(tmp_path,monkeypatch):
    st,client,backend=orbit_scene(tmp_path,monkeypatch)
    planned=client.post('/api/agent/task',json={'text':'Scansionalo.'}).json()
    assert planned['confirmation'],planned
    st.orbits.settings.distance_mm+=1
    denied=client.post('/api/agent/confirm',json={'proposal_id':planned['confirmation']['id']}).json()
    assert denied['status']=='refused' and not backend.calls
    assert 'changed' in denied['reason']
    planned=client.post('/api/agent/task',json={'text':'Scansionalo.'}).json()
    assert planned['confirmation'],planned
    client.post('/api/arm/stop')
    denied=client.post('/api/agent/confirm',json={'proposal_id':planned['confirmation']['id']}).json()
    assert denied['status']=='refused' and not backend.calls


def test_confirmed_orbit_captures_relocalizes_and_proposes_next_view(tmp_path,monkeypatch):
    from trova.server import AgentHost
    st,client,backend=orbit_scene(tmp_path,monkeypatch)
    frame=np.random.default_rng(12).integers(30,220,(800,1280,3),dtype=np.uint8)
    def read(host):
        st.inventory['snapshot']['pose']=st.robot_telemetry()
        return frame,{'kind':'pi','requested_at':time.time(),'width':1280,'height':800}
    monkeypatch.setattr(AgentHost,'read_frame',read)
    first=client.post('/api/agent/task',json={'text':'Scansionalo.'}).json()
    assert first['confirmation'],first
    result=client.post('/api/agent/confirm',json={'proposal_id':first['confirmation']['id']}).json()
    assert len(backend.calls)==1
    assert result['confirmation'] and result['confirmation']['id']!=first['confirmation']['id'],result
    assert result['workflow_state']=='awaiting_operator_confirmation'
    assert result['simulation_moved'] and not result['hardware_moved']
    assert len([v for v in st.scans._session(None)['views'] if v['accepted']])==1
    skills=[row['skill'] for row in result['steps']]
    assert skills.count('locate_object')==2 and skills.count('inspect_scene')==2
    assert 'evaluate_coverage' in skills


def test_voice_retains_next_view_proposal_but_requires_another_yes():
    from test_voice_workflow import node_binary
    import subprocess
    script=r'''
const assert=require('assert');
const voice=require('./trova/static/voice.js');
let confirms=[];
const p=id=>({id,expires_at:Date.now()/1000+60});
global.fetch=async (url,opts)=>({ok:true,json:async()=>{
  if(url==='/api/agent/task')return {confirmation:p('first')};
  if(url==='/api/agent/confirm'){
    confirms.push(JSON.parse(opts.body).proposal_id);
    return {confirmation:confirms.length===1?p('second'):null};
  }
  return {};
}});
voice.configure({stop_words:['stop']});
(async()=>{
  await voice.tools.run_vision_task({text:'Scansionalo.'});
  await voice.onEvent({type:'conversation.item.input_audio_transcription.completed',transcript:'Sì, procedi.'});
  await voice.tools.confirm_motion({proposal_id:'first'});
  assert.equal((await voice.tools.confirm_motion({proposal_id:'second'})).status,'refused');
  assert.deepEqual(confirms,['first']);
  await voice.onEvent({type:'conversation.item.input_audio_transcription.completed',transcript:'Sì, procedi.'});
  await voice.tools.confirm_motion({proposal_id:'second'});
  assert.deepEqual(confirms,['first','second']);
})().catch(e=>{console.error(e);process.exit(1);});
'''
    subprocess.run([node_binary(),'-e',script],check=True,capture_output=True,text=True)
