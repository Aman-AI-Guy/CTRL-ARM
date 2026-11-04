"""M1 integration review regressions. All images/controllers/model outputs are synthetic."""
import threading
import time

import cv2
import numpy as np
import pytest

from test_features import _unique_color
from test_motion_safety import cart_client
from test_pipeline import _client
from test_safety_review import AUTH, confirmation, physical_client
from trova.boxes import Box
from trova.finder import FindResult
from trova.inventory import Inventory


def arm_hardware_policy(st, c):
    assert c.post("/api/arm/recover_lift", headers=AUTH, json=confirmation(
        c, fork_freed=True, note="Synthetic fixture; staff checks simulated")).status_code == 200
    assert c.post("/api/arm/reset", headers=AUTH, json=confirmation(c)).status_code == 200


def test_m1_snapshot_does_not_home_an_armed_physical_session(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    arm_hardware_policy(st, c)
    st.arm.backend.pos = [450.0, 100.0, 200.0]
    st.arm.backend.target = list(st.arm.backend.pos)
    result = c.post("/api/inventory")
    assert result.status_code == 409
    assert "explicitly home" in result.json()["error"]
    assert st.arm.backend.history == [], st.arm.backend.history


def test_selection_does_not_authorize_or_retarget_the_legacy_pending_find(tmp_path, monkeypatch):
    st, c = physical_client(tmp_path, monkeypatch)
    arm_hardware_policy(st, c)
    c.post("/api/background")
    first = c.post("/api/find", data={"text": "the " + _unique_color(c) + " one", "move": "false"}).json()
    assert first["result"]["status"] == "found" and not first["moved"]
    old_target = list(first["table_mm"])
    objects = c.post("/api/inventory").json()["snapshot"]["objects"]
    different = next(obj for obj in objects if obj["box_id"] != first["result"]["box"]["id"])
    selected = c.post("/api/select", data={"track_id": different["track_id"]}).json()
    assert selected["status"] == "selected" and not selected["motion_allowed"]
    assert st.arm.backend.history == []  # selecting itself really was motion-free
    result = c.post("/api/arm/point_last")
    if st.arm_thread:
        st.arm_thread.join(3)
    # point_last explicitly means LAST FIND, not the M1 selection. This is legacy behavior, not a guard bypass.
    assert result.status_code == 200 and result.json()["table_mm"] == old_target
    assert st.arm.backend.history == [(old_target[0], old_target[1], st.arm.z)]


def frame(color):
    image = np.full((200, 300, 3), 220, np.uint8)
    image[40:90, 30:80] = color
    return image


def test_switching_same_resolution_camera_source_resets_track_identity(tmp_path, monkeypatch):
    a, b = tmp_path / "view_a.png", tmp_path / "view_b.png"
    cv2.imwrite(str(a), frame((220, 60, 20)))
    cv2.imwrite(str(b), frame((20, 50, 220)))
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim", TROVA_CAMERA=str(a))
    monkeypatch.setattr(app.state.trova, "proposals", lambda f: ([Box(30, 40, 80, 90, id=1)], False))
    first = c.post("/api/inventory").json()["snapshot"]
    assert c.post("/api/source", data={"spec": str(b)}).status_code == 200
    second = c.post("/api/inventory").json()["snapshot"]
    assert second["tracks_reset"] and second["objects"][0]["track_id"] != first["objects"][0]["track_id"], (
        first["objects"], second["objects"], second["tracks_reset"])
    assert second["tracks_reset"] == "camera source changed"
    assert c.post("/api/inventory").json()["snapshot"]["tracks_reset"] is None


def test_retired_track_id_is_not_silently_resolved_as_a_box_number(tmp_path, monkeypatch):
    folder = tmp_path / "frames"
    folder.mkdir()
    image = frame((220, 60, 20))
    image[110:160, 200:250] = (20, 50, 220)
    cv2.imwrite(str(folder / "00_two.png"), image)
    image[40:90, 30:80] = 220
    cv2.imwrite(str(folder / "01_one.png"), image)
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim", TROVA_CAMERA=str(folder))
    st = app.state.trova
    monkeypatch.setattr(st, "proposals", lambda f: (
        [Box(30, 40, 80, 90, id=1), Box(200, 110, 250, 160, id=2)] if st.source.i == 0
        else [Box(200, 110, 250, 160, id=1)], False))
    first = c.post("/api/inventory").json()["snapshot"]
    removed = first["objects"][0]["track_id"]
    c.post("/api/source/step", data={"delta": 1})
    second = c.post("/api/inventory").json()["snapshot"]
    assert removed not in {o["track_id"] for o in second["objects"]}
    selected = c.post("/api/select", data={"text": f"id {removed}"}).json()
    assert selected["status"] == "not_found", selected
    assert c.post("/api/select", data={"track_id": removed}).json()["status"] == "not_found"
    assert c.post("/api/select", data={"box_id": removed}).status_code == 400  # explicit frame required
    by_box = c.post("/api/select", data={"box_id": removed, "snapshot_id": second["id"]}).json()
    assert by_box["object"]["track_id"] != removed
    assert c.post("/api/select", data={"box_id": removed, "snapshot_id": first["id"]}).status_code == 409


@pytest.mark.parametrize("offline", [False, True])
def test_non_mock_empty_snapshot_is_supported_without_an_api_call(tmp_path, monkeypatch, offline):
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim")
    st = app.state.trova
    monkeypatch.setattr(st, "proposals", lambda f: ([], False))
    monkeypatch.setattr(st.grok, "mock", False)
    st.grok.offline_until = time.time() + 100 if offline else 0
    monkeypatch.setattr(st.grok.session, "post", lambda *a, **k: pytest.fail("No API calls allowed"))
    result = c.post("/api/inventory")
    assert result.status_code == 200 and result.json()["snapshot"]["objects"] == []
    assert result.json()["snapshot"]["grok_mode"] == ("offline" if offline else "live")


def test_offline_selection_is_not_reported_authoritative_after_live_snapshot(tmp_path, monkeypatch):
    import trova.server as server
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim")
    st = app.state.trova
    monkeypatch.setattr(server, "describe_table", lambda grok, f, boxes, lang: Inventory(
        items={b.id: {"name": f"item {b.id}", "category": "other", "note": "synthetic"} for b in boxes},
        grok_mode="live", summary="synthetic live metadata, no network"))
    assert c.post("/api/inventory").json()["snapshot"]["authoritative"]
    monkeypatch.setattr(server, "grok_find", lambda grok, f, boxes, text, **kwargs: FindResult(
        status="found", box=boxes[0], confidence=0.7, mode="offline", reason="Local fallback"))
    result = c.post("/api/select", data={"text": "the object that writes"}).json()
    assert result["status"] == "selected" and result["motion_allowed"] is False
    assert result["authoritative"] is False, result
    assert result["snapshot_grok_mode"] == "live" and result["selection_grok_mode"] == "offline"
    assert result["provenance"] == {"snapshot": "live", "selection": ["offline"]}


@pytest.mark.parametrize("state", ["startup", "STOP", "armed"])
def test_selection_never_moves_or_clears_physical_safety_interlocks(tmp_path, monkeypatch, state):
    st, c = physical_client(tmp_path, monkeypatch)
    if state == "armed":
        arm_hardware_policy(st, c)
    elif state == "STOP":
        st.stop_arm()
    snapshot = c.post("/api/inventory").json()["snapshot"]  # fake starts already parked; no repositioning
    before = (st.stop_latched.is_set(), st.lift_interrupted, st.stop_nonce, st.arm._stop.is_set())

    def forbidden(*a, **k):
        pytest.fail("Selection attempted robot control")

    for name in ("home", "ready", "point_at", "go_joints", "run_waypoints"):
        monkeypatch.setattr(st.arm, name, forbidden)
    obj = snapshot["objects"][0]
    result = c.post("/api/select", data={"track_id": obj["track_id"]}).json()
    assert result["status"] == "selected" and result["motion_allowed"] is False
    assert before == (st.stop_latched.is_set(), st.lift_interrupted, st.stop_nonce, st.arm._stop.is_set())
    assert st.arm.backend.history == []
    assert c.post("/api/arm/point_last").status_code == 400  # M1 did not grant a pending motion
    if state != "armed":
        assert c.post("/api/arm/home").status_code == 409
        assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 401


def test_delayed_selection_stays_motion_free_across_stop_and_reset(tmp_path, monkeypatch):
    import trova.server as server
    st, c = cart_client(tmp_path, monkeypatch)
    c.post("/api/inventory")
    entered, release = threading.Event(), threading.Event()

    def delayed(grok, f, boxes, text, **kwargs):
        entered.set()
        assert release.wait(5)
        return FindResult(status="found", box=boxes[0], confidence=0.8)

    monkeypatch.setattr(server, "grok_find", delayed)
    results = []
    worker = threading.Thread(target=lambda: results.append(c.post("/api/select", data={"text": "my tool"})))
    worker.start()
    try:
        assert entered.wait(3)
        st.stop_arm()
        assert c.post("/api/arm/reset", json=confirmation(c)).status_code == 200
        release.set()
        worker.join(5)
        assert not worker.is_alive() and results[0].status_code == 200
        assert results[0].json()["motion_allowed"] is False and st.arm.backend.history == []
        assert st.pending_move is None
    finally:
        release.set()
        worker.join(5)


@pytest.mark.parametrize("poll_state", [False, True])
@pytest.mark.parametrize("context_state", ["valid", "missing", "stale"])
def test_spoken_selection_answer_stays_readonly_after_state_poll(tmp_path, monkeypatch, poll_state, context_state):
    import json
    import subprocess
    import trova.server as server
    st, c = physical_client(tmp_path, monkeypatch)
    arm_hardware_policy(st, c)
    c.post("/api/background")
    snapshot = c.post("/api/inventory").json()["snapshot"]
    question = c.post("/api/select", data={"text": "black object"}).json()
    assert question["status"] == "clarify"
    if context_state == "missing":
        question.pop("selection_context")
    elif context_state == "stale":
        question["selection_context"]["id"] = "stale"
    state = c.get("/api/state").json()
    state.pop("pending_select")  # prove local context survives even a poll without this field
    script = r'''
const fs = require('fs'), vm = require('vm');
const payload = JSON.parse(process.argv[1]);
const elements = new Map(), requests = [];
const node = id => {
  if (!elements.has(id)) elements.set(id, {value: id==='pttlang'?'en-US':'', checked:false, style:{},
    className:'', innerHTML:'', textContent:'', blur(){}, addEventListener(){},
    classList:{add(){},remove(){},toggle(){}}, querySelector(){return null;}});
  return elements.get(id);
};
const context = vm.createContext({console, FormData, Date, setTimeout(){}, setInterval(){},
  document:{getElementById:node, addEventListener(){}, activeElement:null},
  window:{addEventListener(){}}, TrovaVoice:{running:()=>false},
  fetch:async(url,opts)=>{
    if(opts?.method==='POST') requests.push({url,body:opts.body instanceof FormData?Object.fromEntries(opts.body):null});
    let data = url==='/api/state'?structuredClone(payload.state):
      url==='/api/log?n=30'?[]:url==='/api/zones'?{zones:[],draft:[]}:
      url==='/api/failures'?{total:0}:url==='/api/voice/config'?{stop_words:[]}:
      url==='/api/select'?{status:'selected', object:payload.question.candidates[0],motion_allowed:false,reason:'selection'}:
      {result:{status:'not_found',reason:'mock UI response',confidence:0,n_candidates:0,grok_latency_s:0},total_s:0};
    return {ok:true,json:async()=>data};
  }, structuredClone});
vm.runInContext(fs.readFileSync('trova/static/speech.js','utf8'),context);
let js = fs.readFileSync('trova/static/index.html','utf8').split('<script>')[1].split('</script>')[0];
js = js.replace('refresh(); setInterval(refresh, 1500);','');
vm.runInContext(js,context);
context.payload=payload;context.requests=requests;
vm.runInContext(`(async()=>{
  S=structuredClone(payload.state); showSelection(payload.question);
  const before=!!pendingSelection;
  if(payload.poll) await refresh();
  const after=!!pendingSelection;
  await runSpoken('the second black object');
  return {before,after,requests};
})()`,context).then(r=>console.log(JSON.stringify(r))).catch(e=>{console.error(e);process.exit(1)});
'''
    completed = subprocess.run(["node", "-e", script, json.dumps({"state": state, "question": question,
                                                                "poll": poll_state})],
                               capture_output=True, text=True, check=True)
    observed = json.loads(completed.stdout)
    assert observed["before"]
    assert observed["after"]
    if context_state == "missing":
        assert not any(r["url"] in ("/api/find", "/api/find_all", "/api/clarify", "/api/select")
                       for r in observed["requests"])
        assert st.arm.backend.history == []
        return
    # Replay only the already-observed request against the fake hardware-policy session.
    request = next(r for r in observed["requests"] if r["url"] in ("/api/find", "/api/select"))
    monkeypatch.setattr(server, "grok_find", lambda grok, f, boxes, text, *a, **kw: FindResult(
        status="found", box=boxes[0], confidence=0.8, reason="Synthetic model disambiguation"))
    response = c.post(request["url"], data=request["body"])
    if st.arm_thread:
        st.arm_thread.join(3)
    assert response.status_code == (409 if context_state == "stale" else 200), response.json()
    assert request["url"] == "/api/select" and st.arm.backend.history == [], (observed, response.json(), st.arm.backend.history)


def test_scene_answers_cannot_use_moving_find_routes_or_missing_context(tmp_path, monkeypatch):
    import trova.server as server
    st, c = physical_client(tmp_path, monkeypatch)
    arm_hardware_policy(st, c)
    snapshot = c.post("/api/inventory").json()["snapshot"]
    question = c.post("/api/select", data={"text": "black object"}).json()
    assert question["status"] == "clarify"
    context = question["selection_context"]

    def forbidden(*a, **kw):
        pytest.fail("A scene clarification entered a model or motion path")

    monkeypatch.setattr(server, "grok_find", forbidden)
    for name in ("home", "ready", "point_at", "go_joints", "run_waypoints"):
        monkeypatch.setattr(st.arm, name, forbidden)
    for path, data in [
        ("/api/find", {"text": "the second black object"}),
        ("/api/find", {"text": "the second black object", "move": "false"}),
        ("/api/find_all", {"text": "black objects"}),
        ("/api/clarify", {"answer": "the second black object"}),
        ("/api/select", {"text": "the second black object"}),
        ("/api/select", {"text": "the second black object", "snapshot_id": snapshot["id"],
                         "selection_context": "stale"}),
    ]:
        response = c.post(path, data=data)
        assert response.status_code == 409, (path, response.json())
    assert st.pending_select == context
    answer = {"text": "the second black object", "snapshot_id": snapshot["id"], "selection_context": context["id"]}
    assert c.post("/api/select", data=answer).json()["status"] == "selected"
    assert c.post("/api/select", data=answer).status_code == 409  # one-use context
    # Even a lost/consumed clarification cannot turn scene mode into automatic moving Find.
    assert c.post("/api/find", data={"text": "black object"}).status_code == 409
    assert c.post("/api/find", data={"text": "black object", "move": "false"}).status_code == 409
    assert st.arm.backend.history == []


def test_stale_scene_answer_after_new_snapshot_is_rejected(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch)
    first = c.post("/api/inventory").json()["snapshot"]
    question = c.post("/api/select", data={"text": "black object"}).json()
    second = c.post("/api/inventory").json()["snapshot"]
    assert first["id"] != second["id"]
    result = c.post("/api/select", data={"text": "the second black object", "snapshot_id": first["id"],
                                        "selection_context": question["selection_context"]["id"]})
    assert result.status_code == 409 and st.arm.backend.history == []


@pytest.mark.parametrize("state", ["startup", "STOP", "armed"])
def test_snapshot_away_from_home_never_calls_motion(tmp_path, monkeypatch, state):
    st, c = physical_client(tmp_path, monkeypatch)
    if state == "armed":
        arm_hardware_policy(st, c)
    elif state == "STOP":
        st.stop_arm()
    st.arm.backend.pos = st.arm.backend.target = [450.0, 100.0, 200.0]

    def forbidden(*a, **kw):
        pytest.fail("Read-only inventory attempted robot control")

    for name in ("home", "ready", "point_at", "go_joints", "run_waypoints"):
        monkeypatch.setattr(st.arm, name, forbidden)
    assert c.post("/api/inventory").status_code == 409
    assert c.post("/api/find", data={"text": "the second black object"}).status_code == 409
    assert st.arm.backend.history == []


def test_upload_replacement_and_return_to_live_reset_ids_and_invalidate_snapshot(tmp_path, monkeypatch):
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim")
    st = app.state.trova
    monkeypatch.setattr(st, "proposals", lambda f: ([Box(30, 40, 80, 90, id=1)], False))
    before = c.post("/api/inventory").json()["snapshot"]
    ids = [before["objects"][0]["track_id"]]
    ok, encoded = cv2.imencode(".png", st.frame())
    assert ok
    for reason in ("uploaded scene changed", "uploaded scene changed", "returned to camera source"):
        if reason == "uploaded scene changed":
            response = c.post("/api/upload_scene", files={"photo": ("synthetic.png", encoded.tobytes(), "image/png")})
        else:
            response = c.post("/api/upload_scene/clear")
        assert response.status_code == 200
        assert st.inventory is None and st.pending_select is None
        assert c.post("/api/select", data={"track_id": ids[-1], "snapshot_id": before["id"]}).status_code == 400
        before = c.post("/api/inventory").json()["snapshot"]
        assert before["tracks_reset"] == reason
        ids.append(before["objects"][0]["track_id"])
    assert ids == sorted(set(ids)) and len(ids) == 4


def test_taught_survey_pose_resets_scene_tracks(tmp_path, monkeypatch):
    st, c = cart_client(tmp_path, monkeypatch, TROVA_CAMERA_ON_ARM="1")
    first = c.post("/api/inventory").json()["snapshot"]
    assert c.post("/api/arm/teach_pose", data={"name": "home_joints"}).status_code == 200
    assert st.inventory is None
    second = c.post("/api/inventory").json()["snapshot"]
    assert second["tracks_reset"] == "survey pose changed"
    assert not ({o["track_id"] for o in first["objects"]} & {o["track_id"] for o in second["objects"]})
    assert st.arm.backend.history == []


@pytest.mark.parametrize("snapshot_mode,selection_mode,authoritative", [
    ("live", "live", True), ("live", "mock", False), ("live", "offline", False),
    ("live", "unknown", False), ("mock", "live", False), ("offline", "live", False),
])
def test_snapshot_and_selection_provenance_are_independent(tmp_path, monkeypatch, snapshot_mode, selection_mode,
                                                          authoritative):
    import trova.server as server
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim")
    monkeypatch.setattr(server, "describe_table", lambda grok, f, boxes, lang: Inventory(
        items={b.id: {"name": f"item {b.id}"} for b in boxes}, grok_mode=snapshot_mode))
    c.post("/api/inventory")
    monkeypatch.setattr(server, "grok_find", lambda grok, f, boxes, text, **kwargs: FindResult(
        status="found", box=boxes[0], grok_mode=selection_mode))
    result = c.post("/api/select", data={"text": "the object that writes"}).json()
    assert result["status"] == "selected" and result["motion_allowed"] is False
    assert result["authoritative"] is authoritative
    assert result["provenance"] == {"snapshot": snapshot_mode, "selection": [selection_mode]}
    assert app.state.trova.pending_move is None


def test_offline_clarification_provenance_survives_local_answer(tmp_path, monkeypatch):
    import trova.server as server
    app, c = _client(tmp_path, monkeypatch, TROVA_ARM="sim")
    monkeypatch.setattr(server, "describe_table", lambda grok, f, boxes, lang: Inventory(
        items={b.id: {"name": f"item {b.id}"} for b in boxes}, grok_mode="live"))
    snap = c.post("/api/inventory").json()["snapshot"]
    monkeypatch.setattr(server, "grok_find", lambda grok, f, boxes, text, **kwargs: FindResult(
        status="clarify", candidate_ids=[b.id for b in boxes[:2]], clarifying_question="Which?",
        mode="offline", grok_mode="offline"))
    question = c.post("/api/select", data={"text": "my tool"}).json()
    assert question["status"] == "clarify" and question["authoritative"] is False
    context = question["selection_context"]
    result = c.post("/api/select", data={"track_id": question["candidates"][0]["track_id"],
                                        "snapshot_id": snap["id"], "selection_context": context["id"]}).json()
    assert result["status"] == "selected" and result["authoritative"] is False
    assert result["provenance"] == {"snapshot": "live", "selection": ["offline"]}
    assert app.state.trova.pending_move is None


@pytest.mark.parametrize("state", ["startup", "STOP", "armed"])
def test_explicit_new_find_still_passes_all_motion_guards(tmp_path, monkeypatch, state):
    import trova.server as server
    st, c = physical_client(tmp_path, monkeypatch)
    if state == "armed":
        arm_hardware_policy(st, c)
    elif state == "STOP":
        st.stop_arm()
    assert c.post("/api/inventory").status_code == 200
    monkeypatch.setattr(server, "grok_find", lambda grok, f, boxes, text, *a, **kw: FindResult(
        status="found", box=boxes[0], confidence=0.8, grok_mode="mock"))
    result = c.post("/api/find", data={"text": "my tool", "new_find": "true"})
    assert result.status_code == 200
    if st.arm_thread:
        st.arm_thread.join(3)
    assert result.json()["moved"] is (state == "armed")
    assert bool(st.arm.backend.history) is (state == "armed")
    assert st.stop_latched.is_set() is (state != "armed")


@pytest.mark.parametrize("snapshot_error", [False, True])
def test_realtime_voice_selection_tools_never_fall_through_to_motion(snapshot_error):
    """Exercise the actual voice tool closures with synthetic HTTP responses; no mic or websocket."""
    import json
    import subprocess
    script = r'''
const fs = require('fs'), vm = require('vm');
const requests = [], outputs = [], snapshotError = JSON.parse(process.argv[1]);
let selects = 0;
const context = vm.createContext({FormData, console, fetch: async(url, options) => {
  requests.push({url, body:options.body instanceof FormData ? Object.fromEntries(options.body) : null});
  let data;
  if(url==='/api/inventory') data = snapshotError ? {error:'explicit Home required'} :
    {snapshot:{id:'snapshot-A',objects:[],grok_mode:'mock'}, summary:'synthetic'};
  else if(url==='/api/select') data = ++selects === 1 ?
    {status:'clarify',snapshot_id:'snapshot-A',selection_context:{id:'context-A',snapshot_id:'snapshot-A'},
     candidates:[],question:'Which?',motion_allowed:false} :
    {status:'selected',snapshot_id:'snapshot-A',candidates:[],object:{track_id:2,name:'synthetic'},motion_allowed:false};
  else throw Error('Unexpected route: '+url);
  return {ok:true,json:async()=>data};
}});
let source = fs.readFileSync('trova/static/voice.js','utf8');
// Expose closures only in the VM, keeping the production module's public API unchanged.
source = source.replace('return {start, stop, running:', 'return {tools, start, stop, running:');
vm.runInContext(source,context);
vm.runInContext(`(async()=>{
 const t = TrovaVoice.tools;
 await t.describe_table();
 outputs.push(await t.select_object({description:'black object'}));
 outputs.push(await t.answer_clarification({answer:'the second black object'}));
 outputs.push(await t.find_object({description:'my tool'}));
 outputs.push(await t.find_all({description:'all the tools'}));
 outputs.push(await t.point_at_found());
 return {requests,outputs};
})()`,Object.assign(context,{requests,outputs})).then(r=>console.log(JSON.stringify(r)))
 .catch(e=>{console.error(e);process.exit(1)});
'''
    completed = subprocess.run(["node", "-e", script, json.dumps(snapshot_error)],
                               capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert {r["url"] for r in result["requests"]} <= {"/api/inventory", "/api/select"}
    if snapshot_error:
        assert len(result["requests"]) == 1 and all(o["error"] for o in result["outputs"])
    else:
        assert result["requests"][2]["body"]["selection_context"] == "context-A"
        assert all(r["body"]["snapshot_id"] == "snapshot-A" for r in result["requests"][1:])
        assert result["outputs"][-1]["error"]  # M1 never grants point_at_found permission
