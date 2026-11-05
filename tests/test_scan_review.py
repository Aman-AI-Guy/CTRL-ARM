"""Grok's requests to the operator after a reconstruction (trova/scan_review.py). Offline: mock Grok, no robot."""

import threading

import cv2
import numpy as np
import pytest

from trova import scan_review
from trova.grok import GrokClient


def _job(folder, warnings=(), status="completed", orientation="as_found", session="s1", key="job1",
         renders=("front.png", "top.png")):
    (folder / "renders").mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in renders:
        cv2.imwrite(str(folder / "renders" / name), np.full((60, 80, 3), 128, np.uint8))
        artifacts[name] = f"renders/{name}"
    return {"id": key, "session_id": session, "orientation": orientation, "status": status, "artifacts": artifacts,
            "started_at": 1.0, "reconstructed": status == "completed", "textured_mesh": False, "cad": False,
            "reason": "", "stage": "done",
            "quality": {"warnings": list(warnings), "registered_images": 58, "input_images": 60,
                        "estimated_object_dimensions_mm": {"L": 96.3, "W": 55.6, "H": 37.4}}}


def _renders(job, folder):
    return [(name, cv2.imread(str(folder / rel))) for name, rel in job["artifacts"].items()]


def test_one_orientation_group_asks_to_flip_and_never_moves(tmp_path):
    job = _job(tmp_path)
    rev = scan_review.review(GrokClient(mock=True), job, _renders(job, tmp_path), name="mouse rosso", groups=1, lang="it")
    assert rev["grok_mode"] == "mock" and rev["moves_arm"] is False
    assert rev["surface_coverage_measured"] is False and rev["cross_orientation_registration"] is False
    assert rev["complete"] is False and rev["job_id"] == "job1" and rev["renders"] == ["front.png", "top.png"]
    first = rev["requests"][0]
    assert first["id"] == "R1" and first["action"] == "flip" and first["side"] == "bottom"
    assert first["reorients"] is True and first["status"] == "pending" and first["render"] == "top.png"
    assert "mouse rosso" in first["instruction"]


def test_warnings_add_a_reposition_and_a_second_group_needs_nothing(tmp_path):
    job = _job(tmp_path, warnings=["Some photographs did not register; geometry may be incomplete."])
    rev = scan_review.review(GrokClient(mock=True), job, _renders(job, tmp_path), groups=1, lang="en")
    assert [q["action"] for q in rev["requests"]] == ["flip", "reposition"]
    assert rev["requests"][1]["reorients"] is True and "did not register" in rev["requests"][1]["reason"]
    clean = _job(tmp_path / "b")
    done = scan_review.review(GrokClient(mock=True), clean, _renders(clean, tmp_path / "b"), groups=2)
    assert done["complete"] is True and done["requests"] == []


def test_parse_cleans_the_model_answer():
    data = {"requests": [
        {"action": "teleport", "side": "inside", "instruction": "  Ruota il mouse di 90 gradi. ", "reason": "x", "render": 7},
        {"action": "flip", "side": "bottom", "instruction": "   ", "reason": "empty instruction", "render": 1},
        {"action": "uncover", "side": "left", "instruction": "Sposta il cavo dal lato sinistro", "reason": "y", "render": 1},
    ] + [{"action": "lighting", "side": "top", "instruction": f"Luce {i}", "reason": "", "render": 0} for i in range(6)]}
    out = scan_review.parse(data, ["a.png"])
    assert len(out) <= scan_review.MAX_REQUESTS
    assert out[0]["action"] == "other" and out[0]["side"] == "none" and out[0]["render"] is None
    assert out[0]["instruction"] == "Ruota il mouse di 90 gradi."
    assert out[1]["id"] == "R2" and out[1]["action"] == "uncover" and out[1]["render"] == "a.png"
    assert out[1]["reorients"] is False and all(q["status"] == "pending" for q in out)


def test_review_refuses_without_renders_or_before_the_job_completes(tmp_path):
    job = _job(tmp_path)
    with pytest.raises(ValueError):
        scan_review.review(GrokClient(mock=True), job, [], groups=1)
    with pytest.raises(ValueError):
        scan_review.review(GrokClient(mock=True), dict(job, status="running"), _renders(job, tmp_path))


def test_status_changes_and_persistence(tmp_path):
    job = _job(tmp_path)
    rev = scan_review.review(GrokClient(mock=True), job, _renders(job, tmp_path), groups=1)
    assert scan_review.set_status(rev, "R1", "done")["status"] == "done"
    assert scan_review.pending(rev) == []
    with pytest.raises(KeyError):
        scan_review.set_status(rev, "R9", "done")
    with pytest.raises(ValueError):
        scan_review.set_status(rev, "R1", "moved")
    path = tmp_path / "scan_review.json"
    scan_review.save(path, rev)
    assert scan_review.load(path)["requests"][0]["status"] == "done"
    path.write_text("not json")
    assert scan_review.load(path) is None and scan_review.load(tmp_path / "missing.json") is None


def _install_job(st, folder, **kw):
    job = _job(folder, **kw)
    job["_folder"], job["_cancel"] = folder, threading.Event()
    st.reconstruction.jobs[job["id"]] = job
    return job


def test_api_review_ticks_requests_reorients_and_never_moves(tmp_path, monkeypatch):
    from test_pipeline import _client
    monkeypatch.setenv("TROVA_AUTO_REVIEW", "0")
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    tip = list(st.arm.tip)
    assert c.get("/api/scan/review").json()["requests"] == []
    assert c.post("/api/scan/review", json={}).status_code == 404            # nothing reconstructed yet
    assert c.post("/api/scan/review", json={"job_id": "x", "mode": "hardware"}).status_code == 422
    _install_job(st, tmp_path / "reconstructions" / "job1")
    rev = c.post("/api/scan/review", json={}).json()
    assert rev["job_id"] == "job1" and rev["requests"][0]["action"] == "flip" and rev["moves_arm"] is False
    assert c.get("/api/state").json()["scan_review"]["requests"][0]["id"] == "R1"
    assert (tmp_path / "scan_review.json").exists()

    # without an open scan session a done flip only ticks the request
    out = c.post("/api/scan/review/request", json={"request_id": "R1", "status": "done"}).json()
    assert out["request"]["status"] == "done" and out["requires_reselection"] is False and out["hardware_moved"] is False
    c.post("/api/scan/review/request", json={"request_id": "R1", "status": "pending"})

    # with an open session it opens a new orientation group and asks for reselection
    st.scans.start("1")
    out = c.post("/api/scan/review/request", json={"request_id": "R1", "status": "done"}).json()
    assert out["requires_reselection"] is True and out["orientation"]["old_object_frame_valid"] is False
    assert "uniti" in out["spoken_message"]
    assert c.post("/api/scan/review/request", json={"request_id": "R7", "status": "done"}).status_code == 404
    assert c.post("/api/scan/review/request", json={"request_id": "R1", "status": "moved"}).status_code == 400
    assert list(st.arm.tip) == tip


def test_saying_i_flipped_it_ticks_the_pending_flip(tmp_path, monkeypatch):
    from test_pipeline import _client
    monkeypatch.setenv("TROVA_AUTO_REVIEW", "0")
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    _install_job(st, tmp_path / "reconstructions" / "job1")
    c.post("/api/scan/review", json={})
    st.scans.start("1")
    body = c.post("/api/agent/task", json={"text": "Ho capovolto il mouse"}).json()
    assert body["workflow_state"] == "object_reoriented"
    assert st.scan_review["requests"][0]["status"] == "done"


def test_finished_reconstruction_is_reviewed_automatically(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    job = _install_job(st, tmp_path / "reconstructions" / "job1")
    st._reconstruction_done(st.reconstruction.public(job))
    assert st.scan_review and st.scan_review["job_id"] == "job1" and st.scan_review["requests"]
    monkeypatch.setenv("TROVA_AUTO_REVIEW", "0")
    st.scan_review = None
    st._reconstruction_done(st.reconstruction.public(job))
    assert st.scan_review is None


def test_reorientation_skill_speaks_grok_request_first():
    from types import SimpleNamespace

    from trova.scan3d import skills
    rev = {"job_id": "j", "requests": [{"id": "R1", "status": "pending", "instruction": "Capovolgi il mouse.", "action": "flip"},
                                       {"id": "R2", "status": "done", "instruction": "Sposta il cavo.", "action": "uncover"}]}
    r = skills.reorient(SimpleNamespace(st=SimpleNamespace(scan_review=rev)), {}, "")
    assert r["reason"].startswith("Capovolgi il mouse.") and r["human_assistance_requested"] is True
    assert [q["id"] for q in r["operator_requests"]] == ["R1"] and r["executed"] is False
    plain = skills.reorient(SimpleNamespace(), {}, "")
    assert plain["reason"] == skills.GENERIC_REORIENT and plain["operator_requests"] == []


def test_voice_has_scan_request_tools(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, XAI_API_KEY="k")
    session = c.get("/api/voice/config").json()["session"]
    names = {t["name"] for t in session["tools"]}
    assert {"scan_requests", "scan_request_done"} <= names
    assert "never say the mesh is closed" in session["instructions"]


def test_reorientation_request_refuses_moving_arm_without_consuming_request(tmp_path, monkeypatch):
    from test_pipeline import _client
    monkeypatch.setenv('TROVA_AUTO_REVIEW', '0')
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    _install_job(st, tmp_path / 'reconstructions' / 'job1')
    c.post('/api/scan/review', json={})
    st.scans.start('1')
    monkeypatch.setattr(st.arm, 'status', lambda: {'state': 'moving'})
    response = c.post('/api/scan/review/request', json={'request_id': 'R1', 'status': 'done'})
    assert response.status_code == 409
    assert st.scan_review['requests'][0]['status'] == 'pending'


def test_repeated_done_does_not_create_another_orientation_group(tmp_path, monkeypatch):
    from test_pipeline import _client
    monkeypatch.setenv('TROVA_AUTO_REVIEW', '0')
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    _install_job(st, tmp_path / 'reconstructions' / 'job1')
    c.post('/api/scan/review', json={})
    st.scans.start('1')
    body = {'request_id': 'R1', 'status': 'done'}
    assert c.post('/api/scan/review/request', json=body).json()['requires_reselection']
    orientation = st.scans.sessions[st.scans.active]['orientation']
    assert c.post('/api/scan/review/request', json=body).json()['requires_reselection'] is False
    assert st.scans.sessions[st.scans.active]['orientation'] == orientation
