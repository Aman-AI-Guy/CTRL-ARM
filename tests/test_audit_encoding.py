"""The audit trail is UTF-8 on every platform; lines an older Windows build wrote as cp1252 still read and verify."""

import json

from trova.audit import AuditLog, read_text_lines


def test_audit_log_is_utf8_and_reads_legacy_cp1252_lines(tmp_path):
    log = AuditLog(tmp_path)
    legacy = {"id": "old", "ts": 0, "time": "2026-11-05 11:27:29", "kind": "voice", "who": "trova",
              "text": "L'oggetto 1 è un bicchiere"}
    legacy["hash"] = AuditLog._chain("", AuditLog._canonical(legacy))
    (tmp_path / "events.jsonl").write_bytes((json.dumps(legacy, ensure_ascii=False) + "\n").encode("cp1252"))

    log = AuditLog(tmp_path)                       # reopen: picks up the legacy chain head
    log.event("voice", who="trova", text="Sposto → più vicino, 5°")   # → is not in cp1252

    raw = (tmp_path / "events.jsonl").read_bytes().splitlines()
    assert raw[1].decode("utf-8")                  # the new line is UTF-8
    lines = read_text_lines(tmp_path / "events.jsonl")
    assert "è un bicchiere" in lines[0] and "→ più vicino, 5°" in lines[1]
    assert [e["text"] for e in log.all()] == ["L'oggetto 1 è un bicchiere", "Sposto → più vicino, 5°"]
    assert log.verify()["ok"] and log.verify()["hashed"] == 2
