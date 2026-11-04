"""Tests for the polish pass and the features added after the first seven (see AGENTS.md)."""

import re
import types
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "trova" / "static"

# Inside an inline handler (onclick="..."), names resolve on the element, then its form, then document,
# before the page's own functions. A helper called form() therefore calls button.form (null) and breaks.
SHADOWED = {"form", "submit", "reset", "click", "focus", "blur", "remove", "append", "prepend", "before", "after",
            "replaceWith", "animate", "closest", "matches", "contains", "select", "open", "close", "write",
            "toggle", "labels", "value", "name", "type", "title", "hidden", "style", "dataset", "checked"}


def test_inline_handlers_do_not_call_shadowed_names():
    for page in ("index.html", "jury.html"):
        html = (STATIC / page).read_text()
        for handler in re.findall(r'\son\w+="([^"]*)"', html):
            for fn in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\(", handler):
                assert fn not in SHADOWED, f"{page}: inline handler calls {fn}(), which resolves to an element property"


# ---------------------------------------------------------------- feature 8: offline fallback
def test_offline_fallback_and_circuit_breaker(monkeypatch):
    import requests

    from trova import grok as grok_mod
    from trova.finder import find
    from trova.grok import GrokClient, GrokError, GrokUnavailable
    from trova.proposals import find_blobs
    from trova.synthetic import SyntheticTable

    t = SyntheticTable(seed=3)
    frame, empty = t.render(), t.empty()
    boxes = find_blobs(frame, empty)
    colors = [o.color for o in t.objects]
    unique = next(c for c in colors if colors.count(c) == 1)
    calls = []

    def down(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("no route to host")

    g = GrokClient("xai-test", offline_fallback=True)
    monkeypatch.setattr(g.session, "post", down)
    monkeypatch.setattr(grok_mod.time, "sleep", lambda s: None)
    r = find(g, frame, boxes, f"the {unique} thing")
    assert r.status == "found" and r.mode == "offline" and r.reason.startswith("Grok unreachable")
    assert g.offline and g.stats()["offline_calls"] == 1 and "no route" in g.stats()["last_error"]
    n = len(calls)
    find(g, frame, boxes, f"the {unique} thing")        # inside the hold window: no network attempt at all
    assert len(calls) == n
    g.offline_until = 0                                  # window over: live is tried again
    find(g, frame, boxes, f"the {unique} thing")
    assert len(calls) > n

    def unauthorized(*a, **k):
        return types.SimpleNamespace(status_code=401, text="invalid api key", json=lambda: {})
    g2 = GrokClient("xai-bad", offline_fallback=True)
    monkeypatch.setattr(g2.session, "post", unauthorized)
    with pytest.raises(GrokError, match="401"):          # configuration errors are never hidden
        find(g2, frame, boxes, "the pen")

    g3 = GrokClient("xai-test", offline_fallback=False)   # eval mode: never measure the stand-in
    monkeypatch.setattr(g3.session, "post", down)
    with pytest.raises(GrokUnavailable):
        find(g3, frame, boxes, "the pen")
