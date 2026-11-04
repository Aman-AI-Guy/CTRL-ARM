"""Tests for the polish pass and the features added after the first seven (see AGENTS.md)."""

import re
from pathlib import Path

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
