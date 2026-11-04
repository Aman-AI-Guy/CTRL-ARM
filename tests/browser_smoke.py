"""Browser smoke test: starts Trova in mock mode on a free port and clicks through every feature in Chrome.

    .venv/bin/pip install -e ".[ui]"      # playwright; uses the installed Google Chrome (or: playwright install chromium)
    .venv/bin/python tests/browser_smoke.py [--headed]

Fails (exit 1) on any JavaScript error, any 5xx, an unexpected 4xx, or sideways scrolling at phone width.
Not part of pytest (it needs a browser); run it after UI changes and before the demo.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_4XX = ("/api/calibration/aruco",)       # the demo scene has no ArUco markers


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port, data = free_port(), tempfile.mkdtemp(prefix="trova_smoke_")
    env = {**os.environ, "XAI_API_KEY": "", "TROVA_MOCK": "1", "TROVA_DATA": data, "TROVA_PORT": str(port),
           "TROVA_HOST": "127.0.0.1", "TROVA_CAMERA": "demo", "TROVA_ARM": "sim", "TROVA_DOG": "sim",
           "TROVA_ALLOW_LIFT": "1", "TROVA_TOKEN": ""}
    server = subprocess.Popen([sys.executable, "-m", "trova"], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    problems, log = [], []
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="chrome", headless="--headed" not in sys.argv)
            except Exception:  # noqa: BLE001 - no Google Chrome: the bundled Chromium
                browser = p.chromium.launch(headless="--headed" not in sys.argv)
            ctx = browser.new_context(viewport={"width": 1400, "height": 1000})

            def watch(page, name):
                page.on("pageerror", lambda e: problems.append(f"{name}: JS error: {e}"))
                page.on("response", lambda r: problems.append(f"{name}: HTTP {r.status} {r.request.method} {r.url}")
                        if r.status >= 500 or (r.status >= 400 and not r.url.split("?")[0].endswith(EXPECTED_4XX)) else None)
                page.on("dialog", lambda d: d.accept({"Zone name": "line area"}.get(d.message.split(" (")[0], "")
                                                     if d.type == "prompt" else None))
            pg = ctx.new_page()
            watch(pg, "main")

            def press(name, exact=True):
                """Centre the button first: the sticky header and alarm bar would otherwise cover it."""
                el = pg.get_by_role("button", name=name, exact=exact).first
                el.evaluate("e => e.scrollIntoView({block: 'center'})")
                el.click()
            text = lambda sel: pg.inner_text(sel).replace("\n", " | ")                       # noqa: E731

            def step(name, fn):
                t0 = time.time()
                try:
                    out = fn()
                    log.append(f"ok    {name:<28} {time.time() - t0:4.1f}s  {str(out or '')[:110]}")
                except Exception as e:  # noqa: BLE001
                    problems.append(f"step '{name}' failed: {type(e).__name__}: {str(e)[:200]}")
                    log.append(f"FAIL  {name}")

            def wait_text(sel, gone, timeout=20000):
                pg.wait_for_function(f"!document.querySelector('{sel}').innerText.includes({gone!r})", timeout=timeout)

            def find(q):
                pg.fill("#q", q)
                press("Find")
                pg.wait_for_function("document.getElementById('findbtn').textContent === 'Find'", timeout=20000)
                return text("#findres")

            step("load", lambda: (pg.goto(base + "/"),
                                  pg.wait_for_function("document.getElementById('live').naturalWidth > 0", timeout=15000),
                                  text("#mode"))[2])
            step("preflight", lambda: (press("preflight"), wait_text("#preres", "Checking"), text("#preres")[:80])[2])
            pg.click("summary:has-text('Setup')")
            step("capture empty table", lambda: (press("Capture empty table"), pg.wait_for_timeout(500), text("#bginfo"))[2])
            unique = _unique(pg)
            step("find by words", lambda: find(f"the {unique} one")[:90])
            step("jury: correct", lambda: (press("✓ correct"), pg.wait_for_timeout(400))[1])
            step("find absent object", lambda: find("a stapler")[:60])
            step("clarify + spoken answer", lambda: (find("the black one"),
                                                     pg.evaluate("runSpoken('number 2')"), pg.wait_for_timeout(1200),
                                                     text("#findres")[:60])[3])
            step("find all + space STOP", lambda: (pg.fill("#q", "all the black things"), press("Find all"),
                                                   wait_text("#findres", "Looking for every match"),
                                                   pg.click("body", position={"x": 5, "y": 990}), pg.keyboard.press("Space"),
                                                   pg.wait_for_timeout(600), text("#armst"))[6])
            step("explicit simulator reset", lambda: _reset_simulator(pg))
            step("what's on the table", lambda: (press("What's on the table?"), wait_text("#findres", "Naming every"),
                                                 text("#findres")[:60])[2])
            step("M1 select + poll + speech", lambda: _scene_selection(pg, text))
            step("save reference", lambda: (press("Save reference"), pg.wait_for_timeout(1500), text("#refinfo"))[2])
            step("set batch + old-lot carton", lambda: (pg.fill("#lot", "A123"), press("Set batch"),
                                                        press("+ old-lot carton"), pg.wait_for_timeout(600),
                                                        text("#batchinfo"))[4])
            step("read labels", lambda: (press("Read labels"), wait_text("#checkres", "Reading labels"),
                                         text("#checkres")[:90])[2])
            step("draw a zone", lambda: (press("Draw a zone (click corners)"),
                                         [pg.mouse.click(*xy) for xy in _corners(pg)], press("Finish zone"),
                                         pg.wait_for_timeout(800), text("#zoneinfo"))[4])
            step("check now", lambda: (press("Check now"), wait_text("#checkres", "Checking"), text("#checkres")[:90])[2])
            step("sign off (reject)", lambda: (pg.fill("#operator", "Smoke Test"), press("Reject"),
                                               pg.wait_for_timeout(500))[2])
            step("record + PDF", lambda: (pg.request.get(base + "/api/report").ok and
                                          pg.request.get(base + "/api/report.pdf").body()[:4] == b"%PDF") or _fail("record"))
            step("watch: alarm + acknowledge", lambda: _watch(pg, press, text))
            step("tray lift + STOP", lambda: (find(f"the {_unique(pg)} one"), press("Lift its tray", exact=False),
                                              pg.mouse.click(*_corners(pg)[2]), pg.wait_for_timeout(800), press("STOP"),
                                              pg.wait_for_timeout(500), text("#armst"))[6])
            step("explicit simulator recovery", lambda: _reset_simulator(pg))
            step("arm home", lambda: (press("Arm home"), pg.wait_for_timeout(500), text("#armst"))[2])
            step("dog search", lambda: (press("Send the dog to look"), pg.wait_for_timeout(6000),
                                        text("#dogres")[:60])[2])
            for path, name in (("/jury", "jury"), ("/history", "history")):
                page = ctx.new_page()
                watch(page, name)
                step(f"{name} page", lambda page=page, path=path: (page.goto(base + path), page.wait_for_timeout(1500),
                                                                   page.inner_text("body")[:70].replace("\n", " "))[2])
            hist = ctx.pages[-1]
            step("history: verify log", lambda: (hist.get_by_role("button", name="Verify log integrity").click(),
                                                 hist.wait_for_timeout(600), hist.inner_text("#verify"))[2])
            if "intact" not in hist.inner_text("#verify"):
                problems.append("audit log chain not intact")
            for path in ("/", "/jury", "/history"):
                m = browser.new_page(viewport={"width": 390, "height": 844})
                watch(m, "phone" + path)
                m.goto(base + path)
                m.wait_for_timeout(1200)
                sw, w = m.evaluate("[document.documentElement.scrollWidth, innerWidth]")
                if sw > w:
                    problems.append(f"phone width: {path} scrolls sideways ({sw} > {w} px)")
                m.close()
            browser.close()
    finally:
        server.terminate()
        try:
            _, err = server.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            _, err = server.communicate()
        if "Traceback" in (err or ""):
            problems.append("server traceback:\n" + err[-1500:])
    print("\n".join(log))
    print("\n" + ("PASS: no problems" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems)))
    return 1 if problems else 0


def _reset_simulator(pg):
    """Explicit test-operator decisions, restricted to the built-in offline simulator."""
    state = pg.request.get(pg.url.rstrip("/") + "/api/state").json()
    assert state["arm"]["name"] == "sim" and not state["physical_arm"] and state["stop_latched"]
    base = pg.url.rstrip("/")
    confirmation = {"operator": "Offline Smoke Test", "stop_nonce": state["stop_nonce"],
                    "area_clear": True, "controller_ready": True}
    if state["lift_interrupted"]:
        response = pg.request.post(base + "/api/arm/recover_lift", data=confirmation | {
            "fork_freed": True, "note": "Explicit recovery decision in the synthetic no-load simulator"})
        assert response.ok, response.text()
        confirmation["stop_nonce"] = pg.request.get(base + "/api/state").json()["stop_nonce"]
    response = pg.request.post(base + "/api/arm/reset", data=confirmation)
    assert response.ok, response.text()
    after = pg.request.get(base + "/api/state").json()
    assert not after["stop_latched"]
    assert all(after["arm"][key] == state["arm"][key] for key in ("tip_mm", "target_mm"))  # no motion


def _scene_selection(pg, text):
    before = pg.request.get(pg.url.rstrip("/") + "/api/state").json()
    pg.evaluate("""async () => {
      const f = new FormData(); f.append('text', 'black object'); f.append('snapshot_id', sceneSnapshotId);
      const d = await call('/api/select', {method:'POST', body:f});
      if(d.status !== 'clarify') throw Error('Expected scene clarification');
      showSelection(d);
      await refresh();
      await runSpoken('the second black object');
    }""")
    assert "Selected:" in text("#findres"), text("#findres")
    after = pg.request.get(pg.url.rstrip("/") + "/api/state").json()
    assert before["arm"] == after["arm"] and before["stop_nonce"] == after["stop_nonce"]
    return text("#findres")[:90]


def _corners(pg):
    pg.locator("#live").evaluate("e => e.scrollIntoView({block: 'center'})")
    b = pg.locator("#live").bounding_box()
    return [(b["x"] + b["width"] * fx, b["y"] + b["height"] * fy) for fx, fy in ((0.08, 0.1), (0.35, 0.1), (0.35, 0.4), (0.08, 0.4))]


def _watch(pg, press, text):
    press("Save reference")
    pg.wait_for_timeout(1500)
    press("👁 Watch the line")
    pg.wait_for_timeout(2000)
    pg.evaluate("sendForm('/api/demo/tamper', {kind: 'add'})")
    pg.wait_for_function("document.getElementById('alarm').style.display === 'block'", timeout=10000)
    alarm = text("#alarmtext")
    press("Acknowledge")
    pg.wait_for_timeout(1200)
    press("⏹ Stop watching")
    return alarm[:90]


def _unique(pg):
    colors = [o.split()[0] for o in pg.evaluate("fetch('/api/state').then(r => r.json())")["source"]["objects"]]
    return next(c for c in colors if colors.count(c) == 1)


def _fail(what):
    raise AssertionError(what)


if __name__ == "__main__":
    sys.exit(main())
