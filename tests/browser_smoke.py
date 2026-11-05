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
           "TROVA_ALLOW_LIFT": "1", "TROVA_TOKEN": "", "TROVA_ARM_READ": "off", "TROVA_MOTION_MODE": "simulation", "TROVA_HARDWARE_UNLOCK": ""}
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
                page.on("dialog", lambda d: d.accept({"Nome della zona": "area linea"}.get(d.message.split(" (")[0], "")
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

            def tool(label):
                """Secondary tools live in the Strumenti dropdown; open one the way an operator does."""
                pg.get_by_role("button", name="Strumenti", exact=True).click()
                pg.get_by_role("menuitem", name=label, exact=False).first.click()
                pg.wait_for_timeout(250)

            def find(q):
                if pg.locator("#tool-objects").is_hidden():
                    tool("Oggetti sul tavolo")
                pg.fill("#q", q)
                press("Trova")
                pg.wait_for_function("document.getElementById('findbtn').textContent === 'Trova'", timeout=20000)
                return text("#findres")

            step("load", lambda: (pg.goto(base + "/"),
                                  pg.wait_for_function("document.getElementById('live').naturalWidth > 0", timeout=15000),
                                  text("#mode"))[2])
            step("preflight", lambda: (tool("Controllo preliminare"), wait_text("#preres", "Controllo camera"), text("#preres")[:80])[2])
            step("capture empty table", lambda: (tool("Camera e tavolo"), press("Cattura tavolo vuoto"), pg.wait_for_timeout(500), text("#bginfo"))[3])
            unique = _unique(pg)
            step("find by words", lambda: find(f"the {unique} one")[:90])
            step("jury: correct", lambda: (tool("Giuria e metriche"), press("Corretto"), pg.wait_for_timeout(400))[2])
            step("find absent object", lambda: find("a stapler")[:60])
            step("clarify + spoken answer", lambda: (find("the black one"),
                                                     pg.evaluate("runSpoken('number 2')"), pg.wait_for_timeout(1200),
                                                     text("#findres")[:60])[3])
            step("find all + space STOP", lambda: (pg.fill("#q", "all the black things"), press("Trova tutti"),
                                                   wait_text("#findres", "Cerco tutte"),
                                                   pg.click("body", position={"x": 5, "y": 990}), pg.keyboard.press("Space"),
                                                   pg.wait_for_timeout(600), text("#armst"))[6])
            step("what's on the table", lambda: (press("Cosa c'è sul tavolo?"), wait_text("#findres", "Do un nome"),
                                                 text("#findres")[:60])[2])
            step("spoken selection + approach refusal", lambda: (
                pg.evaluate("async () => { const s = await (await fetch('/api/state')).json(); await runSpoken('seleziona oggetto ' + s.inventory.snapshot.objects[0].track_id); }"),
                pg.evaluate("runSpoken('Ora posizionati sopra.')"), text('#agentblocker'))[2])
            if not pg.locator('#agentblocker').count() or 'calibrazione' not in text('#agentblocker'):
                problems.append('uncalibrated spoken approach did not explain its calibration blocker')
            step("confirmation refused without proposal", lambda: (pg.evaluate("runSpoken('Sì, procedi.')"), text('#agentblocker'))[1])
            if pg.get_by_role('button', name='Conferma il movimento validato', exact=True).count():
                problems.append('confirmation button exists without a valid trajectory')
            step("scan next view remains blocked", lambda: (tool('Scansione e ricostruzione'), press('Pianifica la vista successiva'), pg.wait_for_timeout(300), text('#agentblocker'))[3])
            step("reconstruction capability", lambda: pg.request.get(base + '/api/reconstruction/capabilities').json()['reason'])
            step("save reference", lambda: (tool("Line clearance e lotti"), press("Salva riferimento"), pg.wait_for_timeout(1500), text("#refinfo"))[3])
            step("set batch + old-lot carton", lambda: (pg.fill("#lot", "A123"), press("Imposta lotto"),
                                                        tool("Camera e tavolo"), press("Cartone lotto vecchio"),
                                                        pg.wait_for_timeout(600), text("#batchinfo"))[5])
            step("read labels", lambda: (tool("Line clearance e lotti"), press("Leggi etichette"),
                                         wait_text("#checkres", "Leggo le etichette"), text("#checkres")[:90])[3])
            step("draw a zone", lambda: (press("Disegna una zona (clic sugli angoli)"),
                                         [pg.mouse.click(*xy) for xy in _corners(pg)], press("Chiudi la zona"),
                                         pg.wait_for_timeout(800), text("#zoneinfo"))[4])
            step("check now", lambda: (press("Controlla ora"), wait_text("#checkres", "Controllo"), text("#checkres")[:90])[2])
            step("sign off (reject)", lambda: (pg.fill("#operator", "Smoke Test"), press("Rifiuta"),
                                               pg.wait_for_timeout(500))[2])
            step("record + PDF", lambda: (pg.request.get(base + "/api/report").ok and
                                          pg.request.get(base + "/api/report.pdf").body()[:4] == b"%PDF") or _fail("record"))
            step("watch: alarm + acknowledge", lambda: _watch(pg, press, text))
            step("tray lift + STOP", lambda: (find(f"the {_unique(pg)} one"), press("Solleva il vassoio", exact=False),
                                              pg.mouse.click(*_corners(pg)[2]), pg.wait_for_timeout(800), pg.click("#stop"),
                                              pg.wait_for_timeout(500), text("#armst"))[6])
            tool("Oggetti sul tavolo")
            step("manual image region", lambda: _manual_region(pg, press))
            step("arm home", lambda: (tool("Braccio"), press("Vai a home"), pg.wait_for_timeout(500), text("#armst"))[3])
            step("dog search", lambda: (tool("Cane robot"), press("Manda il cane a cercare"), pg.wait_for_timeout(6000),
                                        text("#dogres")[:60])[3])
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


def _manual_region(pg, press):
    press('Seleziona una regione')
    pg.wait_for_selector('#regionDialog[open]')
    pg.fill('#regionLabel', 'operator test region')
    box = pg.locator('#regionCanvas').bounding_box()
    x, y = box['x'] + box['width']*.15, box['y'] + box['height']*.15
    pg.mouse.move(x, y)
    pg.mouse.down()
    pg.mouse.move(x + box['width']*.1, y + box['height']*.1, steps=6)
    pg.mouse.up()
    pg.wait_for_function("!document.getElementById('regionDialog').open")
    assert 'operator test region' in pg.inner_text('#findres')
    press('Segna il punto sul piano')
    pg.wait_for_selector('#regionDialog[open]')
    pg.mouse.click(x + box['width']*.05, y + box['height']*.05)
    pg.wait_for_function("!document.getElementById('regionDialog').open")
    return 'draw selected a region; click bound a support-plane reference without movement'


def _corners(pg):
    pg.locator("#live").evaluate("e => e.scrollIntoView({block: 'center'})")
    b = pg.locator("#live").bounding_box()
    return [(b["x"] + b["width"] * fx, b["y"] + b["height"] * fy) for fx, fy in ((0.08, 0.1), (0.35, 0.1), (0.35, 0.4), (0.08, 0.4))]


def _watch(pg, press, text):
    press("Salva riferimento")
    pg.wait_for_timeout(1500)
    press("Sorveglia la linea")
    pg.wait_for_timeout(2000)
    pg.evaluate("sendForm('/api/demo/tamper', {kind: 'add'})")
    pg.wait_for_function("document.getElementById('alarm').style.display === 'block'", timeout=10000)
    alarm = text("#alarmtext")
    press("Preso atto")
    pg.wait_for_timeout(1200)
    press("Smetti di sorvegliare")
    return alarm[:90]


def _unique(pg):
    colors = [o.split()[0] for o in pg.evaluate("fetch('/api/state').then(r => r.json())")["source"]["objects"]]
    return next(c for c in colors if colors.count(c) == 1)


def _fail(what):
    raise AssertionError(what)


if __name__ == "__main__":
    sys.exit(main())
