"""Render ad.html frame by frame in headless Chrome.
  python render.py preview 1.5 3 5.2 ...      -> preview/sheet.jpg (contact sheet of those times)
  python render.py frames <worker> <n_workers> -> frames/%05d.jpg at 30 fps (deterministic seek)
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FPS = 30


def open_page(p):
    b = p.chromium.launch(channel="chrome", headless=True, args=["--force-color-profile=srgb", "--hide-scrollbars"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.goto((HERE / "ad.html").as_uri() + "?render=1")
    pg.wait_for_function("window.ready !== undefined")
    pg.evaluate("window.ready")
    pg.wait_for_timeout(500)
    return b, pg


def main():
    mode = sys.argv[1]
    with sync_playwright() as p:
        b, pg = open_page(p)
        if mode == "preview":
            import cv2
            import numpy as np
            out = HERE / "preview"
            out.mkdir(exist_ok=True)
            tiles = []
            for ts in sys.argv[2:]:
                t = float(ts)
                pg.evaluate(f"seek({t})")
                f = out / f"t{t:06.2f}.jpg"
                pg.screenshot(path=str(f), type="jpeg", quality=85)
                im = cv2.resize(cv2.imread(str(f)), (640, 360))
                cv2.putText(im, f"{t:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                tiles.append(im)
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
            cv2.imwrite(str(out / "sheet.jpg"), np.vstack(rows))
        else:
            k, n = int(sys.argv[2]), int(sys.argv[3])
            dur = pg.evaluate("window.DURATION")
            total = int(round(dur * FPS))
            out = HERE / "frames"
            out.mkdir(exist_ok=True)
            t0 = time.time()
            for i in range(k, total, n):
                pg.evaluate(f"seek({i / FPS})")
                pg.screenshot(path=str(out / f"{i:05d}.jpg"), type="jpeg", quality=93)
            print(f"worker {k}: {len(range(k, total, n))} frames in {time.time() - t0:.0f}s", flush=True)
        b.close()


if __name__ == "__main__":
    main()
