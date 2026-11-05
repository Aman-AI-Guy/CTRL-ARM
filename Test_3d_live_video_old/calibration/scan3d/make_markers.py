"""Print-ready A4 PDFs of the ArUco markers (dictionary, ids and size from scan3d/config.json).

    python -m scan3d.make_markers                      # Trova calibration ids -> scan3d/markers_A4.pdf
    python -m scan3d.make_markers --sheet scan         # scan-scale ids -> scan3d/markers_scan_A4.pdf
    python -m scan3d.make_markers --set aruco.marker_mm=40

Markers are drawn as vector shapes (one filled path per marker), so they stay sharp at any print resolution.
The page has a 50 mm reference line to check the print scale. After writing, the script renders the same page to a
300 dpi PNG preview (next to the PDF), detects the markers with OpenCV and checks ids and printed size.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from .config import HERE, add_config_args, config_from_args  # noqa: E402

A4_W, A4_H = 210.0, 297.0     # mm
MM_PER_PT = 25.4 / 72
REF_LINE_MM = 50.0
PREVIEW_DPI = 300
CORNERS = {"top-left": (0, 0), "top-right": (0, 1), "bottom-left": (1, 0), "bottom-right": (1, 1),
           "back-left": (0, 0), "back-right": (0, 1), "front-left": (1, 0), "front-right": (1, 1)}


def marker_bits(dictionary: str, marker_id: int) -> np.ndarray:
    """Module grid (True = black) including the 1-module black border, e.g. 6x6 for a 4x4 dictionary."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    n = d.markerSize + 2
    img = cv2.aruco.generateImageMarker(d, marker_id, n, borderBits=1)
    return img < 128


def draw_marker(ax, bits: np.ndarray, x: float, y: float, side: float) -> None:
    """All black modules as ONE compound path: a single fill, so no hairline seams between modules."""
    n = bits.shape[0]
    m = side / n
    verts, codes = [], []
    for r in range(n):
        for c in range(n):
            if bits[r, c]:
                x0, y0 = x + c * m, y + r * m
                verts += [(x0, y0), (x0 + m, y0), (x0 + m, y0 + m), (x0, y0 + m), (x0, y0)]
                codes += [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="black", edgecolor="none", linewidth=0))


def _corner(placement: dict, i: int):
    text = str(placement.get(str(i), ""))
    return next((v for k, v in CORNERS.items() if text.endswith(k)), None)


def layout(ids: list[int], placement: dict, tile_w: float, tile_h: float, top: float, bottom: float,
           gap: float = 12.0) -> dict[int, tuple[float, float]]:
    """Top-left corner (mm) of each tile. Four markers with corner placements mirror the table layout."""
    cols = max(1, int((A4_W - 2 * 12 + gap) // (tile_w + gap)))
    if len(ids) == 4 and cols >= 2 and all(_corner(placement, i) for i in ids):
        cells = {i: _corner(placement, i) for i in ids}
        cols = 2
    else:
        cells = {i: divmod(k, cols) for k, i in enumerate(ids)}
    rows = 1 + max(r for r, _ in cells.values())
    total_w = cols * tile_w + (cols - 1) * gap
    total_h = rows * tile_h + (rows - 1) * gap
    if total_h > bottom - top:
        raise ValueError(f"{len(ids)} markers of this size do not fit on one A4 page")
    x0 = (A4_W - total_w) / 2
    y0 = top + (bottom - top - total_h) / 2
    return {i: (x0 + c * (tile_w + gap), y0 + r * (tile_h + gap)) for i, (r, c) in cells.items()}


def sheet_spec(cfg: dict, sheet: str) -> tuple[list[int], dict]:
    a = cfg["aruco"]
    if sheet == "scan":
        return [int(i) for i in a["scan_ids"]], a.get("scan_placement", {})
    return [int(i) for i in a["ids"]], a.get("placement", {})


def instructions(sheet: str, side: float, ids: list[int]) -> list[tuple[str, bool]]:
    first = [("How to use", True),
             ("1. Print at 100 % on plain matte paper. Measure the 50 mm line with a ruler; if it measures L mm, the real",
              False),
             (f"    marker side is {side:g} × L / 50 mm: put that value in aruco.marker_mm (scan3d/config.json).", False),
             ("2. Cut along the dashed lines and keep the white border: the detector needs it. Glue flat, no wrinkles.",
              False)]
    if sheet == "scan":
        return first + [
            ("3. 3D scan scale (scan3d): lay the four markers flat around the object, 5-10 cm from it (e.g. the corners",
             False),
            ("    of a 25 × 25 cm square centred on the object), never under it. Exact positions do not matter, only the",
             False),
            ("    size. They must not move during a scan. More markers in view = better scale.", False),
            (f"4. Trova calibration ignores IDs {ids[0]}-{ids[-1]}: it only uses the ids in data/markers.json (0-3).",
             False)]
    return first + [
        ("3. Trova calibration (trova/calibration.py): ID 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left of the",
         False),
        ("    table area as the camera sees it (default: corners of 600 × 400 mm). With the camera on the arm, touch each",
         False),
        ("    centre with the arm tip (Setup: Record marker from arm) to write data/markers.json.", False),
        ("4. The 3D scan scale uses the separate sheet with IDs 4-7, placed around the object.", False)]


def build_page(cfg: dict, sheet: str = "calibration"):
    a = cfg["aruco"]
    dictionary, side, border = a["dictionary"], float(a["marker_mm"]), float(a.get("white_border_mm", 12.0))
    ids, placement = sheet_spec(cfg, sheet)
    label_h = 10.0
    tile_w, tile_h = side + 2 * border, side + 2 * border + label_h

    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(A4_H, 0)            # y grows downwards, like a page
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), A4_W, A4_H, facecolor="white", edgecolor="none"))

    # Header
    title = "Trova ArUco markers: 3D scan scale" if sheet == "scan" else "Trova ArUco markers: calibration"
    ax.text(A4_W / 2, 15, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(A4_W / 2, 22, f"{dictionary}  ·  IDs {', '.join(map(str, ids))}  ·  black square side {side:.1f} mm "
            f"(white border not included)", ha="center", va="center", fontsize=9)
    ax.text(A4_W / 2, 28, "Print at 100 % / \"Actual size\" (never \"Fit to page\"). "
            f"The line below must measure exactly {REF_LINE_MM:.0f} mm.", ha="center", va="center", fontsize=8)

    # Scale reference line with ticks every 10 mm; tick to tick = 50 mm.
    lw = 0.25 / MM_PER_PT            # 0.25 mm line
    rx0, ry = (A4_W - REF_LINE_MM) / 2, 38.0
    ax.plot([rx0, rx0 + REF_LINE_MM], [ry, ry], color="black", lw=lw, solid_capstyle="butt")
    for k in range(0, int(REF_LINE_MM) + 1, 10):
        h = 3.0 if k in (0, int(REF_LINE_MM)) else 1.6
        ax.plot([rx0 + k, rx0 + k], [ry - h, ry + h], color="black", lw=lw, solid_capstyle="butt")
    ax.text(rx0, ry + 5.5, "0", ha="center", va="center", fontsize=7)
    ax.text(rx0 + REF_LINE_MM, ry + 5.5, f"{REF_LINE_MM:.0f} mm", ha="center", va="center", fontsize=7)

    # Marker tiles: dashed cut line, white border, marker, ID underneath.
    where = layout(ids, placement, tile_w, tile_h, top=48.0, bottom=246.0)
    for i in ids:
        tx, ty = where[i]
        ax.add_patch(Rectangle((tx, ty), tile_w, tile_h, facecolor="none", edgecolor="#9a9a9a",
                               lw=0.2 / MM_PER_PT, linestyle=(0, (4, 3))))
        draw_marker(ax, marker_bits(dictionary, i), tx + border, ty + border, side)
        ly = ty + 2 * border + side
        ax.text(tx + tile_w / 2, ly + 2.6, f"ID {i}", ha="center", va="center", fontsize=12, weight="bold")
        where_txt = str(placement.get(str(i), "")).replace("around the object, ", "")
        where_txt = f"  ·  {where_txt}" if where_txt else ""
        ax.text(tx + tile_w / 2, ly + 7.0, f"{dictionary}  ·  {side:g} mm{where_txt}", ha="center", va="center",
                fontsize=6.5, color="#333333")

    y = 252.0
    for text, bold in instructions(sheet, side, ids):
        ax.text(20, y, text, ha="left", va="center", fontsize=7.5, weight="bold" if bold else "normal")
        y += 4.2
    ax.text(A4_W / 2, 289, f"scan3d/make_markers.py --sheet {sheet} · {date.today().isoformat()}", ha="center",
            va="center", fontsize=6, color="#777777")
    return fig


def verify(png: Path, cfg: dict, expected_ids: list[int]) -> list[str]:
    """Detect the markers in the 300 dpi render and check ids and printed side. Returns problems (empty = ok)."""
    a = cfg["aruco"]
    img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a["dictionary"])),
                                  cv2.aruco.DetectorParameters())
    corners, found, _ = det.detectMarkers(img)
    found = [] if found is None else [int(i) for i in found.ravel()]
    problems = []
    if sorted(found) != sorted(expected_ids):
        problems.append(f"detected ids {sorted(found)}, expected {sorted(expected_ids)}")
    for c, i in zip(corners, found):
        q = c.reshape(4, 2)
        side_px = np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)])
        side_mm = side_px / PREVIEW_DPI * 25.4
        print(f"  id {i}: side {side_mm:.2f} mm in the render")
        if abs(side_mm - float(a["marker_mm"])) > 0.3:
            problems.append(f"id {i} measures {side_mm:.2f} mm, expected {a['marker_mm']}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(p)
    p.add_argument("--sheet", choices=("calibration", "scan"), default="calibration",
                   help="calibration = aruco.ids (Trova table calibration); scan = aruco.scan_ids (scan scale)")
    p.add_argument("--out", help="PDF path (default scan3d/markers_A4.pdf or scan3d/markers_scan_A4.pdf)")
    args = p.parse_args()
    cfg = config_from_args(args)
    out = Path(args.out or HERE / ("markers_scan_A4.pdf" if args.sheet == "scan" else "markers_A4.pdf"))

    plt.rcParams["pdf.fonttype"] = 42          # embed TrueType fonts
    fig = build_page(cfg, args.sheet)
    fig.savefig(out, format="pdf")              # no bbox_inches: the page must stay exactly A4
    preview = out.with_name(out.stem + "_preview.png")
    fig.savefig(preview, dpi=PREVIEW_DPI)
    plt.close(fig)

    box = re.search(rb"/MediaBox\s*\[\s*([\d.\s]+)\]", out.read_bytes())
    w_pt, h_pt = (float(v) for v in box.group(1).split()[2:4])
    print(f"wrote {out}  (page {w_pt * MM_PER_PT:.1f} x {h_pt * MM_PER_PT:.1f} mm)")
    problems = verify(preview, cfg, sheet_spec(cfg, args.sheet)[0])
    if abs(w_pt * MM_PER_PT - A4_W) > 0.2 or abs(h_pt * MM_PER_PT - A4_H) > 0.2:
        problems.append("page is not A4")
    print("check: OK" if not problems else "check FAILED: " + "; ".join(problems))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
