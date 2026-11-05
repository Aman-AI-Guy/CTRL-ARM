"""Picture of a plan with the real shapes: the xArm 5 meshes and the camera head at the start pose and at some views,
plus a top view of every camera position. Look at it before --go. Needs matplotlib.

    python tools/render_plan.py                         # the newest runs/*_plan.json -> same name .png
    python tools/render_plan.py runs/20261105-013000_plan.json --views 0,6,12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "model"))

import orbit_here as oh  # noqa: E402
from build_robot_model import MESHES, HEAD_STL, cad_to_head, load_stl  # noqa: E402
from robot_geometry import RobotGeometry  # noqa: E402

LINKS = ("link_base", "link1", "link2", "link3", "link4", "link5")


def placed(q, meshes, head, turn):
    """Triangles of every shape in the robot frame at joints q."""
    fr = oh.fk_frames(q)
    out = []
    for i, name in enumerate(LINKS):
        t = meshes[name]
        if i:
            f = fr[i - 1]
            t = t @ f[:3, :3].T + f[:3, 3]
        out.append(("arm", t))
    f = fr[4]
    out.append(("head", head @ (f[:3, :3] @ turn).T + f[:3, 3]))
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan", nargs="?", help="plan JSON (default: newest runs/*_plan.json)")
    ap.add_argument("--views", help="view numbers to draw, e.g. 0,6,12 (default: 3 spread over the orbit)")
    ap.add_argument("--out", help="PNG (default: next to the plan)")
    a = ap.parse_args(argv)
    path = Path(a.plan) if a.plan else max((KIT / "runs").glob("*_plan.json"), key=lambda p: p.stat().st_mtime)
    plan = json.loads(path.read_text(encoding="utf-8"))
    geo = RobotGeometry.load()
    meshes = {n: load_stl(MESHES / f"{n}.stl")[::2] * 1000.0 for n in LINKS}
    head = cad_to_head(load_stl(HEAD_STL))[::4]
    axis_f = np.array(plan["camera_axis_flange"])
    turn = geo.head_turn(axis_f)
    cam = oh.Camera(plan["camera_offset_flange"], axis_f)
    tgt = np.array(plan["target_mm"])
    ok = [p for p in plan["poses"] if p["q"]]
    if a.views:
        want = {int(v) for v in a.views.split(",")}
        chosen = [p for p in ok if p["i"] in want]
    else:
        chosen = [ok[k] for k in sorted({0, len(ok) // 3, 2 * len(ok) // 3})] if ok else []
    panels = [("start pose", np.array(plan["q0"]))] + [(f"view {p['i']} (az {p['az']:.0f})", np.array(p["q"]))
                                                       for p in chosen]

    fig = plt.figure(figsize=(5.2 * (len(panels) + 1), 5.6))
    lim = max(650.0, float(np.abs(tgt[:2]).max()) + 300)
    for k, (label, q) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels) + 1, k + 1, projection="3d")
        for kind, t in placed(q, meshes, head, turn):
            ax.add_collection3d(Poly3DCollection(t, facecolor="tab:orange" if kind == "head" else "0.78",
                                                 edgecolor="none", alpha=0.9))
        p, v = cam.ray(q)
        ax.plot(*np.array([p, p + v * np.linalg.norm(tgt - p)]).T, color="tab:blue", lw=1.2)
        ax.scatter(*tgt, color="red", s=25)
        th = np.linspace(0, 2 * np.pi, 48)
        r = plan["args"].get("obj_radius", 140)
        ax.plot(tgt[0] + r * np.cos(th), tgt[1] + r * np.sin(th), np.full(48, plan["table_z_mm"]), color="red", lw=0.6)
        ax.set_xlim(-150, lim - 150)
        ax.set_ylim(-lim / 2, lim / 2)
        ax.set_zlim(plan["table_z_mm"] - 20, lim - 170)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_title(f"{label}\nq {np.round(q).astype(int).tolist()}", fontsize=9)
    ax = fig.add_subplot(1, len(panels) + 1, len(panels) + 1)
    ax.add_patch(plt.Circle((0, 0), 70, color="0.6"))
    ax.add_patch(plt.Circle(tgt[:2], plan["args"].get("obj_radius", 140), color="red", alpha=0.15))
    for p in plan["poses"]:
        if p["q"]:
            c = p["camera_mm"]
            ax.plot([c[0], tgt[0]], [c[1], tgt[1]], color="tab:blue", lw=0.4)
            ax.plot(c[0], c[1], "o", color="tab:blue", ms=4)
            ax.annotate(str(p["i"]), c[:2], fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot(tgt[0], tgt[1], "r+", ms=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"top view: {plan['reachable']}/{len(plan['poses'])} views, J5 {plan['j5_span_deg']:.0f} deg", fontsize=9)
    fig.suptitle(f"{path.name}: xArm 5 meshes (grey) + camera head CAD (orange); blue = camera axis; red = object "
                 f"centre and keep-out circle; collision model: {plan.get('collision_model', '?')}", fontsize=10)
    fig.tight_layout()
    out = Path(a.out) if a.out else path.with_suffix(".png")
    fig.savefig(out, dpi=80)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
