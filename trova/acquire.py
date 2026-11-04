"""Multi-view capture sessions for a later reconstruction.

Each accepted view stores the original JPEG and the robot and camera metadata that were
actually read. This module does not build a mesh, does not run COLMAP, and does not claim
that a capture is synchronised more tightly than the timestamps show.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .geom import camera_pose_matrix


def viewpoint_candidates(object_mm, clearance_mm: float | None) -> list[dict]:
    """Candidate camera positions around a plane point. None of them is sent to the robot."""
    if object_mm is None or clearance_mm is None:
        return []
    x, y, z = (float(v) for v in object_mm[:3])
    overhead = {
        "id": "overhead", "xyz_mm": [round(x, 2), round(y, 2), round(z + float(clearance_mm), 2)],
        "executable_cartesian": True, "requires_joint_move": False,
        "reason": "vertical observation pose; a later skill must still submit it to MotionGateway",
    }
    ring = []
    for index, angle in enumerate(range(0, 360, 45)):
        theta = np.deg2rad(angle)
        ring.append({
            "id": f"ring-{index}",
            "xyz_mm": [round(x + 80.0 * np.cos(theta), 2), round(y + 80.0 * np.sin(theta), 2),
                       round(z + float(clearance_mm), 2)],
            "executable_cartesian": False, "requires_joint_move": True,
            "reason": ("a side view needs a tool orientation the xArm 5 Cartesian line does not hold "
                       "(roll 180, pitch 0). This candidate is not a command."),
        })
    return [overhead, *ring]


def score_views(candidates: list[dict], captured: list[dict]) -> dict:
    """Prefer candidates far from poses already captured. Coverage is a fraction of this list, not a surface."""
    taken = []
    for view in captured:
        pose = view.get("robot_tcp_mm")
        if view.get("accepted") and pose:
            taken.append(pose)
    ranked = []
    for candidate in candidates:
        if not taken:
            distance = None
        else:
            distance = min(float(np.linalg.norm(np.array(candidate["xyz_mm"]) - np.array(pose))) for pose in taken)
        ranked.append(dict(candidate) | {"distance_from_captured_mm": None if distance is None else round(distance, 1)})
    ranked.sort(key=lambda item: -(item["distance_from_captured_mm"] or 1e9))
    accepted = len(taken)
    return {
        "next": ranked[0] if ranked else None, "candidates": ranked,
        "coverage_estimate": 0.0 if not candidates else round(min(1.0, accepted / len(candidates)), 3),
        "coverage_is_not_a_reconstruction": True, "reconstructed": False,
    }


def _quaternion(rotation: np.ndarray) -> list[float]:
    """w, x, y, z from a 3x3 rotation. COLMAP wants the world-to-camera quaternion."""
    m = np.asarray(rotation, float)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    return [round(v / norm, 8) for v in (w, x, y, z)]


class ScanBook:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, dict] = {}
        self.active: str | None = None
        self.last: str | None = None

    def start(self, object_id: str) -> dict:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        session_id = f"{stamp}-{len(self.sessions) + 1}"
        folder = self.root / session_id
        folder.mkdir(parents=True, exist_ok=True)
        session = {
            "id": session_id, "object_id": str(object_id or ""), "status": "open",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "views": [], "orientation": "as_found",
            "registration": "not run", "reconstructed": False, "folder": str(folder),
        }
        self.sessions[session_id] = session
        self.active = session_id
        self.last = session_id
        self._write_index(session)
        return self.public(session)

    def _session(self, session_id: str | None) -> dict:
        key = session_id or self.active or self.last
        if not key or key not in self.sessions:
            raise KeyError("no open scan session")
        return self.sessions[key]

    def capture(self, jpeg: bytes, meta: dict, detection: dict | None = None, session_id: str | None = None) -> dict:
        session = self._session(session_id)
        if session["status"] != "open":
            raise RuntimeError("the scan session is already finished")
        meta = dict(meta or {})
        detection = dict(detection or {})
        reasons = []
        if not jpeg or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            reasons.append("the image is not a complete JPEG")
        if meta.get("moving"):
            reasons.append("the robot was moving")
        requested, stopped = meta.get("requested_at"), meta.get("stopped_at")
        if requested is not None and stopped is not None and float(requested) < float(stopped):
            reasons.append("the image was requested before the robot settled")
        if not meta.get("pose6") and not meta.get("robot_tcp_mm"):
            reasons.append("robot pose was not read")
        if meta.get("resolution_mismatch"):
            reasons.append("the JPEG size does not match the camera status")
        view_id = f"v{len(session['views']) + 1:04d}"
        folder = Path(session["folder"])
        image_path = folder / f"{view_id}.jpg"
        if jpeg:
            image_path.write_bytes(jpeg)
        camera_pose = None
        matrix = meta.get("tool_camera")
        pose6 = meta.get("pose6")
        if matrix is not None and pose6 is not None and len(pose6) >= 6:
            camera_pose = np.round(camera_pose_matrix(pose6, matrix), 5).tolist()
        record = {
            "viewpoint_id": view_id, "session_id": session["id"], "object_id": session["object_id"],
            "orientation": session["orientation"],
            "image": str(image_path), "sha256": hashlib.sha256(jpeg or b"").hexdigest(),
            "requested_at": requested, "received_at": meta.get("received_at"),
            "request_s": meta.get("request_s"), "stopped_at": stopped,
            "width": meta.get("width"), "height": meta.get("height"),
            "camera_status": meta.get("camera_status"), "protection": meta.get("protection"),
            "robot_tcp_mm": meta.get("robot_tcp_mm"), "robot_rpy_deg": meta.get("robot_rpy_deg"),
            "joints_deg": meta.get("joints_deg"), "moving": meta.get("moving"),
            "camera_pose": camera_pose, "calibration_version": meta.get("calibration_version"),
            "calibration_valid": meta.get("calibration_valid"),
            "detection": {k: detection.get(k) for k in ("track_id", "name", "box", "association", "confidence")},
            "sync": meta.get("sync") or "unspecified", "sync_precision_claimed": False,
            "accepted": not reasons, "reject_reason": "; ".join(reasons) if reasons else None,
            "provenance": meta.get("provenance"),
        }
        (folder / f"{view_id}.json").write_text(json.dumps(record))
        session["views"].append(record)
        self._write_index(session)
        return record

    def list_views(self, session_id: str | None = None) -> list[dict]:
        return list(self._session(session_id)["views"])

    def mark_orientation(self, name: str, session_id: str | None = None) -> dict:
        """The operator turned or flipped the object. Registration between groups is not run."""
        session = self._session(session_id)
        session["orientation"] = name.strip() or "turned"
        self._write_index(session)
        return {"orientation": session["orientation"], "registration": "not run", "automatic_flip": False,
                "reconstructed": False}

    def candidates(self, object_mm, clearance_mm, session_id: str | None = None) -> dict:
        session = self._session(session_id)
        return score_views(viewpoint_candidates(object_mm, clearance_mm), session["views"])

    def finish(self, session_id: str | None = None) -> dict:
        session = self._session(session_id)
        session["status"] = "finished"
        session["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_index(session)
        if self.active == session["id"]:
            self.active = None
        return self.public(session)

    def export_colmap(self, session_id: str | None = None, dest: Path | None = None) -> dict:
        """Write images and, when a solved camera pose exists, a COLMAP images.txt. No reconstruction."""
        session = self._session(session_id)
        folder = Path(dest) if dest else Path(session["folder"]) / "colmap"
        images = folder / "images"
        images.mkdir(parents=True, exist_ok=True)
        posed = 0
        lines = []
        for index, view in enumerate(session["views"], start=1):
            source = Path(view["image"])
            if source.exists():
                (images / source.name).write_bytes(source.read_bytes())
            pose = view.get("camera_pose")
            if not view.get("accepted") or not pose:
                continue
            rotation = np.array(pose, float)[:3, :3]
            # COLMAP stores world-to-camera. We have camera-to-base, so invert the rotation block.
            world_to_cam = np.linalg.inv(np.array(pose, float))
            quat = _quaternion(world_to_cam[:3, :3])
            translation = world_to_cam[:3, 3]
            name = source.name
            lines.append(f"{index} {' '.join(str(v) for v in quat)} "
                         f"{translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} 1 {name}")
            lines.append("")
            posed += 1
            _ = rotation
        (folder / "images.txt").write_text("# trovacolmap poses are camera-to-base inverted into world-to-camera\n"
                                           "# reconstructed: false\n" + "\n".join(lines) + ("\n" if lines else ""))
        (folder / "points3D.txt").write_text("# no points: reconstruction has not been run\n")
        intrinsics = session.get("intrinsics")
        if intrinsics and intrinsics.get("k"):
            k = intrinsics["k"]
            cameras = (f"1 PINHOLE {intrinsics['width']} {intrinsics['height']} "
                       f"{k[0][0]} {k[1][1]} {k[0][2]} {k[1][2]}\n")
        else:
            cameras = "# camera model unknown: intrinsics were not valid for this session\n"
        (folder / "cameras.txt").write_text(cameras)
        return {"reconstructed": False, "images": len(session["views"]), "posed": posed,
                "folder": str(folder), "textured_mesh": False, "cad": False,
                "note": "images and optional poses only; COLMAP has not been run"}

    def public(self, session: dict) -> dict:
        accepted = [view for view in session["views"] if view.get("accepted")]
        return {
            "id": session["id"], "object_id": session["object_id"], "status": session["status"],
            "orientation": session["orientation"], "registration": "not run", "reconstructed": False,
            "n_views": len(session["views"]), "n_accepted": len(accepted),
            "views": [{k: view.get(k) for k in ("viewpoint_id", "accepted", "reject_reason", "image",
                                                "robot_tcp_mm", "calibration_version", "sync")}
                      for view in session["views"]],
        }

    def _write_index(self, session: dict) -> None:
        payload = {k: v for k, v in session.items() if k != "views"}
        payload["views"] = session["views"]
        payload["reconstructed"] = False
        (Path(session["folder"]) / "session.json").write_text(json.dumps(payload))
