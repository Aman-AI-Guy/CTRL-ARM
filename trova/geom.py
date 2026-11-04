"""Camera-to-robot geometry for the arm-mounted camera.

T_base_camera = T_base_tool @ T_tool_camera.

The fixed-view homography in calibration.py is a different model. It is not reused once the
camera leaves the pose it was measured in. Intrinsics come from cv2.calibrateCamera. The
tool-to-camera mount is solved here by Park and Martin (AX = XB): this OpenCV build exposes
the CALIB_HAND_EYE_* constants and does not provide cv2.calibrateHandEye. A matrix is stored
only when that solve also passes a leave-one-out check, the board-origin consistency limit,
and the pose-diversity limits. Motions that do not determine the mount are refused. A missing
file stays unsolved.

Tool poses use the xArm convention: translation in millimetres, rotation
R = Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees. The camera frame is OpenCV (x right, y down,
z forward). A 5-axis xArm holds roll near 180 and pitch near 0 for a straight Cartesian line.

An object on the calibrated plane is the intersection of a camera ray with that plane.
One RGB image does not measure the object's height, so that point is not a verified tool pose.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

INTRINSIC_MIN = 3
HANDEYE_MIN = 5
TABLE_MIN = 3
MAX_RMS_PX = 1.5
MAX_HANDEYE_MM = 10.0
MAX_PLANE_MM = 5.0
MIN_ROTATION_DEG = 15.0
MIN_TRANSLATION_MM = 30.0

PROCEDURE = [
    "Put the physical e-stop where you can reach it. This procedure never moves the arm.",
    "Start with TROVA_ARM=sim and TROVA_ARM_READ=xarm:<robot-ip>. The reader does not enable motors, clear errors, change the TCP, or move.",
    "Confirm the Pi image. The server already rotates it 180 degrees. Do not rotate it again.",
    "Show the checkerboard. Type the measured square in millimetres. A guess is rejected.",
    "Intrinsics: capture at least 3 tilted views, then solve. The matrix is kept only when reprojection is under 1.5 px and the image size matches.",
    "Hand-eye: fix the board in the robot base. Move the arm by hand between captures. Capture at least 6 telemetry poses that are not about one axis, so one can be held out.",
    "A view is stored only when the pose is still before and after the picture, the controller error is 0, the image size matches the camera, and the corners are found.",
    "Solve hand-eye and the table plane. Consistency and the held-out board origin must be within 10 mm, and the plane residual within 5 mm. Otherwise no matrix is kept.",
    "Set TROVA_ARM_LIMITS, TROVA_CAMERA_BODY_MM, TROVA_ARM_KEEP_OUT and TROVA_OBSERVE_CLEARANCE_MM from measurements. Hardware motion is a separate procedure.",
    "Remove a bad sample from the list if its reprojection is the one above 1.5 px. The gates stay the same.",
    "When the report is valid, type a measured point on the table and read the millimetre error. A centered image is not that check.",
]

DOF_NOTE = (
    "The xArm 5 has five joints. A straight observation pose keeps roll near 180 and pitch near 0, "
    "so those two angles do not give an independent calibration rotation. "
    "Between captures, turn joint 5 and change the shoulder while the board stays fixed in the base. "
    "If the arm still cannot reach 15 degrees about more than one axis, stop the solve. "
    "Measure the flange-to-lens offset with a ruler and keep that note beside the robot. "
    "The note is not a hand-eye matrix, it is not saved as one, and it does not authorize a move."
)

FIRST_MOVE = [
    "This procedure is not run by the software. Keep the physical e-stop in reach. Software STOP is not that e-stop.",
    "Measure the workspace and set TROVA_ARM_LIMITS. Measure the table clearance and set TROVA_OBSERVE_CLEARANCE_MM.",
    "Measure the camera and tool body and set TROVA_CAMERA_BODY_MM. The gateway has no collision mesh, so collision_avoidance stays false.",
    "Write measured keep-out boxes into TROVA_ARM_KEEP_OUT, or set it to none only after checking the path. Empty means not assessed, and a hardware approach is refused.",
    "Confirm the joint and speed limits. Confirm the controller error code is 0 and the arm is still.",
    "Confirm calibration valid from real checkerboard observations: reprojection, held-out hand-eye, and the table residual.",
    "Run the task while motion is still dry-run. Read the proposed target, the straight-line samples, and the safety status before anyone enables motion.",
    "Only then set TROVA_MOTION_MODE=hardware and TROVA_HARDWARE_UNLOCK=MOVE on a process whose motion driver is the real arm. Motion still goes through MotionGateway.",
    "After the move, capture another picture and compare the tool pose with the proposal. A centered image is not a verified 3D position.",
]


def tool_pose_matrix(pose6) -> np.ndarray:
    """T_base_tool. pose6 is x, y, z millimetres and roll, pitch, yaw degrees."""
    x, y, z, roll, pitch, yaw = (float(v) for v in pose6[:6])
    rx, ry, rz = np.deg2rad([roll, pitch, yaw])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rot_z @ rot_y @ rot_x
    matrix[:3, 3] = (x, y, z)
    return matrix


def camera_pose_matrix(pose6, tool_camera: np.ndarray) -> np.ndarray:
    return tool_pose_matrix(pose6) @ np.asarray(tool_camera, float)


def pixel_ray(u: float, v: float, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    points = np.array([[[float(u), float(v)]]], np.float64)
    normalised = cv2.undistortPoints(points, np.asarray(k, float), np.asarray(dist, float).reshape(-1, 1))
    direction = np.array([normalised[0, 0, 0], normalised[0, 0, 1], 1.0], float)
    return direction / np.linalg.norm(direction)


def intersect_ray_plane(origin, direction, plane_point, plane_normal):
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    normal = np.asarray(plane_normal, float)
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-8:
        return None
    distance = float(np.dot(np.asarray(plane_point, float) - origin, normal) / denom)
    if distance <= 1.0:  # the plane has to be in front of the lens, not on it
        return None
    return origin + distance * direction


def _object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), np.float32)
    coords = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid[:, :2] = coords * float(square_mm)
    return grid


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def detect_checkerboard(image: np.ndarray, cols: int, rows: int):
    """Inner corners. None when the board is not visible. Does not invent corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (int(cols), int(rows)), flags)
    if not found or corners is None:
        return None
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return refined.reshape(-1, 2)


def annotate_checkerboard(image: np.ndarray, cols: int, rows: int):
    """Copy of the image with detected corners drawn. The original is left unchanged."""
    corners = detect_checkerboard(image, cols, rows)
    vis = image.copy()
    found = corners is not None
    if found:
        cv2.drawChessboardCorners(vis, (int(cols), int(rows)),
                                 np.asarray(corners, np.float32).reshape(-1, 1, 2), True)
    return vis, {
        "found": found,
        "corners": 0 if corners is None else int(len(corners)),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "cols": int(cols),
        "rows": int(rows),
        "reason": "checkerboard corners found" if found else f"no {cols}x{rows} checkerboard in this image",
    }


def pose_changed(before, after, mm: float = 5.0, deg: float = 2.0) -> bool:
    """True when two tool poses are far enough apart that they are not one stationary sample."""
    if not before or not after or len(before) < 6 or len(after) < 6:
        return False
    if any(abs(float(before[i]) - float(after[i])) > mm for i in range(3)):
        return True
    return any(abs(float(before[i]) - float(after[i])) > deg for i in range(3, 6))


def pose_diversity_report(views) -> dict:
    """How far the stored tool poses spread. This does not solve or store a matrix."""
    poses, sources = [], []
    for view in views or []:
        pose = view.get("pose6")
        if not pose or len(pose) < 6:
            continue
        poses.append(tool_pose_matrix(pose))
        sources.append(view.get("pose_source") or "unspecified")
    count = len(poses)
    base = {"n_poses": count, "telemetry": sources.count("telemetry"),
            "max_rotation_deg": 0.0, "max_translation_mm": 0.0, "one_axis": None,
            "ready_to_solve": False}
    if count < 2:
        return base | {"reason": (
            f"have {count} oriented poses; the held-out check needs {HANDEYE_MIN + 1}")}
    rotation, shift = _pose_diversity(poses)
    moment = np.zeros((3, 3))
    used = 0
    for i in range(count):
        for j in range(i + 1, count):
            relative = _invert_pose(poses[j]) @ poses[i]
            axis, _ = cv2.Rodrigues(relative[:3, :3])
            axis = np.asarray(axis, float).reshape(3)
            if np.linalg.norm(axis) < 1e-6:
                continue
            moment += np.outer(axis, axis)
            used += 1
    one_axis = False
    if used and moment.any():
        singular = np.linalg.svd(moment, compute_uv=False)
        if singular[0] > 1e-9 and float(singular[-1] / singular[0]) < 1e-3:
            one_axis = True
    reasons = []
    if rotation < MIN_ROTATION_DEG or shift < MIN_TRANSLATION_MM:
        reasons.append(
            f"rotation {round(rotation, 2)} deg and translation {round(shift, 2)} mm "
            f"are below {MIN_ROTATION_DEG:g} deg and {MIN_TRANSLATION_MM:g} mm")
    if one_axis:
        reasons.append("the recorded tool rotations stay close to one axis")
    if count < HANDEYE_MIN + 1:
        reasons.append(f"held-out check needs {HANDEYE_MIN + 1} poses, have {count}")
    if any(source != "telemetry" for source in sources):
        reasons.append("at least one pose is not robot telemetry")
    return base | {
        "max_rotation_deg": round(rotation, 2),
        "max_translation_mm": round(shift, 2),
        "one_axis": one_axis,
        "ready_to_solve": not reasons,
        "reason": "; ".join(reasons) if reasons else "the stored poses meet the diversity gates",
    }


def _invert_pose(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    rotation = matrix[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def _park_hand_eye(gripper_poses, target_poses):
    """T_tool_camera from Park and Martin. None when the motions leave the mount free.

    gripper_poses are T_base_tool. target_poses are T_camera_board from solvePnP.
    The returned matrix takes camera-frame points into the tool frame, which is the
    OpenCV cam2gripper result.
    """
    pairs = []
    moment = np.zeros((3, 3))
    for i in range(len(gripper_poses)):
        for j in range(i + 1, len(gripper_poses)):
            gripper_ij = _invert_pose(gripper_poses[j]) @ gripper_poses[i]
            camera_ij = target_poses[j] @ _invert_pose(target_poses[i])
            axis_g, _ = cv2.Rodrigues(gripper_ij[:3, :3])
            axis_c, _ = cv2.Rodrigues(camera_ij[:3, :3])
            axis_g = np.asarray(axis_g, float).reshape(3)
            axis_c = np.asarray(axis_c, float).reshape(3)
            if np.linalg.norm(axis_g) < 1e-6 or np.linalg.norm(axis_c) < 1e-6:
                continue
            moment += np.outer(axis_c, axis_g)
            pairs.append((gripper_ij, camera_ij))
    if len(pairs) < 2:
        return None, "the poses do not contain two distinct rotations"
    singular = np.linalg.svd(moment, compute_uv=False)
    if singular[0] < 1e-6 or float(singular[-1] / singular[0]) < 1e-3:
        return None, "the arm motions stay too close to one axis, so the camera mount is not determined"
    eigenvalues, eigenvectors = np.linalg.eigh(moment.T @ moment)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues <= 1e-12):
        return None, "the arm motions stay too close to one axis, so the camera mount is not determined"
    scale = np.diag(1.0 / np.sqrt(eigenvalues))
    rotation = eigenvectors @ scale @ eigenvectors.T @ moment.T
    left, _values, right = np.linalg.svd(rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    blocks, targets = [], []
    eye = np.eye(3)
    for gripper_ij, camera_ij in pairs:
        blocks.append(eye - gripper_ij[:3, :3])
        targets.append(gripper_ij[:3, 3] - rotation @ camera_ij[:3, 3])
    translation, _residuals, _rank, _sv = np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets), rcond=None)
    if not np.all(np.isfinite(translation)):
        return None, "the hand-eye translation did not converge"
    tool_camera = np.eye(4)
    tool_camera[:3, :3] = rotation
    tool_camera[:3, 3] = translation
    return tool_camera, ""


def _board_origins(gripper_poses, target_poses, tool_camera) -> np.ndarray:
    origins = []
    for tool, target in zip(gripper_poses, target_poses, strict=True):
        point = tool @ tool_camera @ target @ np.array([0.0, 0.0, 0.0, 1.0])
        origins.append(point[:3])
    return np.asarray(origins, float)


def _pose_diversity(gripper_poses) -> tuple[float, float]:
    angles, shifts = [], []
    for i in range(len(gripper_poses)):
        for j in range(i + 1, len(gripper_poses)):
            angles.append(_rotation_angle(gripper_poses[i][:3, :3], gripper_poses[j][:3, :3]))
            shifts.append(float(np.linalg.norm(gripper_poses[i][:3, 3] - gripper_poses[j][:3, 3])))
    return (max(angles) if angles else 0.0, max(shifts) if shifts else 0.0)


def _consistency_mm(gripper_poses, target_poses, tool_camera) -> float:
    origins = _board_origins(gripper_poses, target_poses, tool_camera)
    return float(np.max(np.linalg.norm(origins - origins.mean(axis=0), axis=1)))


def _holdout_mm(gripper_poses, target_poses) -> tuple[float | None, str]:
    """Largest leave-one-out error of the board origin, in the robot base frame.

    A fold that cannot be solved is a failure. Fewer than six poses cannot leave one out
    and still meet the five-pose minimum, so they are not accepted.
    """
    count = len(gripper_poses)
    if count < HANDEYE_MIN + 1:
        return None, f"held-out check needs at least {HANDEYE_MIN + 1} poses, have {count}"
    errors = []
    for index in range(count):
        train_g = gripper_poses[:index] + gripper_poses[index + 1:]
        train_t = target_poses[:index] + target_poses[index + 1:]
        solved, reason = _park_hand_eye(train_g, train_t)
        if solved is None:
            return None, f"held-out pose {index + 1} could not be checked: {reason}"
        trained = _board_origins(train_g, train_t, solved)
        held = _board_origins([gripper_poses[index]], [target_poses[index]], solved)[0]
        errors.append(float(np.linalg.norm(held - trained.mean(axis=0))))
    return float(max(errors)), ""


def _image_record(meta: dict | None, image: np.ndarray) -> dict:
    meta = meta or {}
    return {
        "requested_at": meta.get("requested_at"),
        "received_at": meta.get("received_at"),
        "request_s": meta.get("request_s"),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "rotate_reported": meta.get("rotate_reported"),
        "extra_rotate": meta.get("extra_rotate") or 0,
        "rotated_again": bool(meta.get("rotated_again")),
        "provenance": meta.get("provenance"),
        "kind": meta.get("kind"),
        "preview": meta.get("preview"),
        "sync_precision_claimed": False,
    }


def reference_error(predicted_mm, measured_mm, *, uncertainty_mm=None, calibration_version=None,
                    image_time=None, pose_source=None, calibration_valid=False) -> dict:
    """Compare a calibrated estimate with a point the operator measured. This never authorizes motion."""
    predicted = None if not predicted_mm else [round(float(v), 2) for v in list(predicted_mm)[:3]]
    measured = [round(float(v), 2) for v in list(measured_mm)[:3]]
    report = {
        "predicted_mm": predicted, "measured_mm": measured, "error_mm": None,
        "uncertainty_mm": uncertainty_mm, "calibration_version": calibration_version,
        "image_time": image_time, "pose_source": pose_source,
        "calibration_valid": bool(calibration_valid),
        "estimate_from_calibration": predicted is not None and bool(calibration_valid),
        "centered_image_is_not_localization": True, "accepts_motion": False,
        "moved": False,
    }
    if not calibration_valid or predicted is None:
        report["reason"] = "metric localization is blocked until a validated calibration produces a plane intersection"
        return report
    report["error_mm"] = round(float(np.linalg.norm(np.subtract(predicted, measured))), 2)
    report["reason"] = "difference between the calibrated ray and the operator's measured point"
    return report


def fit_plane(points) -> tuple[np.ndarray, np.ndarray, float] | None:
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return None
    center = pts.mean(axis=0)
    _, singular, vh = np.linalg.svd(pts - center, full_matrices=False)
    if len(singular) < 2 or singular[1] < 1.0:
        return None  # collinear: the plane is not determined
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    residual = float(np.max(np.abs((pts - center) @ normal)))
    return center, normal, residual


class VisionStore:
    """Observations plus the matrices solved from them. Starts empty."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.version = 0
        self.created = ""
        self.intrinsic_views: list[dict] = []
        self.handeye_views: list[dict] = []
        self.rejected: list[dict] = []
        self.intrinsics: dict | None = None
        self.hand_eye: dict | None = None
        self.table: dict | None = None
        self.rpy_convention = "Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees, xArm tool pose"

    def add_intrinsics(self, image: np.ndarray, cols: int, rows: int, square_mm: float,
                       image_meta: dict | None = None) -> dict:
        if square_mm <= 0:
            return {"ok": False, "reason": "square_mm has to be the measured checkerboard square",
                    "count": len(self.intrinsic_views), "corners": 0}
        corners = detect_checkerboard(image, cols, rows)
        if corners is None:
            return {"ok": False, "reason": f"no {cols}x{rows} checkerboard in this image",
                    "count": len(self.intrinsic_views), "corners": 0}
        self.intrinsic_views.append({
            "corners": np.round(corners, 3).tolist(),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "cols": int(cols), "rows": int(rows), "square_mm": float(square_mm),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image": _image_record(image_meta, image),
            "sync_precision_claimed": False,
        })
        self.intrinsics = None  # a new view invalidates a previous solve
        return {"ok": True, "count": len(self.intrinsic_views), "corners": int(len(corners)),
                "reason": "observation stored"}

    def add_handeye(self, image: np.ndarray, pose6, cols: int, rows: int, square_mm: float,
                    on_table: bool = False, pose_source: str = "", robot: dict | None = None,
                    image_meta: dict | None = None, moving: bool | None = None) -> dict:
        if moving:
            return {"ok": False, "reason": "the robot is still moving; the picture was not stored",
                    "count": len(self.handeye_views), "corners": 0}
        if robot and robot.get("error_code") not in (None, 0):
            return {"ok": False, "corners": 0, "count": len(self.handeye_views),
                    "reason": f"controller error code {robot.get('error_code')}; the pose was not stored"}
        if pose6 is None or len(pose6) < 6:
            return {"ok": False, "reason": "the controller did not report roll, pitch and yaw",
                    "count": len(self.handeye_views), "corners": 0}
        if pose_source and pose_source != "telemetry":
            return {"ok": False, "corners": 0, "count": len(self.handeye_views),
                    "reason": f"the pose source is {pose_source}, not telemetry; the view was not stored"}
        if square_mm <= 0:
            return {"ok": False, "reason": "square_mm has to be the measured checkerboard square",
                    "count": len(self.handeye_views), "corners": 0}
        corners = detect_checkerboard(image, cols, rows)
        if corners is None:
            return {"ok": False, "reason": f"no {cols}x{rows} checkerboard in this image",
                    "count": len(self.handeye_views), "corners": 0}
        self.handeye_views.append({
            "corners": np.round(corners, 3).tolist(),
            "pose6": [round(float(v), 4) for v in pose6[:6]],
            "joints": None if not robot else robot.get("joints"),
            "state_code": None if not robot else robot.get("state_code"),
            "error_code": None if not robot else robot.get("error_code"),
            "warn_code": None if not robot else robot.get("warn_code"),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "cols": int(cols), "rows": int(rows), "square_mm": float(square_mm),
            "on_table": bool(on_table), "pose_source": pose_source or "unspecified",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image": _image_record(image_meta, image),
            "read_started_at": None if not robot else robot.get("read_started_at"),
            "read_finished_at": None if not robot else robot.get("read_finished_at"),
            "sync_precision_claimed": False,
        })
        self.hand_eye = None
        self.table = None
        return {"ok": True, "count": len(self.handeye_views), "corners": int(len(corners)),
                "reason": "observation stored"}

    def note_rejection(self, kind: str, reason: str, image_path: str | None = None,
                       pose_source: str | None = None) -> dict:
        """Remember why a capture was not stored. This does not invent corners or a matrix."""
        self.rejected.append({
            "kind": kind, "reason": reason, "accepted": False,
            "image": image_path, "pose_source": pose_source,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        del self.rejected[:-40]
        counted = sum(1 for item in self.rejected if item["kind"] == kind)
        return {"ok": False, "reason": reason, "rejected": counted}

    def drop_sample(self, kind: str, index: int) -> dict:
        """Remove one accepted view and clear every solved matrix. The gates are unchanged."""
        bucket = self.intrinsic_views if kind == "intrinsics" else self.handeye_views if kind == "handeye" else None
        if bucket is None:
            return {"ok": False, "reason": "kind is intrinsics or handeye"}
        if index < 0 or index >= len(bucket):
            return {"ok": False, "reason": "no sample at that index", "count": len(bucket)}
        bucket.pop(index)
        self.intrinsics = None
        self.hand_eye = None
        self.table = None
        return {"ok": True, "reason": "sample removed; solved matrices were cleared", "count": len(bucket),
                "persisted_matrix": False}

    def intrinsic_errors(self) -> list[dict]:
        """Mean reprojection of each stored board. Empty until a solve has produced K."""
        if not self.intrinsics:
            return []
        k = np.array(self.intrinsics["k"], float)
        dist = np.array(self.intrinsics["dist"], float)
        rows = []
        for index, view in enumerate(self.intrinsic_views):
            corners = np.array(view["corners"], np.float32).reshape(-1, 1, 2)
            obj = _object_points(view["cols"], view["rows"], view["square_mm"])
            ok, rvec, tvec = cv2.solvePnP(obj, corners, k, dist)
            if not ok:
                rows.append({"index": index, "mean_px": None, "max_px": None, "reason": "solvePnP failed"})
                continue
            projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
            err = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
            rows.append({"index": index, "mean_px": round(float(err.mean()), 3),
                         "max_px": round(float(err.max()), 3), "reason": ""})
        return rows

    def next_instruction(self, spread: dict | None = None) -> str:
        """What the operator should capture or measure next. It does not relax a gate."""
        if spread is None:
            spread = pose_diversity_report(self.handeye_views)
        if len(self.intrinsic_views) < INTRINSIC_MIN:
            missing = INTRINSIC_MIN - len(self.intrinsic_views)
            return (f"Measure one square and capture {missing} more tilted checkerboard view"
                    f"{'s' if missing != 1 else ''}. The Pi image is already rotated 180 degrees. "
                    "Do not rotate it again.")
        if self.intrinsics is None:
            return ("Solve intrinsics. The matrix is kept only when reprojection is under "
                    f"{MAX_RMS_PX:g} px and every image has the same size.")
        if self.intrinsics["reprojection_rms_px"] > MAX_RMS_PX:
            return (f"Reprojection is {self.intrinsics['reprojection_rms_px']} px, above {MAX_RMS_PX:g}. "
                    "Remove the worst view and capture a sharper one. The gate is not relaxed.")
        if spread.get("one_axis"):
            return (spread.get("reason") or "the rotations stay near one axis") + " " + DOF_NOTE
        if len(self.handeye_views) < HANDEYE_MIN + 1:
            missing = HANDEYE_MIN + 1 - len(self.handeye_views)
            return (f"Fix the board in the robot base. Move the arm by hand, wait until it is still, "
                    f"and capture {missing} more telemetry pose{'s' if missing != 1 else ''}. "
                    "Change the wrist and the shoulder, not only the base. " + DOF_NOTE)
        if not spread.get("ready_to_solve"):
            return (spread.get("reason") or "the poses are not diverse enough") + " " + DOF_NOTE
        on_table = sum(1 for view in self.handeye_views if view.get("on_table"))
        if on_table < TABLE_MIN:
            return (f"Mark at least {TABLE_MIN} hand-eye views as flat on the table. "
                    f"{on_table} are marked. The plane is not invented from the other views.")
        if self.hand_eye is None or self.table is None:
            return (f"Solve hand-eye. Consistency and the held-out error must be within {MAX_HANDEYE_MM:g} mm, "
                    f"and the plane residual within {MAX_PLANE_MM:g} mm. Otherwise no matrix is stored.")
        if not self.assess(self._solved_size())["valid"]:
            return "The stored matrices failed a gate. Read the residuals. A centered image is not a calibration."
        return ("The stored gates passed. Type a measured table point and compare it with the ray. "
                "That comparison does not authorize a move.")

    def samples(self) -> dict:
        errors = {item["index"]: item for item in self.intrinsic_errors()}

        def brief(kind: str, views: list[dict]) -> list[dict]:
            rows = []
            for index, view in enumerate(views):
                image = view.get("image") or {}
                rows.append({
                    "kind": kind, "index": index, "accepted": True,
                    "width": view.get("width"), "height": view.get("height"),
                    "corners": len(view.get("corners") or []),
                    "square_mm": view.get("square_mm"), "pose_source": view.get("pose_source"),
                    "on_table": bool(view.get("on_table")), "time": view.get("time"),
                    "image": image.get("preview"), "rotate_reported": image.get("rotate_reported"),
                    "reprojection_px": None if kind != "intrinsics" else (errors.get(index) or {}).get("mean_px"),
                })
            return rows

        spread = pose_diversity_report(self.handeye_views)
        return {
            "intrinsics": brief("intrinsics", self.intrinsic_views),
            "handeye": brief("handeye", self.handeye_views),
            "rejected": list(self.rejected[-30:]),
            "n_rejected_intrinsics": sum(1 for item in self.rejected if item["kind"] == "intrinsics"),
            "n_rejected_handeye": sum(1 for item in self.rejected if item["kind"] == "handeye"),
            "spread": spread, "next_instruction": self.next_instruction(spread), "dof_note": DOF_NOTE,
            "gates": {"rms_px": MAX_RMS_PX, "handeye_mm": MAX_HANDEYE_MM, "plane_mm": MAX_PLANE_MM},
        }

    def solve(self) -> dict:
        """Solve whatever the stored observations support. Does not invent a missing matrix."""
        report = {"intrinsics": self._solve_intrinsics(), "hand_eye": None, "table": None}
        if self.intrinsics:
            report["hand_eye"] = self._solve_handeye()
        if self.hand_eye:
            report["table"] = self._solve_table()
        if any(item and item.get("ok") for item in report.values()):
            self.version += 1
            self.created = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.save()
        else:
            self.save()  # keep the observations; still no fabricated matrix
        report["version"] = self.version
        report["status"] = self.assess(self._solved_size())
        return report

    def _solve_intrinsics(self) -> dict:
        views = self.intrinsic_views
        if len(views) < INTRINSIC_MIN:
            self.intrinsics = None
            return {"ok": False, "reason": f"need at least {INTRINSIC_MIN} checkerboard images, have {len(views)}"}
        size = (views[0]["width"], views[0]["height"])
        if any((v["width"], v["height"]) != size for v in views):
            self.intrinsics = None
            return {"ok": False, "reason": "checkerboard images have different resolutions"}
        squares = {v["square_mm"] for v in views}
        if len(squares) != 1:
            self.intrinsics = None
            return {"ok": False, "reason": "every intrinsics image must use the same square_mm"}
        obj, img = [], []
        for view in views:
            obj.append(_object_points(view["cols"], view["rows"], view["square_mm"]))
            img.append(np.array(view["corners"], np.float32).reshape(-1, 1, 2))
        rms, k, dist, _rvecs, _tvecs = cv2.calibrateCamera(obj, img, size, None, None)
        self.intrinsics = {
            "k": np.round(k, 6).tolist(), "dist": np.round(dist.reshape(-1), 8).tolist(),
            "width": size[0], "height": size[1], "reprojection_rms_px": round(float(rms), 4),
            "n_images": len(views), "square_mm": views[0]["square_mm"],
        }
        return {"ok": True, "reprojection_rms_px": round(float(rms), 4), "n_images": len(views)}

    def _solve_handeye(self) -> dict:
        views = self.handeye_views
        if len(views) < HANDEYE_MIN:
            self.hand_eye = None
            return {"ok": False, "reason": f"need at least {HANDEYE_MIN} poses, have {len(views)}"}
        k = np.array(self.intrinsics["k"], float)
        dist = np.array(self.intrinsics["dist"], float)
        if any((v["width"], v["height"]) != (self.intrinsics["width"], self.intrinsics["height"]) for v in views):
            self.hand_eye = None
            return {"ok": False, "reason": "hand-eye images do not match the intrinsics resolution"}
        gripper_poses, target_poses = [], []
        for view in views:
            corners = np.array(view["corners"], np.float32).reshape(-1, 1, 2)
            obj = _object_points(view["cols"], view["rows"], view["square_mm"])
            ok, rvec, tvec = cv2.solvePnP(obj, corners, k, dist)
            if not ok:
                continue
            rotation_t, _ = cv2.Rodrigues(rvec)
            tool = tool_pose_matrix(view["pose6"])
            target = np.eye(4)
            target[:3, :3] = rotation_t
            target[:3, 3] = np.asarray(tvec, float).reshape(3)
            gripper_poses.append(tool)
            target_poses.append(target)
        if len(gripper_poses) < HANDEYE_MIN:
            self.hand_eye = None
            return {"ok": False, "reason": "solvePnP failed on too many hand-eye images"}
        tool_camera, reason = _park_hand_eye(gripper_poses, target_poses)
        if tool_camera is None:
            self.hand_eye = None
            return {"ok": False, "reason": reason}
        spread = _consistency_mm(gripper_poses, target_poses, tool_camera)
        max_rotation, max_shift = _pose_diversity(gripper_poses)
        holdout, holdout_reason = _holdout_mm(gripper_poses, target_poses)
        if spread > MAX_HANDEYE_MM:
            self.hand_eye = None
            return {"ok": False, "reason": (
                f"hand-eye consistency {round(spread, 3)} mm is above {MAX_HANDEYE_MM}; "
                "the observations were rejected")}
        if max_rotation < MIN_ROTATION_DEG or max_shift < MIN_TRANSLATION_MM:
            self.hand_eye = None
            return {"ok": False, "reason": (
                "the calibration poses do not move the camera enough "
                f"(rotation {round(max_rotation, 2)} deg, translation {round(max_shift, 2)} mm; "
                f"need {MIN_ROTATION_DEG} deg and {MIN_TRANSLATION_MM} mm)")}
        if holdout is None or holdout > MAX_HANDEYE_MM:
            self.hand_eye = None
            if holdout is None:
                return {"ok": False, "reason": holdout_reason}
            return {"ok": False, "reason": (
                f"held-out board origin error {round(holdout, 3)} mm is above {MAX_HANDEYE_MM}; "
                "the observations were rejected")}
        self.hand_eye = {
            "matrix": np.round(tool_camera, 6).tolist(),
            "method": "park_martin_ax_xb",
            "consistency_mm": round(spread, 3),
            "holdout_mm": round(holdout, 3),
            "n_poses": len(gripper_poses),
            "max_rotation_deg": round(max_rotation, 2),
            "max_translation_mm": round(max_shift, 2),
            "frame": "T_tool_camera, camera points to tool points",
        }
        return {"ok": True, "consistency_mm": round(spread, 3), "holdout_mm": round(holdout, 3),
                "n_poses": len(gripper_poses)}

    def _solve_table(self) -> dict:
        marked = [view for view in self.handeye_views if view.get("on_table")]
        if len(marked) < TABLE_MIN:
            self.table = None
            return {"ok": False, "reason": f"mark at least {TABLE_MIN} hand-eye views as on_table, have {len(marked)}"}
        k = np.array(self.intrinsics["k"], float)
        dist = np.array(self.intrinsics["dist"], float)
        tool_camera = np.array(self.hand_eye["matrix"], float)
        cloud = []
        for view in marked:
            corners = np.array(view["corners"], np.float32).reshape(-1, 1, 2)
            obj = _object_points(view["cols"], view["rows"], view["square_mm"])
            ok, rvec, tvec = cv2.solvePnP(obj, corners, k, dist)
            if not ok:
                continue
            rotation_t, _ = cv2.Rodrigues(rvec)
            board = np.eye(4)
            board[:3, :3] = rotation_t
            board[:3, 3] = np.asarray(tvec, float).reshape(3)
            base_camera = tool_pose_matrix(view["pose6"]) @ tool_camera
            for index in (0, view["cols"] - 1, len(obj) - view["cols"], len(obj) - 1):
                point = base_camera @ board @ np.array([*obj[index], 1.0])
                cloud.append(point[:3])
        fitted = fit_plane(cloud)
        if fitted is None:
            self.table = None
            return {"ok": False, "reason": "the on-table points are collinear, so the plane is not determined"}
        center, normal, residual = fitted
        self.table = {
            "point_mm": np.round(center, 3).tolist(), "normal": np.round(normal, 6).tolist(),
            "residual_mm": round(residual, 3), "n_points": len(cloud),
            "surface": "checkerboard top on the views marked on_table",
            "object_height_measured": False,
        }
        return {"ok": True, "residual_mm": round(residual, 3), "n_points": len(cloud)}

    def _solved_size(self):
        if not self.intrinsics:
            return None
        return [self.intrinsics["width"], self.intrinsics["height"]]

    def assess(self, image_size=None) -> dict:
        """Whether the solved model may be used at this image size. Never upgrades a weak solve."""
        reasons = []
        if not self.intrinsics:
            reasons.append("intrinsics are not solved")
        else:
            if not self.intrinsic_views:
                reasons.append("intrinsics have no captured observations")
            if self.intrinsics["reprojection_rms_px"] > MAX_RMS_PX:
                reasons.append(f"reprojection {self.intrinsics['reprojection_rms_px']} px is above {MAX_RMS_PX}")
            if image_size and list(image_size) != [self.intrinsics["width"], self.intrinsics["height"]]:
                reasons.append(f"calibration is {self.intrinsics['width']}x{self.intrinsics['height']}, "
                               f"the camera is now {image_size[0]}x{image_size[1]}")
        if not self.hand_eye:
            reasons.append("hand-eye is not solved")
        else:
            if not self.handeye_views:
                reasons.append("hand-eye has no captured observations")
            if self.hand_eye["consistency_mm"] > MAX_HANDEYE_MM:
                reasons.append(f"hand-eye consistency {self.hand_eye['consistency_mm']} mm is above {MAX_HANDEYE_MM}")
            if self.hand_eye.get("method") == "park_martin_ax_xb":
                holdout = self.hand_eye.get("holdout_mm")
                if holdout is None:
                    reasons.append("held-out poses were not checked")
                elif holdout > MAX_HANDEYE_MM:
                    reasons.append(f"held-out board origin error {holdout} mm is above {MAX_HANDEYE_MM}")
            if self.hand_eye["max_rotation_deg"] < MIN_ROTATION_DEG or self.hand_eye["max_translation_mm"] < MIN_TRANSLATION_MM:
                reasons.append("the calibration poses do not move the camera enough")
            if any(view.get("pose_source") != "telemetry" for view in self.handeye_views):
                reasons.append("hand-eye poses must be read from robot telemetry")
        if not self.table:
            reasons.append("the table plane is not solved")
        elif self.table["residual_mm"] > MAX_PLANE_MM:
            reasons.append(f"table residual {self.table['residual_mm']} mm is above {MAX_PLANE_MM}")
        return {
            "valid": not reasons, "status": "valid" if not reasons else "not_valid",
            "reasons": reasons, "version": self.version,
            "image_size": self._solved_size(), "rpy_convention": self.rpy_convention,
            "reprojection_rms_px": None if not self.intrinsics else self.intrinsics["reprojection_rms_px"],
            "consistency_mm": None if not self.hand_eye else self.hand_eye["consistency_mm"],
            "holdout_mm": None if not self.hand_eye else self.hand_eye.get("holdout_mm"),
            "plane_residual_mm": None if not self.table else self.table["residual_mm"],
            "n_intrinsics": len(self.intrinsic_views), "n_handeye": len(self.handeye_views),
            "object_height_measured": False,
            "homography_not_used": True,
            "procedure": PROCEDURE,
            "first_move": FIRST_MOVE,
        }

    def localize(self, pose6, u: float, v: float, image_size, box=None) -> dict:
        """Ray against the calibrated plane. No point is returned when the model is not valid."""
        report = self.assess(image_size)
        base = {"calibration_valid": report["valid"], "calibration": report, "verified_tool_position": False,
                "object_height_measured": False, "kind": "plane_intersection"}
        if not report["valid"]:
            return base | {"ok": False, "point_mm": None, "reason": "; ".join(report["reasons"])}
        if pose6 is None or len(list(pose6)) < 6:
            return base | {"ok": False, "point_mm": None, "reason": "robot orientation was not read"}
        k = np.array(self.intrinsics["k"], float)
        dist = np.array(self.intrinsics["dist"], float)
        base_camera = camera_pose_matrix(pose6, self.hand_eye["matrix"])
        direction = base_camera[:3, :3] @ pixel_ray(u, v, k, dist)
        hit = intersect_ray_plane(base_camera[:3, 3], direction, self.table["point_mm"], self.table["normal"])
        if hit is None:
            return base | {"ok": False, "point_mm": None, "reason": "the camera ray does not meet the calibrated plane"}
        uncertainty = None
        if box is not None:
            corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
            hits = []
            for cu, cv in corners:
                ray = base_camera[:3, :3] @ pixel_ray(cu, cv, k, dist)
                corner_hit = intersect_ray_plane(base_camera[:3, 3], ray, self.table["point_mm"], self.table["normal"])
                if corner_hit is not None:
                    hits.append(corner_hit)
            if hits:
                uncertainty = round(float(max(np.linalg.norm(point - hit) for point in hits)), 2)
        return base | {
            "ok": True, "point_mm": [round(float(v), 2) for v in hit],
            "uncertainty_mm": uncertainty, "camera_pose": np.round(base_camera, 4).tolist(),
            "reason": "intersection of the camera ray with the calibrated plane; object height was not measured",
        }

    def observation_target(self, point_mm, clearance_mm: float | None, yaw_deg: float | None = None) -> dict:
        """Tool position above the plane point. Clearance must be an operator measurement.

        The orientation is the xArm 5 straight-line constraint: roll 180, pitch 0, and the
        yaw read from telemetry when the caller has one. Inverse kinematics is not claimed here.
        """
        if self.table is None:
            return {"ok": False, "target_mm": None, "pose6": None, "verified_tool_position": False,
                    "approximate_observation_goal": False, "ik_verified": False,
                    "reason": "the table plane is not solved"}
        if point_mm is None:
            return {"ok": False, "target_mm": None, "pose6": None,
                    "reason": "there is no plane intersection to approach",
                    "verified_tool_position": False, "approximate_observation_goal": False,
                    "ik_verified": False}
        if clearance_mm is None or float(clearance_mm) <= 0:
            return {"ok": False, "target_mm": None, "pose6": None, "verified_tool_position": False,
                    "approximate_observation_goal": False, "ik_verified": False,
                    "reason": "object height is unknown and TROVA_OBSERVE_CLEARANCE_MM is not set"}
        normal = np.array(self.table["normal"], float)
        normal = normal / np.linalg.norm(normal)
        target = np.array(point_mm, float) + normal * float(clearance_mm)
        reach = float(np.linalg.norm(target))
        yaw = 0.0 if yaw_deg is None else float(yaw_deg)
        pose6 = [round(float(v), 2) for v in target] + [180.0, 0.0, round(yaw, 2)]
        reason = "observation pose above the plane by the measured clearance; not a contact pose"
        if yaw_deg is None:
            reason += "; yaw was not read, so 0 is a placeholder until telemetry reports it"
        return {
            "ok": True, "target_mm": [round(float(v), 2) for v in target], "pose6": pose6,
            "clearance_mm": float(clearance_mm), "verified_tool_position": False,
            "approximate_observation_goal": True, "object_height_measured": False,
            "orientation": {"roll": 180.0, "pitch": 0.0, "yaw_deg": round(yaw, 2)},
            "orientation_constraint": "xArm 5 straight line keeps roll near 180 and pitch near 0",
            "yaw_from_telemetry": yaw_deg is not None,
            "dof": 5, "ik_verified": False,
            "reach_from_base_mm": round(reach, 1),
            "within_nominal_700mm": reach <= 700.0,
            "reason": reason,
        }

    def progress(self) -> dict:
        """Counts, diversity, and residuals already measured. A failed gate leaves the matrix empty."""
        views = list(self.intrinsic_views) + list(self.handeye_views)
        squares = sorted({float(view["square_mm"]) for view in views if view.get("square_mm")})
        sizes = sorted({(view.get("width"), view.get("height")) for view in views if view.get("width")})
        spread = pose_diversity_report(self.handeye_views)
        return {
            "n_intrinsics": len(self.intrinsic_views),
            "n_handeye": len(self.handeye_views),
            "n_on_table": sum(1 for view in self.handeye_views if view.get("on_table")),
            "n_rejected_intrinsics": sum(1 for item in self.rejected if item["kind"] == "intrinsics"),
            "n_rejected_handeye": sum(1 for item in self.rejected if item["kind"] == "handeye"),
            "square_mm": squares,
            "image_sizes": [list(size) for size in sizes],
            "spread": spread,
            "intrinsics_rms_px": None if not self.intrinsics else self.intrinsics.get("reprojection_rms_px"),
            "consistency_mm": None if not self.hand_eye else self.hand_eye.get("consistency_mm"),
            "holdout_mm": None if not self.hand_eye else self.hand_eye.get("holdout_mm"),
            "plane_residual_mm": None if not self.table else self.table.get("residual_mm"),
            "per_view_px": self.intrinsic_errors(),
            "next_instruction": self.next_instruction(spread),
            "dof_note": DOF_NOTE,
            "persisted_matrix": self.hand_eye is not None and self.intrinsics is not None,
            "valid": bool(self.assess(self._solved_size())["valid"]),
        }

    def as_dict(self) -> dict:
        return {
            "version": self.version, "created": self.created, "intrinsics": self.intrinsics,
            "hand_eye": self.hand_eye, "table": self.table, "rpy_convention": self.rpy_convention,
            "n_intrinsics": len(self.intrinsic_views), "n_handeye": len(self.handeye_views),
            "status": self.assess(self._solved_size()),
        }

    def save(self) -> None:
        if self.path is None:
            return
        payload = {
            "version": self.version, "created": self.created,
            "intrinsic_views": self.intrinsic_views, "handeye_views": self.handeye_views,
            "rejected": self.rejected,
            "intrinsics": self.intrinsics, "hand_eye": self.hand_eye, "table": self.table,
            "rpy_convention": self.rpy_convention,
            "solved_from_observations": bool(self.intrinsics or self.hand_eye or self.table),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "VisionStore":
        store = cls(path)
        if not path.exists():
            return store
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return store
        if not isinstance(payload, dict):
            return store
        store.version = int(payload.get("version") or 0)
        store.created = str(payload.get("created") or "")
        store.intrinsic_views = list(payload.get("intrinsic_views") or [])
        store.handeye_views = list(payload.get("handeye_views") or [])
        store.rejected = list(payload.get("rejected") or [])
        # A matrix with no observations is ignored. It cannot be checked, so it is not used.
        if store.intrinsic_views:
            store.intrinsics = payload.get("intrinsics")
        if store.handeye_views:
            store.hand_eye = payload.get("hand_eye")
            store.table = payload.get("table")
        return store


def project_observations(k, dist, pose_pairs, cols, rows, square_mm, size) -> list[dict]:
    """Build hand-eye corner lists from a known camera model. Used by tests, not saved as a factory calibration."""
    obj = _object_points(cols, rows, square_mm)
    views = []
    for pose6, tool_camera, board_pose in pose_pairs:
        base_camera = camera_pose_matrix(pose6, tool_camera)
        cam_board = np.linalg.inv(base_camera) @ board_pose
        rvec, _ = cv2.Rodrigues(cam_board[:3, :3])
        image, _ = cv2.projectPoints(obj, rvec, cam_board[:3, 3], np.asarray(k, float), np.asarray(dist, float))
        views.append({
            "corners": np.round(image.reshape(-1, 2), 3).tolist(),
            "pose6": [float(v) for v in pose6],
            "width": int(size[0]), "height": int(size[1]),
            "cols": cols, "rows": rows, "square_mm": float(square_mm),
            "on_table": True, "time": "",
        })
    return views
