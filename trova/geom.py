"""Camera-to-robot geometry for the arm-mounted camera.

T_base_camera = T_base_tool @ T_tool_camera.

The fixed-view homography in calibration.py is a different model. It is not reused once the
camera leaves the pose it was measured in. Intrinsics come from cv2.calibrateCamera. The
tool-to-camera mount is solved here by Park and Martin (AX = XB): this OpenCV build exposes
the CALIB_HAND_EYE_* constants and does not provide cv2.calibrateHandEye. A matrix is stored
only after that solve. Motions that do not determine the mount are refused. A missing file
stays unsolved.

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
    "Pose the arm yourself (pendant or bring-up). Then capture. Trova only reads the pose.",
    "Print a checkerboard. Measure one square and pass that size as square_mm. It is not assumed.",
    "Intrinsics: hold the board in at least 3 different orientations, capture each, then solve.",
    "Hand-eye: fix the board in the workspace. Move the arm to at least 5 well separated poses.",
    "At each pose, capture a hand-eye view. Mark on_table when the board lies flat on the table.",
    "Solve hand-eye, then the table plane. Read the reprojection error and the consistency.",
    "Set TROVA_ARM_LIMITS, TROVA_CAMERA_BODY_MM and TROVA_OBSERVE_CLEARANCE_MM from measurements.",
    "A valid report is required before an object approach can run. Hardware still needs the M2 unlock.",
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
        self.intrinsics: dict | None = None
        self.hand_eye: dict | None = None
        self.table: dict | None = None
        self.rpy_convention = "Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees, xArm tool pose"

    def add_intrinsics(self, image: np.ndarray, cols: int, rows: int, square_mm: float) -> dict:
        if square_mm <= 0:
            return {"ok": False, "reason": "square_mm has to be the measured checkerboard square"}
        corners = detect_checkerboard(image, cols, rows)
        if corners is None:
            return {"ok": False, "reason": f"no {cols}x{rows} checkerboard in this image", "count": len(self.intrinsic_views)}
        self.intrinsic_views.append({
            "corners": np.round(corners, 3).tolist(),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "cols": int(cols), "rows": int(rows), "square_mm": float(square_mm),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.intrinsics = None  # a new view invalidates a previous solve
        return {"ok": True, "count": len(self.intrinsic_views), "reason": "observation stored"}

    def add_handeye(self, image: np.ndarray, pose6, cols: int, rows: int, square_mm: float,
                    on_table: bool = False, pose_source: str = "") -> dict:
        if pose6 is None or len(pose6) < 6:
            return {"ok": False, "reason": "the controller did not report roll, pitch and yaw"}
        if square_mm <= 0:
            return {"ok": False, "reason": "square_mm has to be the measured checkerboard square"}
        corners = detect_checkerboard(image, cols, rows)
        if corners is None:
            return {"ok": False, "reason": f"no {cols}x{rows} checkerboard in this image",
                    "count": len(self.handeye_views)}
        self.handeye_views.append({
            "corners": np.round(corners, 3).tolist(),
            "pose6": [round(float(v), 4) for v in pose6[:6]],
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "cols": int(cols), "rows": int(rows), "square_mm": float(square_mm),
            "on_table": bool(on_table), "pose_source": pose_source or "unspecified",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.hand_eye = None
        self.table = None
        return {"ok": True, "count": len(self.handeye_views), "reason": "observation stored"}

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
        origins = []
        for tool, target in zip(gripper_poses, target_poses, strict=True):
            origin = tool @ tool_camera @ target @ np.array([0.0, 0.0, 0.0, 1.0])
            origins.append(origin[:3])
        spread = float(np.max(np.linalg.norm(np.array(origins) - np.mean(origins, axis=0), axis=1)))
        angles, shifts = [], []
        for i in range(len(gripper_poses)):
            for j in range(i + 1, len(gripper_poses)):
                angles.append(_rotation_angle(gripper_poses[i][:3, :3], gripper_poses[j][:3, :3]))
                shifts.append(float(np.linalg.norm(gripper_poses[i][:3, 3] - gripper_poses[j][:3, 3])))
        self.hand_eye = {
            "matrix": np.round(tool_camera, 6).tolist(),
            "method": "park_martin_ax_xb",
            "consistency_mm": round(spread, 3),
            "n_poses": len(gripper_poses),
            "max_rotation_deg": round(max(angles) if angles else 0.0, 2),
            "max_translation_mm": round(max(shifts) if shifts else 0.0, 2),
            "frame": "T_tool_camera, camera points to tool points",
        }
        return {"ok": True, "consistency_mm": round(spread, 3), "n_poses": len(gripper_poses)}

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
            "plane_residual_mm": None if not self.table else self.table["residual_mm"],
            "n_intrinsics": len(self.intrinsic_views), "n_handeye": len(self.handeye_views),
            "object_height_measured": False,
            "homography_not_used": True,
            "procedure": PROCEDURE,
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

    def observation_target(self, point_mm, clearance_mm: float | None) -> dict:
        """Tool position above the plane point. Clearance must be an operator measurement."""
        if point_mm is None:
            return {"ok": False, "target_mm": None, "reason": "there is no plane intersection to approach",
                    "verified_tool_position": False, "approximate_observation_goal": False}
        if clearance_mm is None or float(clearance_mm) <= 0:
            return {"ok": False, "target_mm": None, "verified_tool_position": False,
                    "approximate_observation_goal": False,
                    "reason": "object height is unknown and TROVA_OBSERVE_CLEARANCE_MM is not set"}
        normal = np.array(self.table["normal"], float)
        normal = normal / np.linalg.norm(normal)
        target = np.array(point_mm, float) + normal * float(clearance_mm)
        reach = float(np.linalg.norm(target))
        return {
            "ok": True, "target_mm": [round(float(v), 2) for v in target],
            "clearance_mm": float(clearance_mm), "verified_tool_position": False,
            "approximate_observation_goal": True, "object_height_measured": False,
            "orientation": {"roll": 180, "pitch": 0, "yaw": "kept from the controller"},
            "dof": 5, "ik_verified": False,
            "reach_from_base_mm": round(reach, 1),
            "within_nominal_700mm": reach <= 700.0,
            "reason": "observation pose above the plane by the measured clearance; not a contact pose",
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
