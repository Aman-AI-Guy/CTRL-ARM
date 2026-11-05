"""Calibration providers over the existing VisionStore and imported flange geometry.

Historical availability is separate from physical qualification and from motion permission.
No provider opens a controller, changes calibration, rescales images or authorizes a move.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .geom import MAX_RMS_PX, VisionStore, pose_changed, tool_pose_matrix
from .scan3d.evidence import flange_to_tool_camera, sha256
from .scan3d.flange import FlangeCalibration, flange_pose


class CalibrationProvider(Protocol):
    def assess(self, image_size=None) -> dict: ...
    def localize(self, robot, pixel, image_size) -> dict: ...


def finite(values, size):
    result = np.asarray(values, float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values")
    return result


def stationary(robot):
    finite(robot.get("joints"), 5)
    finite(robot.get("pose6"), 6)
    if (
        robot.get("source") != "telemetry"
        or robot.get("moving") is not False
        or robot.get("error")
        or robot.get("error_code") != 0
    ):
        raise ValueError(
            "Stationary, fault-free real pose and five joint readings are required"
        )


class CurrentProvider:
    def __init__(self, store):
        self.store = store

    def assess(self, image_size=None):
        report = self.store.assess(image_size)
        return report | {
            "metric_permitted": report["valid"],
            "provider": "current_tool_camera",
            "frame": "robot_base",
            "units": "mm",
            "provenance": str(self.store.path),
            "required_telemetry": "stationary TCP pose, mm/degrees",
            "assumptions": "fixed tool-camera transform, measured support plane, matching calibrated optical profile",
            "depth_measured": False,
        }

    def localize(self, robot, pixel, image_size):
        return self.store.localize(robot.get("pose6"), *pixel, image_size)


class ImageSpaceProvider:
    def assess(self, image_size=None):
        return {
            "provider": "image_space",
            "valid": False,
            "metric_permitted": False,
            "units": "pixels",
            "frame": "image",
            "image_size": image_size,
            "reasons": [
                "No physically qualified metric calibration provider is active."
            ],
        }

    def localize(self, robot, pixel, image_size):
        return {
            "ok": False,
            "point_mm": None,
            "calibration": self.assess(image_size),
            "reason": "Only image-space localization is available.",
        }


class HistoricalProvider:
    """An imported model can qualify only against bound current measurements, never replay alone."""

    def __init__(self, path, directory):
        self.path, self.directory = Path(path), Path(directory)
        self.cal = FlangeCalibration.load(self.path)
        self.digest = sha256(self.path)
        cal = self.cal
        if (
            cal.width <= 0
            or cal.height <= 0
            or cal.K.shape != (3, 3)
            or cal.T_fc.shape != (4, 4)
            or not np.isfinite(cal.K).all()
            or not np.isfinite(cal.T_fc).all()
            or cal.dist.shape != (5,)
            or not np.isfinite(cal.dist).all()
            or not np.allclose(cal.K[2], [0, 0, 1])
            or cal.K[0, 0] <= 0
            or cal.K[1, 1] <= 0
            or not np.allclose(cal.T_fc[3], [0, 0, 0, 1])
            or not np.allclose(
                cal.T_fc[:3, :3].T @ cal.T_fc[:3, :3], np.eye(3), atol=1e-5
            )
            or not np.isclose(np.linalg.det(cal.T_fc[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError("Invalid imported intrinsic or rigid flange-camera matrix")
        finite(cal.plane, 3)
        self.camera = {}
        self.robot = {}

    def station(self):
        path = self.directory / "calibration_station.json"
        value = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(value, dict):
            raise ValueError("Physical station binding must be a JSON object")
        return value

    def references(self):
        path = self.directory / "calibration_references.jsonl"
        rows = (
            [json.loads(line) for line in path.read_text().splitlines() if line]
            if path.exists()
            else []
        )
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Independent references must be JSON objects")
        return rows

    def assess(self, image_size=None):
        reasons = []
        metrics = {}
        data = self.cal.data
        station = {}
        try:
            if sha256(self.path) != self.digest:
                reasons.append(
                    "Candidate artifact changed since loading; restart/review the model."
                )
            station = self.station()
            if station.get("provider") != "historical_flange":
                reasons.append(
                    "Historical provider is not explicitly selected by the operator in calibration_station.json."
                )
            if station.get("calibration_sha256") != self.digest:
                reasons.append(
                    "Current physical binding is missing or belongs to another calibration artifact."
                )
            for name in (
                "camera_identity",
                "optical_profile",
                "mount_id",
                "robot_base_id",
                "table_id",
                "measurement_note",
            ):
                if not str(station.get(name) or "").strip():
                    reasons.append(f"Current {name} has not been measured/identified.")
            if station.get("units") != "mm" or station.get("frame") != "robot_base":
                reasons.append("Physical binding must use robot_base and millimetres.")
            if list(image_size or []) != [self.cal.width, self.cal.height]:
                reasons.append(
                    f"Image resolution {image_size} differs from {[self.cal.width, self.cal.height]}; no rescale or crop is assumed."
                )
            if (
                self.camera.get("rotate_reported") != 180
                or self.camera.get("extra_rotate") != 0
            ):
                reasons.append(
                    "Current 180-degree server rotation and zero extra rotation are not verified."
                )
            if self.camera.get("kind") != "pi":
                reasons.append("Current source is not the physically bound Pi camera.")
            # The optical profile/mount identity is operator measured, never inferred from host/model name.
            if station.get("camera_host") != self.camera.get("host"):
                reasons.append(
                    "Current camera endpoint does not match the operator physical binding."
                )
            tcp = finite(station.get("tcp_offset_mm"), 6)
            stationary(self.robot)
            expected = flange_pose(self.robot["joints"]) @ tool_pose_matrix(tcp)
            actual = tool_pose_matrix(self.robot["pose6"])
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(expected[:3, :3].T @ actual[:3, :3]) - 1) / 2, -1, 1
                    )
                )
            )
            metrics["tcp_fk_mm"] = float(
                np.linalg.norm(expected[:3, 3] - actual[:3, 3])
            )
            metrics["tcp_fk_deg"] = float(angle)
            if metrics["tcp_fk_mm"] > 2 or angle > 2:
                reasons.append(
                    "Measured flange-to-TCP transform disagrees with live controller FK (2 mm / 2 degree gate)."
                )
            refs = self.references()
            errors, pixels, points, joints, poses = [], [], [], [], []
            binding_hash = self.binding_hash(station)
            for row in refs:
                if (
                    row.get("calibration_sha256") != self.digest
                    or row.get("binding_sha256") != binding_hash
                ):
                    continue
                if list(row.get("image_size") or []) != list(image_size or []):
                    continue
                if row.get(
                    "provenance"
                ) != "operator_independent_reference" or not row.get(
                    "measurement_note"
                ):
                    raise ValueError(
                        "Reference lacks independent physical measurement provenance"
                    )
                profile = row.get("camera_profile") or {}
                if (
                    profile.get("rotate_reported") != 180
                    or profile.get("extra_rotate") != 0
                    or profile.get("kind") != "pi"
                    or profile.get("host") != station.get("camera_host")
                ):
                    raise ValueError("Reference optical rotation/profile is not bound")
                stationary(row["robot"])
                image_path = self.directory / row["image"]
                if (
                    not image_path.resolve().is_relative_to(self.directory.resolve())
                    or sha256(image_path) != row["image_sha256"]
                ):
                    raise ValueError(
                        "Independent validation image is missing or changed"
                    )
                p = finite(row["measured_mm"], 3)
                uv = finite(row["pixel"], 2)
                q = finite(row["robot"]["joints"], 5)
                expected_reference = flange_pose(q) @ tool_pose_matrix(tcp)
                actual_reference = tool_pose_matrix(row["robot"]["pose6"])
                rotation_error = np.degrees(
                    np.arccos(
                        np.clip(
                            (
                                np.trace(
                                    expected_reference[:3, :3].T
                                    @ actual_reference[:3, :3]
                                )
                                - 1
                            )
                            / 2,
                            -1,
                            1,
                        )
                    )
                )
                if (
                    np.linalg.norm(expected_reference[:3, 3] - actual_reference[:3, 3])
                    > 2
                    or rotation_error > 2
                ):
                    raise ValueError(
                        "Independent reference TCP/joints disagree with the bound flange model"
                    )
                predicted = self.cal.pixel_to_plane(*uv, q, float(p[2]))
                projected = self.cal.robot_to_pixel(p, q)
                errors.append(float(np.linalg.norm(predicted - p)))
                pixels.append(float(np.linalg.norm(np.asarray(projected) - uv)))
                points.append(p)
                joints.append(q)
                poses.append(row["robot"]["pose6"])
            metrics["independent_samples"] = len(errors)
            if len(errors) < 6:
                reasons.append(
                    "Need six independent current reference observations: three measured non-collinear points from at least two diverse stationary poses."
                )
            else:
                metrics.update(
                    reference_median_mm=float(np.median(errors)),
                    reference_p95_mm=float(np.percentile(errors, 95)),
                    reference_rms_px=float(np.sqrt(np.mean(np.square(pixels)))),
                )
                if (
                    metrics["reference_p95_mm"] > 5
                    or metrics["reference_rms_px"] > MAX_RMS_PX
                ):
                    reasons.append(
                        "Independent reference validation exceeds 5 mm p95 or the existing 1.5 px reprojection gate."
                    )
                xy = np.asarray(points)[:, :2]
                if np.linalg.svd(xy - xy.mean(0), compute_uv=False)[-1] < 20:
                    reasons.append(
                        "Independent reference points do not span the work area in two axes (20 mm minimum)."
                    )
                diversity = any(
                    pose_changed(a, b, mm=30, deg=15) for a in poses for b in poses
                )
                if not diversity:
                    reasons.append(
                        "Independent references do not span two camera poses (30 mm or 15 degrees)."
                    )
                # Permit only interpolation inside the independently checked joint/XY envelope.
                current_q = finite(self.robot["joints"], 5)
                qs = np.asarray(joints)
                if (current_q < qs.min(0) - 0.1).any() or (
                    current_q > qs.max(0) + 0.1
                ).any():
                    reasons.append(
                        "Current robot joints lie outside the independently validated pose range."
                    )
                metrics["qualified_xy_mm"] = [xy.min(0).tolist(), xy.max(0).tolist()]
            measured_table = station.get("table_plane")
            if not measured_table:
                reasons.append(
                    "Current table plane requires independent measured a,b,c and residual_mm; historical table height is insufficient."
                )
            else:
                abc = finite([measured_table[k] for k in ("a", "b", "c")], 3)
                if (
                    not np.allclose(abc, self.cal.plane, atol=0.001)
                    or not 0 <= float(measured_table["residual_mm"]) <= 5
                ):
                    reasons.append(
                        "Measured current support plane differs from the candidate model or exceeds 5 mm residual."
                    )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            reasons.append(f"Physical binding/validation incomplete: {exc}")
        if "independent_samples" not in metrics:
            reasons.append(
                "Current independent physical reference measurements are not yet validated."
            )
        if not station.get("table_plane"):
            reasons.append(
                "Current table orientation/height has not been measured in robot_base millimetres."
            )
        try:
            rms = float(
                data.get("quality", {}).get("reprojection_rms_px", float("inf"))
            )
        except (ValueError, TypeError, AttributeError):
            rms = float("inf")
        if not np.isfinite(rms) or rms > MAX_RMS_PX:
            reasons.append(
                f"Candidate reprojection RMS {rms:g} px exceeds the unchanged {MAX_RMS_PX:g} px calibration gate; refine the original model with measured constraints before qualification."
            )
        return {
            "provider": "historical_flange",
            "available": True,
            "valid": not reasons,
            "metric_permitted": not reasons,
            "reasons": reasons,
            "frame": "robot_base",
            "units": "mm",
            "image_size": [self.cal.width, self.cal.height],
            "rotation_deg": 180,
            "required_telemetry": "five stationary joints in degrees plus TCP pose for FK consistency",
            "assumptions": "rigid flange-camera mounting; unchanged optics; measured support plane; no monocular depth",
            "pose_relation": "FK_flange(joints_deg) @ T_flange_camera",
            "provenance": str(self.path),
            "sha256": self.digest,
            "historical_quality": data.get("quality"),
            "validation": metrics,
            "historical_replay_is_independent_validation": False,
            "uncertainty_mm": metrics.get("reference_p95_mm"),
            "motion_authorized": False,
        }

    @staticmethod
    def binding_hash(station):
        return hashlib.sha256(json.dumps(station, sort_keys=True).encode()).hexdigest()

    def localize(self, robot, pixel, image_size):
        self.robot = robot
        report = self.assess(image_size)
        base = {
            "provider": "historical_flange",
            "calibration": report,
            "calibration_valid": report["valid"],
            "point_mm": None,
            "ok": False,
            "verified_tool_position": False,
            "frame": "robot_base",
            "units": "mm",
        }
        if not report["valid"]:
            return base | {"reason": "; ".join(report["reasons"])}
        uv = finite(pixel, 2)
        if not (0 <= uv[0] < self.cal.width and 0 <= uv[1] < self.cal.height):
            return base | {"reason": "Pixel is outside the calibrated image."}
        point = self.cal.pixel_to_table(*uv, robot["joints"])
        lo, hi = np.asarray(report["validation"]["qualified_xy_mm"])
        if (point[:2] < lo).any() or (point[:2] > hi).any():
            return base | {
                "reason": "Point lies outside the independently validated work area."
            }
        return base | {
            "ok": True,
            "point_mm": point.tolist(),
            "camera_pose": self.cal.camera_pose(robot["joints"]).tolist(),
            "uncertainty_mm": report["uncertainty_mm"]
            + 2 * self.cal.mm_per_pixel(robot["joints"], point[2]),
            "uncertainty_kind": "independent_reference_p95_plus_two_pixel_reference_allowance_not_a_global_guarantee",
            "kind": "support_plane_ray_intersection",
            "object_height_measured": False,
            "reason": "Joint-dependent distorted ray intersects the independently qualified support plane.",
        }


class CalibrationRouter:
    """Keep existing calibration editing/storage; route localization through qualified providers."""

    def __init__(self, current, historical=None):
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "historical", historical)
        object.__setattr__(self, "image_size", None)

    def __setattr__(self, name, value):
        if name in {"current", "historical", "image_size"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.current, name, value)

    def __getattr__(self, name):
        historical = self.historical
        if (
            name in {"intrinsics", "hand_eye", "table"}
            and historical
            and self._historical_active()
        ):
            if name == "intrinsics":
                return {
                    "k": historical.cal.K.tolist(),
                    "dist": historical.cal.dist.tolist(),
                    "width": historical.cal.width,
                    "height": historical.cal.height,
                }
            if name == "hand_eye":
                return {
                    "matrix": flange_to_tool_camera(
                        historical.cal.T_fc, historical.station()["tcp_offset_mm"]
                    ).tolist()
                }
            a, b, c = historical.cal.plane
            return {
                "point_mm": [0, 0, c],
                "normal": [-a, -b, 1],
                "residual_mm": historical.station()["table_plane"]["residual_mm"],
            }
        return getattr(self.current, name)

    def set_context(self, camera, robot):
        if self.historical:
            self.historical.camera, self.historical.robot = dict(camera), dict(robot)

    def _historical_active(self):
        return (
            not self.current.assess(self.image_size)["valid"]
            and self.historical.assess(self.image_size)["valid"]
        )

    def assess(self, image_size=None):
        if image_size is not None:
            self.image_size = list(image_size)
        native = CurrentProvider(self.current).assess(image_size)
        historical = self.historical.assess(image_size) if self.historical else None
        active = (
            native
            if native["valid"]
            else (historical if historical and historical["valid"] else None)
        )
        report = dict(active or native)
        report.update(
            active_provider=(active or {"provider": "image_space"})["provider"],
            providers=[
                native,
                *([historical] if historical else []),
                ImageSpaceProvider().assess(image_size),
            ],
        )
        if not active and historical:
            report["reasons"] = [*native["reasons"], *historical["reasons"]]
        error = getattr(self.current, "candidate_error", None)
        if error:
            report["reasons"] = [*report.get("reasons", []), error]
        return report

    def binding_fingerprint(self):
        if not self.historical:
            return None
        historical = self.historical
        return {
            "calibration_sha256": sha256(historical.path),
            "station": historical.station(),
            "references": historical.references(),
            "camera": {
                k: historical.camera.get(k)
                for k in ("kind", "host", "rotate_reported", "extra_rotate")
            },
        }

    def localize(self, pose6, u, v, image_size, box=None):
        if self.current.assess(image_size)["valid"]:
            return self.current.localize(pose6, u, v, image_size, box)
        return {
            "ok": False,
            "point_mm": None,
            "calibration_valid": False,
            "reason": "; ".join(self.assess(image_size)["reasons"]),
            "verified_tool_position": False,
        }

    def localize_observation(self, robot, selected, image_size):
        report = self.assess(image_size)
        reference = selected.get("geometric_reference")
        if not report["valid"]:
            return {
                "ok": False,
                "point_mm": None,
                "reason": "; ".join(report["reasons"]),
                "calibration": report,
                "calibration_valid": False,
                "verified_tool_position": False,
            }
        if not reference or reference.get("kind") != "support_plane_point":
            return {
                "ok": False,
                "point_mm": None,
                "calibration_valid": True,
                "reason": "Object height/reference is unknown. Mark a visible support-plane point for this track; a bounding-box center is not its foot or 3D center.",
                "object_height_measured": False,
                "object_center_mm": None,
                "verified_tool_position": False,
            }
        try:
            stationary(robot)
        except (ValueError, TypeError) as exc:
            return {
                "ok": False,
                "point_mm": None,
                "reason": str(exc),
                "calibration_valid": True,
            }
        pixel = finite(reference["pixel"], 2)
        box = finite(selected["box"], 4)
        if not (box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3]):
            return {
                "ok": False,
                "point_mm": None,
                "reason": "Geometric reference is outside the selected object region.",
            }
        if report["active_provider"] == "historical_flange":
            result = self.historical.localize(robot, pixel, image_size)
        else:
            result = self.current.localize(
                robot.get("pose6"), *pixel, image_size, box=selected["box"]
            )
        return result | {
            "reference_kind": "operator_marked_support_plane_point",
            "object_center_mm": None,
            "object_height_measured": False,
            "provider": report["active_provider"],
        }

    def observation_target(self, *args, **kwargs):
        return VisionStore.observation_target(self, *args, **kwargs)

    def as_dict(self):
        return self.current.as_dict() | {"status": self.assess(self.image_size)}


def make_router(store, directory, candidate=None):
    if candidate is None:
        from .scan3d.evidence import workspace

        try:
            candidate = workspace() / "calib/camera_robot.json"
        except FileNotFoundError:
            return CalibrationRouter(store)
    try:
        return CalibrationRouter(store, HistoricalProvider(candidate, directory))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        store.candidate_error = f"Imported calibration cannot be loaded: {exc}"
        return CalibrationRouter(store)
