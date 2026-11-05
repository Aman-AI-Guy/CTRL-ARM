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

from .geom import VisionStore, pose_changed, tool_pose_matrix
from .scan3d.evidence import flange_to_tool_camera, sha256
from .scan3d.flange import FlangeCalibration, flange_pose


STATION_TEXT_FIELDS = ("camera_identity", "optical_profile", "mount_id", "robot_base_id",
                       "table_id", "measurement_note", "camera_host")


def station_binding_errors(station, digest):
    errors = []
    if not isinstance(station, dict):
        return ["station: expected a JSON object"]
    if station.get("provider") != "historical_flange":
        errors.append("provider: select historical_flange (not hardware activation)")
    if station.get("calibration_sha256") != digest:
        errors.append("calibration_sha256: must match the selected candidate")
    for name in STATION_TEXT_FIELDS:
        if not isinstance(station.get(name), str) or not station[name].strip():
            errors.append(f"{name}: independently identify/measure this field")
    if station.get("units") != "mm" or station.get("frame") != "robot_base":
        errors.append("units/frame: expected mm / robot_base")
    try:
        finite(station.get("tcp_offset_mm"), 6)
    except (ValueError, TypeError):
        errors.append("tcp_offset_mm: expected [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg] (six finite numbers); XYZ alone is insufficient, no zero offset is assumed")
    plane = station.get("table_plane")
    try:
        finite([plane[k] for k in ("a", "b", "c")], 3)
        residual = float(plane["residual_mm"])
        if not np.isfinite(residual) or not 0 <= residual <= 5:
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        errors.append("table_plane: independently measured a,b,c for z=a*x+b*y+c and residual_mm within 0..5; saved fit is not current measurement")
    return errors


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


class ProviderOperations:
    def validate_current_setup(self, image_size=None):
        return self.assess(image_size)

    def get_validity_reason(self, image_size=None):
        return self.assess(image_size).get("reasons", [])

    def get_uncertainty(self, result):
        return {
            "uncertainty_mm": result.get("uncertainty_mm"),
            "kind": result.get("uncertainty_kind", "provider_estimate"),
        }

    def localize_region(self, robot, selected, image_size):
        reference = selected.get("geometric_reference") or {}
        if reference.get("kind") not in {"support_plane_point", "known_height_point"}:
            return {
                "ok": False,
                "point_mm": None,
                "reason": "A measured geometric reference is required; a bounding-box center is not depth.",
            }
        return self.pixel_to_base(
            robot,
            reference["pixel"],
            image_size,
            height_mm=reference.get("height_mm", 0),
            height_uncertainty_mm=reference.get("height_uncertainty_mm", 0),
        )


class CurrentProvider(ProviderOperations):
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

    def is_compatible(self, observation, robot_state):
        size = observation.get("image_size") or observation.get("size")
        report = self.assess(size)
        return {"compatible": report["valid"], "reasons": report["reasons"]}

    def pixel_to_base(
        self, robot, pixel, image_size, *, height_mm=0, height_uncertainty_mm=0
    ):
        height = float(height_mm)
        height_error = float(height_uncertainty_mm)
        if (
            not np.isfinite([height, height_error]).all()
            or height < 0
            or height_error < 0
        ):
            raise ValueError(
                "Measured height and uncertainty must be finite and nonnegative"
            )
        result = self.store.localize(robot.get("pose6"), *pixel, image_size)
        result["reference_height_mm"] = height
        if not result.get("ok") or not height_mm:
            return result
        from .geom import camera_pose_matrix, pixel_ray, intersect_ray_plane

        height = float(height_mm)
        if not np.isfinite(height) or height < 0:
            raise ValueError("Measured object height must be finite and nonnegative")
        matrix = camera_pose_matrix(robot["pose6"], self.store.hand_eye["matrix"])
        ray = matrix[:3, :3] @ pixel_ray(
            *pixel,
            np.asarray(self.store.intrinsics["k"]),
            np.asarray(self.store.intrinsics["dist"]),
        )
        origin = np.asarray(self.store.table["point_mm"], float) + [0, 0, height]
        point = intersect_ray_plane(
            matrix[:3, 3], ray, origin, self.store.table["normal"]
        )
        if point is None:
            return result | {
                "ok": False,
                "point_mm": None,
                "reason": "Ray does not reach the measured height plane",
            }
        return result | {
            "point_mm": point.tolist(),
            "object_height_measured": True,
            "uncertainty_mm": result.get("uncertainty_mm", 0)
            + float(height_uncertainty_mm),
            "kind": "known_height_plane_intersection",
        }

    localize = pixel_to_base


class ImageSpaceProvider(ProviderOperations):
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

    def localize(self, robot, pixel, image_size, **_kwargs):
        return {
            "ok": False,
            "point_mm": None,
            "calibration": self.assess(image_size),
            "reason": "Only image-space localization is available.",
        }

    def is_compatible(self, observation, robot_state):
        return {
            "compatible": False,
            "reasons": ["Image-space observations have no metric calibration."],
        }

    pixel_to_base = localize


class HistoricalProvider(ProviderOperations):
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

    def station_template(self):
        return dict(provider="historical_flange", calibration_sha256=self.digest,
                    **{name: None for name in STATION_TEXT_FIELDS},
                    units="mm", frame="robot_base", tcp_offset_mm=None, table_plane=None)

    def configure_station(self, values, *, operator, confirm):
        if not confirm or not operator.strip():
            raise ValueError("Named operator confirmation of independent station measurements is required")
        allowed = set(self.station_template()) | {"recorded_by"}
        if set(values) - allowed:
            raise ValueError("Unknown station fields: " + ", ".join(sorted(set(values)-allowed)))
        errors = station_binding_errors(values, self.digest)
        if errors:
            raise ValueError("; ".join(errors))
        values = dict(values, recorded_by=operator.strip())
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "calibration_station.json").write_text(json.dumps(values,indent=2)+"\n")
        return {"station":values,"metric_permitted":False,"hardware_authorized":False,
                "reason":"Station binding saved; independent image references and separate qualification approval are still required."}

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

    def is_compatible(self, observation, robot_state):
        size = observation.get("image_size") or observation.get("size")
        reasons = []
        if list(size or []) != [self.cal.width, self.cal.height]:
            reasons.append(
                "Exact calibrated optical profile is required; no implicit resize/crop."
            )
        if (
            observation.get("rotate_reported") != 180
            or observation.get("extra_rotate") != 0
        ):
            reasons.append(
                "Calibrated server rotation is 180 degrees with zero extra rotation."
            )
        if observation.get("kind") != "pi":
            reasons.append("A fresh physically bound Pi observation is required.")
        try:
            stationary(robot_state)
        except (ValueError, TypeError) as exc:
            reasons.append(str(exc))
        return {"compatible": not reasons, "reasons": reasons}

    def assess(self, image_size=None, *, require_approval=True):
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
            binding_errors = station_binding_errors(station, self.digest)
            if binding_errors:
                raise ValueError("; ".join(binding_errors))
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
            observations = []
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
                plane = station["table_plane"]
                predicted = self.cal.pixel_to_plane(*uv,q,float(plane["c"]),(float(plane["a"]),float(plane["b"])))
                observations.append({"reference_id":row.get("reference_id"),"image":row["image"],
                    "predicted_mm":predicted.tolist(),"measured_mm":p.tolist(),
                    "signed_error_mm":(predicted-p).tolist(),"error_mm":float(np.linalg.norm(predicted-p)),
                    "measurement_uncertainty_mm":row.get("measurement_uncertainty_mm"),
                    "joints_deg":q.tolist(),"T_base_camera":self.cal.camera_pose(q).tolist(),
                    "plane_residual_mm":float(p[2]-(plane["a"]*p[0]+plane["b"]*p[1]+plane["c"]))})
                projected = self.cal.robot_to_pixel(p, q)
                errors.append(float(np.linalg.norm(predicted - p)))
                pixels.append(float(np.linalg.norm(np.asarray(projected) - uv)))
                points.append(p)
                joints.append(q)
                poses.append(row["robot"]["pose6"])
            metrics["observations"] = observations
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
                allowances = [
                    float(row["measurement_uncertainty_mm"])
                    for row in refs
                    if row.get("binding_sha256") == binding_hash
                    and row.get("calibration_sha256") == self.digest
                ]
                if any(not np.isfinite(v) or v < 0 or v > 2 for v in allowances):
                    reasons.append(
                        "Independent measurement uncertainty must be between 0 and 2 mm."
                    )
                allowance = max(allowances, default=0)
                metrics["measurement_uncertainty_mm"] = allowance
                metrics["reference_max_mm"] = max(errors)
                metrics["metric_error_bound_mm"] = (
                    metrics["reference_p95_mm"] + allowance
                )
                if (
                    metrics["metric_error_bound_mm"] > 5
                    or metrics["reference_max_mm"] + allowance > 8
                ):
                    reasons.append(
                        "Independent metric validation exceeds 5 mm p95 or 8 mm maximum, including measurement uncertainty."
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
                residual = float(measured_table["residual_mm"])
                if not 0 <= residual <= 5:
                    reasons.append(
                        "Measured current support plane exceeds 5 mm residual."
                    )
                metrics["active_plane_abc"] = abc.tolist()
                metrics["table_correction_from_saved_abc"] = (
                    abc - np.asarray(self.cal.plane)
                ).tolist()
                if points:
                    plane_errors = [
                        abs(float(p[2] - (abc[0] * p[0] + abc[1] * p[1] + abc[2])))
                        for p in points
                    ]
                    if max(plane_errors) > 5:
                        reasons.append(
                            "Independently measured support references disagree with the current plane by more than 5 mm."
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
        # Original bundle-fit pixel residual and checkerboard intrinsic RMS have different
        # nuisance parameters, data and evaluation scopes. Do not transfer its pixel gate.
        if not np.isfinite(rms) or rms < 0:
            reasons.append("Imported model lacks a finite reprojection diagnostic.")
        approval = self.qualification()
        if require_approval and not self.approval_matches(approval):
            reasons.append(
                "Current independent evidence requires an explicit operator-approved qualification record."
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
            "uncertainty_mm": metrics.get("metric_error_bound_mm"),
            "validation_policy": {
                "name": "independent_metric_references_v1",
                "p95_limit_mm": 5,
                "max_limit_mm": 8,
                "checkerboard_pixel_gate_applied": False,
            },
            "qualification": approval,
            "historical_pixel_rms_is_activation_evidence": False,
            "motion_authorized": False,
        }

    def qualification(self):
        path = self.directory / "calibration_qualification.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (ValueError, OSError):
            return {}

    def evidence_digest(self):
        path = self.directory / "calibration_references.jsonl"
        return sha256(path) if path.exists() else None

    def approval_matches(self, approval):
        try:
            return bool(
                approval.get("operator")
                and approval.get("approved_at")
                and approval.get("calibration_sha256") == self.digest
                and approval.get("binding_sha256") == self.binding_hash(self.station())
                and approval.get("evidence_sha256") == self.evidence_digest()
                and approval.get("policy") == "independent_metric_references_v1"
            )
        except (ValueError, OSError):
            return False

    def qualify(self, image_size, *, operator, confirm):
        if not confirm or not str(operator).strip():
            raise ValueError(
                "An operator must explicitly approve current independent physical qualification."
            )
        report = self.assess(image_size, require_approval=False)
        if not report["valid"]:
            raise ValueError("; ".join(report["reasons"]))
        import time

        record = {
            "operator": str(operator).strip(),
            "approved_at": time.time(),
            "calibration_sha256": self.digest,
            "binding_sha256": self.binding_hash(self.station()),
            "evidence_sha256": self.evidence_digest(),
            "policy": "independent_metric_references_v1",
            "validation": report["validation"],
            "frame": "robot_base",
            "units": "mm",
            "hardware_authorized": False,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "calibration_qualification.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return record

    @staticmethod
    def binding_hash(station):
        return hashlib.sha256(json.dumps(station, sort_keys=True).encode()).hexdigest()

    def pixel_to_base(
        self, robot, pixel, image_size, *, height_mm=0, height_uncertainty_mm=0
    ):
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
        height = float(height_mm)
        height_error = float(height_uncertainty_mm)
        if (
            not np.isfinite([height, height_error]).all()
            or height < 0
            or height_error < 0
        ):
            return base | {
                "reason": "A finite independently measured height and uncertainty are required."
            }
        a, b, c = report["validation"]["active_plane_abc"]
        point = self.cal.pixel_to_plane(*uv, robot["joints"], c + height, (a, b))
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
            + 2 * self.cal.mm_per_pixel(robot["joints"], point[2])
            + height_error,
            "uncertainty_kind": "independent_reference_p95_plus_two_pixel_reference_allowance_not_a_global_guarantee",
            "kind": "support_plane_ray_intersection",
            "object_height_measured": height > 0,
            "reference_height_mm": height,
            "reason": "Joint-dependent distorted ray intersects the independently qualified known-height plane.",
        }

    localize = pixel_to_base


class CalibrationRouter(ProviderOperations):
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
            measured = historical.station()["table_plane"]
            a, b, c = (float(measured[k]) for k in ("a", "b", "c"))
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
            "qualification": historical.qualification(),
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

    def experimental_localize_observation(self,robot,selected,image_size,*,mode="dry_run"):
        """Unqualified saved-plane estimates only; never an active calibration or motion receipt."""
        if mode not in {"dry_run","simulation"}:
            raise ValueError("Experimental estimates are restricted to read-only/dry-run/simulation")
        provider=self.historical
        if provider is None:raise ValueError("Selected imported candidate unavailable")
        compatibility=provider.is_compatible(dict(provider.camera,image_size=image_size),robot)
        if not compatibility['compatible']:raise ValueError('; '.join(compatibility['reasons']))
        if sha256(provider.path)!=provider.digest:raise ValueError("Candidate artifact changed")
        reference=selected.get('geometric_reference') or {}
        if reference.get('kind') not in {'support_plane_point','known_height_point'}:
            raise ValueError("Explicit support/known-height pixel reference is required; box center is not depth")
        uv=finite(reference.get('pixel'),2)
        if not 0<=uv[0]<provider.cal.width or not 0<=uv[1]<provider.cal.height:raise ValueError('Pixel outside image')
        box=finite(selected.get('box'),4)
        if not box[0]<=uv[0]<=box[2] or not box[1]<=uv[1]<=box[3]:raise ValueError('Reference pixel is outside selected region')
        height=float(reference.get('height_mm',0))
        if not np.isfinite(height) or height<0:raise ValueError('Invalid reference height')
        point=provider.cal.pixel_to_object_top(*uv,robot['joints'],height)
        return {'ok':True,'experimental':True,'calibration_valid':False,'metric_permitted':False,
                'hardware_authorized':False,'motion_authorized':False,'verified_tool_position':False,
                'provider':'historical_flange','calibration_sha256':provider.digest,
                'object_id':selected.get('track_id'),'source_image':reference.get('snapshot_id'),
                'box':selected['box'],'pixel':uv.tolist(),'point_mm':point.tolist(),
                'frame':'robot_base','units':'mm','robot_joints_deg':list(robot['joints']),
                'reference_height_mm':height,'T_base_camera':provider.cal.camera_pose(robot['joints']).tolist(),
                'plane_abc':list(provider.cal.plane),'plane_source':'saved_V2_fit_not_current_measurement',
                'uncertainty_mm':None,'historical_p95_mm':provider.cal.data['quality'].get('holdout_p95_mm'),
                'reason':'Experimental saved-plane coordinate; current independent accuracy, plane/mount continuity and object support assumptions are unverified.'}

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
        if not reference or reference.get("kind") not in {
            "support_plane_point",
            "known_height_point",
            "measured_base_point",
        }:
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
        if reference["kind"] == "measured_base_point":
            point = finite(reference.get("point_mm"), 3)
            uncertainty = float(reference.get("point_uncertainty_mm", float("nan")))
            if (
                not np.isfinite(uncertainty)
                or not 0 <= uncertainty <= 5
                or not reference.get("measurement_note")
            ):
                raise ValueError(
                    "A measured base point needs independent measurement provenance and uncertainty"
                )
            projected = self.base_to_pixel(point, robot, image_size)
            if not (
                box[0] <= projected[0] <= box[2] and box[1] <= projected[1] <= box[3]
            ):
                return {
                    "ok": False,
                    "point_mm": None,
                    "reason": "The measured base point does not project into the selected image region.",
                }
            if report["active_provider"] == "historical_flange":
                lo, hi = np.asarray(report["validation"]["qualified_xy_mm"])
                if (point[:2] < lo).any() or (point[:2] > hi).any():
                    return {
                        "ok": False,
                        "point_mm": None,
                        "reason": "Measured point leaves the qualified metric domain.",
                    }
            plane = self.table
            normal = np.asarray(plane["normal"], float)
            origin = np.asarray(plane["point_mm"], float)
            plane_z = (
                origin[2]
                - (
                    normal[0] * (point[0] - origin[0])
                    + normal[1] * (point[1] - origin[1])
                )
                / normal[2]
            )
            return {
                "ok": True,
                "point_mm": point.tolist(),
                "uncertainty_mm": uncertainty,
                "uncertainty_kind": "independent_operator_measurement",
                "reference_kind": "measured_base_point",
                "reference_height_mm": float(point[2] - plane_z),
                "object_height_measured": False,
                "object_center_mm": None,
                "object_id": selected.get("track_id"),
                "source_image": reference.get("snapshot_id"),
                "image_region": selected.get("box"),
                "geometric_reference": dict(reference),
                "robot_joints_deg": list(robot["joints"]),
                "frame": "robot_base",
                "units": "mm",
                "provider": report["active_provider"],
                "calibration_valid": True,
                "verified_tool_position": False,
                "rejection_reasons": [],
                "reason": "Independently measured base point corroborated against the selected calibrated image region.",
            }
        if report["active_provider"] == "historical_flange":
            provider = self.historical
        else:
            provider = CurrentProvider(self.current)
        height = (
            reference.get("height_mm", 0)
            if reference["kind"] == "known_height_point"
            else 0
        )
        result = provider.pixel_to_base(
            robot,
            pixel,
            image_size,
            height_mm=height,
            height_uncertainty_mm=reference.get("height_uncertainty_mm", 0),
        )
        return result | {
            "reference_kind": reference["kind"],
            "object_center_mm": None,
            "object_id": selected.get("track_id"),
            "source_image": reference.get("snapshot_id"),
            "image_region": selected.get("box"),
            "geometric_reference": dict(reference),
            "robot_joints_deg": list(robot.get("joints") or []),
            "frame": "robot_base",
            "units": "mm",
            "object_height_measured": reference["kind"] == "known_height_point",
            "reference_height_mm": height,
            "provider": report["active_provider"],
            "calibration_valid": bool(report["valid"]),
            "rejection_reasons": [] if result.get("ok") else [result.get("reason")],
        }

    def is_compatible(self, observation, robot_state):
        if self.historical:
            return self.historical.is_compatible(observation, robot_state)
        return CurrentProvider(self.current).is_compatible(observation, robot_state)

    def pixel_to_base(self, robot, pixel, image_size, **kwargs):
        report = self.assess(image_size)
        if report["active_provider"] == "historical_flange":
            return self.historical.pixel_to_base(robot, pixel, image_size, **kwargs)
        if report["active_provider"] == "current_tool_camera":
            return CurrentProvider(self.current).pixel_to_base(
                robot, pixel, image_size, **kwargs
            )
        return {"ok": False, "point_mm": None, "reason": "; ".join(report["reasons"])}

    localize_region = localize_observation

    def configuration_compatible(self, joints):
        q = finite(joints, 5)
        if self.current.assess(self.image_size)["valid"]:
            return True
        if not self.historical or not self.historical.approval_matches(
            self.historical.qualification()
        ):
            return False
        bound = self.historical.binding_hash(self.historical.station())
        rows = [
            r for r in self.historical.references() if r.get("binding_sha256") == bound
        ]
        if not rows:
            return False
        qs = np.asarray([r["robot"]["joints"] for r in rows], float)
        return bool((q >= qs.min(0) - 0.1).all() and (q <= qs.max(0) + 0.1).all())

    def camera_pose(self, robot):
        from .geom import camera_pose_matrix

        if self.assess(self.image_size)["active_provider"] == "historical_flange":
            return self.historical.cal.camera_pose(robot["joints"])
        if self.current.assess(self.image_size)["valid"]:
            return camera_pose_matrix(robot["pose6"], self.current.hand_eye["matrix"])
        raise ValueError("No qualified camera-to-base provider is active")

    def base_to_pixel(self, point_mm, robot, image_size):
        import cv2

        self.image_size = list(image_size)
        matrix = np.linalg.inv(self.camera_pose(robot))
        point = finite(point_mm, 3)
        camera = matrix[:3, :3] @ point + matrix[:3, 3]
        if camera[2] <= 0:
            raise ValueError("Point is behind the camera")
        intr = self.intrinsics
        uv, _ = cv2.projectPoints(
            camera.reshape(1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            np.asarray(intr["k"]),
            np.asarray(intr["dist"]),
        )
        return uv.reshape(2).tolist()

    def observation_target(self, *args, **kwargs):
        return VisionStore.observation_target(self, *args, **kwargs)

    def as_dict(self):
        return self.current.as_dict() | {"status": self.assess(self.image_size)}


def make_router(store, directory, candidate=None):
    if candidate is None:
        import os

        candidate = os.getenv("TROVA_CALIBRATION_CANDIDATE") or None
    if candidate is None:
        manifest = Path(__file__).resolve().parents[1] / "config/calibration_candidate.json"
        if manifest.exists():
            try:
                selection = json.loads(manifest.read_text())
                candidate = manifest.parent.parent / selection["candidate"]
                if sha256(candidate) != selection["sha256"]:
                    raise ValueError("Primary candidate SHA-256 differs from the explicitly selected artifact")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                store.candidate_error = f"Explicit calibration candidate unavailable: {exc}"
                return CalibrationRouter(store)
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
