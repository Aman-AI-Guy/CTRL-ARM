"""Read-only comparison of retained calibration fits; never independent physical qualification."""

import argparse
import json
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
from scipy.optimize import least_squares

from .scan3d import marker_fit as fit
from .scan3d.flange import FlangeCalibration, flange_pose
from .scan3d.evidence import sha256

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "Test 3d live video/calib/v2/camera_robot.json"
OLD = REPO / "Test_3d_live_video_old/calibration/calib/camera_robot.json"


def stats(values):
    a = np.asarray(values, float)
    return (
        {
            "samples": len(a),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "rms": float(np.sqrt(np.mean(a * a))),
        }
        if len(a)
        else {"samples": 0}
    )


def observations(root, data):
    # Preserve legs (the cubes were moved between A/B/C), original start-image de-duplication
    # and exact recorded outlier exclusions; no writes or original SDK imports.
    groups = {}
    for source in data["source"]["snapshots"]:
        parts = PureWindowsPath(source).parts
        index = parts.index("calib2")
        groups.setdefault(parts[index + 1], []).append(
            root / "arm_pc/runs" / Path(*parts[index:])
        )
    images, obs, size, side = fit.load_legs(
        list(groups.values()),
        {
            "aruco_scan": {
                "dictionary": "DICT_ARUCO_MIP_36h12",
                "ids": [2, 3, 4],
                "marker_mm": 39,
            }
        },
    )
    dropped = {
        (row[0], tuple(row[1]))
        for row in data["source"].get("dropped_observations", [])
    }
    obs = [row for row in obs if (images[row[0]]["file"], row[1]) not in dropped]
    return images, obs, size, side


def marker_transform(corners, side):
    corners = np.asarray(corners, float)
    center = corners.mean(0)
    u, _, vt = np.linalg.svd((corners - center).T @ fit.corners_obj(side))
    transform = np.eye(4)
    transform[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    transform[:3, 3] = center
    return transform


def project(cal, q, points):
    inv = np.linalg.inv(cal.camera_pose(q))
    camera = points @ inv[:3, :3].T + inv[:3, 3]
    uv, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), cal.K, cal.dist)
    return uv.reshape(-1, 2)


def compare(v2=V2, old=OLD):
    v2, old = Path(v2), Path(old)
    newdata, olddata = json.loads(v2.read_text()), json.loads(old.read_text())
    root = v2.parents[2] if v2.parent.name == "v2" else v2.parents[1]
    images, obs, size, side = observations(root, newdata)
    result = {
        "provenance": "recorded_offline_comparison",
        "independent_current_validation": False,
        "valid_for_motion": False,
        "new_sha256": sha256(v2),
        "old_sha256": sha256(old),
        "new_root_byte_identical": v2.read_bytes()
        == (root / "calib/camera_robot.json").read_bytes(),
        "images": len(images),
        "observations": len(obs),
        "corner_samples": 4 * len(obs),
        "models": {},
        "full_parameters": {"v2": newdata, "old": olddata},
        "parameter_delta": {
            key: (np.asarray(newdata[key]) - np.asarray(olddata[key])).tolist()
            for key in ["K", "dist", "T_flange_camera"]
        },
        "ruler_fit_residual_mm": stats(
            [abs(row["fitted_gap_mm"] - row["gap_mm"]) for row in newdata["ruler"]]
        ),
        "ruler_status": "All seven ruler constraints informed the fit/plane; not independent validation.",
    }
    rotation = (
        np.asarray(olddata["T_flange_camera"])[:3, :3].T
        @ np.asarray(newdata["T_flange_camera"])[:3, :3]
    )
    result["mount_rotation_delta_deg"] = float(
        np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    )
    geometry = {
        mid: marker_transform(
            newdata["markers_by_leg"][fit.marker_name(mid)]["corners_mm"], side
        )
        for mid in sorted({o[1] for o in obs})
    }
    raw_errors = []
    newcal = FlangeCalibration(newdata)
    for index, mid, pixels in obs:
        points = fit.corners_obj(side) @ geometry[mid][:3, :3].T + geometry[mid][:3, 3]
        raw_errors.extend(
            np.linalg.norm(project(newcal, images[index]["q"], points) - pixels, axis=1)
        )
    result["v2_saved_fit_reprojection_px"] = stats(raw_errors)
    # Fair same-image comparison: each fixed camera model gets train-only nuisance marker
    # poses for every leg. Neither optical/mount parameters nor ruler constraints are refit.
    for label, data in [("v2", newdata), ("old", olddata)]:
        cal = FlangeCalibration(data)
        all_mm, all_px = [], []
        per_leg = {}
        for split in (0, 1):
            for mid, initial in geometry.items():
                train = [o for o in obs if o[1] == mid and o[0] % 2 == split]
                test = [o for o in obs if o[1] == mid and o[0] % 2 != split]
                if len(train) < 2 or not test:
                    continue

                def residual(x):
                    tm = fit.T_from(x[:3], x[3:])
                    points = fit.corners_obj(side) @ tm[:3, :3].T + tm[:3, 3]
                    return np.concatenate(
                        [
                            (project(cal, images[i]["q"], points) - uv).ravel()
                            for i, _, uv in train
                        ]
                    )

                solved = least_squares(
                    residual, fit.rt_of(initial), loss="huber", f_scale=2, max_nfev=150
                )
                tm = fit.T_from(solved.x[:3], solved.x[3:])
                points = fit.corners_obj(side) @ tm[:3, :3].T + tm[:3, 3]
                normal = tm[:3, 2]
                for index, _, uv in test:
                    px = np.linalg.norm(
                        project(cal, images[index]["q"], points) - uv, axis=1
                    )
                    all_px.extend(px)
                    for pixel, reference in zip(uv, points):
                        c, d = cal.ray(*pixel, images[index]["q"])
                        scale = float(normal @ (reference - c) / (normal @ d))
                        error = float(np.linalg.norm(c + scale * d - reference))
                        all_mm.append(error)
                        per_leg.setdefault(str(mid[0]), []).append(error)
        result["models"][label] = {
            "same_image_heldout_px": stats(all_px),
            "same_image_heldout_plane_mm": stats(all_mm),
            "by_leg_mm": {k: stats(v) for k, v in per_leg.items()},
            "excluded_corners_insufficient_training_views": 4 * len(obs) - len(all_mm),
            "procedure": "Even/odd split; fixed optical/mount calibration; train-only per-leg marker poses fitted with Huber loss. Fitted marker-plane errors are predictive consistency, not independent metric accuracy.",
        }
    # Replay original split with saved camera/marker seed and the SAME fit-derived ruler priors.
    problem = fit.Problem(
        [flange_pose(row["q"]) for row in images],
        obs,
        side,
        size,
        (size[0] / 2) / np.tan(np.radians(62.2 / 2)),
    )
    x = np.array(
        [
            newcal.K[0, 0],
            newcal.K[0, 2],
            newcal.K[1, 2],
            *newcal.dist[:2],
            *fit.rt_of(newcal.T_fc),
            *[v for mid in problem.mids for v in fit.rt_of(geometry[mid])],
        ]
    )
    ruler = [
        (tuple(row["key"]), row["head_bottom_mm"][2] - row["gap_mm"], 1.5)
        for row in newdata["ruler"]
        if row["key"] is not None
    ]
    heldout = []
    for part in (0, 1):
        train = [o for o in obs if o[0] % 2 == part]
        test = [o for o in obs if o[0] % 2 != part]
        p = fit.Problem(problem.Tbf, train, side, size, problem.f0)
        p.mids, p.ruler = problem.mids, ruler
        xf = fit.lm(p, x, iters=30)
        heldout.extend(
            fit.backproject_errors(
                p,
                xf,
                problem.Tbf,
                test,
                {
                    m: (fit.corners_obj(side) @ t[:3, :3].T + t[:3, 3], t[:3, 2])
                    for m, t in geometry.items()
                },
            )
        )
    result["v2_original_split_saved_seed_mm"] = stats(heldout)
    result["original_split_limit"] = (
        "Full-fit initialization, full-fit rounded marker planes and fitted ruler priors remain in both splits; not independent held-out physical references."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, default=V2)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.new, args.old)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "full_parameters"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
