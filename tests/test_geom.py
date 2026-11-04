"""Camera geometry. Matrices here are test fixtures, not a calibration of the real arm."""

import json

import cv2
import numpy as np

from trova.boxes import Box
from trova.geom import (
    FIRST_MOVE, PROCEDURE, VisionStore, annotate_checkerboard, camera_pose_matrix,
    detect_checkerboard, intersect_ray_plane, pixel_ray, pose_changed, pose_diversity_report,
    project_observations, tool_pose_matrix,
)
from trova.scene import TrackState, assign_tracks


def _checker(cols=5, rows=4, square=40, margin=50):
    height = (rows + 1) * square + 2 * margin
    width = (cols + 1) * square + 2 * margin
    image = np.full((height, width, 3), 255, np.uint8)
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2 == 0:
                y0, x0 = margin + row * square, margin + col * square
                image[y0:y0 + square, x0:x0 + square] = 0
    return image


def _projected_board_views():
    """Exact corners of one board seen from several angles. A drawn frontal copy is degenerate."""
    k = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    obj = np.zeros((5 * 6, 3), np.float32)
    obj[:, :2] = np.mgrid[0:6, 0:5].T.reshape(-1, 2) * 20.0
    poses = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 400.0)),
        ((0.35, 0.15, 0.05), (-30.0, 15.0, 380.0)),
        ((-0.25, 0.40, -0.10), (25.0, -20.0, 420.0)),
        ((0.15, -0.30, 0.20), (-15.0, 35.0, 360.0)),
        ((-0.40, -0.10, 0.12), (40.0, 10.0, 450.0)),
    ]
    views = []
    for rvec, tvec in poses:
        image, _jac = cv2.projectPoints(obj, np.array(rvec, float), np.array(tvec, float), k, np.zeros(5))
        views.append({
            "corners": np.round(image.reshape(-1, 2), 3).tolist(),
            "width": 640, "height": 480, "cols": 6, "rows": 5, "square_mm": 20.0, "time": "",
        })
    return views


def test_camera_to_base_is_the_tool_pose_times_the_mount():
    pose = (10.0, -20.0, 300.0, 180.0, 0.0, 25.0)
    mount = tool_pose_matrix((0.0, 0.0, 40.0, 0.0, 0.0, 0.0))
    camera = camera_pose_matrix(pose, mount)
    assert np.allclose(camera, tool_pose_matrix(pose) @ mount)
    ray = pixel_ray(320.0, 240.0, np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]]), np.zeros(5))
    assert abs(np.linalg.norm(ray) - 1.0) < 1e-6


def test_checkerboard_annotation_draws_only_detected_corners():
    image = _checker()
    drawn, info = annotate_checkerboard(image, 5, 4)
    assert info["found"] is True and info["corners"] == 20
    assert drawn.shape == image.shape
    assert not np.array_equal(drawn, image)
    blank, missing = annotate_checkerboard(np.full_like(image, 255), 5, 4)
    assert missing["found"] is False and missing["corners"] == 0
    assert np.array_equal(blank, np.full_like(image, 255))


def test_pose_diversity_rejects_one_axis_and_a_short_set():
    assert pose_changed([0, 0, 200, 180, 0, 0], [4, 0, 200, 180, 0, 0]) is False
    assert pose_changed([0, 0, 200, 180, 0, 0], [0, 0, 200, 180, 0, 3]) is True
    narrow = [{"pose6": [i * 40, 0, 200, roll, 0, 0], "pose_source": "telemetry"}
              for i, roll in enumerate((0, 20, 40, 60, 80, 100))]
    report = pose_diversity_report(narrow)
    assert report["n_poses"] == 6 and report["one_axis"] is True
    assert report["ready_to_solve"] is False and "one axis" in report["reason"]
    mixed = [dict(view, pose_source="simulated") for view in narrow]
    assert "not robot telemetry" in pose_diversity_report(mixed)["reason"]
    short = pose_diversity_report(narrow[:2])
    assert short["ready_to_solve"] is False and "held-out" in short["reason"]


def test_observation_pose_keeps_the_xarm_orientation_constraint():
    store = _fixture_store()
    missing = store.observation_target([100.0, 50.0, 0.0], 150.0)
    assert missing["yaw_from_telemetry"] is False and missing["ik_verified"] is False
    assert missing["pose6"][3:] == [180.0, 0.0, 0.0]
    assert "placeholder" in missing["reason"]
    aimed = store.observation_target([100.0, 50.0, 0.0], 150.0, yaw_deg=25.0)
    assert aimed["yaw_from_telemetry"] is True
    assert aimed["pose6"] == [100.0, 50.0, 150.0, 180.0, 0.0, 25.0]
    assert aimed["orientation_constraint"]
    assert "never moves" in PROCEDURE[0]
    assert any("physical e-stop" in step for step in FIRST_MOVE)
    assert any("MotionGateway" in step for step in FIRST_MOVE)


def test_handeye_capture_rejects_a_moving_or_faulted_controller():
    image = _checker()
    store = VisionStore()
    moving = store.add_handeye(image, [0, 0, 200, 180, 0, 0], 5, 4, 25,
                               pose_source="telemetry", moving=True)
    assert moving["ok"] is False and store.handeye_views == []
    fault = store.add_handeye(image, [0, 0, 200, 180, 0, 0], 5, 4, 25, pose_source="telemetry",
                              robot={"error_code": 22, "joints": [1, 2, 3, 4, 5]})
    assert fault["ok"] is False and "22" in fault["reason"]
    simulated = store.add_handeye(image, [0, 0, 200, 180, 0, 0], 5, 4, 25, pose_source="simulated")
    assert simulated["ok"] is False and "not telemetry" in simulated["reason"]
    stored = store.add_handeye(
        image, [0, 0, 200, 180, 0, 0], 5, 4, 25, pose_source="telemetry", on_table=True,
        robot={"error_code": 0, "joints": [1, 2, 3, 4, 5], "state_code": 2, "read_finished_at": 3.0},
        image_meta={"rotate_reported": 180, "extra_rotate": 0, "provenance": "raspberry_pi", "kind": "pi"})
    assert stored["ok"] and stored["corners"] == 20
    view = store.handeye_views[0]
    assert view["pose_source"] == "telemetry" and view["joints"] == [1, 2, 3, 4, 5]
    assert view["image"]["rotate_reported"] == 180 and view["sync_precision_claimed"] is False
    progress = store.progress()
    assert progress["n_handeye"] == 1 and progress["n_on_table"] == 1
    assert progress["persisted_matrix"] is False and progress["valid"] is False


def test_checkerboard_detection_and_intrinsics():
    corners = detect_checkerboard(_checker(), 5, 4)
    assert corners is not None and len(corners) == 20
    assert np.allclose(corners[0], [90.0, 90.0], atol=2.0)
    frontal = VisionStore()
    for shift in (0, 12, 24):
        image = _checker()
        moved = np.full_like(image, 255)
        moved[shift:, shift:] = image[:image.shape[0] - shift, :image.shape[1] - shift]
        added = frontal.add_intrinsics(moved, 5, 4, square_mm=25)
        assert added["ok"]
    frontal.solve()
    assert frontal.intrinsics["reprojection_rms_px"] > 1.5
    assert frontal.assess(None)["valid"] is False
    store = VisionStore()
    store.intrinsic_views = _projected_board_views()
    solved = store.solve()
    assert solved["intrinsics"]["ok"]
    assert store.intrinsics["reprojection_rms_px"] < 1.5
    assert solved["status"]["valid"] is False  # no hand-eye yet, and these poses are not telemetry


def test_empty_store_has_no_matrix_and_ignores_a_matrix_without_observations(tmp_path):
    path = tmp_path / "vision_calib.json"
    store = VisionStore(path)
    assert store.intrinsics is None and store.hand_eye is None
    assert store.assess(None)["valid"] is False
    store.save()
    payload = json.loads(path.read_text())
    payload["intrinsics"] = {"k": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                             "width": 10, "height": 10, "reprojection_rms_px": 0.01}
    payload["hand_eye"] = {"matrix": np.eye(4).tolist(), "consistency_mm": 0,
                           "max_rotation_deg": 90, "max_translation_mm": 100}
    path.write_text(json.dumps(payload))
    loaded = VisionStore.load(path)
    assert loaded.intrinsics is None and loaded.hand_eye is None


def _fixture_store():
    """A known downward camera at the tool, looking at z=0. Not a saved robot calibration."""
    store = VisionStore()
    store.intrinsics = {
        "k": [[800.0, 0.0, 640.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]],
        "dist": [0, 0, 0, 0, 0], "width": 1280, "height": 800,
        "reprojection_rms_px": 0.1, "n_images": 3, "square_mm": 25.0,
    }
    store.intrinsic_views = [{"width": 1280, "height": 800}] * 3
    store.hand_eye = {
        "matrix": np.eye(4).tolist(), "method": "fixture", "consistency_mm": 0.0, "n_poses": 5,
        "max_rotation_deg": 30.0, "max_translation_mm": 80.0, "frame": "T_tool_camera",
    }
    store.handeye_views = [{"pose_source": "telemetry", "width": 1280, "height": 800}] * 5
    store.table = {"point_mm": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "residual_mm": 0.0,
                   "n_points": 12, "surface": "fixture", "object_height_measured": False}
    return store


def test_ray_meets_the_table_plane_and_clearance_is_not_invented():
    store = _fixture_store()
    assert store.assess([1280, 800])["valid"] is True
    hit = store.localize((0, 0, 400, 180, 0, 0), 840, 300, [1280, 800])
    assert hit["ok"] and hit["kind"] == "plane_intersection"
    assert hit["verified_tool_position"] is False and hit["object_height_measured"] is False
    assert np.allclose(hit["point_mm"], [100, 50, 0], atol=1.0)
    missing = store.observation_target(hit["point_mm"], None)
    assert missing["ok"] is False and missing["target_mm"] is None
    goal = store.observation_target(hit["point_mm"], 150)
    assert goal["ok"] and goal["approximate_observation_goal"] is True
    assert goal["verified_tool_position"] is False
    assert np.allclose(goal["target_mm"], [100, 50, 150], atol=1.0)


def test_handeye_solve_uses_only_supplied_observations():
    k = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], float)
    dist = np.zeros(5)
    mount = tool_pose_matrix((15, -10, 40, 8, -12, 20))
    board = np.eye(4)
    board[:3, 3] = (200.0, 0.0, 0.0)
    poses = [
        (300, -80, 280, 170, -15, -25), (360, 40, 340, 185, 12, 20), (250, 90, 300, 165, -8, 5),
        (400, -30, 380, 190, 18, 35), (280, 20, 250, 175, -20, -40), (340, -70, 320, 160, 10, 15),
        (220, -20, 360, 188, -5, -15), (380, 60, 290, 172, 22, 50),
    ]
    views = project_observations(k, dist, [(pose, mount, board) for pose in poses], 6, 5, 20.0, (640, 480))
    store = VisionStore()
    store.intrinsics = {"k": k.tolist(), "dist": dist.tolist(), "width": 640, "height": 480,
                        "reprojection_rms_px": 0.01, "n_images": 3, "square_mm": 20.0}
    store.intrinsic_views = [{"width": 640, "height": 480}] * 3
    store.handeye_views = views
    report = store._solve_handeye()
    assert report["ok"] and store.hand_eye is not None
    recovered = np.array(store.hand_eye["matrix"], float)
    assert np.allclose(recovered[:3, 3], mount[:3, 3], atol=1.0)
    assert np.allclose(recovered[:3, :3], mount[:3, :3], atol=1e-3)
    assert store.hand_eye["consistency_mm"] < 1.0
    assert store.hand_eye["holdout_mm"] < 1.0
    assert store.hand_eye["method"] == "park_martin_ax_xb"
    # These poses were projected in the test. They are not robot telemetry, so the model stays invalid.
    assert store.assess([640, 480])["valid"] is False
    assert any("telemetry" in reason for reason in store.assess([640, 480])["reasons"])
    yaw_only = [
        (160, -40, 280, 180, 0, -25), (210, 30, 340, 180, 0, 20), (180, 70, 300, 180, 0, 0),
        (250, -20, 360, 180, 0, 35), (130, 10, 260, 180, 0, -10), (190, -60, 320, 180, 0, 15),
    ]
    flat = project_observations(k, dist, [(pose, np.eye(4), np.eye(4)) for pose in yaw_only], 6, 5, 20.0, (640, 480))
    refused = VisionStore()
    refused.intrinsics = store.intrinsics
    refused.handeye_views = flat
    missed = refused._solve_handeye()
    assert missed["ok"] is False and refused.hand_eye is None
    assert "one axis" in missed["reason"]
    short = VisionStore()
    short.intrinsics = store.intrinsics
    short.intrinsic_views = store.intrinsic_views
    short.handeye_views = views[:5]
    few = short._solve_handeye()
    assert few["ok"] is False and short.hand_eye is None
    assert "held-out" in few["reason"]


def _handeye_store(views):
    k = [[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]]
    store = VisionStore()
    store.intrinsics = {"k": k, "dist": [0, 0, 0, 0, 0], "width": 640, "height": 480,
                        "reprojection_rms_px": 0.01, "n_images": 3, "square_mm": 20.0}
    store.intrinsic_views = [{"width": 640, "height": 480}] * 3
    store.handeye_views = views
    return store


def _known_mount_views():
    k = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], float)
    mount = tool_pose_matrix((15, -10, 40, 8, -12, 20))
    board = np.eye(4)
    board[:3, 3] = (200.0, 0.0, 0.0)
    poses = [
        (300, -80, 280, 170, -15, -25), (360, 40, 340, 185, 12, 20), (250, 90, 300, 165, -8, 5),
        (400, -30, 380, 190, 18, 35), (280, 20, 250, 175, -20, -40), (340, -70, 320, 160, 10, 15),
        (220, -20, 360, 188, -5, -15), (380, 60, 290, 172, 22, 50),
    ]
    views = project_observations(k, np.zeros(5), [(pose, mount, board) for pose in poses], 6, 5, 20.0, (640, 480))
    return views, mount


def test_handeye_rejects_noise_inconsistency_and_a_small_motion():
    """These views are projected fixtures. They do not make the calibration valid."""
    views, mount = _known_mount_views()
    rng = np.random.default_rng(7)
    noisy = []
    for view in views:
        corners = np.array(view["corners"], float) + rng.normal(0.0, 0.25, size=(len(view["corners"]), 2))
        noisy.append(dict(view) | {"corners": np.round(corners, 3).tolist()})
    quiet = _handeye_store(noisy)._solve_handeye()
    solved = _handeye_store(noisy)
    solved._solve_handeye()
    assert quiet["ok"] and solved.hand_eye is not None
    assert np.allclose(np.array(solved.hand_eye["matrix"])[:3, 3], mount[:3, 3], atol=5.0)
    assert solved.hand_eye["holdout_mm"] <= 10.0
    assert solved.assess([640, 480])["valid"] is False
    shifted = [dict(view) for view in views]
    corners = np.array(shifted[-1]["corners"], float)
    corners[:, 0] += 40.0
    shifted[-1] = dict(shifted[-1]) | {"corners": np.round(corners, 3).tolist()}
    bad = _handeye_store(shifted)
    refused = bad._solve_handeye()
    assert refused["ok"] is False and bad.hand_eye is None
    assert "consistency" in refused["reason"] or "held-out" in refused["reason"]
    tiny = [
        (200, 0, 300, 180, 0, 0), (204, 2, 302, 181, 1, 1), (198, -2, 301, 179, -1, -1),
        (203, 1, 299, 180.5, 0.4, -0.6), (201, -1, 303, 179.4, -0.5, 0.8),
        (202, 3, 300, 180.2, 0.2, -0.3),
    ]
    mount_i = np.eye(4)
    board = np.eye(4)
    board[:3, 3] = (80.0, 0.0, 0.0)
    k = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], float)
    small = project_observations(k, np.zeros(5), [(pose, mount_i, board) for pose in tiny], 6, 5, 20.0, (640, 480))
    narrow = _handeye_store(small)._solve_handeye()
    assert narrow["ok"] is False
    pure = [
        (180, -40, 280, 180, 0, 0), (240, 30, 320, 180, 0, 0), (200, 80, 300, 180, 0, 0),
        (280, -20, 360, 180, 0, 0), (150, 10, 260, 180, 0, 0), (220, -60, 340, 180, 0, 0),
    ]
    flat = project_observations(k, np.zeros(5), [(pose, mount_i, board) for pose in pure], 6, 5, 20.0, (640, 480))
    translated = _handeye_store(flat)._solve_handeye()
    assert translated["ok"] is False
    assert "rotation" in translated["reason"] or "axis" in translated["reason"]


def test_nonidentity_mount_round_trips_a_plane_point():
    """Frame check only. This does not write a calibration and does not mark one valid."""
    mount = tool_pose_matrix((15, -10, 40, 8, -12, 20))
    k = np.array([[800.0, 0, 640.0], [0, 800.0, 400.0], [0, 0, 1]])
    pose6 = (0, 0, 400, 180, 0, 0)
    base_camera = camera_pose_matrix(pose6, mount)
    point = np.array([100.0, 50.0, 0.0, 1.0])
    camera_point = np.linalg.inv(base_camera) @ point
    pixel = k @ camera_point[:3]
    pixel = pixel[:2] / pixel[2]
    direction = base_camera[:3, :3] @ pixel_ray(pixel[0], pixel[1], k, np.zeros(5))
    hit = intersect_ray_plane(base_camera[:3, 3], direction, [0, 0, 0], [0, 0, 1])
    assert hit is not None
    assert np.allclose(hit, [100, 50, 0], atol=0.5)


def test_moving_camera_does_not_trust_image_overlap():
    frame = np.zeros((200, 240, 3), np.uint8)
    frame[40:100, 50:120] = (20, 20, 220)
    box = Box(50, 40, 120, 100)
    state = TrackState()
    first, _reason = assign_tracks(state, [box], frame, pose_mm=(0, 0, 400))
    second, reason = assign_tracks(state, [Box(50, 40, 120, 100)], frame, pose_mm=(200, 0, 400))
    assert second[0].extra["track_id"] == first[0].extra["track_id"]
    assert second[0].extra["association"] == "uncertain"
    assert "camera moved" in reason
    state = TrackState()
    marked = Box(50, 40, 120, 100, extra={"world_mm": [10, 20, 0]})
    assign_tracks(state, [marked], frame, pose_mm=(0, 0, 400))
    again = Box(50, 40, 120, 100, extra={"world_mm": [12, 21, 0]})
    matched, _reason = assign_tracks(state, [again], frame, pose_mm=(200, 0, 400))
    assert matched[0].extra["association"] == "geometric"


def test_no_default_square_size():
    image = _checker()
    store = VisionStore()
    refused = store.add_intrinsics(image, 5, 4, square_mm=0)
    assert refused["ok"] is False and store.intrinsic_views == []
