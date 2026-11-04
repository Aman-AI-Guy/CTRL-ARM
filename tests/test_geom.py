"""Camera geometry. Matrices here are test fixtures, not a calibration of the real arm."""

import json

import cv2
import numpy as np

from trova.boxes import Box
from trova.geom import VisionStore, detect_checkerboard, project_observations, tool_pose_matrix
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
