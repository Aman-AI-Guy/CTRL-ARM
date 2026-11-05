"""Real Pi pixels, deterministic replay of vision responses; no live API or robot."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from trova.agent import skill_locate, skill_select
from trova.boxes import Box
from trova.finder import find
from trova.geom import VisionStore
from trova.grok import GrokResult
from trova.proposals import find_blobs
from trova.scene import TrackState, adopt_found, assign_tracks, build_objects

FIXTURES = Path(__file__).parent / "fixtures" / "selection"


class RecordedVision:
    mock = False

    def __init__(self, region=(469, 211, 591, 553), confidence=0.95, live=True):
        self.region, self.confidence, self.live = region, confidence, live
        self.calls = []

    def json(self, *, name, images=(), **kwargs):
        self.calls.append(name)
        if name == "trova_select":
            data = dict(
                box_id=None,
                visible_but_unboxed=True,
                candidate_ids=[],
                clarifying_question=None,
                label="mouse rosso",
                reason="A red computer mouse is visible but has no numbered box.",
                confidence=0.98,
            )
        elif name == "trova_direct":
            data = dict(
                found=True,
                box_2d=self.region,
                label="mouse rosso",
                reason="Red mouse image region.",
                confidence=self.confidence,
            )
        else:
            raise AssertionError(name)
        return GrokResult(data, 0, "recorded-response", mock=not self.live)


def frame():
    return cv2.imread(str(FIXTURES / "red_mouse_close.jpg"))


@pytest.mark.parametrize("empty", [False, True])
def test_unboxed_red_mouse_reaches_metric_calibration_gate(empty, tmp_path):
    from test_agent import _Host, _arm

    host = _Host(_arm(), VisionStore(tmp_path / "unsolved.json"))
    host.frame = frame()
    host.image_size = [1600, 900]
    host.boxes = (
        [] if empty else [Box(348, 363, 640, 650, id=1)]
    )  # real marker, mouse deliberately missed
    host.grok = RecordedVision()
    task = dict(
        goal="Posizionati sopra il mouse rosso.",
        objects=[],
        phase="inspected",
        snapshot_pose={},
        steps=[],
        timings=[],
        motion_lock=None,
        authoritative=True,
    )
    if host.boxes:
        host.boxes, _ = assign_tracks(TrackState(), host.boxes, host.frame)
        task["objects"] = build_objects(host.boxes, host.frame, {})
    before = host.arm.tcp_mm()
    result = skill_select(host, task, "")
    assert result["ok"] and task["phase"] == "selected"
    obj = task["selected"]
    assert obj["name"] == "mouse rosso" and obj["track_id"] > 0
    assert Box.from_list(obj["box"]).iou(Box(750, 190, 946, 498)) > 0.9
    assert obj["localization_provenance"] == "grok_direct_region"
    assert obj["recognition_state"] == "recognized"
    assert obj["image_localization_state"] == "region_identified"
    assert obj["metric_localization_state"] == "not_localized"
    located = skill_locate(host, task, "")
    assert not located["ok"] and task["phase"] == "blocked"
    assert (
        "intrinsics" in located["reason"]
        and "hand-eye" in located["reason"]
        and "plane" in located["reason"]
    )
    assert not task["location"].get("point_mm")
    assert obj["metric_localization_state"] == "blocked"
    assert host.arm.tcp_mm() == before
    assert host.grok.calls == (
        ["trova_direct"] if empty else ["trova_select", "trova_direct"]
    )


@pytest.mark.parametrize(
    "region,confidence",
    [
        ([-1, 200, 600, 550], 0.9),
        ([600, 200, 400, 550], 0.9),
        ([0, 0, 1000, 1000], 0.9),
        ([400, 200, 401, 201], 0.9),
        ([469, 211, 591, 553], 0.3),
        ([469, float("nan"), 591, 553], 0.9),
        ([469, 211, 591, 553], float("nan")),
    ],
)
def test_invalid_or_uncertain_regions_never_become_selections(region, confidence):
    result = find(RecordedVision(region, confidence), frame(), [])
    assert result.status == "not_found" and result.box is None
    assert "draw a box" in result.reason


def test_direct_region_tracking_requires_fresh_pixels():
    im = frame()
    state = TrackState()
    result = find(RecordedVision(), im, [])
    boxes, objects = adopt_found(state, im, [], [], result)
    tid = objects[0]["track_id"]
    boxes, _ = assign_tracks(state, [], im.copy())
    assert len(boxes) == 1 and boxes[0].extra["track_id"] == tid
    assert boxes[0].extra["localization_provenance"] == "visual_template_reobservation"
    boxes, _ = assign_tracks(state, [], np.full_like(im, 120))
    assert not boxes  # the label and old position cannot keep a missing object alive


def test_camera_rotation_prevents_template_reuse():
    im = frame()
    state = TrackState()
    result = find(RecordedVision(), im, [])
    boxes, _ = adopt_found(state, im, [], [], result)
    assign_tracks(state, boxes, im, pose_mm=[0, 0, 300, 180, 0, 0])
    moved, _ = assign_tracks(state, [], im, pose_mm=[0, 0, 300, 180, 0, 10])
    assert moved == []


def test_real_low_contrast_ruler_is_not_mistaken_for_semantic_localization():
    im = cv2.imread(str(FIXTURES / "table_mouse_ruler.jpg"))
    ruler = Box(95, 527, 340, 785)
    boxes = find_blobs(im)
    assert not any(
        b.iou(ruler) > 0.35 for b in boxes
    )  # actual OpenCV miss on patterned table
    from trova.regions import validated_region

    manual = validated_region(ruler.to_list(), im, label="transparent ruler")
    manual.extra["localization_provenance"] = "operator_drawn_region"
    from trova.finder import FindResult

    result = FindResult(status="found", box=manual, label=manual.label, mode="direct")
    boxes, objects = adopt_found(TrackState(), im, boxes, [], result)
    chosen = next(o for o in objects if o["box_id"] == result.box.id)
    assert chosen["localization_provenance"] == "operator_drawn_region"
    assert not chosen["label_authoritative"]
    assert chosen["metric_localization_state"] == "not_localized"


def test_operator_snapshot_regions_and_stale_id(tmp_path, monkeypatch):
    from test_pipeline import _client

    app, client = _client(tmp_path, monkeypatch)
    st = app.state.trova
    st.clear_view = lambda: pytest.fail("selection must not park the robot")
    snap = client.post("/api/scene").json()["snapshot"]
    result = client.post(
        "/api/select/region",
        json=dict(snapshot_id=snap["id"], box=[40, 40, 140, 140], label="mouse"),
    )
    assert result.status_code == 200, result.text
    selected = result.json()
    assert selected["status"] == "selected" and not selected["motion_allowed"]
    assert selected["object"]["localization_provenance"] == "operator_drawn_region"
    assert not selected["object"]["label_authoritative"]
    click = client.post(
        "/api/select/region", json=dict(snapshot_id=snap["id"], point=[90, 90])
    )
    assert click.json()["object"]["track_id"] == selected["object"]["track_id"]
    stale = client.post(
        "/api/select/region", json=dict(snapshot_id="old", box=[40, 40, 140, 140])
    )
    assert stale.status_code == 409
    bad = client.post(
        "/api/select/region", json=dict(snapshot_id=snap["id"], box=[-40, 40, 140, 140])
    )
    assert bad.status_code == 400


def test_conflicting_live_object_type_gets_a_new_track():
    im = frame()
    state = TrackState()
    boxes, _ = assign_tracks(state, [Box(750, 190, 946, 498, id=1)], im)
    first = build_objects(
        boxes, im, {1: dict(name="red wireless mouse", label_authoritative=True)}, state
    )[0]
    boxes, _ = assign_tracks(state, [Box(750, 190, 946, 498, id=1)], im)
    later = build_objects(
        boxes, im, {1: dict(name="green paper cup", label_authoritative=True)}, state
    )[0]
    assert later["track_id"] != first["track_id"]
    assert later["identity_uncertain"] and later["association"] == "semantic_conflict"
