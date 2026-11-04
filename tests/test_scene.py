"""M1: a frozen snapshot, stable track ids, and selection that does not move the arm."""

import cv2
import numpy as np

from trova.boxes import Box
from trova.proposals import find_blobs
from trova.scene import TrackState, assign_tracks, build_objects, decide_text, from_finder, mark_ambiguous, pack
from trova.synthetic import ObjSpec, SyntheticTable


def _frame_with_box():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[40:90, 30:80] = (200, 80, 30)
    return frame


def test_track_ids_follow_a_shift_and_reset_when_the_view_changes():
    state = TrackState()
    frame = _frame_with_box()
    first, reason = assign_tracks(state, [Box(30, 40, 80, 90, id=1)], frame)
    assert reason is None and first[0].extra["track_id"] == 1
    second, reason = assign_tracks(state, [Box(40, 48, 90, 98, id=1)], frame)
    assert reason is None and second[0].extra["track_id"] == 1
    wider = cv2.resize(frame, (400, 280))
    third, reason = assign_tracks(state, [Box(40, 48, 90, 98, id=1)], wider)
    assert reason == "frame size changed" and third[0].extra["track_id"] == 2


def test_ordinal_and_id_selection_stay_in_code():
    objects = [
        {"track_id": 4, "box_id": 1, "name": "blue object", "note": "Mock: blue", "color": "blue",
         "confidence": 0.7, "ambiguous": True},
        {"track_id": 5, "box_id": 2, "name": "blue object", "note": "Mock: blue", "color": "blue",
         "confidence": 0.7, "ambiguous": True},
        {"track_id": 6, "box_id": 3, "name": "red object", "note": "Mock: red", "color": "red",
         "confidence": 0.8, "ambiguous": False},
    ]
    assert decide_text("the second blue pen", objects).track_ids == [5]
    assert decide_text("seconda penna blu", objects).track_ids == [5]
    assert decide_text("oggetto 4", objects).track_ids == [4]
    assert decide_text("oggetto 1", objects).track_ids == [4]   # no track 1: the reading-order box
    assert decide_text("blue pen", objects).status == "defer"
    assert decide_text("oggetto 9", objects).status == "not_found"
    mark_ambiguous(objects)
    shared = from_finder("found", 1, [1], 0.8, "the blue one", objects, "the blue pen")
    assert shared.status == "clarify" and set(shared.track_ids) == {4, 5}
    packed = pack(shared, objects, authoritative=False, snapshot_id="s")
    assert packed["motion_allowed"] is False and packed["authoritative"] is False
    assert "cannot authorize motion" in packed["reason"] and packed["object"] is None


def _table(shift=(0, 0), extra=False):
    table = SyntheticTable(width=800, height=600, n_objects=0, seed=3)
    dx, dy = shift
    table.objects = [
        ObjSpec("pen", "blue", "square", 180 + dx, 160 + dy, 120, 0),
        ObjSpec("pen", "blue", "square", 560 + dx, 210 + dy, 120, 0),
        ObjSpec("mug", "red", "circle", 380 + dx, 450 + dy, 130, 0),
    ]
    if extra:
        table.objects.append(ObjSpec("cube", "yellow", "square", 380 + dx, 70 + dy, 100, 0))
    return table


def _shot(table):
    frame = table.render(shadows=False, noise=0)
    boxes = find_blobs(frame, table.empty())
    return frame, boxes


def test_two_blue_pens_keep_ids_and_a_new_object_gets_a_new_one():
    state = TrackState()
    frame, boxes = _shot(_table())
    assert len(boxes) == 3
    tracked, _ = assign_tracks(state, boxes, frame)
    objects = build_objects(tracked, frame, {b.id: {"name": f"{b.extra['color']} object", "category": "other",
                                                    "note": "m"} for b in tracked})
    assert len({o["track_id"] for o in objects}) == 3
    blues = [o for o in objects if o["color"] == "blue"]
    assert len(blues) == 2 and all(o["ambiguous"] for o in blues)
    assert not any(o["ambiguous"] for o in objects if o["color"] == "red")

    frame2, boxes2 = _shot(_table(shift=(12, 8)))
    tracked2, reason = assign_tracks(state, boxes2, frame2)
    assert reason is None
    objects2 = build_objects(tracked2, frame2, {b.id: {"name": f"{b.extra['color']} object", "category": "other",
                                                       "note": "m"} for b in tracked2})
    assert sorted(o["track_id"] for o in objects if o["color"] == "blue") == \
        sorted(o["track_id"] for o in objects2 if o["color"] == "blue")
    assert [o["track_id"] for o in objects if o["color"] == "red"] == \
        [o["track_id"] for o in objects2 if o["color"] == "red"]

    frame3, boxes3 = _shot(_table(shift=(12, 8), extra=True))
    tracked3, _ = assign_tracks(state, boxes3, frame3)
    assert len(tracked3) == 4
    new_ids = {b.extra["track_id"] for b in tracked3} - {b.extra["track_id"] for b in tracked2}
    assert len(new_ids) == 1


def test_select_on_a_snapshot_does_not_move_the_arm(tmp_path, monkeypatch):
    folder = tmp_path / "frames"
    folder.mkdir()
    empty = _table().empty()
    cv2.imwrite(str(folder / "00_empty.png"), empty)
    cv2.imwrite(str(folder / "01_scene.png"), _table().render(shadows=False, noise=0))
    cv2.imwrite(str(folder / "02_shift.png"), _table(shift=(12, 8)).render(shadows=False, noise=0))
    cv2.imwrite(str(folder / "03_extra.png"), _table(shift=(12, 8), extra=True).render(shadows=False, noise=0))

    from test_pipeline import _client
    app, c = _client(tmp_path, monkeypatch, TROVA_CAMERA=str(folder))
    refused = c.post("/api/select", data={"text": "the blue pen"})
    assert refused.status_code == 400 and "snapshot" in refused.json()["detail"].lower()

    assert c.post("/api/background", data={"mode": "capture"}).json()["background"]
    assert c.post("/api/source/step", data={"delta": 1}).json()["index"] == 1
    before = app.state.trova.arm.status()
    inv = c.post("/api/inventory").json()
    snap = inv["snapshot"]
    assert inv["summary"] and snap["grok_mode"] == "mock" and snap["authoritative"] is False
    assert snap["motion_allowed"] is False and snap["pose"]["state"] == "idle"
    assert (tmp_path / snap["image"]).exists() and (tmp_path / snap["map"]).exists()
    objs = snap["objects"]
    assert len(objs) == 3
    blues = sorted((o for o in objs if o["color"] == "blue"), key=lambda o: o["box_id"])
    assert len(blues) == 2 and all(o["ambiguous"] for o in blues)
    assert len({o["track_id"] for o in objs}) == 3

    def choose(data):
        out = c.post("/api/select", data=data).json()
        assert app.state.trova.arm.status() == before
        assert out["motion_allowed"] is False and out["authoritative"] is False
        assert "xarm.wrapper" not in __import__("sys").modules
        return out

    second = blues[1]
    for data in ({"text": "the second blue pen"}, {"text": "seconda penna blu"},
                 {"text": f"oggetto {second['track_id']}"}, {"track_id": str(second["track_id"])}):
        picked = choose(data)
        assert picked["status"] == "selected" and picked["object"]["track_id"] == second["track_id"]
        assert picked["object"]["color"] == "blue"

    ambiguous = choose({"text": "blue pen"})
    assert ambiguous["status"] == "clarify" and ambiguous["object"] is None
    assert sorted(o["track_id"] for o in ambiguous["candidates"]) == sorted(o["track_id"] for o in blues)

    assert c.post("/api/source/step", data={"delta": 1}).json()["index"] == 2
    shifted = c.post("/api/inventory").json()["snapshot"]["objects"]
    assert sorted(o["track_id"] for o in shifted if o["color"] == "blue") == \
        sorted(o["track_id"] for o in objs if o["color"] == "blue")

    assert c.post("/api/source/step", data={"delta": 1}).json()["index"] == 3
    grown = c.post("/api/inventory").json()["snapshot"]["objects"]
    assert len(grown) == 4
    assert len({o["track_id"] for o in grown} - {o["track_id"] for o in objs}) == 1
    assert app.state.trova.arm.status() == before


def test_empty_snapshot_reports_no_objects(tmp_path, monkeypatch):
    folder = tmp_path / "empty"
    folder.mkdir()
    cv2.imwrite(str(folder / "00_empty.png"), _table().empty())
    from test_pipeline import _client
    _app, c = _client(tmp_path, monkeypatch, TROVA_CAMERA=str(folder))
    c.post("/api/background", data={"mode": "capture"})
    inv = c.post("/api/inventory").json()
    assert inv["n_objects"] == 0 and inv["snapshot"]["objects"] == []
    assert "empty" in inv["summary"].lower() or "vuot" in inv["summary"].lower()
    missing = c.post("/api/select", data={"text": "the blue pen"}).json()
    assert missing["status"] == "not_found" and missing["motion_allowed"] is False
