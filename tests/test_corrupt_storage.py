import json
from pathlib import Path

from trova.calibration import Calibration, load_marker_positions
from trova.zones import Zone, load_zones, save_zones


def test_calibration_load_corrupt_files(tmp_path: Path):
    empty_file = tmp_path / "empty_calib.json"
    empty_file.write_text("")
    assert Calibration.load(empty_file) is None

    malformed_json = tmp_path / "bad_json.json"
    malformed_json.write_text("{bad json")
    assert Calibration.load(malformed_json) is None

    three_number_point = tmp_path / "three_nums.json"
    three_number_point.write_text(json.dumps({"points": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]}))
    assert Calibration.load(three_number_point) is None


def test_calibration_save_atomic(tmp_path: Path):
    calib = Calibration.default(640, 480, 600, 400)
    dest = tmp_path / "calibration.json"
    calib.save(dest)
    assert dest.exists()
    loaded = Calibration.load(dest)
    assert loaded is not None
    assert loaded.ok


def test_load_zones_corrupt_files(tmp_path: Path):
    empty_file = tmp_path / "empty_zones.json"
    empty_file.write_text("")
    assert load_zones(empty_file) == []

    bad_json = tmp_path / "bad_zones.json"
    bad_json.write_text("{not a list")
    assert load_zones(bad_json) == []

    not_a_list = tmp_path / "dict_zones.json"
    not_a_list.write_text(json.dumps({"key": "val"}))
    assert load_zones(not_a_list) == []


def test_save_zones_atomic(tmp_path: Path):
    zones = [Zone(name="z1", rule="keep_out", polygon=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])]
    dest = tmp_path / "zones.json"
    save_zones(dest, zones)
    assert dest.exists()
    loaded = load_zones(dest)
    assert len(loaded) == 1
    assert loaded[0].name == "z1"


def test_load_marker_positions_corrupt_files(tmp_path: Path):
    empty_file = tmp_path / "markers.json"
    empty_file.write_text("")
    pos = load_marker_positions(empty_file, 600.0, 400.0)
    assert 0 in pos and 1 in pos and 2 in pos and 3 in pos
