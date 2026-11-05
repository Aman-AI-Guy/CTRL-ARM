"""V2 selection is configuration, never current physical qualification or actuation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trova.calibration_compare import REPO, V2, OLD, compare
from trova.geom import VisionStore
from trova.localization import HistoricalProvider, make_router, station_binding_errors
from trova.scan3d.flange import FlangeCalibration


@pytest.fixture
def v2():
    if not V2.is_file():
        pytest.skip("Optional real V2 calibration archive absent")
    return FlangeCalibration.load(V2)


def test_explicit_primary_v2_hash_and_identical_root_without_activation(
    tmp_path, v2, monkeypatch
):
    monkeypatch.delenv("TROVA_CALIBRATION_CANDIDATE", raising=False)
    router = make_router(VisionStore(), tmp_path)
    assert router.historical.path.resolve() == V2
    assert V2.read_bytes() == (V2.parents[2] / "calib/camera_robot.json").read_bytes()
    assert router.historical.cal.data["quality"]["pictures"] == 57
    assert not router.assess([1600, 900])["valid"]
    assert router.assess([1600, 900])["active_provider"] == "image_space"
    assert not np.allclose(v2.T_fc, FlangeCalibration.load(OLD).T_fc)


def test_manifest_hash_change_never_falls_back_to_old_calibration(
    tmp_path, monkeypatch, v2
):
    import trova.localization as module

    monkeypatch.delenv("TROVA_CALIBRATION_CANDIDATE", raising=False)
    monkeypatch.setattr(module, "sha256", lambda path: "changed")
    router = make_router(VisionStore(), tmp_path)
    assert router.historical is None and "SHA-256" in router.current.candidate_error


def test_v2_full_tilted_plane_reproduces_recorded_hover_pixel(v2):
    record = json.loads(
        (
            V2.parents[2] / "arm_pc/runs/20261105-044620_hover/hover_target.json"
        ).read_text()
    )
    point = v2.pixel_to_object_top(*record["pixel"], record["q0"], record["height_mm"])
    np.testing.assert_allclose(point, record["target_mm"], atol=1e-8)
    a, b, c = v2.plane
    assert point[2] == pytest.approx(
        a * point[0] + b * point[1] + c + record["height_mm"]
    )
    np.testing.assert_allclose(
        v2.robot_to_pixel(point, record["q0"]), record["pixel"], atol=1e-4
    )
    flat = v2.pixel_to_plane(
        *record["pixel"], record["q0"], v2.table_z + record["height_mm"]
    )
    assert np.linalg.norm(flat - point) > 0.01


def test_station_schema_lists_all_missing_fields_without_guessing_tcp(tmp_path, v2):
    provider = HistoricalProvider(V2, tmp_path)
    template = provider.station_template()
    assert template["tcp_offset_mm"] is None and template["table_plane"] is None
    errors = station_binding_errors(template, provider.digest)
    assert any("roll_deg" in e and "tcp_offset_mm" in e for e in errors)
    assert any("table_plane" in e for e in errors)
    assert any("mount_id" in e for e in errors)
    with pytest.raises(ValueError, match="tcp_offset_mm"):
        provider.configure_station(template, operator="operator", confirm=True)
    assert not (tmp_path / "calibration_station.json").exists()


def test_station_api_requires_measurements_and_does_not_qualify(tmp_path, monkeypatch):
    from test_pipeline import _client
    from test_localization_providers import qualified

    app, client = _client(tmp_path, monkeypatch)
    provider = qualified(tmp_path)
    # The synthetic calibration is deliberately bound only for exercising the schema endpoint.
    app.state.trova.vision.historical = provider
    response = client.get("/api/vision/provider/station").json()
    assert response["tcp_order"] == [
        "x_mm",
        "y_mm",
        "z_mm",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]
    station = provider.station()
    bad = client.post(
        "/api/vision/provider/station",
        json={
            "operator": "test",
            "confirm": True,
            "station": station | {"tcp_offset_mm": [0, 0, 0]},
        },
    )
    assert bad.status_code == 400 and "tcp_offset_mm" in bad.json()["detail"]
    denied = client.post(
        "/api/vision/provider/station", json={"operator": "test", "station": station}
    )
    assert denied.status_code == 400
    saved = client.post(
        "/api/vision/provider/station",
        json={"operator": "test", "confirm": True, "station": station},
    ).json()
    assert not saved["metric_permitted"] and not saved["hardware_authorized"]
    assert not provider.assess([1600, 900])[
        "valid"
    ]  # changed binding invalidates old approval/evidence
    forbidden = client.post(
        "/api/vision/provider/station",
        json={
            "operator": "test",
            "confirm": True,
            "station": station | {"qualified": True},
        },
    )
    assert forbidden.status_code == 400


def test_workspace_candidate_tcp_fits_but_head_does_not(tmp_path, v2):
    from test_localization_providers import robot
    from trova.diagnose import diagnose

    reader = SimpleNamespace(
        read=lambda: robot([0] * 5) | {"state_code": 5, "warn_code": 0},
        configuration=lambda: {"mode": 0},
    )
    result = diagnose(
        reader=reader,
        data=tmp_path,
        workspace_candidate=REPO / "config/workspace_candidate.json",
    )
    stages = {r["stage"]: r for r in result["stages"]}
    assert stages["workspace_candidate"]["status"] == "NOT_VERIFIED"
    current = stages["current_workspace_geometry"]
    assert current["status"] == "BLOCKED"
    assert current["actual_input"]["tcp_inside_candidate"]
    assert not current["actual_input"]["model_inside_candidate"]["head"]
    assert not result["ready_for_physical_execution"] and not result["motion_sent"]
    assert stages["controller_readiness"]["status"] == "BLOCKED"


def test_v2_comparison_replays_internal_tail_and_fixed_model_holdout(v2):
    report = compare()
    assert (report["images"], report["observations"], report["corner_samples"]) == (
        57,
        92,
        368,
    )
    assert report["v2_saved_fit_reprojection_px"]["rms"] == pytest.approx(8.4, abs=0.02)
    internal = report["v2_original_split_saved_seed_mm"]
    assert internal["median"] == pytest.approx(1.44, abs=0.03)
    assert internal["p95"] == pytest.approx(6.15, abs=0.04)
    models = report["models"]
    assert models["v2"]["same_image_heldout_plane_mm"]["samples"] == 344
    assert (
        models["v2"]["same_image_heldout_plane_mm"]["p95"]
        < models["old"]["same_image_heldout_plane_mm"]["p95"]
    )
    assert (
        not report["independent_current_validation"] and not report["valid_for_motion"]
    )


def test_v2_approach_is_recorded_numerical_evidence_only(v2):
    from trova.scan3d.orbit_planner import recorded_context, JointGeometry
    from trova.scan3d.approach import propose

    _, old = recorded_context()
    record = json.loads(
        (
            V2.parents[2] / "arm_pc/runs/20261105-044620_hover/hover_target.json"
        ).read_text()
    )
    top = v2.pixel_to_object_top(*record["pixel"], record["q0"], record["height_mm"])
    old.settings.object_height_mm = record["height_mm"]
    old.settings.object_radius_mm = 45
    a, b, c = v2.plane
    context = JointGeometry(
        v2.T_fc,
        {"normal": [-a, -b, 1], "point_mm": [0, 0, c]},
        top - [0, 0, record["height_mm"]],
        old.settings,
        old.gateway,
        qualified=False,
    )
    result = propose(context, record["q0"], record["above_mm"])
    candidate = result["next"]
    assert candidate, result
    assert not candidate["executable"] and candidate["gateway"]["status"] == "dry_run"
    center, direction = context.cam.ray(candidate["joints_deg"])
    assert center[2] - top[2] == pytest.approx(120, abs=1)
    np.testing.assert_allclose(
        direction, (top - center) / np.linalg.norm(top - center), atol=0.01
    )

    # A reachable recorded camera pose is not a qualified path in the requested box.
    from trova.arm import Limits

    context.gateway.limits = Limits.parse("-500,500,-500,500,100,1000")
    certificate = context.swept_check(record["q0"], candidate["joints_deg"])
    assert not certificate["ok"] and "workspace/head" in certificate["reason"]
