import json
import pytest
from trova.calibration_qualify import import_ledger, inventory


def test_metric_report_does_not_force_predicted_z_to_reference_height(tmp_path):
    from test_localization_providers import qualified

    provider = qualified(tmp_path)
    path = tmp_path / "calibration_references.jsonl"
    rows = provider.references()
    for row in rows:
        row["measured_mm"][2] = 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = provider.assess([1600, 900], require_approval=False)
    assert len(report["validation"]["observations"]) == 6
    for row in report["validation"]["observations"]:
        assert row["predicted_mm"][2] == pytest.approx(0)
        assert row["signed_error_mm"][2] == pytest.approx(-1)
        assert row["error_mm"] >= 1


def test_import_preserves_source_and_never_grants_approval(tmp_path):
    from test_localization_providers import qualified

    provider = qualified(tmp_path)
    rows = provider.references()
    for n, row in enumerate(rows):
        row.update(
            reference_id=f"point-{n % 3}",
            used_for_fit=False,
            independent_measurement_source="synthetic fixture only",
        )
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "calibration_references.jsonl").unlink()
    result = import_ledger(provider, source, confirm=True)
    assert not result["metric_permitted"] and not result["hardware_authorized"]
    assert len(provider.references()) == 6
    assert all(
        r["original_observation"]["used_for_fit"] is False and r["import_source_sha256"]
        for r in provider.references()
    )
    assert not provider.assess([1600, 900])[
        "valid"
    ]  # old approval no longer matches ledger
    assert import_ledger(provider, source, confirm=True)["observations"] == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("used_for_fit", True),
        ("binding_sha256", "another installation"),
        ("measurement_uncertainty_mm", None),
    ],
)
def test_import_rejects_training_unknown_measurements_and_rebinding(
    tmp_path, field, value
):
    from test_localization_providers import qualified

    provider = qualified(tmp_path)
    rows = provider.references()
    for row in rows:
        row.update(
            reference_id="p",
            used_for_fit=False,
            independent_measurement_source="synthetic",
        )
    rows[0][field] = value
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = provider.evidence_digest()
    with pytest.raises((ValueError, TypeError)):
        import_ledger(provider, source, confirm=True)
    assert before == provider.evidence_digest()


def test_inventory_classifies_real_ruler_and_hover_without_promoting_them():
    from trova.calibration_compare import V2

    if not V2.exists():
        pytest.skip("Optional archive absent")
    report = inventory(V2.parents[2])
    assert len(report["ruler_measurements"]) == 7
    assert report["current_independent_observations_verified"] == 0
    assert any(
        r["classification"] == "historical_internal_consistency"
        and r["path"].endswith("hover_check.json")
        for r in report["files"]
    )
    assert report["fit_points_are_not_independent"]
