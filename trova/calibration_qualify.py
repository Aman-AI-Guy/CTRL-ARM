"""V2 evidence inventory and qualification CLI. No motion driver or motion endpoints."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from .localization import make_router, station_binding_errors, finite, stationary
from .geom import VisionStore
from .scan3d.evidence import sha256

REPO = Path(__file__).resolve().parents[1]


def inventory(root):
    root = Path(root)
    rows = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in {"node_modules", "__pycache__"}
        ]
        for name in files:
            if not name.endswith((".json", ".jsonl", ".csv")):
                continue
            path = Path(base) / name
            category = "unclassified_retained_metadata"
            reason = "No independent robot-base measurement schema established; not importable."
            if "calib2" in path.parts or "calib" in path.parts:
                category = "calibration_training_fitting"
                reason = "Marker/ruler observations or fitted model/report; not independent V2 validation."
            elif name == "hover_check.json" or name == "hover_target.json":
                category = "historical_internal_consistency"
                reason = "Target coordinates computed by calibration; hover pixel displacement is not independent measured XYZ."
            elif name == "calibration_references.jsonl":
                category = "potential_independent_validation"
                reason = "Requires row-by-row provenance, image integrity and current binding validation; filename alone is insufficient."
            rows.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "classification": category,
                    "reason": reason,
                }
            )
    # Explicitly recover actual physical measurements, not just filenames.
    candidate = root / "calib/v2/camera_robot.json"
    if not candidate.exists():
        candidate = root / "calib/camera_robot.json"
    cal = json.loads(candidate.read_text())
    hovers = [
        {
            "source": str(p),
            "observation": json.loads(p.read_text()),
            "classification": "historical_internal_consistency",
            "independent_measured_xyz_mm": None,
            "measurement_uncertainty_mm": None,
        }
        for p in (root / "arm_pc/runs").glob("*_hover/hover_check.json")
    ]
    return {
        "hover_checks": hovers,
        "files": rows,
        "ruler_measurements": cal.get("ruler", []),
        "fitted_plane": cal.get("table_plane"),
        "historical_quality": cal["quality"],
        "artifact": str(candidate),
        "current_independent_observations_verified": 0,
        "fit_points_are_not_independent": True,
    }


def import_ledger(provider, source, confirm=False):
    """Reuse canonical independent evidence only with the SAME independently established binding.

    Never rebind historical observations or promote training rows. Source bytes and original
    rows remain auditable; images are copied with content-addressed names. No approval written.
    """
    if not confirm:
        raise ValueError("Explicit confirmation is required to import evidence")
    source = Path(source).resolve()
    station = provider.station()
    errors = station_binding_errors(station, provider.digest)
    if errors:
        raise ValueError("; ".join(errors))
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    prepared = []
    for row in rows:
        if (
            row.get("provenance") != "operator_independent_reference"
            or row.get("used_for_fit") is not False
        ):
            raise ValueError(
                "Each source observation must declare independent provenance and used_for_fit: false"
            )
        if row.get("calibration_sha256") != provider.digest or row.get(
            "binding_sha256"
        ) != provider.binding_hash(station):
            raise ValueError(
                "Candidate/current physical binding differs; independent operator continuity evidence is required before reuse"
            )
        if (
            not row.get("reference_id")
            or not row.get("measurement_note")
            or not row.get("independent_measurement_source")
        ):
            raise ValueError(
                "Reference identity, independent measurement source and note are required"
            )
        uncertainty = float(row["measurement_uncertainty_mm"])
        if not np.isfinite(uncertainty) or not 0 <= uncertainty <= 2:
            raise ValueError("Measured uncertainty must be within 0..2 mm")
        stationary(row["robot"])
        finite(row["pixel"], 2)
        finite(row["measured_mm"], 3)
        image = (source.parent / row["image"]).resolve()
        if (
            not image.is_relative_to(source.parent)
            or sha256(image) != row["image_sha256"]
        ):
            raise ValueError(
                "Source image is missing, changed or escapes the source directory"
            )
        if list(row["image_size"]) != [provider.cal.width, provider.cal.height]:
            raise ValueError("Image profile differs")
        profile = row["camera_profile"]
        if (
            profile.get("kind") != "pi"
            or profile.get("host") != station["camera_host"]
            or profile.get("rotate_reported") != 180
            or profile.get("extra_rotate") != 0
        ):
            raise ValueError("Source optical profile/endpoint is not bound")
        # Reject images in the actual V2 training snapshots even if relabelled independent.
        training_root = REPO / "Test 3d live video/arm_pc/runs/calib2"
        if training_root.exists():
            if row["image_sha256"] in {sha256(p) for p in training_root.rglob("*.jpg")}:
                raise ValueError(
                    "Source image was used for V2 fitting; cannot import as independent"
                )
        destination = Path("imported_references") / (row["image_sha256"] + image.suffix)
        prepared.append(
            (
                image,
                destination,
                dict(
                    row,
                    image=str(destination),
                    import_source_sha256=sha256(source),
                    original_observation=row,
                ),
            )
        )
    current = provider.references()
    for image, destination, row in prepared:
        duplicate = [
            r
            for r in current
            if r.get("image_sha256") == row["image_sha256"]
            and r.get("pixel") == row["pixel"]
        ]
        if duplicate:
            if any(r.get("measured_mm") != row["measured_mm"] for r in duplicate):
                raise ValueError(
                    "Conflicting independent measurement for an existing image/pixel"
                )
            continue
        target = provider.directory / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, target)
        current.append(row)
    ledger = provider.directory / "calibration_references.jsonl"
    temporary = ledger.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in current))
    os.replace(temporary, ledger)
    return {
        "observations": len(current),
        "metric_permitted": False,
        "hardware_authorized": False,
    }


def api(base, route, body=None):
    token_path = Path(
        os.getenv("TROVA_TOKEN_FILE", "~/.config/trova/token")
    ).expanduser()
    token = (
        token_path.read_text().strip()
        if token_path.is_file()
        else os.getenv("TROVA_TOKEN", "")
    )
    req = urllib.request.Request(
        base.rstrip("/") + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={"x-trova-token": token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "snapshot",
            "inventory",
            "inspect",
            "validate",
            "import",
            "station",
            "reference",
            "approve",
            "live",
        ],
    )
    parser.add_argument("--data", type=Path, default=Path("data/m61"))
    parser.add_argument("--root", type=Path, default=REPO / "Test 3d live video")
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robot-ip", default="192.168.1.111")
    parser.add_argument("--camera-host", default="192.168.5.160")
    parser.add_argument("--operator")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--pixel", type=float, nargs=2)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(".env")
    router = make_router(VisionStore.load(args.data / "vision_calib.json"), args.data)
    provider = router.historical
    if provider is None:
        parser.error("Pinned V2 candidate unavailable")
    if args.action == "snapshot":
        result = api(args.url, "/api/scene", {})
    elif args.action == "inventory":
        result = inventory(args.root)
    elif args.action == "inspect":
        result = api(args.url, "/api/vision/provider/station")
    elif args.action == "station":
        if not args.file or not args.operator or not args.confirm:
            parser.error("Provide measured station --file, --operator and --confirm")
        result = api(
            args.url,
            "/api/vision/provider/station",
            {
                "station": json.loads(args.file.read_text()),
                "operator": args.operator,
                "confirm": True,
            },
        )
    elif args.action == "reference":
        if not args.file:
            parser.error("--file must contain a measured reference payload")
        result = api(
            args.url,
            "/api/vision/provider/reference",
            json.loads(args.file.read_text()),
        )
    elif args.action == "import":
        if not args.file:
            parser.error("--file must be a canonical independent JSONL ledger")
        result = import_ledger(provider, args.file, args.confirm)
    elif args.action in {"validate", "live"}:
        from .arm.telemetry import open_reader
        from .pi_cam import PiCam

        reader = open_reader("xarm:" + args.robot_ip)
        try:
            camera = PiCam(args.camera_host, timeout=4, retries=0, status_timeout=2)
            before = reader.read()
            cap = camera.capture()
            after = reader.read()
            from .geom import pose_changed

            stationary(before)
            stationary(after)
            if (
                pose_changed(before["pose6"], after["pose6"], mm=1, deg=0.1)
                or max(abs(a - b) for a, b in zip(before["joints"], after["joints"]))
                > 0.1
            ):
                raise ValueError("Robot moved during capture")
            import cv2

            image = cv2.imdecode(np.frombuffer(cap.jpeg, np.uint8), cv2.IMREAD_COLOR)
            size = [image.shape[1], image.shape[0]]
            profile = dict(cap.meta, kind="pi", host=args.camera_host)
            router.set_context(profile, after)
            result = {
                "assessment": provider.assess(size, require_approval=False),
                "active_provider": router.assess(size)["active_provider"],
                "references": provider.references(),
                "controller_configuration": reader.configuration(),
                "source_image_sha256": hashlib.sha256(cap.jpeg).hexdigest(),
                "source_metadata": cap.meta,
                "robot": after,
                "hardware_authorized": False,
            }
            if args.output:
                args.output.with_suffix(".jpg").write_bytes(cap.jpeg)
            if args.action == "live":
                if args.pixel is None:
                    parser.error("Supply an operator-identified support --pixel U V")
                result["location"] = provider.pixel_to_base(after, args.pixel, size)
        finally:
            reader.close()
    else:
        # Never ask/approve before the local independent validation passes.
        from .arm.telemetry import open_reader

        reader = open_reader("xarm:" + args.robot_ip)
        try:
            provider.robot = reader.read()
            from .pi_cam import PiCam

            status = PiCam(args.camera_host).status()
            provider.camera = {
                "kind": "pi",
                "host": args.camera_host,
                "rotate_reported": status.get("rotate"),
                "extra_rotate": 0,
            }
            report = provider.assess(
                [status["width"], status["height"]], require_approval=False
            )
            if not report["valid"]:
                result = {
                    "approved": False,
                    "reasons": report["reasons"],
                    "validation": report["validation"],
                }
            elif not args.operator or not args.confirm:
                result = {
                    "approved": False,
                    "validation_passed": True,
                    "next": "Operator reviews report, then supplies --operator NAME --confirm",
                }
            else:
                result = api(
                    args.url,
                    "/api/vision/provider/qualify",
                    {"operator": args.operator, "confirm": True},
                )
                result["final_providers"] = api(args.url, "/api/vision/providers")
        finally:
            reader.close()
    raw = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
