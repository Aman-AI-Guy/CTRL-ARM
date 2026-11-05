# M6.2 — V2 independent physical qualification

The pinned V2 model remains selected. **Qualification is incomplete; active provider is image_space, metric permission is false, hardware authorization is false.** No independent robot-base reference ledger was recovered in the supplied workspace or `data/m61`. No measurements or approval flags were invented. MotionGateway, trajectory execution, UI, reconstruction and scanning code were not changed.

## Recovered evidence

`python -m trova.calibration_qualify inventory --output /tmp/trova-m62-inventory.json` inventories the retained workspace, hashes each JSON/JSONL/CSV, and preserves the seven ruler rows and hover checks. The current archive contains 149 such files excluding environments/cache: 60 fitting records, eight hover target/check records of internal consistency, and 81 remaining settings/telemetry/processing metadata records. These counts describe the supplied files, not independent observations.

- `arm_pc/runs/calib2/session.json`, `ruler.json`: three legs A/B/C with two non-collinear cube-top markers per leg and one direct table gap. The original readings are recovered below. Their measured quantity is a vertical **gap**, not independently measured full robot-base XYZ. Saved head-bottom coordinates use robot FK/CAD; marker X/Y are fitted. These rows entered V2's solve or fitted plane. Do not repeat these training experiments merely because `data/m61` is empty, but do not use them as independent validation of the resulting fit.
- `calib/v2/camera_robot.json`, `report.json`: 57 images, 92 retained marker views, nine marker poses across three cube arrangements. The six non-collinear `table_plane.points` are cube-foot coordinates derived from fitted markers plus a table point derived using FK/head geometry and a ruler gap. Their 1.28 mm fit RMS is not an independently measured current-plane residual. The source images and joints remain under `arm_pc/runs/calib2/{A,B,C}`. Known black-square size is 39 mm, attributed to a ruler measurement by Giacomo in `scan3d/config.json`; that sets training scale.
- Original even/odd validation shares full-fit initialization, marker planes and ruler priors: historical holdout/internal consistency, not independent validation. M6.1 reproduced median 1.4359 / p95 6.1526 mm.
- Hover runs `20261105-031058`, `043942`, `044620`: original stills, joints, predicted and detected pixels are retained. Reported offsets are 8.2 mm (clipped view), 6.0 mm and 2.7 mm. `scan3d/hover_check.py:analyse` computes the target from the calibration and converts image displacement using that calibration. These are physical cross-view experiments, **not independently measured target XYZ**. The later runs use a supplied 55 mm object height. Measurement uncertainty and external XYZ ground truth are absent. Thus the per-reference independent XYZ-error table currently has **zero rows**, not zero error.
- Scan telemetry/CSV and camera metadata retain capture information but do not supply independently surveyed robot-base targets. Reconstruction coordinates and default marker-layout coordinates are not robot-base measurements. No historical qualification ledger was found. `data/m61` has no station/reference/approval record.

| Original ruler observation | Measured gap mm | Fit-implied gap mm | Signed fit-minus-measurement mm | Classification |
|---|---:|---:|---:|---|
| A/check-2 | 151 | 149.5 | -1.5 | training |
| A/check-4 | 151 | 147.3 | -3.7 | training |
| A/check-table | 200 | 197.2 | -2.8 | plane fitting |
| B/check-2 | 153 | 150.9 | -2.1 | training |
| B/check-4 | 153 | 147.9 | -5.1 | training |
| C/check-2 | 181 | 180.2 | -0.8 | training |
| C/check-4 | 180 | 184.2 | +4.2 | training |

These scalar residuals are not Euclidean localization errors and cannot fill the qualification ledger. No targeted parameter refinement is justified from an empty independent residual set: refitting these same rows and validating against them would remain circular.

## Confirmed current setup

A fresh stationary Pi capture and guarded xArm reads succeeded. Camera: 1600×900, server rotation 180°, extra rotation 0, endpoint 192.168.5.160. Controller mode 0/state 5, TCP offset `[0,0,0,0,0,0]`, world offset `[0,0,0,0,0,0]`. These zeros were **read**, not assumed or commanded. They agree with the original flange-based convention, but controller/model agreement does not independently establish tool geometry or unchanged physical mounting/table.

Authenticated `GET http://127.0.0.1:8010/api/vision/provider/station` succeeded using the operator-specified token file, without printing/copying the secret. The station is empty and the candidate hash matches V2. Identified/confirmed fields are candidate hash/provider type, endpoint/profile, units/frame and controller-configured offsets. Camera serial/physical identity, unchanged mount/base/table identity and an independently measured current plane are still missing. No complete operator-provided binding exists, so no incomplete record was POSTed or saved as verified.

Read-only validation output is `/tmp/trova-m62-validation.json`, with the corresponding fresh JPEG. It reports no independent samples/qualified uncertainty, all missing bindings, actual joints/controller configuration and image identity. There is no legitimate visible measured reference to assign an arbitrary pixel to; live metric output must remain refused until qualification passes. State 5 and movement geometry are separate future motion blockers and were not modified.

## Reproducible CLI

Use `/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify` with these actions. Authentication reads `TROVA_TOKEN_FILE` (default `~/.config/trova/token`) when present, otherwise the existing `TROVA_TOKEN`. It never uses XAI_API_KEY. API operations require the existing authenticated station and never disable authentication.

```bash
# Read-only evidence discovery and authenticated binding inspection
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify inventory \
  --output /tmp/trova-m62-inventory.json
TROVA_TOKEN_FILE=/home/simostocco/.config/trova/token \
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify inspect \
  --output /tmp/trova-m62-station.json

# Fresh Pi capture and guarded telemetry; no motion driver, Grok, planner or motor initialization
TROVA_ARM=sim TROVA_HARDWARE_UNLOCK= OPENBLAS_NUM_THREADS=1 \
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify validate \
  --data data/m61 --camera-host 192.168.5.160 --robot-ip 192.168.1.111 \
  --output /tmp/trova-m62-validation.json
```

The local `validate` action lists retained ledger rows and returns both unapproved qualification assessment and actual active provider. Per-observation results contain predicted/measured XYZ, signed XYZ error, Euclidean error, independent measurement uncertainty, joints, camera matrix and plane residual; aggregate uncertainty remains the existing empirical p95 plus measurement allowance. Scoring now intersects the **bound tilted plane**, not a plane forced through the measured reference's Z, so Z errors cannot silently disappear. References are table-support references, not arbitrary unknown-height RGB targets.

## Minimum next operator steps

1. First recover any separately retained surveyed XYZ notes or files from the already performed experiments. Supply their file path if they exist outside this workspace. Reuse is possible only when their coordinates, pixels/images, stationary joints, uncertainty and independent measurement source exist and their physical installation continuity is established. The recovered ruler gaps and fitted points above cannot supply the missing X/Y ground truth or uncertainties.
2. Complete the template returned by `inspect` with identified camera/mount/base/table and operator measurement notes. The controller TCP offset is confirmed above; distinguish controller configuration from physical tool verification. Measure the current table plane independently; do not copy fitted coefficients as new measurements. Preserve the exact V2 hash, units mm and frame robot_base. Save the completed template as `/tmp/measured-station.json`, then:

```bash
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify station \
  --file /tmp/measured-station.json --operator 'ACTUAL_OPERATOR_NAME' --confirm
```

3. If no reusable independent XYZ measurements exist, acquire three non-collinear identified table reference points with independently measured robot-base XYZ and declared uncertainty ≤2 mm. Observe those same points at two stationary diverse camera configurations (≥30 mm or ≥15° diversity), at least six observations. Reuse existing raw images/joints where they show those exact independently surveyed points and their physical identities/continuity are known; pixels can be redetected rather than photographs repeated. The operator alone arranges camera poses through their separate physical procedure; this CLI never moves the robot. Include the intended localization region/joint range. Current images without known point coordinates do not qualify accuracy.
4. For fresh observations, obtain `snapshot_id` from `POST /api/scene`. Each `/tmp/reference.json` contains `snapshot_id`, `pixel:[u,v]`, independent `measured_mm:[x,y,z]`, `measurement_note`, `measurement_uncertainty_mm`, and optional `reference_id`. Submit through:

```bash
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify reference \
  --file /tmp/reference.json
```

To reuse an existing **canonical independent JSONL ledger** instead, stop concurrent qualification writes, then run `import --data data/m61 --file PATH --confirm`. Each row must preserve the original independent measurement, reference_id, uncertainty, image/hash, robot/optical profile, calibration/binding hashes, `used_for_fit:false`, and `independent_measurement_source`. Import rejects fitting images, missing provenance and different installation bindings; it never automatically rebinds old observations. Source rows and ledger SHA-256 are retained, images are copied by content hash, duplicate rows are not counted twice, and changed evidence invalidates old approval. No recovered archive row currently meets these requirements. Noncanonical old notes require an explicit auditable transcription of their actual measurements, not synthesized coordinates.
5. Run `validate` and review every observation. Policy remains p95+uncertainty ≤5 mm, max+uncertainty ≤8 mm, plane residual ≤5 mm and existing FK consistency/domain checks. If these fail, inspect signed residuals before any targeted correction; keep new refinement data separate from validation. There is currently no evidence requiring a checkerboard recalibration.
6. `approve` checks the independent evidence **before** presenting approval as an available next step. Only after PASS does the operator rerun it with name and confirmation:

```bash
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify approve --data data/m61
# Only after its validation passes and the operator reviews it:
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_qualify approve \
  --data data/m61 --operator 'ACTUAL_OPERATOR_NAME' --confirm
```

It uses the existing qualification endpoint, then reads final providers. Desired result is historical_flange/V2, metric permitted, hardware unauthorized. Restart/reload the **read-only qualification server** on the updated code before using the new per-reference XYZ scoring; never start a hardware driver to reload qualification code.
7. After PASS only, use `live --pixel U V` with the operator-identified visible support point, the same `--data`, camera/robot arguments and `--output /tmp/trova-m62-point.json`. It captures a new stationary image and uses the qualified provider; without approval it returns the provider rejection. Do not use arbitrary pixels to claim success. The stationary reference needs no motion authorization; first motion qualification remains out of scope.

## Obtaining the missing XYZ without repeating the calibration fit

The old workspace was also searched in full before that copy was removed from the tree (`inventory --root Test_3d_live_video_old/calibration`): 52 metadata records, including old marker fits, capture/robot logs and `data/checks/hover_target.json`. The latter's `top_mm` is a calibration-derived prediction, not a surveyed target. No `measured_mm`/measurement-uncertainty/independent-reference ledger or actual touched-marker coordinate file was found. References to `data/markers.json` in configuration are instructions/defaults; that file with physical measurements is absent. The old fit and synthetic truth cannot substitute for current independent XYZ.

An independent way to obtain XYZ is an existing surveyed reference fixture registered to the robot-base datums, or a calibrated metrology instrument surveying three marked table points in that frame. Another operator-only option is a previously calibrated physical probe: the operator touches each mark through their separately authorized robot procedure, records the stationary contact joints, and computes the contact point as `FK_flange(q_contact) @ T_flange_probe @ [0,0,0,1]` using an independently measured probe offset. This does not use V2 or its camera pixels, and requires no controller TCP change. Do not treat the reported TCP as the contact point unless the real probe tip is demonstrably that TCP. Include robot/probe/base-registration/contact uncertainty; if it exceeds 2 mm, these references cannot meet the current measurement policy. No such movement/contact experiment was run here.

Label the marks P1/P2/P3, retain their independently measured XYZ, measurement method/source and uncertainty. At stationary pose A, run `python -m trova.calibration_qualify snapshot --output /tmp/pose-a.json`; preserve its snapshot/image/joints and identify all three pixels on the decoded 1600x900 image. Submit three `reference` payload files using that snapshot ID, the same measured XYZ and reference_id P1/P2/P3. Repeat at pose B with new snapshot/pixels (two fresh images suffice if all three points are visible). This registers six observations, not six newly fitted points. Camera repositioning is an operator-only procedure outside this coding task. Never move the marked fixture between surveying and either photograph.

Use the existing authenticated API/CLI after the read-only server reloads the code. `validate` gives the per-point XYZ-error table and missing requirements. `approve` remains a precheck unless all requirements pass and the operator subsequently supplies explicit confirmation. No controller readiness reset, movement limit change or hardware authorization is involved in metric qualification.
