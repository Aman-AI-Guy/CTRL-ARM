# M6.1 — V2 candidate and Z=100 dry-run envelope

V2 is the primary imported candidate through `config/calibration_candidate.json`, with its SHA-256 pinned. `make_router` honors an explicit argument or `TROVA_CALIBRATION_CANDIDATE` first, then the manifest. A missing/changed manifest artifact fails closed; it does not silently fall back to the old model. This selection supplies a candidate, not current physical qualification or hardware authorization. The old bundle remains available to the explicit historical replay functions.

`config/workspace_candidate.json` records the requested X/Y [-500,500], Z [100,1000] mm, speed 20 mm/s and jog cap 300 mm. It is marked `dry_run_only`, contains no measurement confirmation, and is accepted only by the driverless diagnostic. The operating `station_workspace.json` remains at Z=200 because the new envelope has not passed physical qualification. It must not be promoted simply because the TCP fits.

## Artifact identity and numerical changes

The new root and `calib/v2/camera_robot.json` are byte-identical, SHA-256:
`59c1afadeaebaafb66fa6caa165440fea9ca8a3bde2e2d7d4db6321387ec6f4d`.
The old artifact SHA-256 is `94edc7b5dd10a78796ac8ad5c7a588ef2577e23bf83b0c5f0294ebe42832c6ff`.

| Parameter | Old | V2 |
|---|---|---|
| Image | 1600×900 | 1600×900 |
| fx = fy, pixels | 1263.920570005984 | 1304.925634628151 |
| cx, cy, pixels | 834.221495338442, 430.258009775308 | 829.281958865646, 467.686112900349 |
| k1, k2 | 0.098497239085, -0.254803502298 | 0.082162261165, -0.154143898762 |
| p1, p2, k3 | 0, 0, 0 | 0, 0, 0 |
| Camera translation in flange, mm | -34.865810830852, 22.617350508868, 33.855401122507 | -26.820788268115, 25.374786618643, 39.095741846943 |
| Constant table summary, mm | -34.1 | -40.97 (summary only) |
| Plane z = a*x + b*y + c | a=b=0, c=-34.1 | a=-0.004281709384781, b=0.000736015707124, c=-40.689342354919 |
| Plane fit | absent | tilt 0.249°, RMS 1.28 mm, six fit-derived points |
| Head sphere | center [0,0,45], radius 60 mm | unchanged |
| Cube size | 53 mm | unchanged |
| Images / marker views | 19 / 48 | 57 / 92 |
| Reported RMS / median / p95 | 6.148 px / 1.84 mm / 4.67 mm | 8.4 px / 1.44 mm / 6.15 mm |

Intrinsic matrices are `[[f,0,cx],[0,f,cy],[0,0,1]]`. Full flange-camera matrices (row-major, translation in mm):

```text
Old:
[-0.707885920913 -0.023549224865 -0.705934102436 -34.865810830852]
[-0.706326721991  0.023189215320  0.707506057991  22.617350508868]
[-0.000291161350  0.999453697928 -0.033048765846  33.855401122507]
[ 0               0               0               1             ]
V2:
[-0.693078237672  0.003777629926 -0.720852471715 -26.820788268115]
[-0.720746133335  0.014325685335  0.693051070284  25.374786618643]
[ 0.012944796146  0.999890246103 -0.007206108583  39.095741846943]
[ 0               0               0               1             ]
```

The fitted mount changes by [8.0450,2.7574,5.2403] mm and 2.0368°. This is a change in estimated parameters, not proof the physical mount moved. V2 also has nine separate marker poses across three legs; the cubes moved between legs, so those coordinates must not be treated as the old marker coordinates or fixed current references.

`FlangeCalibration.pixel_to_table/pixel_to_object_top` use the full tilted plane. The qualified provider uses the independently measured current plane, including both slopes, through `pixel_to_plane(u,v,q,c+height,(a,b))`. The stored scalar table summary is not substituted for that plane. The live profile was confirmed as 1600×900, server rotation 180°, extra rotation zero; no resizing or second rotation was applied. Camera/mount identity and optical continuity remain operator evidence, not consequences of matching image dimensions.

## Recomputed offline comparison

Run without camera, SDK or actuation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/simostocco/hackathon/.venv/bin/python -m trova.calibration_compare \
  --output /tmp/trova-m61-comparison.json
```

The JSON contains the full original parameters, numerical matrix/vector deltas and evaluation details. Real retained images are redetected; duplicate start images and recorded outlier exclusions follow the original method. Results:

- 57 images, 92 retained marker views, 368 corners; saved-fit reprojection RMS **8.39945 px**, median 5.07388 px, p95 18.74963 px.
- Original even/odd procedure from saved rounded marker geometry: **1.43586 mm median / 6.15262 mm p95 / 15.34569 mm maximum**, 368 errors. Full-fit initialization, full-fit planes and the fitted ruler priors remain shared. This reproduces the reported values but is not independent validation.
- Fair same-image predictive comparison: keep each camera model fixed; fit only per-leg marker poses on training images, then evaluate held-out images against those fitted planes. V2: **1.57231 mm median / 6.00702 mm p95**; old: **2.33623 / 13.54780 mm**. Both use the same 344 held-out corners. Twenty-four corners are excluded because the relevant split has fewer than two training views. Neither set is physical ground truth independent of fitting; V2 optical parameters were themselves originally fitted using this dataset.
- V2 predictive tails by leg: A p95 7.40101 mm, B 3.27158 mm, C 5.26572 mm. The wider three-leg dataset has a heavy tail; the lower median does not imply uniformly lower error. Comparing old/new headline p95 on different fitting datasets would not establish degradation or improvement.
- All seven ruler answers informed the fit or table plane. Their saved absolute fit residual median is 2.8 mm, p95 4.83 mm, maximum 5.1 mm. They are not new independent validation. No threshold was relaxed: current independent qualification still requires p95 plus declared measurement uncertainty ≤5 mm and maximum plus uncertainty ≤8 mm. If current errors reproduce the V2 tail, qualification must fail; refine the model or validate an explicitly restricted operating domain with new evidence.

V2 recorded disk hover `20261105-044620_hover` uses pixel [813.36611047,535.82104910], recorded joints and supplied height 55 mm. Full-plane conversion reproduces [-0.47727163,-477.90845759,13.96095305] base mm. A scalar-Z approximation differs by 0.06922 mm for this point; other pixels/poses need not have that small an error. Round-trip projection is checked. Imported five-axis IK finds an 85° view at nominal 120 mm vertical camera clearance, joints approximately [-91.54268,44.33057,-49.41929,-82.98055,48.29209]. This is a recorded numerical proposal only. Applying the proposed Z=100 bounds to the full interval certificate rejects this path at `workspace/head, workspace/camera`.

## Current envelope and live blockers

At q=[0,0,0,0,0], TCP is [207,0,112] mm, giving 12 mm point clearance to Z=100. V2's optical center is [180.1792,-25.3748,72.9043] mm. Imported CAD sphere extents include head Z=[5.9,126.1] mm and link5 Z=[97.5,152.5] mm. Thus the camera/head and link5 do not lie wholly within this proposed box; actual tool geometry is missing. This is model evidence, not a new physical measurement. The fixed base naturally extends below the TCP workspace; the gateway's workspace bounds apply to TCP/head/camera/tool, while link collisions use table/obstacles and self checks. The diagnostic lists every link extent so these meanings remain explicit.

Read-only live results: camera/Grok/telemetry available, exact resolution/rotation compatible, selected V2 hash correct, stationary controller mode 0/state **5** with error/warning 0. No red mouse visible; recognition does not silently select the cup instead. Station identity/TCP/plane binding, independent image references and approval are missing. Camera/tool measurements, keep-out assessment, observation clearance, object geometry and certified motion envelope remain unconfigured. No metric live target or executable live proposal exists. No state reset is attempted.

Exact driverless command (no motion driver initialized):

```bash
TROVA_ARM=sim TROVA_MOTION_MODE=dry_run TROVA_HARDWARE_UNLOCK= \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/simostocco/hackathon/.venv/bin/python -m trova.diagnose \
  --live --camera-host 192.168.5.160 --robot-ip 192.168.1.111 \
  --data data/m61 --candidate 'Test 3d live video/calib/v2/camera_robot.json' \
  --workspace-candidate config/workspace_candidate.json --text 'mouse rosso' \
  --check-recorded-fk --recorded-plan --output /tmp/trova-m61-live.json
```

`--recorded-plan` remains explicitly labelled original replay, never a live or V2-qualified trajectory. Independent stages are reported even when localization is blocked. Workspace candidate statuses are NOT_VERIFIED; they do not become physical PASS results.

## Minimum operator qualification workflow

1. Keep hardware disabled. Identify the installed camera/optical mode, flange mounting, robot base/table and exact artifact. Inspect mounting continuity against the physical setup, not just model names. Read the controller's TCP/world settings; independently verify the flange-to-TCP transform and base convention. Six TCP values mean `[x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg]`; the live TCP position [207,0,112] is **not** the flange-to-TCP offset. Do not replace a missing offset with zeros inferred from FK agreement.
2. Measure at least three non-collinear support points in robot-base mm with a suitable independent reference/probe and declared uncertainty ≤2 mm. Derive the current plane `z=a*x+b*y+c` and measurement residual. Do not copy V2's fitted table coefficients into the current station as if they were measured today. Plan an independent work-area/pose domain that covers the intended camera viewpoints; the provider refuses extrapolation outside qualified joint/XY ranges.
3. Start the existing read-only station with the same data directory (this opens telemetry and a simulated motion driver, never a physical motion driver):

```bash
TROVA_DATA=data/m61 TROVA_HARDWARE_UNLOCK= \
/home/simostocco/hackathon/.venv/bin/python -m trova.station \
  --profile qualify --camera-on-arm --camera-fallback 192.168.5.160 \
  --robot-ip 192.168.1.111 --port 8000
```

4. `GET /api/vision/provider/station` returns the exact candidate hash, current binding, editable `template`, required TCP order and all field errors. Fill `camera_identity`, `optical_profile`, `mount_id`, `robot_base_id`, `table_id`, `measurement_note`, `camera_host`, the measured six-value `tcp_offset_mm`, and `table_plane` (`a,b,c,residual_mm`). The template leaves measurements null, deliberately. Preserve `provider=historical_flange`, the candidate hash, `units=mm`, `frame=robot_base`. POST to the same URL with `{"operator":"actual name","confirm":true,"station":{...completed template...}}`. Extra qualification/hardware fields are rejected. This only saves a binding; changing it invalidates old evidence/approval. The previous generic `Expected 6 finite values` error now names the TCP field, units/order and missing other station measurements.
5. Use `POST /api/scene` for a fresh stationary image. For each of the three measured support points, identify the exact corresponding pixel; POST `/api/vision/provider/reference` with `snapshot_id`, `pixel:[u,v]`, independently `measured_mm:[x,y,z]`, `measurement_note` and `measurement_uncertainty_mm`. Repeat at a second independently arranged stationary camera pose (≥30 mm or ≥15° diversity) through the robot owner's separate physical procedure. Keep the same point identities. This gives at least six real reference observations. No coding-agent command moves the robot to obtain them. Use additional viewpoints/references where needed to bound the intended approach domain; do not omit failing observations to make qualification pass.
6. Recheck the binding and all errors, then POST `/api/vision/provider/qualify` with operator and confirm. It requires the independent 5 mm p95 / 8 mm maximum bounds including uncertainty, non-collinearity/domain checks, pose/FK agreement and unchanged evidence hashes. It never unlocks hardware. A failing result stays inactive and reports the targeted mismatch; do not ask for a new checkerboard solve merely to replace missing station metadata.
7. Before any first approach, physically measure camera/tool/body geometry, object height/radius and support point, table clearance, keep-outs, speed and qualified joint/object envelope. Have the robot owner review the wrist contact exclusion and establish controller interpolation/tracking bounds. Assess whether Z=100 is appropriate for the **whole commanded path and head/tool**, not the TCP alone. The current zero-joint pose and the tested recorded path fail those model workspace checks. Any safe repositioning/recovery and envelope qualification are separate operator-controlled procedures; do not bypass the failing start check or silently widen bounds. Only after physical evidence supports the change should the operator promote reviewed workspace limits through the existing safety mechanism.
8. The robot owner resolves state 5/faults and establishes ready position mode through their authorized procedure. Place/identify the intended object in a fresh image. Select it, mark its measured geometric reference, localize, and review the five-axis plan and complete gateway certificate. Hardware mode/unlock remains a separate human action; confirm exactly one unexpired proposal only after all checks pass. Review final real telemetry and the fresh photograph. The supervised approach/orbit procedures in `M6_OPERATOR_GUIDE.md` still apply, using this V2 candidate and qualified data directory. No physical approach was executed in M6.1.

API calls use the existing authenticated browser session or `x-trova-token` header when configured. Credentials are not part of station evidence, reports or commits.

## Verification

Complete Trova offline suite: **264 passed** using `/home/simostocco/hackathon/.venv/bin/python -m pytest -q -rs` (bundled Node on PATH, BLAS threads limited, camera host supplied for read-only coverage). Original imported tests: **64 passed** in the isolated temporary source copy with FakeArm fixtures. Static undefined/unused-name checks and `git diff --check` passed. No automated actuator test connected a physical driver. Live reads confirmed the stated V2/profile/workspace blockers; no physical qualification flags were populated.
