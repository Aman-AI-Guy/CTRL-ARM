# Live localization and historical calibration audit

M6 update: the imported bundle-fit pixel RMS and checkerboard intrinsic RMS are both image residuals, but use different observations and fitted nuisance parameters (robot FK, marker poses and camera mounting versus intrinsic board fits). A shared activation threshold was inappropriate. The imported provider now qualifies using current independent metric references (p95 ≤5 mm, maximum ≤8 mm including declared measurement uncertainty), measured plane/FK consistency, exact optical compatibility and an explicit operator-approved record bound to model/station/evidence hashes. Its 6.148 px RMS remains visible as a diagnostic, not an activation gate. Native checkerboard gates are unchanged. Earlier audit statements below describe the pre-M6 rejection. No current physical references or approval have been fabricated.

The retained calibration is now a discoverable localization provider. It is **not active on this station**. A historical replay, an image region and an independently validated metric point are separate evidence states. No hardware motion was sent during this work.

## Actual execution path and original rejection

| Stage | Actual implementation and data |
|---|---|
| Pi acquisition | `camera.PiSource.read_capture` → `pi_cam.PiCam.get_jpeg/read_capture`, fresh decoded BGR/JPEG plus request metadata. Pi applies 180°; extra rotation must be zero. Display MJPEG is not the localization capture. |
| Fast regions | `State.proposals` → `proposals.find_blobs`, estimated median background or captured empty table, lighting-normalized LAB difference, threshold/morphology/contours/area filtering, merge and reading-order `Box.id`. Patterned/low-contrast objects can disappear; large objects can enter the estimated background. |
| Semantics | `inventory.describe` names existing regions. `finder.find` supplies annotated image and contact sheet to `GrokClient.json` (`trova_select`). Recognition never supplies robot coordinates. |
| Missing region | Both selection paths previously passed `direct_fallback=False`; the agent also rejected zero proposals. They now reuse `trova_direct`, validate finite ordered in-frame coordinates and confidence ≥0.65, and adopt the region into the frozen scene. `/api/select/region` supports existing-box clicks or operator drawings. |
| Persistent selection | `scene.assign_tracks`, `build_objects`, `adopt_found`; transient `box_id` differs from stable `track_id`. Same-view overlap now also needs image evidence. Fallback regions survive only a unique strong template match. Pose/rotation/profile changes prevent template reuse. Conflicting live mouse/cup labels allocate a new ID rather than attaching a cup to the mouse ID. |
| Autonomous loop | `server.AgentHost` → `agent.run`, `skill_inspect`, `skill_select`, mandatory `skill_locate`. `workflow.refers_to_selection` binds pronoun follow-ups; a missing/lost target is refused instead of silently reselected. |
| Telemetry | `State.robot_telemetry` → `arm.telemetry.LazyXArmTelemetry/XArmTelemetry.read`: TCP mm/degrees, five joints in degrees, moving/state/error. Guarded SDK reads; no enable, clear, TCP change or move. Snapshot reference evidence now requires a fresh live source (uploaded scenes are refused) and matching stationary telemetry before and after capture; wall-clock pairing is not hardware synchronization. |
| Calibration load | Previously only `VisionStore.load(TROVA_DATA/vision_calib.json)`. Missing file or matrices without captured observations leave intrinsics/hand-eye/table unset. The imported `calib/camera_robot.json` was only available through `scan3d.evidence` and M5 replay. |
| Provider routing | `localization.make_router` wraps that same store, discovers the imported candidate, and exposes current tool-camera, historical flange-camera and image-space providers. Editing/saving current calibration remains in `VisionStore`. No fabricated checkerboard observations or auto activation. |
| Pixel → metric | `CalibrationRouter.localize_observation` requires a geometric reference, then routes to `VisionStore.localize` or the qualified `HistoricalProvider.localize`/`FlangeCalibration.pixel_to_table`. A box center is not assumed to be a cup's foot or 3D center. |
| Observation goal | `agent._target_from_location` → existing `VisionStore.observation_target` with measured clearance, fixed xArm-5 roll/pitch and telemetry yaw. Support point, unmeasured object center and desired TCP observation pose remain distinct. |
| IK | `AgentHost.inverse` → guarded controller inverse-kinematics query. Hardware requires sampled path IK and driver-compatible orientation. |
| Trajectory | `skill_plan` → `MotionGateway.authorize_cartesian`; workspace, speed, camera body, keep-outs, uncertainty, joints/orientation, STOP and authorization remain enforced. A provider cannot authorize motion. Confirmation fingerprints also bind provider artifact, physical binding and independent evidence. |

The original invalid-calibration message was correct for the loaded current store, but omitted the retained alternative and its compatibility failures. The provider report now explains those failures. Missing checkerboard artifacts alone do not disqualify the alternative provider: a separately qualified synthetic test model reaches metric localization and the existing observation planner without checkerboard artifacts, then is rejected by a restrictive MotionGateway.

## Fallback alternatives evaluated

| Method | Result and choice |
|---|---|
| OpenCV background difference | Retained as the fast first path. The recorded transparent ruler is not covered by a matching blob; labels cannot repair a missing region by themselves. |
| Existing Grok direct-region call | Reused for the requested unboxed object, including zero proposals. No second detector service or extra model dependency; strict coordinate/confidence checks and explicit `grok_direct_region` provenance. Offline replay validates integration with annotated real Pi pixels, not new live model accuracy. |
| GrabCut segmentation | Actually evaluated on the wide recorded image with rectangular seeds, three iterations. Mouse: 22,619 foreground pixels, bounding box (304,108,166,178); ruler: 34,614 pixels, box (90,520,222,249). It depends on a correct seed and absorbed surrounding table for the transparent ruler. It cannot independently decide which object was requested, so it is not used as automatic semantic or metric evidence. It remains a possible future refinement after a reviewed region. |
| Operator click/draw | Implemented on a frozen snapshot with exact snapshot ID and validated pixel bounds. Clicking selects an existing region; drawing adds a new track. Operator labels remain non-authoritative model recognition. The headless browser regression exercises both box drawing and support-point marking. |

## What `pix2robot.py` actually implements

This is a pinhole/distortion model with a rigid flange-camera transform and robot forward kinematics, not a fixed empirical pixel regression. For five measured joint angles:

```
T_base_camera = FK_flange(q_degrees) @ T_flange_camera
ray_camera = undistortPoints(pixel, K, dist)
ray_base = transform(ray_camera, T_base_camera)
point = ray_base intersect z = a*x + b*y + c + measured_height
```

Coordinates are xArm robot-base millimetres; OpenCV camera axes are right/down/forward. Camera rotation is already included in the calibration images and fitted transform; applying 180° a second time is wrong. `for_size` in the original accepts same-aspect scaling, but Trova intentionally requires the exact optical profile and does not assume equal aspect ratio proves an identical crop.

The saved artifact contains 1600×900 intrinsics with fx=fy=1263.92057 px, cx=834.22150, cy=430.25801; k1=0.09849724, k2=-0.25480350 and zero tangential/k3 terms. The fitted camera translation in the flange frame is [-34.86581,22.61735,33.85540] mm. It is **not** a TCP-camera transform. Conversion requires the actual six-value flange-to-TCP transform: `T_tool_camera = inverse(T_flange_tool) @ T_flange_camera`.

`scan3d/handeye.py` initializes nominal optics from 62.2° horizontal FOV, solves marker PnP, initializes Park/Martin hand-eye, and jointly fits f,cx,cy,k1,k2, flange-camera pose and marker poses using Huber-weighted Levenberg–Marquardt with weak CAD/optical priors. The marker side is a supplied 39 mm physical measurement. The 53 mm cube height is supplied, not inferred monocular depth. Table feet are derived from fitted upward marker planes minus that height. The saved horizontal table is z=-34.1 mm; it has no saved tilted `table_plane` or scene-fact measurements. The notes mention a 40 mm pedestal. These do not measure today's table orientation/origin.

Joints determine the camera pose; there is no single fixed observation height. Metric output is a ray-plane intersection at a supplied support/object-top height, not measured arbitrary 3D depth. The original `pixel_to_object_top` explicitly warns that an oblique silhouette center need not be the object's top center.

## Reproduced historical evaluation

| Procedure | Images | Retained marker views | Held-out corner distances | Median mm | p95 mm | RMS px |
|---|---:|---:|---:|---:|---:|---:|
| Saved report | 19 | 48 | 192 by procedure | 1.84 | 4.67 | 6.148 |
| Migrated saved-seed replay | 19 | 48 | 192 | 1.85038 | 4.67249 | 6.15006 |
| Original source, fresh initialization and solve | 19 | 48 | 192 | 1.83 | 4.66 | 6.137 |

49 marker views are initially detected; `015.jpg`, marker 4 (23.9 px residual), is excluded by the original outlier policy. The even-image fit tests odd images and vice versa. **Both fits initialize from the full fit and score against full-fit marker planes.** The 192 distances are four corners ×48 views, not 192 independently measured physical reference points. The result demonstrates internal consistency across recorded poses, not an independent workspace accuracy guarantee.

Recorded joint ranges in degrees: J1 [-124.18,-52.38], J2 [-0.2409,58.61], J3 [-119.47,-32.6], J4 [-66.15,20.56], J5 [-0.59,86.26]. Unseen joint/height ranges and a changed mount are unvalidated. The physical hover record reports an 8.2 mm offset, with the mouse clipped at the top of the image; it cannot validate full lateral accuracy.

Original source was imported read-only with bytecode disabled; its evaluation outputs are in `/tmp/trova-original-calibration-eval`. Migrated replay is in `/tmp/trova-calibration-replay-audit.json`. Neither run changed or activated the retained calibration.

## Current physical compatibility

Read-only live audit on 2026-11-05 reached Pi 192.168.5.160:8080 and xArm 192.168.1.111. The Pi returned 1600×900, rotate 180, protection normal. Robot telemetry was stationary, state 5, error 0, joints [-90.907,-15.152,-66.708,137.849,-135.061]. This differs substantially from the historical J2/J4/J5 ranges. No controller state was changed.

| Parameter | Evidence and disposition |
|---|---|
| Robot model / joints | Controller reports five-axis XF1203; flange FK agrees with reported TCP at this one pose to 0.0014 mm for an inferred near-identity offset. This supports the joint convention at that pose; it is not an independent measurement of the tool or global base frame. |
| Image dimensions / server rotation | Confirmed current 1600×900 and 180°, compatible nominal profile. Earlier 1640×1232 is a different aspect ratio and incompatible without a newly qualified profile. |
| Camera sensor identity / crop / optics | IMX219 documented; serial/physical identity and unchanged optical crop are not independently bound to the artifact. Matching dimensions/model/host do not prove this. |
| Camera mounting | Historical flange mounting and CAD priors exist. Continuity, absence of remount/slip and present transform are unverified. |
| TCP | Controller-based inferred flange→TCP is near identity at the live pose. Actual physical TCP/pointer configuration still needs operator confirmation/measurement; inference is explicitly labelled. No TCP was set. |
| Base/world frame | FK/pose agree locally; unchanged base placement, configured world offset and physical reference origin remain unverified. |
| Table | Historical z=-34.1 mm, no measured slope in saved artifact. Current height/orientation/base relationship unverified. |
| Accuracy gate | Historical RMS 6.148 px exceeds the unchanged 1.5 px calibration gate even if dimensions match. Current independent references are absent. |

**Reuse decision:** reuse the algorithms and load the parameters for diagnostics/replay. Do not activate this physical artifact now. Refine the original model with targeted measured constraints if needed, retain its flange convention, and qualify the resulting candidate independently. There is no reason to begin by building another calibration framework.

## Targeted independent qualification

First record the station's actual camera/optical profile, mount continuity, base/table identities and measured flange-to-TCP transform in local `TROVA_DATA/calibration_station.json`. The file is an explicit operator physical binding, never written by Grok or auto generated as valid. Required fields are documented by `HistoricalProvider.assess`: provider=`historical_flange`, exact candidate `calibration_sha256`, camera_identity, optical_profile, mount_id, robot_base_id, table_id, camera_host, measurement_note, tcp_offset_mm (six measured mm/degrees values), units=`mm`, frame=`robot_base`, and measured table_plane {a,b,c,residual_mm}. Do not fill missing measurements with guesses.

The minimum validation set is three independently measured, non-collinear support/reference points spanning the intended work area, observed at two diverse **operator-positioned** stationary camera poses (six observations). Measure coordinates in the robot base frame with a known reference/tool/ruler; fitted marker poses and model predictions are not ground truth. The three support points can also establish the current table plane. Verify the actual marker side/cube height if refining the original solver. No robot repositioning is automated by this workflow.

1. `POST /api/scene` captures without parking/moving the arm and binds a snapshot to stationary before/after telemetry.
2. `POST /api/vision/provider/reference` accepts snapshot_id, pixel [u,v], measured_mm [x,y,z], and a measurement_note. It checks source, exact resolution/rotation, image and pose/joints, stores image hashes and bound independent evidence, and returns a diagnostic error. It **does not activate** a provider.
3. `GET /api/vision/providers` evaluates candidate RMS ≤1.5 px, independent reprojection RMS ≤1.5 px, reference p95 ≤5 mm, plane residual ≤5 mm, 2 mm/2° FK consistency, spatial/pose diversity, unchanged image/binding hashes, exact optical profile and qualified joint/XY domain. It exposes every rejection.
4. A candidate must incorporate the measured table plane and pass the original quality gate; changing a reported quality number is not calibration. The existing artifact still fails. Editing a binding invalidates references collected under the old binding; collect references after finalizing it.
5. Only a genuinely qualified candidate can supply a metric support point. The router derives TCP-camera/plane representations for existing geometry consumers without populating the current store with invented captured observations.

Observed reference p95 plus a two-pixel allowance is reported as empirical uncertainty, **not a statistical guarantee outside the sampled domain**. The qualification domain does not prove continuous collision clearance or all five-axis poses.

For object tasks, click an existing region or draw one with **Select image region**. **Mark support-plane point** marks a visible support-plane point inside the selected region (`POST /api/select/reference`). A raised cup rim or silhouette center is not such a point. When a support point is hidden or height is unknown, the task remains blocked and asks for that geometric information; it does not claim an object center. A reference may follow a stationary same-view track only with visual association and ≤5% region-size change. A changed view requires marking it again.

## Object switching audit

`data/events.jsonl` records the earlier red-mouse requests. `data/m5/ui-demo/events.jsonl` at 05:59:56 records the explicit goal **"posizionati sopra il bicchiere verde"**, with a Grok selection of green cup box 3 / track 8. That logged run changed target because the task named a new object. There is no evidence in that run of a mouse track silently becoming the cup. Broader identity hardening still addresses plausible failure modes: typed/voice pronouns require retained context, fresh observations must recover that ID, image matches need more than histogram/overlap, and authoritative semantic type conflicts cannot keep the previous ID.

## Read-only demonstration

No credentials are changed. These commands do not enable motors, clear errors, set TCP, or execute robot movements. No push is required.

```bash
# Retained recorded evaluation; no network or robot.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m trova.integration_m5 calibration-replay --output /tmp/trova-calibration-replay.json

# Fresh Pi image and guarded robot reads, independent of any motion driver.
TROVA_ARM=sim TROVA_HARDWARE_UNLOCK= .venv/bin/python -m trova.calibration_audit \
  --live --camera-host 192.168.5.160 --robot-ip 192.168.1.111 \
  --data data/m5/ui-demo --output /tmp/trova-calibration-live.json

# Station server with live recognition if the existing environment has its valid API key.
TROVA_DATA="$PWD/data/m5/ui-demo" TROVA_HOST=127.0.0.1 TROVA_PORT=8012 \
TROVA_CAMERA=pi:192.168.5.160 TROVA_CAMERA_ON_ARM=1 TROVA_CAM_EXTRA_ROTATE=0 \
TROVA_ARM=sim TROVA_ARM_READ=xarm:192.168.1.111 \
TROVA_MOTION_MODE=dry_run TROVA_HARDWARE_UNLOCK= .venv/bin/python -m trova

curl --silent --show-error -X POST http://127.0.0.1:8012/api/scene
curl --silent --show-error http://127.0.0.1:8012/api/vision/providers
curl --silent --show-error -X POST http://127.0.0.1:8012/api/agent/task \
  -H 'Content-Type: application/json' -d '{"text":"Posizionati sopra il mouse rosso."}'
```

The expected current result is recognized/region-selected when the mouse is visible, then a specific calibration/measurement rejection, **no validated metric target and no object-directed trajectory**. Do not interpret that refusal as a new request to move or clear controller state. The read-only audit can optionally compare a genuinely measured reference using `--pixel U V --measured X Y Z`; values must come from the operator, never from the model's prediction.

The standalone live check succeeded in reading both devices. Selection regressions use real retained Pi pixels with deterministic recorded-model responses; they do not establish new live Grok accuracy. The complete offline suite and headless browser smoke are required before committing changes.

## Final verification

- Repository `.venv`: **234 passed, 6 skipped**; skips were Node, SciPy/Open3D and unresolved camera mDNS.
- Existing complete environment: **240 passed, no skips**, in 164.83 s, including the read-only Pi camera check at the confirmed DHCP address. Two dependency deprecation warnings remain.
- Chromium feature smoke, including frozen-image drawing and support-point selection: **PASS**, no JavaScript errors, unexpected HTTP failures or mobile overflow.
- Python compilation, unused/undefined-name checks and staged whitespace checks passed.

Exact complete-suite command (uses already installed dependencies; no installation):

```bash
PATH=/home/simostocco/hackathon/.venv/lib/python3.10/site-packages/playwright/driver:$PATH \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TROVA_CAM_HOST=192.168.5.160 \
/home/simostocco/hackathon/.venv/bin/python -m pytest -q -rs

PLAYWRIGHT_BROWSERS_PATH=/tmp/trova-playwright .venv/bin/python tests/browser_smoke.py
```
