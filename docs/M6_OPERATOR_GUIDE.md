# M6: qualified pixel-to-base approaches and bounded scanning

M6.1 update: the pinned primary provider candidate is now V2. See [V2 qualification and Z=100 dry-run results](M61_V2_QUALIFICATION.md) before operation; the operating workspace has not been lowered without physical qualification.

The existing Trova stack now supports imported `pix2robot` calibration, camera-centered five-axis approaches, whole-robot trajectory checks, the existing confirmation/executor path, and bounded scan scopes. Physical qualification remains operator work. No physical movement, motor enable, error clear, TCP change or original-framework actuator entry point was run during development.

## Data flow and frames

`static/voice.js` / `speech.js` → `/api/agent/task` → `AgentHost.snapshot` → `State.snapshot_scene/capture_frame` → `proposals.find_blobs` → `finder.find` / direct fallback → `scene.assign_tracks/build_objects` → `agent.skill_select` → `skill_locate` → `CalibrationRouter.localize_observation` → `OrbitStore.approach` → `scan3d.approach.propose` → imported `orbit_math.solve_one/solve_view` → `MotionGateway.authorize_joints` → bound receipt → `/api/agent/confirm` → `scan3d.skills.observe` → `AgentHost.execute_orbit` under `State.start_arm` → `MotionGateway.execute_joints` → existing `CartesianArm.go_joints` / xArm driver → final telemetry → fresh capture/evaluation.

| Quantity | Frame, units and convention |
|---|---|
| Region/reference pixel | Image, u right / v down, decoded pixels, exact optical profile |
| Undistorted camera ray | OpenCV camera: x right, y down, z forward |
| Imported camera pose | `T_base_camera(q) = FK_flange(q_degrees) @ T_flange_camera`; translations mm |
| Controller tool pose | `T_base_tool = FK_flange(q) @ T_flange_tool`; xArm roll/pitch/yaw degrees, `Rz @ Ry @ Rx` |
| Tool-camera conversion | `T_tool_camera = inverse(T_flange_tool) @ T_flange_camera`; never assume TCP equals flange |
| Metric reference | Robot base, mm; ray intersects measured support or known-height plane, or corroborates an independently measured base point |
| Desired observation | Lens center/axis relative to the object reference; image roll is free. Joint IK avoids arbitrary six-DOF Cartesian orientation requests |
| Object volume | Measured height and enclosing radius around the selected reference; the box center is not a measured 3D center |

The Pi source confirms that 1600×900 is a 16:9 ISP crop of the binned sensor image; 1640×1232 is the 4:3 profile. These are not interchangeable uniform scales. The server rotates 180°. No second rotation or automatic intrinsic rescaling is applied.

## Calibration qualification, not another calibration framework

The imported calibration uses fitted intrinsics/radial distortion, measured joint-dependent flange FK, a fitted rigid flange-camera transform and a known plane. Bundle fitting also estimates marker poses and includes robot/CAD uncertainty. Its corner RMS and checkerboard intrinsic RMS have the same pixel units but different fit/data scopes. The native checkerboard provider retains its 1.5 px gate; the imported provider uses current independent metric validation instead of borrowing that gate.

The original rejection had two distinct causes. `VisionStore` loaded only Trova's native `vision_calib.json` intrinsics/tool-hand-eye/table representation; the flange calibration was retained as replay evidence. Provider routing then exposed the model but still required explicit station binding and independent current references, and initially compared the imported bundle RMS with the native checkerboard gate. M6 removes that architectural mismatch while retaining the missing-current-evidence rejection. Changing provider does not establish an unchanged physical mount or measured TCP.

`pix2robot.Calibration.ray` uses `cv2.undistortPoints`, transforms `[x,y,1]` with `flange_pose(q) @ T_flange_camera` and normalizes it. `pixel_to_plane` intersects that base-frame ray with `z = a*x + b*y + c + known_height`. This is joint-dependent plane localization across recorded arm configurations, not unconstrained full 3D recovery or a fixed-camera homography. The mapping itself uses flange FK and is independent of controller TCP; planning and controller cross-checks require the actual flange-to-TCP transform. The fit migrated in `scan3d.marker_fit.Problem/solve_all` estimates focal/principal-point/radial parameters, rigid camera mounting and marker poses from marker corners across joint poses; known marker size sets scale, while the robot chain dimensions are fixed model inputs. `scan3d.flange.FlangeCalibration` preserves the original ray/FK equations.

Original half-split replay: **19 images, 48 retained marker observations, 192 held-out corner errors; median 1.8504 mm, p95 4.6725 mm**. One recorded outlier is excluded. Each split starts from the full fit and uses full-fit marker planes; these results are internal consistency evidence, not an independent current ruler check. The recorded mouse hover supplied object height and 120 mm clearance; the reported ~8 mm lateral error is not depth inferred from RGB.

Imported qualification requires:

- Exact optical profile/rotation, identified unchanged camera/mount/base, actual measured flange-to-TCP transform and a measured table plane.
- At least six independent observations: three non-collinear measured support points from two diverse stationary poses. Declare measurement uncertainty (0..2 mm), not a model-predicted coordinate. Include the intended observation pose range and work area.
- Reference p95 plus measurement uncertainty ≤5 mm; maximum error plus measurement uncertainty ≤8 mm. Pixel residual is retained as a diagnostic. Plane residual ≤5 mm and controller/local FK consistency ≤2 mm / 2° remain enforced.
- `POST /api/vision/provider/qualify` with an explicit operator name/confirmation. The record binds the exact calibration artifact, physical station binding and image/reference ledger hashes. New evidence, changed settings or a different artifact invalidate the approval. This never enables hardware.

Use the existing `calibration_station.json` and `/api/vision/provider/reference` workflow described in `LIVE_LOCALIZATION_AUDIT.md`. The station record's measured plane is an explicit site correction to the original rays; it need not numerically equal the old saved table height. Do not repeat a full checkerboard solve merely to manufacture its artifact structure.

A newer supplied workspace is available at `Test 3d live video/`, with both `calib/camera_robot.json` and `calib/v2/camera_robot.json`. Discovery supports that layout and the earlier nested layout; explicit `TROVA_3D_WORKSPACE` and `TROVA_CALIBRATION_CANDIDATE` take priority. V2 reports 57 images / 92 marker views, RMS 8.4 px, internal holdout median 1.44 mm / p95 6.15 mm, and a measured tilted plane. Seven ruler answers constrain its fit, so they are not independent validation of that fit. Two saved later hover checks report 6.0 and 2.7 mm offsets at a supplied 55 mm object height. They are reported physical experiments, not measurements repeated during this coding task. Neither candidate is auto activated. Historical evidence replay discovery preserves the original M5 bundle (now under `Test_3d_live_video_old/calibration`) for reproducible replay and calibration hashes; select the newer workspace/artifact explicitly. The newly supplied root calibration currently contains the newer 57-image fit as well, so filenames alone do not establish version identity.

`/api/select/reference` supports `support_plane_point`, `known_height_point` (height and uncertainty) and `measured_base_point` (independent point, uncertainty and measurement note). All are bound to one snapshot/track and corroborated against its region. RGB does not invent depth. References can be carried through a new view only with reliable visual/geometric association and the explicitly qualified static-object assumption. Identity loss stops continuation.

## Approach and trajectory qualification

The approach planner tries a directly overhead camera view, then bounded 85°, 75° and 65° alternatives around the current azimuth. It reuses the original five-axis camera IK and checks numerical task conditioning. It retains joint-jump/cable limits. With the recorded hover inputs, it finds a 75° camera-centered view at the requested 120 mm vertical clearance; the overhead solution is rejected by the existing 45° jump limit. This is a recorded numerical result, not current station qualification.

Sampled checks remain visible. Hardware additionally requires an adaptive interval certificate: sphere-center motion bounds cover the unsampled joint boxes between commanded checkpoints. The executor actually commands the certified checkpoints, revalidating each through the same gateway. TCP speed and acceleration have analytic conservative bounds. The certificate only covers the supplied model/environment and qualified controller tracking bounds.

Enforced geometry includes robot links, rotated camera housing, configured camera sphere, measured tool spheres, object cylinder plus localization uncertainty, table, known keep-out boxes and non-neighbour self pairs. The inflated head/link4 covers overlap at their connected wrist assembly; that contact exclusion requires explicit operator review. Adjacent mechanical contacts and unknown obstacles are not a full collision model. `collision_avoidance` stays false.

Before hardware planning, `orbit_settings.json` must contain measured object height/radius, six TCP values, geometry confirmation, tool spheres (or explicit assessment that no additional tool exists), and `motion_qualification` with:

- `operator`, `measurement_note`, matching `model_sha256` and `mount_sha256` exposed by the geometry proposal;
- `head_link4_contact_reviewed`, `tool_body_assessed`, `static_object_assessed`;
- `controller_segment_box_verified` and measured `tracking_error_mm` (0..2 mm);
- `joint_min_deg` / `joint_max_deg`, `object_foot_min_mm` / `object_foot_max_mm`, `max_object_height_mm` / `max_object_radius_mm`.

These are physical assertions requiring evidence, not switches to set merely to pass. The current model and tool must fit the real installation. If no qualified controller interpolation/tracking bounds exist, stop and have the robot owner qualify them within a separately assessed envelope before the first object approach. Do not mark this verified to try an unknown path. Workspace limits, assessed keep-outs, measured camera body, operator clearance, STOP/fault checks and independent hardware unlock remain required. State 5 (`MODE_CHANGED`), paused/stopped states, warnings, missing mode/error reads and pose changes block execution; no auto reset occurs.

## Read-only end-to-end command

This command initializes no motion driver, even if another shell has hardware variables set. It evaluates each stage independently and writes PASS / FAIL / BLOCKED / NOT_CONFIGURED / NOT_VERIFIED with function names, inputs, reasons, categories and minimum actions. Credentials are not printed.

```bash
TROVA_ARM=sim TROVA_MOTION_MODE=dry_run TROVA_HARDWARE_UNLOCK= \
OPENBLAS_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m trova.diagnose \
  --live --camera-host 192.168.5.160 --robot-ip 192.168.1.111 \
  --data data/m6 --safety station_workspace.json --text 'mouse rosso' \
  --check-recorded-fk --recorded-plan --output /tmp/trova-m6-live.json
```

Select an alternate artifact explicitly using `--candidate "$PWD/Test 3d live video/calib/v2/camera_robot.json"`. Add `--pixel U V` only for an operator-marked visible support point. For an elevated reference also supply `--height-mm H --height-uncertainty-mm E`, using measured values. Recorded planning is labelled separately and never becomes a live executable path.

Live checks during development reached 1600×900/180° Pi capture, Grok and five-joint xArm telemetry. The actual image viewed the table from its side; Grok reported that no red mouse was visible. The controller was mode 0, state 5, error/warning 0. Read-only controller FK on 19 recorded configurations agreed with the imported convention to 0.000062 mm translation spread and 0.000013° rotation spread for a near-identity inferred TCP offset. This is controller/model agreement, not an independent physical tool measurement. No metric target or executable live trajectory was claimed.

## Operator-only first approach and orbit

The following is a separate physical procedure; it was not executed during development.

1. Verify the real camera/mount/TCP/base/table and independent reference evidence, qualify the chosen provider, and fill the measured geometry/envelope fields above. Have the robot owner establish ready position mode and resolve state 5 or faults through their authorized controller procedure. Confirm physical clearance and an accessible e-stop. Do not widen the workspace to make a path pass.
2. Start the existing station in read-only mode first, with the Pi and `TROVA_ARM=sim`, real telemetry and no hardware unlock. Use the diagnostic command and review every stage. Mark the object region/reference and run the task. A missing measurement, target, IK solution or geometry certificate must remain a refusal.
3. Only the human operator starts the existing real driver with `TROVA_ARM=xarm:192.168.1.111`, `TROVA_ARM_READ=off`, `TROVA_MOTION_MODE=hardware` and their `TROVA_HARDWARE_UNLOCK=MOVE`. Keep `TROVA_ARM_TCP_MM=` empty so connecting does not alter the already qualified controller TCP. Use the same qualified data directory, measured safety settings and candidate path. The coding agent must not run this command.
4. Say/type “Seleziona il mouse rosso” and “Ora posizionati sopra.” Review the current-image region, reference, uncertainty, camera target, joints, interval clearance certificate and gateway verdict. Confirm **that specific** unexpired proposal once. Any changed pose/settings/scene or stale image requires a new proposal. Observe final joint/TCP telemetry and the fresh picture; centering alone is not movement verification.
5. For first scanning, say “Scansionalo.” Review and confirm each view separately. Each move settles, captures, saves real joints/TCP/calibrated camera pose, checks quality/identity and replans from the new actual configuration. A lost target, rejected capture or unavailable view stops the loop. Record supervised orbit qualification only after this physical review succeeds.
6. Optional bounded autonomy requires that recorded supervised success (`supervised_orbit_verified` and `verification_note`) in the same motion qualification. After reviewing the first pending scan proposal, the operator calls `POST /api/scan/orbit/authorize` with operator, confirm, max_views (1..12) and ttl_s (30..600). This sends no motion. The first proposal still needs explicit confirmation. Subsequent views consume the bounded scope and still recheck every new receipt, calibration/model/safety binding, identity, trajectory and capture. STOP/cancel, budget/time exhaustion, new tasks or changed settings end the scope. Grok Voice cannot issue this authorization or unlock hardware.
7. Finish the accepted scan and use the existing reconstruction action. COLMAP must actually be installed for a fresh full reconstruction. Retained dense-cloud replay remains postprocessing; mesh vertex colours/texture do not make a parametric CAD model. Coverage is viewpoint/image evidence, not a proof of complete surfaces.

Remaining blockers are separated in the diagnostic: configuration/software errors; camera/frame/calibration mismatch; unmeasured physical geometry/reference constraints; hardware state; and intentional read-only authorization. No valid-current physical measurement or first supervised movement is fabricated by tests.

Human-only station launch, **after** the qualifications above pass (not run by this coding task):

```bash
TROVA_DATA=data/m6 \
TROVA_CALIBRATION_CANDIDATE="$PWD/Test 3d live video/calib/camera_robot.json" \
TROVA_ARM_TCP_MM= TROVA_MOTION_MODE=hardware TROVA_HARDWARE_UNLOCK=MOVE \
/home/simostocco/hackathon/.venv/bin/python -m trova.station \
  --profile operate --camera-on-arm --camera-fallback 192.168.5.160 \
  --robot-ip 192.168.1.111 --port 8000
```

Use the artifact actually bound to the approved qualification, rather than changing candidates between qualification and operation. `data/m6` must contain the reviewed station/reference/provider approval and measured orbit/safety settings; an empty directory will refuse the task. The station also loads the operator's `station_workspace.json`. Its current minimum TCP Z is 200 mm: the recorded 120 mm camera-clearance example does not qualify a path within that workspace. Choose a physically measured feasible clearance and review the resulting proposal; do not lower the workspace boundary to replay the old experiment. The physical driver may initialize controller motion state, so only the human operator launches `operate`; diagnostics never do so.

## Verification and current outcome

The complete Trova suite passed **256 tests** with `/home/simostocco/hackathon/.venv/bin/python -m pytest -q -rs` (Node's bundled driver directory on PATH, live camera address supplied, BLAS threads limited). The original newly supplied suite passed **64 tests** in an isolated temporary copy using fake arms; retained originals were not edited. The red-mouse and low-contrast recorded-image regressions, provider qualification, metric data flow, five-axis approach, bounded simulated orbit, expired proposals and reconstruction replay are included in Trova's suite.

The live camera/service/FK stages pass; the requested mouse is absent from the current view, current provider qualification is missing, and controller state 5 blocks movement. Consequently there is **no validated live mouse coordinate or executable live trajectory**. Recorded inputs yield reachable joint approaches and 12 orbit candidates, independently labelled as replay. The original 60-image benchmark remains postprocessing/replay; full live COLMAP reconstruction needs the actual backend and newly acquired qualified images.

Chromium smoke passed with `PLAYWRIGHT_BROWSERS_PATH=/tmp/trova-playwright /home/simostocco/hackathon/.venv/bin/python tests/browser_smoke.py`, using a temporary mock server and simulated arms.
