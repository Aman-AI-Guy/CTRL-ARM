# M5 integration record

The retained workspace was discovered at `Test_3d_live_video/calibration`. Original code, reports, JPEGs, meshes and run logs stay in place. Its old instructions to keep Trova read-only and defer integration are superseded by the current M5 integration request. No original hardware execution entry point is imported into Trova.

## Compatibility and disposition

| Imported component | Integration decision |
|---|---|
| `calib/pix2robot.py` | Reuse flange FK, distorted rays, table-plane intersections and projection in `trova.scan3d.flange`; convert flange/TCP transforms explicitly. Historical parameters are not activated. |
| `scan3d/handeye.py` | Reuse marker bundle fitting, Park/Martin initialization, tilted-plane fitting and original half-split evaluation in `marker_fit`; expose candidate fitting through `VisionStore`. |
| `arm_pc/orbit_here.py`, `robot_geometry.py`, `robot_model.json` | Reuse planning/IK and sphere geometry; omit SDK connection, motor enable, movement and CLI execution functions. Joint movement belongs to MotionGateway. |
| `scan3d/orbit.py` | Review alternative route selection, branch/cable limits and collision sampling; preserve as evidence/reference where not reused. |
| `scan3d/reconstruct.py` | Reuse ArUco metric scale, dense reconstruction, Open3D object isolation/Poisson, robust dimensions, GLB export and software renders inside existing reconstruction jobs. Replace Windows-specific process launching. |
| `scan3d/receiver.py`, capture and camera code | Extend the existing Pi client/session infrastructure; no second camera client, driver or agent runtime. Retain raw motion-video timing as historical, not precise robot/image synchronization. |
| `trova-stream/` | Protocol reference only; no Pi deployment or default-profile changes during development. |
| datasets, reports and CAD | Local read-only evidence. No bulk Git add/upload. Raw robot meshes/CAD remain outside Git; reviewed compact sphere geometry and attribution may be packaged. |

A pre-integration SHA-256 manifest covers 754 retained non-cache, non-Windows-sidecar files (323,516,988 bytes), stored locally at `data/m5/original_manifest.json`. Imported tests run from `/tmp/trova-m5-imported-tests`, with outputs and caches outside the retained workspace.

## A: camera and robot calibration

The saved model is **1600×900**, radial distortion k1/k2, an OpenCV camera frame (x right, y down, z forward), millimetres, and `T_base_camera = FK_flange(joints_deg) @ T_flange_camera`. It is not a TCP-to-camera matrix. A measured six-value flange-to-TCP offset is needed to convert it into Trova's tool convention. The stored table is horizontal at z=-34.1 mm; notes discuss a measured pedestal of 40 mm and an unresolved tilted table. No slope or identity metadata is invented.

The IMX219 and 180-degree server rotation are documented historically, but current camera serial identity, mount continuity, TCP configuration and independent reference checks are not bound to this file. The scan profile 1640×1232 has a different aspect ratio. Matching aspect ratio alone also does not prove an identical crop/optical profile. No automatic live scaling/rotation or calibration activation is performed.

The historical reprojection RMS **6.148 px** exceeds Trova's existing **1.5 px** active-calibration gate. The integration keeps that gate. The historical matrix can be used for explicitly recorded replay and benchmarking, with provenance and uncertainty labelled; it cannot authorize a move.

Recomputed from 19 original JPEG/joint observations, with the recorded outlier excluded and the reused fitter/evaluation procedure: RMS **6.1501 px**, held-out median **1.8504 mm**, p95 **4.6725 mm** (reported 6.148 px, 1.84 mm, 4.67 mm), runtime 2.23 s in the requested Linux Python environment. This reproduces the original internal procedure, whose half-split fits start from the full fit and use full-fit marker planes. It is not an independent physical ruler measurement or a new live validation.

Commands (no robot connection):

```bash
/home/simostocco/hackathon/.venv/bin/python -m trova.integration_m5 audit
OPENBLAS_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m trova.integration_m5 calibration-replay --output data/m5/calibration_replay.json
```

The supplied archive reports robust object dimensions **96.3×55.7×37.4 mm**, extremes **102.8×63.4×41.3 mm**, and mesh extents **106.1×66.2×38.6 mm**. These are distinct estimators. Ruler values 98×58×35 mm are historical operator notes, not newly measured ground truth. The recorded hover reports an 8.2 mm offset and a clipped mouse image; full lateral accuracy cannot be inferred from it.

## B: orbit and robot geometry

The migrated numerical camera-position/direction IK is extracted from `arm_pc/orbit_here.py`, with its source hash retained. Controller connection, enable, halt, move, and executable CLI code are omitted. It shares flange FK with calibration. Candidate solutions retain five-axis constraints, the narrowed J2 lower limit, joint-branch handling and CAD-derived collision checks; Trova additionally rejects excessive joint jumps and J5 cable travel instead of merely printing warnings.

The compact `robot_model.json` contains sphere centers/radii in millimetres in the URDF link frames. Its surfaces were checked against every vertex of the retained UFACTORY STL files (metres, converted by ×1000) and the team's camera-head STL (millimetres, CAD-to-flange/head transform preserved). Raw STL/STEP assets remain outside Git. UFACTORY's [upstream BSD license](https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE) is retained as `LICENSE-UFACTORY`; the team camera-head geometry is operator-supplied, with no new public license asserted for its raw CAD.

MotionGateway checks the geometry inside joint authorization, including every sampled pose, even if a planner's earlier report said it was valid. It checks TCP workspace/keep-outs, rotated head spheres, table/object exclusion, selected self-collision pairs and link keep-outs. It compares controller and local FK/TCP within 2 mm in hardware mode. Requested joint speed is bounded against sampled TCP speed; joint acceleration is carried in the driver ticket. A qualified rotated-head check replaces the fixed-orientation camera offset only for that joint path. Modes and the independent hardware unlock are unchanged.

The checks sample joint-linear paths every 2 degrees. They are not continuous swept-volume checks, the imported self-pair set is incomplete, and controller interpolation equivalence is not proven. `collision_avoidance` remains false. Offline proposals can pass geometric checks but cannot execute until the current model, mounting/TCP, active calibration and operator settings are qualified.

The original 220-degree orbit was replayed from its saved starting joints/target: **12/12** generated views passed the integrated sampled geometry/gateway checks, with no robot opened. This uses the recorded scene assumptions and a deliberately broad offline workspace; it does not qualify the station's current workspace or a physical route. Reproduce with `python -m trova.integration_m5 orbit-replay --output data/m5/orbit_replay.json`.

The original imported suite now runs **50 tests**, rather than the old report's 47. All 50 passed in the preserved temporary copy after installing matplotlib/Open3D. A transient fake-camera server-startup failure occurred in an earlier concurrent run and passed in the serial rerun. Source data and reports were not changed.

## C — acquisition and orbit adapter

The original multipart parser and reconnect/backfill receiver use Trova's existing PiCam.
No second camera client or controller is added. Pi scan profile changes are explicit;
stationary fresh captures remain the default and streamed moving frames are not silently
accepted as telemetry-synchronized observations. Clock offsets are wall-clock estimates.

`POST /api/scan/replay/mouse` copies exactly the 60 registered names from the retained
COLMAP text model, verifies JPEG decoding and hashes, and records arbitrary SfM poses.
These images are reconstruction inputs, **not** accepted metric robot captures.
Reorientation keeps separate groups. The orbit adapter requires valid active calibration,
measured object size/TCP, telemetry/model agreement and operator model qualification.
View ranking uses camera-angle novelty, measured-plane overlap and joint travel;
newly visible mesh surface area is unknown.

## D — real reconstruction and verified replay

`TROVA_COLMAP` accepts an executable or installation directory; otherwise PATH is searched.
Windows drive paths are translated to `/mnt/<drive>` on Linux/WSL and validated with `-h`.
COLMAP 3 uses legacy Sift flags; 4 uses Feature flags. Sparse GPU use is optional through
`TROVA_COLMAP_USE_GPU=1`; dense PatchMatch CUDA failures remain visible. No Windows
installation path is baked into source. COLMAP is absent on this workstation.

Original feature/matching/SfM/dense algorithms, triangulated ArUco scale, table removal,
clustering, robust dimensions, Poisson, decimation, exports and software rendering are
reused. The unsafe assumed-camera-distance metric fallback was removed.
Jobs distinguish normal reconstruction from explicitly requested retained-data replay.
Cancellation kills COLMAP processes; native Open3D stages finish before cancellation
is acknowledged, and cancelled work is never reported as a completed reconstruction.

Executed `python -m trova.integration_m5 mouse-postprocess --output data/m5/mouse-postprocessing`:
60 retained registered camera poses, 10,246 sparse points, 839,668 retained dense points;
recomputed marker scale 53.4097165 mm/unit, robust object size **96.3×55.6×37.4 mm**;
new mesh bbox extent **104.3349×66.6040×38.6206 mm**. Runtime 10.35 s.
New PLY/STL/GLB and four PNG renders were written under that new output directory.
A separate browser replay produced robust 96.3×55.6×37.2 mm and mesh bbox extent
105.9592×66.2446×38.2922 mm; these are separately recorded runs. Native plane/mesh
processing is not bitwise reproducible.
Historical robust size was 96.3×55.7×37.4 and historical mesh report 106.1×66.2×38.6;
Poisson/RANSAC/version differences change bounding extents. Reported ruler size 98×58×35
was not remeasured. Mesh vertex colours are not UV textures or parametric CAD.

This validates loading the retained 60-image camera model, redetecting markers in the
original JPEGs and actually rerunning dense-cloud postprocessing, not a new COLMAP
feature/SfM/dense run. The latter cannot run until COLMAP is installed.

## E — agent, confirmations and UI

The existing skill registry now includes start_scan, plan_orbit, observe_from_viewpoint,
capture_viewpoint, evaluate_coverage, choose_next_view, reconstruct_object,
inspect_reconstruction, request_object_reorientation and finish_task. Speech and typed
scan goals use this registry. Compound Italian/English selection-plus-model requests
use scene/selection/localization before starting the scan. Grok still supplies no poses.
The old Cartesian parallax helper remains for legacy isolated tests; the application
uses the imported five-axis planner. A scan with missing measured orbit settings now
blocks instead of offering a parallax confirmation.

Each accepted joint proposal has its own expiring, single-use confirmation. The realtime
voice adapter retains the next viewpoint proposal and demands another explicit spoken yes. The receipt
binds calibration/safety/camera configuration, operator orbit settings, geometry, selected
identity, exact joint/TCP target, stationary pose and joints. Revalidation happens after
State.start_arm takes ownership and again in MotionGateway/the existing joint driver.
The current calibrated camera-centre sphere retains the configured radius alongside
CAD head spheres. Joint speed/acceleration and 2-degree path checks remain bounded.
J5 cable reference persists for the whole orientation group, rather than resetting
at each viewpoint. Changing settings, pose, calibration, target, STOP or cancellation
invalidates confirmation. A simulated driver is reported as simulation.

After a confirmed orbit move, the bounded skill loop settles, captures, evaluates
viewpoint coverage, obtains a fresh tracked scene and localization, and proposes the
next view. That next movement requires another confirmation. At the viewpoint budget
(with at least three accepted images) it can finish acquisition and queue reconstruction;
this budget is not a proof of complete surface coverage. Tracking/quality/localization
failures stop continuation. If no useful safe view exists, the reason and human-assistance
need are reported. Turning/flipping creates a new group and requires reselection;
registration between object orientations is not implemented. Parking for human access
requires its own validated operator action; no automatic unvalidated parking is added.

The existing page now has a 3D integration disclosure with historical calibration
compatibility, read-only orbit preview, operator measurements, recorded mouse import,
explicit retained-cloud replay, persisted job history, actual orthographic mesh renders
and downloadable GLB/PLY/STL/cloud files. No mock rendering is presented as reconstruction.
Vertex colours remain distinct from UV textures, and meshes from parametric CAD.

## Verification and practical limits

- Imported original tests: **50 passed** from an isolated copy. The old 47-pass report
  describes an earlier suite, not the currently retained one.
- Trova offline suite after the main integration: **205 passed, 2 skipped** in the final run (125.06 s), including continuation,
  quality and repeated voice-confirmation regressions.
  Tests force simulated arm, telemetry off and hardware unlock empty.
- Existing Chromium feature smoke: PASS. Dedicated M5 Chromium workflow: PASS for
  60-image import, actual Open3D mesh generation, four renders, GLB download, persisted
  job listing and 390 px mobile layout.
- Preserved archive: **754/754 SHA-256 checks passed**, 323,516,988 bytes unchanged.
- Read-only live camera: successful JPEG, status **1600×900, rotate 180, protection normal**,
  no extra client rotation. Read-only xArm: five joints and actual TCP, stationary,
  controller state 2, error 0. Logs are local `data/m5/live-readonly.json`.
  No motor enable, actuator, TCP change, bring-up or physical motion occurred.

Current resolution agrees with the retained calibration profile. Camera serial/mount
continuity, current TCP offset, independent reference measurements and the historical
6.148 px reprojection error still prevent promoting that calibration. The 1.5 px gate
is unchanged. Offline IK success in a broad replay workspace does not qualify today's
workspace. Model/controller FK agreement, measured tool/camera/keep-outs, current
calibration and operator hardware authorization remain required before physical scanning.
No continuous collision proof, complete self-pair set, controller interpolation proof,
true surface coverage estimator, cross-orientation registration or parametric CAD is claimed.
Full COLMAP reconstruction has **not** been rerun here because COLMAP is unavailable.
Open3D replay uses retained sparse/dense inputs and labels this explicitly. The historical
120 mm hover / roughly 8 mm lateral offset remains a recorded physical experiment,
not a movement reproduced by this integration.

Raw datasets/CAD remain local and unmodified. Package dependencies are optional via
`.[reconstruction]`. For sharing the approximately 315 MB evidence use an artifact store
or reviewed Git LFS rules; nothing was uploaded. Windows metadata, caches, archives
and setup credentials are excluded. The compact derived link sphere data retains
UFACTORY attribution; original team camera-head CAD is not redistributed as a raw mesh.

## Reproduce and demonstrate

Offline commands (no robot connection):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m trova.integration_m5 audit
OPENBLAS_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m trova.integration_m5 calibration-replay --output data/m5/calibration-replay-new.json
OPENBLAS_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m trova.integration_m5 orbit-replay --output data/m5/orbit-replay-new.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m trova.integration_m5 mouse-postprocess --output data/m5/mouse-replay-new
PLAYWRIGHT_BROWSERS_PATH=/tmp/trova-playwright OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python tests/browser_m5_smoke.py
```

To rerun the original suite without writing caches/output into the archive, make an
isolated copy (the destination must be a new directory):

```bash
/home/simostocco/hackathon/.venv/bin/python -c "from trova.scan3d.evidence import workspace; from shutil import copytree, ignore_patterns; copytree(workspace(), '/tmp/trova-m5-original-validation', ignore=ignore_patterns('data','runs','__pycache__','*.pyc','*:Zone.Identifier'))"
cd /tmp/trova-m5-original-validation
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/simostocco/hackathon/.venv/bin/python -m pytest -q -o cache_dir=/tmp/trova-m5-original-pytest-cache tests
```

The already generated browser demonstration is retained in `data/m5/ui-demo`. To open
it with the real Pi camera and read-only robot telemetry (use the current DHCP address
if it changes):

```bash
TROVA_DATA="$PWD/data/m5/ui-demo" TROVA_HOST=127.0.0.1 TROVA_PORT=8010 \
TROVA_CAMERA=pi:192.168.5.160 TROVA_CAMERA_ON_ARM=1 TROVA_CAM_EXTRA_ROTATE=0 \
TROVA_ARM=sim TROVA_ARM_READ=xarm:192.168.1.111 \
TROVA_MOTION_MODE=dry_run TROVA_HARDWARE_UNLOCK= \
/home/simostocco/hackathon/.venv/bin/python -m trova
```

Open `http://localhost:8010`, then **Saved reconstruction jobs** to inspect the actual
recorded-data replay. Preview orbit/calibration blockers without authorizing a move.
Use a valid live xAI key for authoritative recognition; mock/offline selections cannot
authorize physical motion. This demo's historical session is reconstruction-only.

A **separate operator-supervised hardware validation**, only after the physical
calibration/reference/model/workspace qualification, uses a separate station data
directory containing that newly validated calibration and measurements. This command
was **not run** during implementation. It opens the existing real driver; do not run
it as a read-only demonstration and do not set TROVA_ARM_TCP_MM to invent/change a TCP.

```bash
TROVA_DATA="$PWD/data/qualified-station" TROVA_HOST=127.0.0.1 TROVA_PORT=8011 \
TROVA_CAMERA=pi:192.168.5.160 TROVA_CAMERA_ON_ARM=1 TROVA_CAM_EXTRA_ROTATE=0 \
TROVA_ARM=xarm:192.168.1.111 TROVA_ARM_READ=off \
TROVA_MOTION_MODE=hardware TROVA_HARDWARE_UNLOCK=MOVE \
/home/simostocco/hackathon/.venv/bin/python -m trova
```

The operator must load their measured workspace/clearances/keep-outs, actual TCP and
qualified model settings; the command does not supply those measurements. Start with
a single proposed view, inspect the trajectory and all blockers, then confirm that
specific proposal with the physical emergency stop accessible. Every subsequent view
has a new proposal. Missing qualifications must refuse; do not relax gates to pass.

## Source changes

New reusable modules: `scan3d/flange.py`, `marker_fit.py`, `evidence.py`, `orbit_math.py`,
`robot_geometry.py`, `orbit_planner.py`, `service.py`, `streaming.py`, `receiver.py`,
`sparse_model.py`, `photogrammetry.py`, `benchmark.py`, `executables.py`,
`reconstruct_pipeline.py`, `skills.py`; curated `robot_model.json`, `mouse_profile.json`,
`provenance.json` and UFACTORY BSD attribution.
Changed primary modules: `geom.py`, `motion.py`, `arm/cartesian.py`, `arm/xarm.py`,
`pi_cam.py`, `acquire.py`, `reconstruction.py`, `agent.py`, `workflow.py`, `server.py`,
`voice.py`, `static/speech.js`, `static/index.html`, `integration_m5.py`,
`pyproject.toml`, `.gitignore`, `AGENTS.md`. Tests: safety fixture, M5 calibration, orbit,
capture, reconstruction, agent and recorded-data browser tests; updated the former
parallax speech regression to expect missing measured joint-orbit settings.

Existing unrelated workspace edits remain outside these integration commits.
