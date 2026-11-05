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
