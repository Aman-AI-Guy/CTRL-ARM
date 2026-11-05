# Scanning and reconstruction

The focused voice fix is committed separately. This continuation adds software for further M4 work, not a physical qualification.

## Operator flow

1. Select an object in an inventory snapshot.
2. Say “Scansionalo” or click **Plan next scan view**. The backend observes afresh and retains the selected identity only when association remains reliable. It localizes through real calibration before generating a target. Missing geometry produces the same explicit calibration refusal as an approach.
3. Inspect the candidate target, path, visibility, IK and gateway result. Confirm the exact proposal if it is executable. Hardware mode and the independent operator unlock remain necessary. A dry-run result cannot be confirmed into hardware execution.
4. After execution, a fresh stationary capture is checked and recorded. A failed capture blocks progression and preserves its JPEG and rejection reason. Request the next view to repeat. Every new move needs a new confirmation; this is supervised exploration, not an unattended sweep.
5. If useful supported views are exhausted, the planner asks for human assistance. The current driver holds tool orientation, so it proposes only limited parallax translations and never claims side/underside coverage. It does not automatically park or flip the object.
6. Before manually turning an object, keep the robot stationary at an operator-validated safe position. Click **Object turned / flipped**, or say “L'ho capovolto” / “I flipped it.” Localization and pending proposals are invalidated. Capture the scene and explicitly select the object again. The new track can be associated with the same session but belongs to a separate acquisition orientation. No automatic geometric registration between groups is claimed.
7. Click **Reconstruct dataset** or say “Ricostruiscilo.” A local job snapshots the current orientation group. Its status, failed stage, log and outputs appear in the UI. **Cancel reconstruction** terminates the local subprocess; it sends no robot command.

## Viewpoint validation

Nine candidates derive from the calibrated plane point and operator-provided observation clearance. Translation radius is capped at 30 mm. The planner predicts the selected point in the calibrated image and checks at least 50% measured-plane footprint overlap with an accepted view when one exists. That overlap is not object-surface coverage. Controller IK queries and sampled TCP/camera-sphere workspace/keep-out checks determine eligibility; localization uncertainty cannot widen the workspace. A candidate already captured within 10 mm is not useful. Ranking rewards camera-position novelty and penalizes travel.

No arbitrary tool orientation or new joint-space executor is added. Hardware approaches also check that the driver's commanded orientation matches telemetry and that controller IK accepts sampled positions along the path. The existing operate-driver SDK can now answer read-only telemetry/IK through `GuardedSDK`, using the backend lock. This does not open a new connection, enable motors, clear faults or change the TCP.

## Acquisition quality and provenance

Production scan captures require a decoded JPEG with matching dimensions, finite six-value TCP pose, five joint angles, stationary telemetry, no controller error, timing metadata, valid calibration and a reliable visible selected object. A Laplacian-variance threshold of 20 rejects blurry or textureless images; a clipped-pixel fraction above 98% rejects unusable exposure. These are heuristics, not a measured reconstruction-quality guarantee. Rejected views remain on disk and have no metric camera pose. Wall-clock skew never becomes a synchronization claim.

Scan indexes reload after a server restart. Raw JPEGs, camera poses, calibration version, object/group identity and detection boxes remain associated. COLMAP pose export separates object orientations and preserves nonzero camera distortion.

## Optional COLMAP backend

The job follows the [official COLMAP CLI pipeline](https://colmap.github.io/cli.html): feature extraction, exhaustive matching, sparse mapping, undistortion, dense stereo, fusion and Poisson meshing. It uses calibrated FULL_OPENCV intrinsics and bounding-box masks. The original JPEGs are preserved. Masks exclude much background but are not semantic object segmentation, so reconstructed geometry can include background within a box.

At least three different, calibrated, telemetry-posed captures from one orientation are required. Duplicate photographs, insufficient camera translation, modified JPEG hashes and mixed calibration versions refuse reconstruction. COLMAP then determines whether features actually match. Each command has a bounded timeout, logs to a file, and supports cancellation. Commands use argument arrays, not a shell or model-provided command text.

The installed binary must support the selected commands. Dense stereo may require a compatible GPU/build. If `mesh_texturer` is advertised, the runner attempts it and marks texture present only when OBJ, material and texture artifacts exist; otherwise it reports an untextured PLY mesh. This workstation has no COLMAP executable. Tests use a fixture executable to verify sequencing, arguments, failures, cancellation and artifact handling; no real photogrammetry result is claimed.

Outputs include an accepted-image dataset ZIP with masks, capture metadata and calibration; a job log; a JSON quality report; and a PLY mesh/point cloud only when generated. A textured bundle is available only after successful texturing. Reports include registration fraction, sparse point count, reprojection error, mesh vertex/face counts and explicit limitations. Artifacts and reports survive a restart. Partial registration and high reprojection residuals generate warnings.

The initial COLMAP mapper estimates its own scene scale. Recorded robot poses are retained but are not currently used to align the reconstructed model to the robot frame. Metric scale, closed/watertight surfaces, missing-surface detection, cross-orientation registration, automatic object flipping and parametric CAD are not claimed. The acquisition-session `reconstructed` flag remains false; reconstruction results belong to separate jobs.

## Verification

`tests/test_voice_workflow.py` exercises spoken selection, follow-up reference, planning/refusal and proposal confirmation through JavaScript tools and the actual backend using fake robot hardware. `tests/test_scanning_reconstruction.py` checks deterministic view qualification, capture rejection, orientation invalidation/reselection, grouped exports, subprocess stages, failure/cancellation, artifact downloads and restart recovery. The expanded Chromium smoke test covers visible calibration blockers and confirmation refusal. Physical calibration, localization error, object approaches and real reconstruction quality remain operator tests.
