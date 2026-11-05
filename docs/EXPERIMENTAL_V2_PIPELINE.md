# Live V2 experimental pipeline readiness

V2 is the working imported candidate. `CalibrationRouter.experimental_localize_observation` exposes its saved tilted-plane transformation in read-only/dry-run/simulation, without changing provider validity, independent qualification, motion authorization or remembered metric anchors. The normal metric operation stays fail-closed. Agent localization responses now carry an independent `experimental_location` alongside the blocked qualified location; an experimental result never becomes an executable target or confirmation receipt.

The diagnostic uses the same recognition, region, V2 provider, imported camera-axis IK, orbit planner and MotionGateway. It continues experimental numerical stages even when current physical qualification is missing. `--manual-region` is an explicit image-space input, not automatic semantic verification. A pixel must lie inside the selected region. The supplied support/height remains a hypothesis until independently verified. The saved table-plane source and historical p95 are identified separately from current uncertainty, which remains unknown.

## Live run and limitations

The saved run `/tmp/trova-experimental-final.json` and JPEG used fresh 1600×900/180° Pi imagery, live Grok and actual stationary five-joint telemetry. The cup moved between two earlier observations, so the old pixel was correctly refused. The final run explicitly selected the visible green cup at the front left and supplied region [0,550,200,780], support pixel [100,738]. This is a diagnostic target, not a silent switch of a persisted mouse task. Reinspect the current image and change those inputs if the object moves again.

| Stage | Evidence/result |
|---|---|
| Fresh camera and stationary telemetry | Live PASS; no motion driver initialized |
| Grok recognition | Live green paper cup; authoritative semantic response |
| Pixel/region association | Explicit diagnostic image region and support pixel; not independently measured support geometry |
| V2 transformation | Experimental base estimate [-20.9879,-679.8110,-41.0998] mm; saved tilted plane, q approximately zero; current uncertainty unknown |
| Numerical five-axis camera IK | 22 mathematical solutions exposed; never marked executable when gateway rejects |
| Trajectory generation | Joint samples generated with numerical solutions; original jump/cable and workspace checks retained |
| Approach / orbit gateway | Refused: existing 45° joint-jump/cable budget; estimated target also lies beyond Y=-500 workspace |
| Live-target simulated execution | BLOCKED by those same trajectory constraints; a successful live-target simulation is not claimed |
| Simulated execution/feedback/orbit | Existing qualified **synthetic/recorded fixtures** pass approach confirmation/execution, subsequent capture/evaluation and bounded two-view scan. These are not acceptance of the current live path |
| Physical feedback | NOT_VERIFIED; real camera cannot photograph a simulated arm pose that the physical robot never reached |
| Acquisition → reconstruction | Existing `scan3d.skills.coverage` sets scan_ready_for_reconstruction; `agent.run` dispatches reconstruct_object, which finishes the session and calls the existing reconstruction job manager |
| Real reconstruction | COLMAP absent; retained 60-image postprocessing and subprocess fixtures are labelled replay/tests, not a fresh live reconstruction |
| Hardware execution | Always unauthorized in this diagnostic; state 5, incomplete qualification and geometry remain blocking |

The optional reference-model file supplies **recorded** height/diameter/clearance (55/44/120 mm) purely to exercise numerical scenario planning. These are not claimed measurements of the live cup. No physical qualification flag or clearance measurement is invented. Changing the recorded model does not change operating limits or authorization.

## Reproduce safely

```bash
TROVA_ARM=sim TROVA_MOTION_MODE=dry_run TROVA_HARDWARE_UNLOCK= \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/simostocco/hackathon/.venv/bin/python -m trova.diagnose \
  --live --camera-host 192.168.5.160 --robot-ip 192.168.1.111 --data data/m61 \
  --text 'green paper cup at the front left' \
  --manual-region 0 550 200 780 --pixel 100 738 \
  --workspace-candidate config/workspace_candidate.json \
  --experimental-reference-model 'Test 3d live video/arm_pc/runs/20261105-044620_hover/hover_target.json' \
  --output /tmp/trova-experimental-final.json
```

This does not bypass the gateway to force an otherwise rejected simulated approach. Numerical IK feasibility is separate from a safe path. A meaningful subsequent live-target execution requires a reachable observation start and target inside the independently assessed workspace; those cannot be created merely by editing software safety gates.

For a first supervised physical run: independently qualify V2 as described in `M62_QUALIFICATION.md`; identify an in-workspace object/support point; have the robot owner arrange a safe qualified observation pose; measure actual object, camera, tool and surrounding clearance; certify the motion envelope and controller tracking assumptions; verify a physical E-stop and operator supervision; have the owner establish controller readiness without automatic resets; and review/authorize one gateway-accepted proposal through the existing hardware path. No motors, controller state, errors, TCP or operating workspace were changed here.

The software route is shared; current real-world constraints do not permit describing only the hardware unlock as missing. State, calibration, reachable start/target and clearance are independently reported prerequisites.

Final complete workspace regression run: **281 passed** with `/home/simostocco/hackathon/.venv/bin/python -m pytest -q -rs` (bundled Node on PATH, BLAS threads limited, Pi host supplied). Focused experimental/approach/agent/bounded-orbit suite: **14 passed**. This does not turn the rejected live-target path into a successful simulated or physical execution.
