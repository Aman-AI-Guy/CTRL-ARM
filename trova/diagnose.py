"""End-to-end perception/planning diagnostic with camera and guarded telemetry only.

No arm driver is initialized. No motor, TCP, state-reset or actuator operation exists.
Each independently testable stage reports its own status and minimum corrective action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from .finder import find
from .geom import VisionStore, pose_changed, tool_pose_matrix
from .localization import make_router, stationary
from .motion import MotionGateway
from .proposals import find_blobs
from .scene import TrackState, adopt_found
from .scan3d.flange import flange_pose
from .scan3d.service import OrbitStore

STATUSES = {"PASS", "FAIL", "BLOCKED", "NOT_CONFIGURED", "NOT_VERIFIED"}


def diagnose(
    *,
    camera=None,
    reader=None,
    grok=None,
    data=Path("data"),
    text="mouse rosso",
    pixel=None,
    height_mm=None,
    height_uncertainty_mm=None,
    output=None,
    check_recorded_fk=False,
    recorded_plan=False,
    safety_file=None,
    workspace_candidate=None,
    experimental_reference_model=None,
    manual_region=None,
    gateway=None,
    vision=None,
    orbits=None,
):
    data = Path(data)
    vision = vision or make_router(VisionStore.load(data / "vision_calib.json"), data)
    orbits = orbits or OrbitStore(data / "orbit_settings.json")
    if gateway is None:
        env = dict(os.environ, TROVA_MOTION_MODE="dry_run", TROVA_HARDWARE_UNLOCK="")
        gateway = MotionGateway.standalone(environ=env)
        from .arm import Limits

        gateway.limits = Limits.parse(os.environ.get("TROVA_ARM_LIMITS", ""))
    configured_safety = None
    if safety_file:
        from .motion import apply_operator_safety

        configured_safety = apply_operator_safety(
            gateway, json.loads(Path(safety_file).read_text())
        )
    if gateway.arm is not None or gateway.mode not in {"dry_run", "simulation"}:
        raise ValueError(
            "Diagnostics require a gateway with no attached motion driver and no hardware mode"
        )
    candidate_workspace = None
    if workspace_candidate:
        # Deliberately separate requested bounds from operator-measured safety evidence.
        from .arm import Limits
        from .motion import HARD_SPEED_MM_S, HARD_JOG_MAX_MM
        candidate_workspace = json.loads(Path(workspace_candidate).read_text())
        if gateway.arm is not None or gateway.mode != 'dry_run' or candidate_workspace.get('dry_run_only') is not True:
            raise ValueError('Workspace candidates require a driverless dry-run gateway and dry_run_only: true')
        if set(candidate_workspace)-{'limits','speed_mm_s','jog_max_mm','dry_run_only','measurement_note'}:
            raise ValueError('Workspace candidate contains unsupported safety/authorization fields')
        limits = Limits.parse(candidate_workspace['limits'])
        if limits is None or not np.isfinite([*limits.x,*limits.y,*limits.z]).all() or any(lo>=hi for lo,hi in [limits.x,limits.y,limits.z]):
            raise ValueError('Workspace candidate needs finite increasing bounds')
        speed, jog = float(candidate_workspace['speed_mm_s']),float(candidate_workspace['jog_max_mm'])
        if limits is None or not np.isfinite([speed,jog]).all() or not 0 < speed <= HARD_SPEED_MM_S or not 3 <= jog <= HARD_JOG_MAX_MM:
            raise ValueError('Invalid diagnostic workspace limits/speed/jog')
        gateway.limits, gateway.max_speed_mm_s, gateway.jog_max_mm = limits,speed,jog
    stages = []

    def record(
        name, status, module, required, actual, reason="", category=None, action=""
    ):
        if status not in STATUSES:
            raise ValueError("Unknown diagnostic status")
        stages.append(
            {
                "stage": name,
                "status": status,
                "module": module,
                "required_input": required,
                "actual_input": actual,
                "reason": reason,
                "category": category,
                "minimum_action": action,
                "motion_sent": False,
            }
        )

    if candidate_workspace is not None:
        record('workspace_candidate','NOT_VERIFIED','trova.diagnose.diagnose',
               'independently measured and operator-qualified physical envelope',candidate_workspace,
               'Requested bounds apply only to this driverless dry-run process; they are not physical safety evidence.',
               'physical_measurement','Measure/qualify the envelope before changing the operating workspace.')
    if configured_safety is not None:
        record(
            "operator_safety_configuration",
            "PASS" if configured_safety.get("ok") else "FAIL",
            "trova.motion.apply_operator_safety",
            "existing confirmed measured safety JSON",
            configured_safety,
            configured_safety.get("reason", ""),
            None if configured_safety.get("ok") else "configuration",
            "Correct the measured safety fields; diagnostics never change hardware mode.",
        )
    before = robot = {}
    frame = None
    size = None
    meta = {}
    image_id = None
    profile = {}
    # Robot connectivity and camera connectivity are evaluated independently.
    if reader:
        try:
            before = reader.read()
            configuration = (
                reader.configuration() if hasattr(reader, "configuration") else {}
            )
            robot = before | configuration
            valid = (
                before.get("source") == "telemetry"
                and bool(before.get("joints"))
                and bool(before.get("pose6"))
            )
            record(
                "robot_telemetry",
                "PASS" if valid else "FAIL",
                "trova.arm.telemetry.XArmTelemetry.read",
                "five real joints, TCP pose and controller state",
                robot,
                before.get("error") or "",
                None if valid else "hardware_state",
                "Check the robot Ethernet address and read-only connection."
                if not valid
                else "",
            )
        except Exception as exc:  # noqa: BLE001 - each diagnostic stage must still run
            record(
                "robot_telemetry",
                "FAIL",
                "trova.arm.telemetry.XArmTelemetry.read",
                "read-only telemetry",
                {},
                str(exc),
                "hardware_state",
                "Restore read-only robot connectivity.",
            )
    else:
        record(
            "robot_telemetry",
            "NOT_CONFIGURED",
            "trova.arm.telemetry.open_reader",
            "robot IP",
            None,
            "No robot connection requested.",
            "configuration",
            "Supply --robot-ip; this opens telemetry only.",
        )
    if camera:
        try:
            status = camera.status()
            record(
                "camera_connectivity",
                "PASS",
                "trova.pi_cam.PiCam.status",
                "Pi HTTP status",
                status,
            )
        except Exception as exc:  # noqa: BLE001
            record(
                "camera_connectivity",
                "FAIL",
                "trova.pi_cam.PiCam.status",
                "Pi HTTP status",
                None,
                str(exc),
                "network",
                "Check the current DHCP address; no camera buffer is reused.",
            )
        try:
            cap = camera.capture()
            meta = dict(cap.meta)
            frame = cv2.imdecode(np.frombuffer(cap.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Fresh camera JPEG did not decode")
            size = [frame.shape[1], frame.shape[0]]
            image_id = "sha256:" + hashlib.sha256(cap.jpeg).hexdigest()
            if output:
                image_path = Path(output).with_suffix(".jpg")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(cap.jpeg)
                image_id = str(image_path)
            profile = {
                "kind": meta.get("kind", "pi"),
                "host": getattr(camera, "host", None),
                "size": size,
                "rotate_reported": meta.get("rotate_reported"),
                "extra_rotate": meta.get("extra_rotate", 0),
            }
            record(
                "fresh_image",
                "PASS",
                "trova.pi_cam.PiCam.capture",
                "fresh decoded JPEG",
                {"image": image_id, "image_size": size, "metadata": meta},
            )
        except Exception as exc:  # noqa: BLE001
            record(
                "fresh_image",
                "FAIL",
                "trova.pi_cam.PiCam.capture",
                "fresh decoded JPEG",
                None,
                str(exc),
                "network",
                "Restore the camera stream; do not use an old frame.",
            )
    else:
        record(
            "camera_connectivity",
            "NOT_CONFIGURED",
            "trova.pi_cam.PiCam",
            "camera endpoint",
            None,
            "No camera requested.",
            "configuration",
            "Use --live and --camera-host.",
        )
        record(
            "fresh_image",
            "BLOCKED",
            "trova.pi_cam.PiCam.capture",
            "fresh image",
            None,
            "No camera input.",
            "upstream",
            "Supply a fresh observation.",
        )
    if reader:
        try:
            robot = reader.read() | (
                reader.configuration() if hasattr(reader, "configuration") else {}
            )
        except Exception as exc:  # noqa: BLE001
            robot = {}
            record(
                "telemetry_after_capture",
                "FAIL",
                "trova.arm.telemetry.XArmTelemetry.read",
                "pose after image",
                None,
                str(exc),
                "hardware_state",
                "Restore telemetry.",
            )
    still = False
    try:
        stationary(before)
        stationary(robot)
        still = (
            not pose_changed(before["pose6"], robot["pose6"], mm=1, deg=0.1)
            and max(abs(a - b) for a, b in zip(before["joints"], robot["joints"]))
            <= 0.1
        )
    except (ValueError, KeyError, TypeError):
        pass
    record(
        "stationary_capture",
        "PASS" if still and frame is not None else "BLOCKED",
        "trova.agent._settled",
        "stationary before/after pose and joint agreement",
        {"stationary": still},
        "" if still else "Image and real stationary telemetry are not bound.",
        "hardware_state" if not still else None,
        "Have the operator establish a stationary pose, then recapture."
        if not still
        else "",
    )
    ready = (
        robot.get("state_code") in (0, 2)
        and robot.get("mode") == 0
        and not robot.get("error_code")
        and not robot.get("warn_code")
    )
    record(
        "controller_readiness",
        "PASS"
        if ready
        else ("NOT_VERIFIED" if robot.get("mode") is None else "BLOCKED"),
        "trova.arm.xarm.XArmBackend.read_status",
        "position mode 0, state 0/2, no errors/warnings",
        {k: robot.get(k) for k in ("mode", "state_code", "error_code", "warn_code")},
        ""
        if ready
        else "Controller readiness is not established; state 5 means MODE_CHANGED.",
        "hardware_state" if not ready else None,
        "Operator checks controller mode/state; this command never clears/enables anything."
        if not ready
        else "",
    )
    vision.set_context(profile, robot)
    report = vision.assess(size)
    record(
        "calibration_loading",
        "PASS" if vision.historical or vision.current.intrinsics else "NOT_CONFIGURED",
        "trova.localization.make_router",
        "current or imported calibration artifact",
        {
            "providers": [p.get("provider") for p in report.get("providers", [])],
            "active": report.get("active_provider"),
            "candidate": str(vision.historical.path) if vision.historical else None,
            "sha256": vision.historical.digest if vision.historical else None,
        },
    )
    if vision.historical:
        from .localization import station_binding_errors
        errors = station_binding_errors(vision.historical.station(),vision.historical.digest)
        record('station_binding','NOT_CONFIGURED' if errors else 'PASS',
               'trova.localization.station_binding_errors',vision.historical.station_template(),
               vision.historical.station(),'; '.join(errors),'physical_measurement' if errors else None,
               'GET /api/vision/provider/station, enter independent measured fields, then POST with named operator confirmation; references and qualification remain separate.')
    if candidate_workspace is not None and robot.get('joints') and vision.historical:
        from .scan3d.robot_geometry import RobotGeometry
        from .scan3d.orbit_math import fk_frames
        geometry = RobotGeometry.load()
        cal = vision.historical.cal
        world = geometry.world(fk_frames(robot['joints']),geometry.head_turn(cal.T_fc[:3,2]))
        extents = {name:{'min_mm':(centers-radii[:,None]).min(0).tolist(),
                         'max_mm':(centers+radii[:,None]).max(0).tolist()}
                   for name,(centers,radii) in world.items()}
        fits = {name: all(gateway.limits.check(*point) is None for point in [row['min_mm'],row['max_mm']])
                for name,row in extents.items()}
        tcp_fits = bool(robot.get('xyz_mm') and gateway.limits.check(*robot['xyz_mm']) is None)
        record('current_workspace_geometry','NOT_VERIFIED' if all(fits.values()) and tcp_fits else 'BLOCKED',
               'trova.scan3d.robot_geometry.RobotGeometry.world',
               'measured body geometry within qualified operating envelope',
               {'tcp_inside_candidate':tcp_fits,'model_sphere_extents':extents,'model_inside_candidate':fits,
                'geometry_is_independently_qualified':False,'tool_geometry_configured':bool(orbits.settings.tool_spheres_tcp_mm)},
               'TCP inclusion does not establish camera/head/tool or complete robot-body clearance. Base/link envelopes are diagnostic; gateway applies workspace bounds to TCP/head/camera/tool, not the fixed robot base.',
               'physical_measurement','Review body extents and define/measure permitted body regions; do not lower limits just to fit the TCP.')
    compatible = vision.is_compatible(profile, robot)
    record(
        "calibration_compatibility",
        "PASS" if compatible["compatible"] else "BLOCKED",
        "trova.localization.CalibrationRouter.is_compatible",
        "exact optical profile, rotation, real joints",
        profile,
        "; ".join(compatible["reasons"]),
        "calibration_mismatch" if not compatible["compatible"] else None,
        "Use the calibrated optical mode; verify mount and TCP without implicit image scaling."
        if not compatible["compatible"]
        else "",
    )
    record(
        "setup_qualification",
        "PASS" if report["valid"] else "NOT_CONFIGURED",
        "trova.localization.HistoricalProvider.assess",
        "independent physical references and operator-approved qualification",
        report,
        "; ".join(report.get("reasons", [])),
        None if report["valid"] else "physical_measurement",
        "Measure three support points from two diverse poses; bind mount/TCP/base/table and approve the evidence."
        if not report["valid"]
        else "",
    )
    selected = None
    found = None
    if frame is not None and grok is not None:
        try:
            boxes = find_blobs(frame)
            found = find(grok, frame, boxes, text)
            record(
                "grok_vision_connectivity",
                "PASS" if found.recognition_live else "NOT_VERIFIED",
                "trova.grok.GrokClient.json",
                "live vision service response",
                {"mode": found.mode, "recognition_live": found.recognition_live},
                ""
                if found.recognition_live
                else "Mock/offline response; live recognition not verified.",
            )
            record(
                "semantic_recognition",
                "PASS"
                if found.recognition_live and found.recognized
                else ("BLOCKED" if found.recognition_live else "NOT_VERIFIED"),
                "trova.finder.find / trova.grok.GrokClient.json",
                "live semantic response",
                {
                    "label": found.label,
                    "target_recognized": found.recognized,
                    "mode": found.mode,
                    "recognition_live": found.recognition_live,
                    "reason": found.reason,
                },
                ""
                if found.recognition_live
                else "Mock/offline recognition is not authoritative.",
                None if found.recognition_live else "configuration",
                "Use an existing valid API key for live recognition."
                if not found.recognition_live
                else "",
            )
            if found.status == "found":
                tracked, objects = (
                    adopt_found(TrackState(), frame, [], [], found)
                    if found.mode == "direct"
                    else adopt_found(TrackState(), frame, boxes, [], found)
                )
                selected = next(obj for obj in objects if obj["box_id"] == found.box.id)
            record(
                "image_localization",
                "PASS" if selected else "BLOCKED",
                "trova.finder.find / trova.scene.adopt_found",
                "validated image region and track identity",
                selected,
                "" if selected else found.reason,
                "upstream" if not selected else None,
                "Clarify the target or draw its region on /api/scene."
                if not selected
                else "",
            )
        except Exception as exc:  # noqa: BLE001
            record(
                "semantic_recognition",
                "FAIL",
                "trova.finder.find",
                "vision response",
                None,
                str(exc),
                "recognition_or_network",
                "Inspect the model/network error and use manual region selection when needed.",
            )
    else:
        record(
            "semantic_recognition",
            "NOT_CONFIGURED" if grok is None else "BLOCKED",
            "trova.finder.find",
            "image and vision client",
            None,
            "Image or vision client missing.",
            "configuration",
            "Supply a camera image and vision client.",
        )
        record(
            "image_localization",
            "BLOCKED",
            "trova.scene.adopt_found",
            "validated object region",
            None,
            "No grounded recognition.",
            "upstream",
            "Localize an image region.",
        )
    if selected is not None and manual_region is not None:
        from .localization import finite
        region=finite(manual_region,4)
        if not (0<=region[0]<region[2]<=size[0] and 0<=region[1]<region[3]<=size[1]):
            raise ValueError('Manual region must be inside this decoded image')
        selected=dict(selected,box=region.tolist(),localization_provenance='explicit_diagnostic_manual_region')
    if selected is not None and pixel is not None:
        reference = {
            "kind": "known_height_point"
            if height_mm is not None
            else "support_plane_point",
            "pixel": list(pixel),
            "snapshot_id": image_id,
            "provenance": "explicit_diagnostic_input_not_independent_measurement",
            "height_mm": height_mm or 0,
            "height_uncertainty_mm": height_uncertainty_mm or 0,
        }
        selected["geometric_reference"] = reference
    reference = (selected or {}).get("geometric_reference")
    record(
        "geometric_reference",
        "PASS" if reference else "NOT_CONFIGURED",
        "trova.localization.CalibrationRouter.localize_region",
        "operator-marked support point or measured height reference",
        reference,
        ""
        if reference
        else "A bounding-box center is not a support point or object depth.",
        "physical_measurement" if not reference else None,
        "Use --pixel U V; for an elevated visible surface also supply measured height and uncertainty."
        if not reference
        else "",
    )
    location = None
    if selected is not None and still:
        try:
            location = vision.localize_observation(robot, selected, size)
        except (ValueError, KeyError, TypeError) as exc:
            location = {"ok": False, "point_mm": None, "reason": str(exc)}
    record(
        "pixel_to_base",
        "PASS" if location and location.get("ok") else "BLOCKED",
        "trova.localization.CalibrationRouter.localize_observation",
        "qualified provider, region, geometric reference and image-bound joints",
        location,
        ""
        if location and location.get("ok")
        else (location or {}).get("reason", "Missing upstream inputs."),
        "physical_measurement" if not (location or {}).get("ok") else None,
        "Resolve provider/reference blockers; no metric coordinate is fabricated."
        if not (location or {}).get("ok")
        else "",
    )
    record(
        "localization_uncertainty",
        "PASS"
        if location
        and location.get("ok")
        and location.get("uncertainty_mm") is not None
        else "NOT_VERIFIED",
        "trova.localization.CalibrationRouter.get_uncertainty",
        "measured error bound and reference allowance",
        vision.get_uncertainty(location or {}),
        category=None if (location or {}).get("ok") else "upstream",
    )
    task = {
        "selected": selected,
        "location": location,
        "authoritative": bool(found and found.recognition_live),
        "goal": text,
    }
    host = SimpleNamespace(
        vision=vision,
        image_size=size,
        gateway=gateway,
        robot=lambda: robot,
        observe_clearance_mm=float(os.environ["TROVA_OBSERVE_CLEARANCE_MM"])
        if os.getenv("TROVA_OBSERVE_CLEARANCE_MM")
        else None,
        cancel=threading.Event(),
        orbits=orbits,
    )
    safety_inputs = [
        (
            "workspace_limits",
            gateway.limits is not None,
            "trova.motion.MotionGateway._limits_and_body",
            repr(gateway.limits),
            "Measure and configure the operator workspace limits.",
        ),
        (
            "known_obstacles",
            gateway.keepout_assessed,
            "trova.motion.MotionGateway._limits_and_body",
            {"assessed": gateway.keepout_assessed, "boxes": len(gateway.keepouts)},
            "Measure keep-outs or explicitly confirm none after inspecting the entire robot path.",
        ),
        (
            "camera_body_dimensions",
            gateway.camera is not None,
            "trova.scan3d.orbit_planner.JointGeometry.check",
            repr(gateway.camera),
            "Measure the camera housing offset/radius.",
        ),
        (
            "observation_clearance",
            host.observe_clearance_mm is not None,
            "trova.scan3d.approach.propose",
            host.observe_clearance_mm,
            "Supply the independently measured camera observation clearance.",
        ),
        (
            "object_geometry",
            orbits.settings.object_height_mm is not None
            and orbits.settings.object_radius_mm is not None,
            "trova.scan3d.orbit_planner.OrbitSettings.validate",
            {
                "height_mm": orbits.settings.object_height_mm,
                "radius_mm": orbits.settings.object_radius_mm,
            },
            "Measure the object height and enclosing radius relative to the selected reference.",
        ),
        (
            "motion_envelope",
            bool(orbits.settings.motion_qualification),
            "trova.scan3d.orbit_planner.JointGeometry.hardware_blockers",
            orbits.settings.motion_qualification,
            "Qualify robot/tool geometry, contact exclusions, tracking error and bounded joint/object envelope.",
        ),
    ]
    for name, configured, module, actual, action in safety_inputs:
        if name == 'workspace_limits' and candidate_workspace is not None:
            record(name,'NOT_VERIFIED',module,'independently measured physical operating envelope',actual,
                   'Requested Z=100 candidate is configured for dry-run only; physical qualification is absent.',
                   'physical_measurement',action)
            continue
        record(
            name,
            "PASS" if configured else "NOT_CONFIGURED",
            module,
            "operator-measured/qualified setting",
            actual,
            "" if configured else "Required physical input is not configured.",
            None if configured else "physical_measurement",
            "" if configured else action,
        )
    planning = (
        orbits.approach(host, task)
        if location and location.get("ok")
        else {"next": None, "reason": "Metric target is not qualified."}
    )
    next_ = planning.get("next")
    record(
        "five_axis_ik",
        "PASS" if next_ else "BLOCKED",
        "trova.scan3d.approach.propose",
        "metric reference, measured object/head/tool geometry and clearance",
        planning,
        planning["reason"],
        "upstream" if not next_ else None,
        "Supply the reported measurements or choose a reachable qualified viewpoint."
        if not next_
        else "",
    )
    record(
        "path_generation",
        "PASS" if next_ else "BLOCKED",
        "trova.scan3d.approach.propose",
        "current joints and feasible camera goal",
        None if not next_ else next_["joint_path"],
        "" if next_ else "No qualified approach goal.",
        "upstream" if not next_ else None,
    )
    decision = (next_ or {}).get("gateway") or {}
    record(
        "motion_gateway",
        "PASS" if decision.get("ok") else "BLOCKED",
        "trova.motion.MotionGateway.authorize_joints",
        "limits, geometry, keep-outs, envelope and speed/acceleration bounds",
        decision,
        decision.get("reason", planning["reason"]),
        "physical_constraint" if not decision.get("ok") else None,
        "Configure/qualify the exact reported safety inputs; do not widen them to pass."
        if not decision.get("ok")
        else "",
    )
    record(
        "robot_body_clearance",
        "PASS"
        if decision.get("safety", {}).get("trajectory_certificate", {}).get("ok")
        else "NOT_VERIFIED",
        "trova.scan3d.trajectory.IntervalValidator.validate",
        "qualified model, tracking error and bounded segment certificate",
        decision.get("safety", {}),
        "Finite samples alone do not establish controller path or unknown-obstacle safety.",
        "physical_constraint",
        "Complete the operator motion-envelope qualification before actuation.",
    )
    if check_recorded_fk:
        recorded_fk_stage(reader, robot, record)
    if recorded_plan:
        from .scan3d.orbit_planner import recorded_context, plan_orbit

        retained, context = recorded_context()
        replay = plan_orbit(retained["q0"], context, sequential=True)
        record(
            "recorded_orbit_replay",
            "NOT_VERIFIED",
            "trova.scan3d.orbit_planner.plan_orbit",
            "separately labelled recorded data",
            {
                "reachable": replay["reachable"],
                "geometry": replay["geometry"],
                "provenance": "recorded_not_live",
            },
            "Recorded numerical readiness is not a live executable trajectory.",
            "recorded_evidence",
        )
    experimental = None
    if experimental_reference_model and selected and still and reference:
        try:
            experimental=vision.experimental_localize_observation(robot,selected,size,mode=gateway.mode)
            record('experimental_pixel_to_base','NOT_VERIFIED','trova.localization.CalibrationRouter.experimental_localize_observation',
                   'explicit pixel/region and compatible live optical profile/joints',experimental,experimental['reason'],'experimental')
            from .scan3d.orbit_planner import JointGeometry,OrbitSettings,plan_orbit
            from .scan3d.approach import propose
            model_path=Path(experimental_reference_model)
            model=json.loads(model_path.read_text())
            # Recorded object geometry is an explicit scenario assumption, never a current measurement.
            settings=OrbitSettings(object_height_mm=model['height_mm'],object_radius_mm=model['diameter_mm']/2,
                                   tcp_offset_mm=robot.get('tcp_offset'))
            a,b,c=experimental['plane_abc']
            context=JointGeometry(vision.historical.cal.T_fc,{'normal':[-a,-b,1],'point_mm':[0,0,c]},
                                  experimental['point_mm'],settings,gateway,qualified=False)
            planned=propose(context,robot['joints'],model['above_mm'])
            numerical=bool(planned.get('next') or any(c.get('numerical_ik_solution') for c in planned['candidates']))
            record('experimental_numerical_ik','PASS' if numerical else 'BLOCKED','trova.scan3d.orbit_math.solve_one',
                   'five-axis numerical camera pose (separate from path authorization)',
                   {'solutions':[c for c in planned['candidates'] if c.get('numerical_ik_solution')],
                    'candidate_count':len(planned['candidates'])},
                   'Numerical solution does not authorize its rejected trajectory.','experimental')
            experimental['scenario']={'source':str(model_path),'sha256':__import__("hashlib").sha256(model_path.read_bytes()).hexdigest(),
                'current_object_dimensions_verified':False,'settings':context.signature()}
            record('experimental_approach','PASS' if planned.get('next') else 'BLOCKED','trova.scan3d.approach.propose',
                   'V2 estimate and explicitly labelled recorded geometry assumptions',planned,planned['reason'],'experimental')
            if planned.get('next'):
                candidate=planned['next']
                certificate=context.swept_check(robot['joints'],candidate['joints_deg'])
                record('experimental_interval_clearance','PASS' if certificate['ok'] else 'BLOCKED',
                       'trova.scan3d.trajectory.IntervalValidator.validate','complete modeled path inside configured limits',
                       certificate,certificate.get('reason',''),'physical_constraint')
            orbit=plan_orbit(robot['joints'],context,cancel=threading.Event())
            record('experimental_orbit','PASS' if orbit.get('next') else 'BLOCKED','trova.scan3d.orbit_planner.plan_orbit',
                   'actual start joints and same V2 object-centered scenario',orbit,orbit.get('reason',''),'experimental')
            record('simulated_execution','BLOCKED','trova.motion.MotionGateway.execute_joints',
                   'gateway-accepted trajectory and simulated driver',None,
                   'Unqualified geometry is a dry-run proposal only; no acceptance or motion was fabricated. Existing fake-arm execution/feedback/scan regressions exercise the downstream qualified-fixture path.',
                   'physical_constraint')
        except (ValueError,KeyError,TypeError) as exc:
            record('experimental_pipeline','BLOCKED','trova.diagnose.diagnose','compatible pixel/model inputs',None,str(exc),'experimental')
    from .scan3d.executables import discover

    discovery = discover()
    record(
        "reconstruction_backend",
        "PASS" if discovery.get("path") else "NOT_CONFIGURED",
        "trova.scan3d.executables.discover",
        "COLMAP executable; Open3D optional postprocessing",
        discovery,
        discovery.get("reason", ""),
        "dependency" if not discovery.get("path") else None,
        "Use the existing COLMAP integration when a qualified scan dataset is ready.",
    )
    record(
        "closed_loop_feedback",
        "NOT_VERIFIED",
        "trova.agent.skill_capture / skill_evaluate",
        "movement then fresh image and final joints",
        None,
        "No movement is performed by diagnostics.",
        "operator_test",
        "Perform the separate supervised physical qualification after all prerequisites pass.",
    )
    record(
        "hardware_authorization",
        "BLOCKED",
        "trova.motion.MotionGateway.resolve_mode",
        "independent operator hardware authorization",
        {
            "requested_mode": os.getenv("TROVA_MOTION_MODE", "dry_run"),
            "operator_unlock_present": os.getenv("TROVA_HARDWARE_UNLOCK") == "MOVE",
            "diagnostic_mode": gateway.mode,
            "driver_initialized": False,
        },
        "Read-only diagnostics never attach a motion driver or execute proposals.",
        "intentional_read_only",
    )
    result = {
        "stages": stages,
        "planning_ready": bool(next_ and decision.get("ok")),
        "ready_for_physical_execution": False,
        "active_provider": report.get("active_provider"),
        "motion_sent": False,
        "motors_enabled": False,
        "tcp_changed": False,
        "errors_cleared": False,
        "source_image": image_id,
        "selected": selected,
        "location": location,
        "approach": planning,
        "experimental_location": experimental,
        "counts": {
            status: sum(r["status"] == status for r in stages) for status in STATUSES
        },
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def recorded_fk_stage(reader, robot, record):
    from .scan3d.evidence import workspace

    if reader is None or not hasattr(reader, "forward"):
        record(
            "recorded_fk_conventions",
            "NOT_CONFIGURED",
            "trova.arm.telemetry.XArmTelemetry.forward",
            "controller read-only FK queries",
            None,
            "No guarded controller FK reader.",
            "configuration",
            "Supply --robot-ip and --check-recorded-fk.",
        )
        return
    try:
        root = workspace()
        rows = json.loads(
            (root / "arm_pc/runs/20261105-023325_snaps/snapshots.json").read_text()
        )["snapshots"]["rows"]
        offsets = []
        for row in rows:
            reply = reader.forward(row["q"])
            if not reply.get("ok"):
                raise ValueError(reply.get("reason"))
            offsets.append(
                np.linalg.inv(flange_pose(row["q"])) @ tool_pose_matrix(reply["pose6"])
            )
        anchor = offsets[0]
        translation = max(
            float(np.linalg.norm(m[:3, 3] - anchor[:3, 3])) for m in offsets
        )
        angles = [
            float(
                np.degrees(
                    np.arccos(
                        np.clip((np.trace(anchor[:3, :3].T @ m[:3, :3]) - 1) / 2, -1, 1)
                    )
                )
            )
            for m in offsets
        ]
        ok = translation <= 2 and max(angles) <= 2
        record(
            "recorded_fk_conventions",
            "PASS" if ok else "FAIL",
            "trova.scan3d.flange.flange_pose / XArmTelemetry.forward",
            "constant flange-to-controller-TCP transform over recorded joints",
            {
                "poses": len(offsets),
                "translation_spread_mm": translation,
                "rotation_spread_deg": max(angles),
                "controller_inferred_flange_tcp": anchor.tolist(),
                "independent_tool_measurement": False,
            },
            "" if ok else "Original FK and controller TCP conventions disagree.",
            "coordinate_mismatch" if not ok else None,
            "Measure/verify the actual tool and controller offsets; inferred agreement is not physical calibration.",
        )
    except Exception as exc:  # noqa: BLE001
        record(
            "recorded_fk_conventions",
            "FAIL",
            "trova.arm.telemetry.XArmTelemetry.forward",
            "read-only FK on original recorded poses",
            None,
            str(exc),
            "coordinate_or_network",
            "Check SDK query support and the retained pose data.",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera-host", default="trova-cam.local")
    parser.add_argument("--camera-port", type=int, default=8080)
    parser.add_argument("--robot-ip")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Explicit imported calibration artifact, e.g. calib/v2/camera_robot.json",
    )
    parser.add_argument(
        "--safety",
        type=Path,
        help="Existing operator-measured safety JSON; applied only to the driverless dry-run gateway",
    )
    parser.add_argument("--workspace-candidate", type=Path, help="Unmeasured limits for driverless dry-run only")
    parser.add_argument("--experimental-reference-model",type=Path,help="Recorded hover target supplies explicitly hypothetical object dimensions/clearance; never qualifies hardware")
    parser.add_argument("--text", default="mouse rosso")
    parser.add_argument("--pixel", type=float, nargs=2)
    parser.add_argument("--manual-region",type=float,nargs=4,help="Explicit image-space region for the recognized target; no metric qualification")
    parser.add_argument("--height-mm", type=float)
    parser.add_argument("--height-uncertainty-mm", type=float)
    parser.add_argument("--check-recorded-fk", action="store_true")
    parser.add_argument("--recorded-plan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.height_mm is not None and args.height_uncertainty_mm is None:
        parser.error("A measured height requires its measurement uncertainty.")
    from .config import load_settings
    from .grok import GrokClient

    settings = load_settings()
    camera = reader = None
    if args.live:
        from .pi_cam import PiCam
        from .arm.telemetry import open_reader

        camera = PiCam(
            args.camera_host, args.camera_port, timeout=4, retries=0, status_timeout=2
        )
        reader = open_reader("xarm:" + args.robot_ip) if args.robot_ip else None
    grok = GrokClient(
        settings.xai_api_key,
        model=settings.model,
        effort=settings.effort,
        mock=settings.mock,
    )
    try:
        result = diagnose(
            camera=camera,
            reader=reader,
            vision=make_router(
                VisionStore.load(args.data / "vision_calib.json"),
                args.data,
                args.candidate,
            ),
            grok=grok,
            data=args.data,
            text=args.text,
            pixel=args.pixel,
            height_mm=args.height_mm,
            height_uncertainty_mm=args.height_uncertainty_mm,
            check_recorded_fk=args.check_recorded_fk,
            safety_file=args.safety,
            workspace_candidate=args.workspace_candidate,
            experimental_reference_model=args.experimental_reference_model,
            manual_region=args.manual_region,
            recorded_plan=args.recorded_plan,
            output=args.output,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        if reader:
            reader.close()


if __name__ == "__main__":
    main()
