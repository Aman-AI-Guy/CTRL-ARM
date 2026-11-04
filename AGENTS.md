# AGENTS.md: read this first (people and AI agents)

This file is the single source of truth for the state of the repo. **Anyone who changes the code, person or
agent, updates this file in the same commit**: the feature table, the API list if endpoints changed, and one
line in the changelog. If what you read here disagrees with the code, the code wins; fix this file.

## What Trova is

Grok Hackathon Pavia (4-5 Nov 2026), Track A "CTRL+F: SEA" (SEA Vision). You ask for an object (typed, spoken in
English or Italian, or as a phone photo); Trova finds it on the table and a cobot points at it. It also compares the
table with an approved reference and reports what was added, removed or moved (pharma **line clearance**), with an
operator-signed record. Nothing is trained: plain code finds objects, Grok decides what the person means.

- Team plan (roles, schedule, risks, market): https://claude.ai/artifact/Xc5nfd5Q41HmeHHwScSxmb
- Morning guide for the team: `docs/Trova_Morning_Guide.pdf` (regenerate: `python docs/make_guide.py docs/Trova_Morning_Guide.pdf`)
- Concept video: `media/trova_concept.mp4` (regenerate: `python media/make_concept_video.py`)

## Run and test

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,robots,ui]"
cp .env.example .env            # XAI_API_KEY empty = MOCK mode (Grok simulated)
.venv/bin/python -m trova       # http://localhost:8000 ; phones: the printed LAN URL
.venv/bin/pytest -q             # must pass before every commit
.venv/bin/python tests/browser_smoke.py   # after UI changes: clicks through every feature in Chrome (needs .[ui])
```

## Feature status

| Feature | Status | Where | Notes |
|---|---|---|---|
| Object proposals (empty-table difference, lighting normalisation, occlusion) | Done, tested | `proposals.py` | Tune `TROVA_BLOB_THRESH` on the real table |
| Find by words / photo, clarifying question, direct-box fallback | Done, tested (mock + request format) | `finder.py` | Needs live key to verify accuracy |
| What changed (match + 2-of-3 persistence + Grok verify + naive baseline) | Done, tested | `changes.py` | |
| Line-clearance record + sign-off (HTML) | Done | `server.py` `/api/report` | |
| Cobot drivers UR / Elite (sockets), xArm (SDK), safety box, stop-and-hold | Done, tested vs fakes | `arm/`, `server.py`, `safety.py`, `tests/test_motion_safety.py`, `tests/test_safety_review.py` | **The team has the xArm 5 (XF1203, 700 mm reach, 3 kg).** Bring-up remains staff-guided. Every physical boot is disarmed; explicit authenticated rearming is required. xArm construction never enables/recovers the controller, STOP only requests stop state and checks SDK results. `TROVA_ARM_TCP_MM` sets pointer length. Software reset uses the current STOP nonce and never moves or clears a lift interlock. Central authorization guards home/camera/room/watch motion; Cartesian limits + home are required and unavailable FK refuses joint movement. `move=false` never repositions. Only explicit built-in simulators bypass hardware startup/authentication requirements. No real hardware validation yet |
| Calibration (ArUco, clicks, arm-tip clicks) | Done, tested | `calibration.py`, `server.py` | Real arm refuses to move until calibrated with the arm. Arm points (`st.arm_points`) are never mixed with table clicks; duplicate / collinear points and fits worse than 15 mm are rejected (the bad point is dropped); the image size is stored and an uploaded photo or another resolution blocks the arm |
| Grok Voice (realtime, confirm-before-move, stop word) | Done; config/token tested | `voice.py`, `static/voice.js` | Not yet heard live |
| Optional dog room search (sim) | Done, tested | `dog.py` | Go2 driver is a template |
| Jury mode + eval script | Done, tested | `server.py`, `eval/run_eval.py` | |
| 1. Table inventory map ("what's on the table?") | Done, tested offline | `inventory.py`, `scene.py`, `/api/inventory`, `/api/select`, `tests/test_scene.py`, `tests/test_m1_integration.py` | M1 frozen snapshots and selection are motion-free, including camera preparation (409 if explicit Home is needed). Stable track IDs are separate from frame-local box IDs; changing source/upload/survey view retires tracks without reusing IDs. Voice selection context survives polling; snapshot-bound one-use clarification tokens are checked by the server. Snapshot and selection provenance remain separate: mock/offline/unverified decisions never become authoritative. Saving a reference also names its objects (`describe=false` skips) |
| M1 scene snapshot and select without moving | Done, tested (mock, recorded images) | `scene.py`, `POST /api/select`, inventory `snapshot`, voice tool `select_object` | Stable `track_id` across shots of the same view (reading-order `box_id` unchanged). Uncertainty, similar-object clarification, click or EN/IT phrase. Selection never starts the arm. Mock/offline names set `authoritative` false and cannot authorize motion. If the camera pose moves more than 25 mm, image overlap is ignored: appearance alone is `uncertain`, and a plane point within 40 mm can mark the match `geometric` |
| M2 unified motion gateway | Done, tested on simulated backends | `motion.py`, `arm/cartesian.py`, `POST /api/arm/jog`, `tests/test_merge_safety.py` | Every jog, Cartesian move, joint move, point, tour, room scan, and waypoint path passes gateway checks in addition to the existing State authorization. Hardware needs `TROVA_MOTION_MODE=hardware`, `TROVA_HARDWARE_UNLOCK=MOVE`, and an authenticated, rearmed server session; pytest cannot select hardware mode. SDK-wrapper/loopback heuristics cannot exempt a physical server session. Physical Cartesian dispatch requires a live server cancellation ticket; standalone `xarm_voice_execute.py` retains planning/dry-run but cannot bypass startup/STOP/lift recovery to execute. Requested jog millimetres are kept when the gateway rejects them. No collision mesh. A software or voice STOP is not a safety-rated e-stop |
| M3 Pi camera (IMX219 on the arm) | Client done and tested; live reads succeeded, mDNS did not | `pi_cam.py`, `camera.PiSource`, `trova_cam.py`, `trova-stream/` | Sony IMX219 on a Pi Zero 2 W, Wi-Fi port 8080, not the robot Ethernet port and not `/dev/video`. `TROVA_CAMERA=pi`, `pi:host`, or `pi:host:port`. Default host `trova-cam.local` (it did not resolve here). DHCP `192.168.5.160` answered again on 2026-11-04: status `streaming`, protection `normal`, 1640×1232, rotate 180. That address is not the default. `GET /snapshot.jpg` is captured after the request; use it after the arm has stopped. `GET /stream.mjpg` is for display. A failed request drops the previous buffer. The server already rotates 180; `TROVA_CAM_EXTRA_ROTATE` is the only extra rotation. HTTP 503 (thermal pause) is not retried. `trova-stream/install.sh` and the systemd units were not run |
| M3 read-only xArm telemetry | Done; fake SDK tested, one live read on 2026-11-04 | `arm/telemetry.py`, `TROVA_ARM_READ`, `State.robot_telemetry` | Independent of the motion driver. Reads pose, joints, state, and error, and may ask inverse kinematics. It does not call `motion_enable`, `set_state`, `set_position`, `set_tcp_offset`, or `clean_error`. Empty, `off`, and `sim` open nothing. A live read is not replaced by a simulated pose. `sync_precision_claimed` stays false: `skew_s` is only the wall-clock gap. With `TROVA_ARM=sim` the process cannot command the physical arm. A real motion driver is not labeled read-only |
| M3 camera-to-robot geometry | Software done; no real calibration has been solved | `geom.py`, `POST /api/vision/preview`, UI checkerboard row | `T_base_camera = T_base_tool @ T_tool_camera`. The UI shows the live frame, detected checkerboard corners, sample counts, pose diversity, and residuals. Intrinsics use `cv2.calibrateCamera`. Hand-eye uses Park and Martin here, because OpenCV 5.0.0 has the `CALIB_HAND_EYE_*` constants and no `cv2.calibrateHandEye`. A matrix is stored only when the solve passes leave-one-out, board-origin consistency within 10 mm, and pose diversity (15 deg and 30 mm, not one axis). That needs at least 6 telemetry poses so one can be held out. The board stays fixed; the operator moves the arm by hand. A capture is kept only when the pose is still across the picture, the controller error is 0, the decoded size matches the camera, and the corners are found. Reprojection must be under 1.5 px and the table plane within 5 mm. The Pi image is already rotated 180. Simulated observations do not set the valid flag. A table object is a calibrated ray against that plane. One RGB image does not measure height. An observation pose holds roll 180 and pitch 0; yaw comes from telemetry, or 0 is named as a placeholder. `TROVA_OBSERVE_CLEARANCE_MM` is an operator measurement |
| M3 vision skills and closed loop | Software done; the arm has not approached an object | `agent.py`, `POST /api/agent/task`, UI "Run task", voice tool `run_vision_task` | Skills: `inspect_scene`, `list_objects`, `select_object`, `locate_object`, `observe_object`, `plan_approach`, `move_towards_object`, `move_above_object`, `capture_image`, `evaluate_goal`, `request_clarification`, `report_status`. Grok returns a skill name. An argument that contains coordinates is rejected, and the motion skills recompute the target from the ray. The planned start is the telemetry pose, not the simulated tip. The UI shows the target, the straight-line samples, inverse-kinematics status, and the gateway safety line before any move. Controller inverse kinematics is a query. It is required before hardware execution, and it is not claimed until the controller returns a pose inside the xArm 5 joint ranges. Unset `TROVA_CAMERA_BODY_MM` refuses the approach because clearance cannot be established. Hardware execution also refuses an unassessed keep-out (`TROVA_ARM_KEEP_OUT` empty; `none` means the operator checked and there are no boxes). A read-only reader plus a simulated motion driver does not command the physical arm. A picture whose pose changed, or a moving arm, is not localized. Cancel before the first skill runs nothing. Every proposal goes through `MotionGateway`. `collision_avoidance` stays false. An invalid calibration does not invent a metric target. A rejected move locks the task. A centered image is not a reached 3D goal. `verified_tool_position` stays false. The next physical sentence is "Grok, trova l'evidenziatore rosa e posizionati sopra." It has not been executed on the robot |
| M3 multi-view capture | Session format done; reconstruction not run | `acquire.py`, `/api/scan*` | `start`, `capture`, list, `finish`, an orientation mark, and a COLMAP pose export. Each view stores the JPEG, request and record times, resolution, TCP, joints, pose source, controller state and error, camera pose when the view is accepted and the hand-eye exists, calibration version, object id, and viewpoint metadata. A moving robot, a stale request, a bad JPEG, a frame without a size, a missing pose, a non-zero controller error, or a pose whose source is not telemetry is kept and marked rejected. A rejected view does not get a metric camera pose, and rejected views are left out of the COLMAP export. `sync_precision_claimed`, `reconstructed`, `textured_mesh`, and `cad` stay false. Side viewpoints are candidates, not Cartesian commands |
| 2. Line-clearance zones | Done, tested (mock) | `zones.py`, `/api/zones*`, UI "Draw a zone" | Rules: `empty` or `allowed` (list). Evaluated in every change check; a violation makes the line NOT clear. Zones are in image pixels: redraw if the camera moves |
| 3. PDF clearance record | Done, tested | `record_pdf.py`, `GET /api/report.pdf`, UI "Download PDF" | Verdict, images, changes, zones, rejected candidates, sign-off, audit id. Copy saved in `data/records/`. Uses reportlab (now a dependency) |
| 4. Full-screen jury scoreboard | Done, tested | `static/jury.html`, `GET /jury`, `POST /api/metrics/reset` | Found x/N, median find time, false alarms, checks; latest request + verdict; live camera. Reset before the demo. Link "scoreboard" in the UI header |
| 5. Failure log (cause of each wrong find) | Done, tested | `failures.py`, `/api/label`, `/api/judge` (cause), `GET /api/failures`, `run_eval.py failures` | Causes: missed_box (P2), wrong_pick / said_not_found / picked_absent_object / asked_unnecessarily / slow (P3), arm_missed (P1). Stored in `data/cases/failures.jsonl`. Clicking outside every box now records missed_box instead of an error |
| 6. "Point at all the X" | Done, tested (mock) | `finder.find_all`, `POST /api/find_all`, `State.run_tour`, UI "Find all", voice tool `find_all` | Grok returns every matching box; the arm visits each (1 s pause). STOP ends the tour. With move=false, `/api/arm/point_last` starts the tour after a yes |
| 7. Tray lift with the fork (experimental) | Done, tested vs fake robot; OFF by default | `arm/skills.py`, `safety.py`, `POST /api/arm/lift_tray` | 12 straight moves; whole path checked against safety box. A durable atomic in-progress marker precedes the FIRST physical waypoint. Missing/corrupt/ambiguous/incomplete records require authenticated staff recovery on restart; storage failures block dispatch/retain the interlock. Home/reset cannot clear it; `/api/arm/recover_lift` requires the current nonce, fork-freed confirmation and written note, keeps STOP latched and never moves |
| 8. Offline fallback (Wi-Fi or xAI down) | Done, tested | `grok.py` (`GrokUnavailable`, circuit breaker), `finder.py` (mode `offline`), header badge | Network error, timeout or 429/5xx after retries: the call is answered by the local handler and marked offline; for 30 s calls skip the network, then live is retried. 400/401/403/422 still raise. Change verification offline keeps every candidate as real (never a false 'clear'). `TROVA_OFFLINE_FALLBACK=0` disables; eval always disables it. `TROVA_GROK_TIMEOUT_S` (60), connect timeout 5 s |
| 9. Preflight check | Done, tested | `preflight.py`, `GET /api/preflight`, header button "preflight", `python -m trova.preflight [url]` | ok / warn / fail per item with the fix: camera (frozen, resolution), light, focus, empty table, objects / occlusion, calibration (real arm), arm state, safety box, home pose, tray lift, Wi-Fi access, Grok ping (real network attempt, latency), voice token, disk. Run it each morning and before the jury; exit code 1 on any failure |
| 10. Watch the line (live guard) | Done, tested (mock) | `watch.py`, `POST /api/watch` (on, point), `/api/watch/ack`, UI "Watch the line" + red alarm bar with beep, jury page banner, voice tool `watch_line` | Twice a second, plain code compares the live view with the reference (2-of-3 persistence) and checks 'must be empty' zones; an alarm appears in ~1.5 s, Grok explains it in the background and can close a false alarm. Hand or moving arm pauses it. Optional: the arm points at the problem (homes again on acknowledge). An alarm stays until acknowledged; the same situation does not re-alarm; a new reference resets it |
| 11. Labels, lots and batch check | Done, tested (mock) | `labels.py`, `POST /api/batch` (lot), `/api/labels`, `/api/labels/sheet`, UI batch row + "Read labels" + "Print labels", demo "+ old-lot carton", voice tool `read_labels` | QR codes and barcodes decoded locally by OpenCV (second pass per object when several codes are in view); Grok reads printed product / lot / expiry from close-ups ('?' when unreadable). Once a batch is set, every change check also reads labels: a different lot, a code that disagrees with the print, an expired date, an unreadable or missing lot make the line NOT clear. In the HTML and PDF record. Print the label sheet (3 good, 1 old lot, 1 expired, 1 mismatch) for the demo. `TROVA_BATCH` sets a default lot |
| 12. Browser speech fallback + spoken results | Done; parser tested (node) | `static/speech.js`, UI row "Push to talk (browser)" + EN/IT + "speak results" | When Grok Voice is not available (mock, token error, Wi-Fi): the browser's speech recognition (Chrome / Edge / Safari; Chrome sends audio to Google) + a small EN/IT command parser: find, find all, check, labels, inventory, select, jog (a direction plus mm or cm), a vision task when the phrase says to move above an object (`posizionati`, `move above`), watch on/off, save reference, home, and answers to Trova's question. A stop word stops the arm on the interim transcript already. Results are spoken with speechSynthesis (off while Grok Voice runs, which speaks itself) |
| 13. Audit trail page + tamper evidence | Done, tested | `history.py`, `static/history.html`, `GET /history`, `/api/history`, `/api/history.csv`, `/api/history/verify`, header link "history"; hash chain in `audit.py` | One plain line per event with its photo, newest first; filter by group (finds, line clearance, alarms, arm, setup, voice, dog, vision, errors), search, live refresh, CSV export. Vision events are `vision_task`, `vision_capture`, and `scan`. Every event now stores a SHA-256 chained to the previous one: editing, inserting or deleting a line is detected by Verify (older events without a hash are counted, not checked) |
| Camera on the arm (the event's xArm 5 sphere camera) | Done, tested vs simulated arm + camera | `TROVA_CAMERA_ON_ARM=1`, `State.at_survey/clear_view`, `arm/cartesian.py` (`go_joints`, `at_home`), `arm/xarm.py` (`move_joints`, `fk_mm`, joint-planned first move), `camera.ArmCameraSim` (`TROVA_CAMERA=armsim`), `POST /api/arm/teach_pose`, UI "Teach survey / look pose here", `python -m trova.camera --list` | The sphere camera looks sideways from the flange. Home becomes the SURVEY pose, taught by hand as joint angles (data/arm_poses.json or `TROVA_ARM_HOME_JOINTS`); every picture is taken there after `TROVA_CAMERA_SETTLE_S`. Pixel-based moves, watch mode and live-view overlays only run at the survey pose ("moving view" otherwise). Calibrate with ArUco markers recorded by touching them with the arm. Joint targets are checked against the safety box with forward kinematics. A new survey pose deletes the old calibration. Bring-up: `--camera-on-arm` |
| 14. Look around the room | Done, tested vs simulated arm + camera (mock Grok) | `room.py`, `POST /api/room/find` (poll `state.room`), UI "Look around the room", voice tool `look_around`, speech "look around for ..." | From the taught LOOK pose the arm turns one joint (`TROVA_PAN_JOINT`, default 1) over `TROVA_ROOM_PAN` (-90,90,30), one photo per step; Grok picks the photo, then boxes the object; pixels per degree are measured from neighbouring photos (ORB), else `TROVA_CAMERA_HFOV_DEG`; the arm turns to face it. STOP aborts. The Track A brief: Waldo in a real room, without the dog |
| LED ring on the arm camera (optional) | Done, tested vs fake serial; OFF by default | `light.py`, `hardware/led_ring/led_ring.ino`, `TROVA_LED=serial:/dev/tty.usbmodemXXXX` | Only if the ring is driven by a microcontroller you may flash (WS2812 needs one). White light for pictures, green found, amber not found, red flashing alarm, orange STOP. A missing or unplugged ring never stops Trova |

## Architecture (data flow)

```
camera.py (demo, webcam, file, or pi:8080 /snapshot.jpg) ──frame──> proposals.py (numbered boxes) ──> finder.py ──Grok (grok.py)──> box
                                   │                                                  │
                                   └──> changes.py (reference vs now) ──Grok verify──>│
calibration.py: pixel -> robot mm  <──────────────────────────────────────────────────┘
arm/ (ArmDriver: sim | cartesian: ur, elite, xarm | taught_grid) points, stops, parks
motion.py MotionGateway: every move is dry_run, simulation, or operator-unlocked hardware
geom.py: T_base_tool @ T_tool_camera, then a ray against the table plane
agent.py: Grok picks a skill; plain code plans and the gateway decides
acquire.py: scan sessions (images + poses; no mesh)
server.py (FastAPI) + static/index.html (UI) + static/voice.js (Grok Voice) + audit.py (events.jsonl)
```

### Rules that keep it safe and testable
1. **Grok never moves the arm.** Grok returns structured JSON; plain code validates it, converts pixels to mm and
   calls the ArmDriver. Every move is a proposal to `MotionGateway` (`trova/motion.py`): the path is sampled,
   then the workspace, keep-outs, speed, acceleration, and joint limits are applied. Grok cannot set the mode or
   those limits. `dry_run` is the default for a real controller. `hardware` runs only when the operator set
   `TROVA_MOTION_MODE=hardware` and `TROVA_HARDWARE_UNLOCK=MOVE`. A software STOP holds position and is not a
   safety-rated e-stop. There is no collision mesh; a camera sphere is checked only when `TROVA_CAMERA_BODY_MM` is set.
   A vision skill cannot change the mode, the limits, or the unlock. After the gateway rejects an approach,
   that task does not get another proposal by asking for a different distance.
2. **STOP never waits for a model**: UI button, space bar, Escape, and the stop word in the live voice transcript all
   call `/api/arm/stop` directly. Stop = hold position, never torque off. The independent software STOP latch and
   request cancellation epoch are set before driver STOP, without waiting for arm/model locks. Only explicit
   operator `/api/arm/reset` calls `arm.ready()`; it checks quiescence, configuration and operator confirmations,
   performs no controller recovery/motion, and preserves interrupted lifts. Every physical boot is disarmed;
   reset and lift recovery require configured TROVA_OPERATOR_NAME/TROVA_OPERATOR_TOKEN and an
   x-trova-operator-token header. Identity comes from authentication, never a submitted operator name. Both
   confirmations must match the current /api/state.stop_nonce (rotated on boot/STOP/recovery/reset).
   Lift recovery is a separate explicit
   staff decision via `/api/arm/recover_lift`, with STOP latched. Every automatic motion goes through
   `State.start_arm` / `authorize_motion`, including home/camera preparation (one at a time; a running tray lift is
   never cut, 409 `ArmBusy`). `move=false` never initiates camera preparation motion.
   Watch runs retain immutable generations, cancellation tickets, worker identities and their own histories;
   callbacks carry the original alert/run, with authority rechecked on admission and worker dispatch. Replacing
   a watch permanently invalidates its predecessor even if a callback takes longer than the join timeout.
3. **Every Grok call has a mock handler** (`register_mock(name, fn)` next to the call) so the app and tests run
   without a key. New Grok call = new schema + prompt + mock + test.
4. **English and Italian**: prompts answer in the language of the request; reports follow `TROVA_LANG`.
5. **Everything is logged** to `data/events.jsonl` with saved images in `data/runs/` (traceability for the jury).
6. `data/` and `.env` are never committed. On the venue Wi-Fi set `TROVA_TOKEN` (the API then needs `?token=` once per device). Copy the final test set to `eval/data/` before the deadline.
7. **Request models (pydantic) live at module level in `server.py`**, never inside `create_app`: FastAPI cannot
   resolve them under postponed annotations (a test enforces this).
8. **Scene selection is read-only.** Inventory never implicitly homes; unavailable parked views return 409.
   A selection never grants or retargets `point_last` permission (that remains the explicit last-Find action).
   Browser and realtime voice retain their own scene context outside state polling. Clarification answers use
   `/api/select` with the original `snapshot_id` and one-use `selection_context` ID; missing/stale context is refused.
   Starting a moving Find after scene mode requires an explicit new Find action (`new_find=true`), then all
   existing STOP, cancellation, startup, lift and motion authorization guards still apply.
9. **The M2 gateway adds checks; it never replaces State authorization.** Metric jog/approach requests use
   `pixels=false` only because their targets are in robot mm; STOP, home, workspace, lift and cancellation checks
   still apply. Cartesian physical dispatch requires the live session ticket installed by `State.start_arm`.
   Vision tasks retain immutable STOP/view/calibration context, cannot overlap, and recheck it before dispatch.
   STOP cancels the task immediately; cancelling an active approach also latches STOP and holds the arm.
   Task provenance includes every skill/selection response, so a live snapshot cannot hide offline fallback.

## Module map

| File | Owner (role) | What it does |
|---|---|---|
| `trova/proposals.py` | P2 | Empty-table difference in LAB, merge boxes, occlusion check |
| `trova/camera.py`, `trova/pi_cam.py` | P2 | Webcam, files, demo, and the Pi camera (`TROVA_CAMERA=pi`). The Pi server sources live in `trova-stream/` |
| `trova/calibration.py` | P2 | Homography pixel <-> mm, ArUco, click calibration |
| `trova/finder.py` | P3 | Set-of-marks selection prompt + schema, clarification, direct fallback, mocks |
| `trova/scene.py` | P3 | Snapshot track ids, uncertainty, select-without-motion, cross-view association |
| `trova/geom.py` | P2 | Arm-camera intrinsics, hand-eye, table plane, ray localisation. No factory matrix is shipped |
| `trova/motion.py` | P1 | Motion gateway: dry-run, simulation, hardware, trajectory checks, driver tickets |
| `trova/grok.py` | P3 | xAI Responses API client (vision + JSON schema), retries, field dropping, mock mode |
| `trova/changes.py` | P4 | Diff, persistence, Grok verification, naive baseline |
| `trova/arm/` | P1 | `base.py` interface; `cartesian.py` safety + reached check; `ur.py`, `elite.py`, `xarm.py`, `rtde_reader.py`; `sim.py`; `taught_grid.py`; `telemetry.py` read-only xArm pose |
| `trova/agent.py` | P3 | Skill loop. Grok names a skill; geometry and MotionGateway own the motion |
| `trova/acquire.py` | P2 | Multi-view capture sessions and a COLMAP pose export. No reconstruction |
| `trova/voice.py`, `trova/static/voice.js` | P5 | Grok Voice session config, tools, token; browser client. `run_vision_task` sends text only |
| `trova/dog.py`, `trova/dog_go2.py` | P5 | Optional dog route search (sim), Go2 template |
| `trova/server.py`, `trova/static/index.html` | all | API + UI |
| `trova/audit.py` | all | Append-only event log |
| `eval/run_eval.py` | P4 | Measurements against baselines |
| `tests/` | all | Offline tests (`fake_robot.py` = fake UR/Elite controller) |

## HTTP API

`GET /` UI · `GET /api/state` · `GET /api/frame.jpg` · `GET /api/log`
Find: `POST /api/find` (text, photo, move, new_find=false) · `POST /api/clarify` · `POST /api/arm/point_last`. `new_find=true` explicitly leaves scene mode; it does not bypass motion safety guards.
Changes: `POST /api/reference` · `POST /api/changes` · `POST /api/signoff` · `GET /api/report`
Setup: `POST /api/background` · `/api/source` · `/api/source/step` · `/api/upload_scene` · `/api/upload_scene/clear` · `/api/tune`
Calibration: `POST /api/calibration/aruco` · `/click` · `/arm_click` · `/marker_from_arm` · `/reset` · `/markers`
Arm: `POST /api/arm/stop` · `/home` · `/point` · `/jog` (direction, distance_mm only; mode overrides rejected) · `/freedrive` · `/clear_errors` · `/teach` · `/teach_home` · `/compliant`
Software reset: `POST /api/arm/reset` JSON (operator, stop_nonce, area_clear=true, controller_ready=true), no motion; `GET /api/state` includes `stop_latched`, `stop_nonce`, `physical_arm`, `safety_persistence_error`. Lift recovery: `POST /api/arm/recover_lift` same JSON + fork_freed=true + note, requires STOP latched, clears only the lift interlock. Hardware reset/recovery/clear_errors/freedrive/compliant require x-trova-operator-token with configured operator credentials (403 if unconfigured, 401 if invalid); stale nonce 409. Successful recovery/reset rotates the nonce. Software reset never recovers controller/physical stops. Watch acknowledgement dismisses normally and returns `motion_blocked` when its guarded homing is refused.
Jury: `POST /api/judge` · `/api/label`
Voice: `GET /api/voice/config` · `POST /api/voice/session` · `/api/voice/log`
Inventory: `POST /api/inventory` (read-only frozen snapshot, stable track IDs, labelled map; 409 if repositioning is needed) · `POST /api/select` (text or track_id, optional snapshot_id; box_id requires snapshot_id; pending clarification requires original snapshot_id and selection_context ID). Selection returns motion_allowed=false, snapshot_grok_mode, selection_grok_mode and cumulative provenance; a clarification also returns selection_context. `GET /api/state` exposes pending_select and selection_active. `POST /api/reference` takes `describe` (default true).
Zones: `GET /api/zones` · `POST /api/zones/point` · `/api/zones/finish` (name, rule, allowed) · `/api/zones/cancel` · `DELETE /api/zones/{name}`
Record: `GET /api/report.pdf` (PDF of the last check; copy in data/records/)
Jury: `GET /jury` (scoreboard page) · `POST /api/metrics/reset`
Failures: `GET /api/failures` (counts by cause, owner, next fix) · `POST /api/judge` takes `cause` (arm_missed)
Find all: `POST /api/find_all` (text, move, new_find=false) · `/api/arm/point_last` also starts a pending tour
Tray: `POST /api/arm/lift_tray` (place_u, place_v; pick defaults to the last find; 403 unless TROVA_ALLOW_LIFT=1)
Preflight: `GET /api/preflight` (grok=1 pings Grok) · CLI `python -m trova.preflight [url]`
Watch: `POST /api/watch` (on, point) · `/api/watch/ack` (who) · state field `watch`
Labels: `POST /api/batch` (lot) · `/api/labels` · `/api/labels/sheet` · demo `/api/demo/tamper` kind=label
History: `GET /history` · `/api/history` (groups, q, n) · `/api/history.csv` · `/api/history/verify`
Camera on arm: `POST /api/arm/teach_pose` (name=home_joints|look_joints) · Room: `POST /api/room/find` (text; poll state.room)
Dog: `POST /api/dog/search` · `/api/dog/stop` · Demo: `POST /api/demo/tamper` · `/api/demo/hand`
Camera: `GET /api/camera/status` · `GET /api/camera/stream` (Pi MJPEG only; other sources return 404). `/api/state` does not take a Pi snapshot
Vision: `GET /api/vision/calibration` (includes `progress` and `first_move`) · `POST /api/vision/preview` (cols, rows; draws corners, stores nothing) · `POST /api/vision/intrinsics` (cols, rows, square_mm) · `POST /api/vision/intrinsics/solve` · `POST /api/vision/handeye` (plus on_table) · `POST /api/vision/handeye/solve`. These capture and read the pose. They do not move the arm. `/api/state` includes `vision_progress` and `robot` (`process_can_move` is false while the motion driver is simulated)
Agent: `POST /api/agent/task` (text only; a mode or unlock field is rejected; 409 if another task or scene clarification is pending) · `POST /api/agent/cancel` (cancels queued skills; an active approach also receives STOP). Tasks expose cumulative decision provenance. `/api/arm/jog` uses both State and gateway authorization and refuses pending scene clarifications.
Scan: `POST /api/scan/start` (object_id) · `POST /api/scan/capture` · `GET /api/scan` · `GET /api/scan/candidates` · `POST /api/scan/orientation` · `POST /api/scan/finish` · `POST /api/scan/export`

## M3 launch, calibration, and physical qualification

The robot has not approached an object. These commands do not unlock hardware and do not send a motion.
`TROVA_MOCK=1` forces simulated Grok. Leave it unset when the check is live recognition. `TROVA_ARM=sim`
keeps the motion driver off the controller. `TROVA_ARM_READ=xarm:<robot-ip>` reads pose and joints only.
The camera host is DHCP; `trova-cam.local` did not resolve here. `192.168.5.160` answered on 2026-11-04
and must be confirmed before the next boot.

```bash
# 1. Real camera, live Grok, read-only robot. Bind loopback. Do not set TROVA_MOCK. Do not set TROVA_ARM=xarm.
TROVA_CAMERA=pi TROVA_CAM_HOST=192.168.5.160 TROVA_ARM=sim \
  TROVA_ARM_READ=xarm:<robot-ip> TROVA_HOST=127.0.0.1 TROVA_PORT=8765 \
  .venv/bin/python -m trova

# Read-only camera check. Skips with the real error when the host does not answer.
TROVA_CAM_HOST=192.168.5.160 .venv/bin/python -m pytest -q tests/test_live_camera.py

# 2. Snapshot, inventory, selection. A JPEG is not recognition: read grok_mode on the JSON.
curl -s -X POST http://127.0.0.1:8765/api/inventory
curl -s http://127.0.0.1:8765/api/frame.jpg -o /tmp/trova-frame.jpg
curl -s -X POST http://127.0.0.1:8765/api/select -F 'text=oggetto 1'

# 3. Operator calibration captures. Pose the arm yourself first. These do not move it.
# square_mm is the measured square. 25 below is an example, not a measurement of this board.
curl -s -X POST http://127.0.0.1:8765/api/vision/preview -H 'content-type: application/json' \
  -d '{"cols":9,"rows":6}'
curl -s -X POST http://127.0.0.1:8765/api/vision/intrinsics -H 'content-type: application/json' \
  -d '{"cols":9,"rows":6,"square_mm":25}'
curl -s -X POST http://127.0.0.1:8765/api/vision/handeye -H 'content-type: application/json' \
  -d '{"cols":9,"rows":6,"square_mm":25,"on_table":true}'

# 4. Solve and read the report. valid stays false until the real observations pass.
curl -s -X POST http://127.0.0.1:8765/api/vision/intrinsics/solve
curl -s -X POST http://127.0.0.1:8765/api/vision/handeye/solve
curl -s http://127.0.0.1:8765/api/vision/calibration

# 5. Dry-run proposal. hardware_moved must stay false. Read blockers when calibration is missing.
curl -s -X POST http://127.0.0.1:8765/api/agent/task -H 'content-type: application/json' \
  -d '{"text":"Grok, trova l'\''evidenziatore rosa e posizionati sopra."}'

# 6. Multi-view session. A simulated pose is stored and marked rejected. reconstructed stays false.
curl -s -X POST http://127.0.0.1:8765/api/scan/start -H 'content-type: application/json' \
  -d '{"object_id":"pen"}'
curl -s -X POST http://127.0.0.1:8765/api/scan/capture
curl -s http://127.0.0.1:8765/api/scan
curl -s -X POST http://127.0.0.1:8765/api/scan/export
```

Calibration captures only take a picture and read telemetry. Pose the arm yourself between captures. Measure
one checkerboard square and pass that `square_mm`. Do not construct `XArmBackend` for a read-only session:
its constructor applies `TROVA_ARM_TCP_MM` when that variable is set. `TROVA_ARM_READ` never does. If
`TROVA_TOKEN` is set, add `-H "x-trova-token: $TROVA_TOKEN"` to each `/api/` request.

Operator procedure for collecting calibration. `trova/geom.py` `PROCEDURE` is the same list. It does not move the arm.

1. Keep the physical e-stop in reach. Software STOP and voice STOP are not that e-stop.
2. Start the command above. Confirm `/api/state` shows `robot.source` `telemetry`, `movements_enabled` false, `process_can_move` false, and `motion.mode` `simulation`.
3. Confirm the Pi answers `GET /status.json` and `GET /snapshot.jpg`. The server already rotates 180. Do not set `TROVA_CAM_EXTRA_ROTATE` unless a later calibration says the image is still upside down.
4. In the UI, choose the inner-corner counts and press Show checkerboard. Corners are drawn only when OpenCV finds them.
5. Measure one square with a ruler and type that size. Intrinsics: at least three tilted views of the board, then Solve intrinsics. The matrix is kept only when the reprojection is under 1.5 px and every image has the same size.
6. Fix the checkerboard in the robot base. Move the arm by hand to at least six poses. Rotations must not all be about one axis, and the tool must move at least 30 mm and 15 degrees, so one pose can be held out. At each pose, wait until the arm is still, then Capture hand-eye view. Mark "board is flat on the table" for the views that will define the plane. A view is rejected if the pose changes while the picture is taken, the controller error is not 0, or the JPEG size disagrees with the camera status.
7. Solve hand-eye. Read the consistency, the held-out error, and the plane residual. Above 10 mm, 10 mm, and 5 mm respectively, no matrix is stored. A simulated pose does not make the report valid.
8. Compare a localized point with a physical mark on the table when you have one. The software does not invent that mark. The held-out board origin and the plane residual are the checks it can compute from the captures.

Operator-supervised first movement. This was not run. `trova/geom.py` `FIRST_MOVE` is the same list. Do not start it until the calibration report is valid.

1. Keep the physical e-stop in reach. The gateway is not a collision model. `collision_avoidance` stays false, and a path whose clearance cannot be checked is refused.
2. Measure the workspace and set `TROVA_ARM_LIMITS`. Measure the gap from the table plane to the tool and set `TROVA_OBSERVE_CLEARANCE_MM`. Measure the camera and tool body and set `TROVA_CAMERA_BODY_MM`.
3. Write keep-out boxes into `TROVA_ARM_KEEP_OUT`, or set it to `none` only after you have checked the path by hand. Empty means not assessed, and hardware execution is refused.
4. Confirm the controller error code is 0, the arm is still, and the joint and speed caps are the ones you intend. Confirm calibration valid on the real image size.
5. Run the highlighter sentence while `TROVA_ARM` is still `sim` or the motion mode is still `dry_run`. Read the proposed target, the straight-line samples, the inverse-kinematics line, and the safety status. There is no metric proposal until calibration is valid.
6. Only then start a separate process with the real motion driver, `TROVA_MOTION_MODE=hardware`, and `TROVA_HARDWARE_UNLOCK=MOVE`. Motion still goes through MotionGateway. No other script sends the move.
   The server still starts disarmed. Configure operator authentication and a measured home pose, complete any
   required staff lift recovery, and explicitly reset with the current STOP nonce as described above. The hardware
   unlock alone does not authorize a move. The standalone voice executor cannot replace this server session.
7. After the arm stops, capture another picture. Compare the reported tool pose with the proposal. A centered object in the image is not a verified 3D position. If the target is lost, the pose changed after the picture, or the gateway rejects the path, stop. Do not change a distance to get past a rejection.

## Known gaps

- M1 snapshots are frozen observations, not live motion targets. Track association uses image overlap/colour,
  not guaranteed physical identity; unannounced camera displacement is not detected as a source change.
  Retake/calibrate after moving the camera. Scene selection never grants motion permission.
- Every physical restart is disarmed and lift state is durable in data/arm_safety.json, but application persistence
  cannot stop an abandoned controller command after a hard process crash: hardware watchdog/controller-side
  safeguards are separately required. Software STOP is not safety-rated. Use one server process per robot with
  exclusive ownership of its data directory (no multiple Uvicorn workers). Reset/recovery still rely on staff's
  physical checks; target safety boxes do not validate swept volume. Hardware/communication stop latency and
  actual SDK stop/hold behavior require independent review and staff-supervised validation. Controller protective
  stops and physical emergency stops remain separate procedures.
- The motion gateway checks an axis-aligned workspace, optional keep-out boxes, and an optional camera sphere
  (`TROVA_CAMERA_BODY_MM`). That is not a collision model of the robot, the tool, or the room. `collision_avoidance`
  in every result is false. Jog directions are the robot base frame and have not been checked against the operator's left and right.
- A software or voice STOP holds position. It is not a safety-rated emergency stop. The physical e-stop is unchanged.
- Hardware mode is never enabled by the test suite, and M2 has not been run on the physical arm. `python -m trova.arm.bringup`
  remains an operator procedure: it measures the box before `TROVA_ARM_LIMITS` exists, so it caps speed at 30 mm/s and
  still calls the SDK itself. It is not available to Grok. `xarm_voice_preview.py` and the telemetry scripts only read the arm when someone runs them.
- Live Grok Vision ran on 2026-11-04 through the Trova app against a Pi frame. Inventory `grok_mode` was `live` (not mock or offline): it named two black caps, a tall black bottle, a pink highlighter, a white earbud case, and a small white sticker. One local proposal kept an empty name. A find for "the blue pen" returned `not_found` and named the pink highlighter. Grok Voice was not opened; there is no browser on this machine. The speech parser passed with a temporary Node binary that was not installed into the environment.
- Arm drivers are tested against fakes, not real robots or simulators. Elite is the least certain.
- Go2 dog driver is a template.
- Elite: the freedrive script (`while True:` inside `def ... end`) and the `a=`/`v=` keywords of `movel` are unverified:
  run them once on the Elite simulator or with staff. xArm start latency is handled (0.5 s grace) but not measured.
- Voice: `audio.input.transcription` (grok-transcribe + keyterms) is not in the public session docs; if the session
  rejects it, drop that block in `voice.py`. Browser speech fallback in Chrome needs internet (Google recognition).
- Label OCR (feature 11) and watch explanations (feature 10) are tested in mock only; QR decoding is real (OpenCV).
- The Pi camera client is in the app. `trova-cam.local` still did not resolve. `192.168.5.160:8080` answered on 2026-11-04: status `streaming`, protection `normal`, 1640×1232, rotate 180. A later read the same day through `POST /api/vision/preview` returned that size, `rotated_again` false, and no checkerboard. `/api/state` reports the camera online only after a snapshot. That address is DHCP and is not the default. The boot service has not been installed.
- No `data/vision_calib.json` has been solved from the real arm. An empty file is unsolved and contains no matrix. A matrix without the observations that produced it is ignored. On 2026-11-04 `TROVA_ARM_READ=xarm:192.168.1.111` with `TROVA_ARM=sim` returned a 6-value pose and 5 joints, error code 0, warn code 0, state code 2, not moving. The only SDK calls were `get_position`, `get_servo_angle`, `get_state`, `get_err_warn_code`, and `get_is_moving`, then `disconnect`. Those numbers were not stored as a calibration. A preview paired that pose with a 1640×1232 frame; the wall-clock gap was about 0.02 s and sync was not claimed. No checkerboard was in view, so no hand-eye sample was stored. OpenCV 5.0.0 here has no `cv2.calibrateHandEye`. The Park and Martin solve was checked on projected observations. Those checks do not set the calibration valid. It has not been run on a physical board.
- An autonomous approach stays blocked until calibration is valid, `TROVA_OBSERVE_CLEARANCE_MM` and `TROVA_CAMERA_BODY_MM` are measured, workspace limits exist, keep-outs are assessed, the pose source is telemetry, recognition is live, and the operator has set the M2 hardware unlock. There is no metric dry-run target from the live scene until those checkerboard captures exist. The gateway is still not a collision model. Watch mode and preflight call `frame()`, which on the Pi is a fresh snapshot and is slow. On 2026-11-04 the sentence "Grok, trova l'evidenziatore rosa e posizionati sopra." was sent to `/api/agent/task` with the Pi camera, live Grok, and `TROVA_ARM_READ` while `TROVA_ARM` stayed `sim`. Status was `held`, phase `reported`. Skills were `inspect_scene`, `select_object`, `locate_object`, and `report_status`. `hardware_moved`, `motion_proposed`, `verified_tool_position`, and `goal_achieved` were false. `planned_target` was empty. `calibration_valid` was false. `grok_mode` was `live` and `authoritative` was true. `collision_avoidance` was false. The reason was that intrinsics, hand-eye, and the table plane are not solved. The blockers also named `TROVA_OBSERVE_CLEARANCE_MM`, `TROVA_ARM_LIMITS`, `TROVA_CAMERA_BODY_MM`, and that read-only telemetry on a simulated driver will not command the physical arm. No metric proposal was produced and the arm was not moved.
- The arm camera: which way it looks, what the sphere's cable plugs into (USB to the Mac?), and whether the LED ring is
  controllable are read from photos, not confirmed. Check with `python -m trova.camera --list` and `ls /dev/tty.usb*`.
  Room scan pan direction and pixels/degree are measured from the photos, but only in simulation so far.
- Change verification sends at most 12 candidates to Grok; more means "the camera probably moved" (not clear).

## Changelog (newest first)

- 2026-11-04 PR2 merge integration: resolved all seven conflicts by retaining P0/P1 State safety, all six M1 integration fixes, and incoming M2/M3 Pi camera, read-only telemetry, geometry, calibration, vision skills and scan sessions. Inventory shares motion-free snapshot code; strict FK validation and gateway sampling both run. New motion paths use State tickets; agent STOP/cancel, overlapping tasks, source/calibration generations and cumulative model provenance are guarded. Physical Cartesian dispatch additionally requires a server admission ticket, so standalone gateway/CLI execution cannot bypass authenticated recovery. Added 18 merge regressions and extended unchanged M1 safety assertions to incoming voice commands. Offline pytest: 247 passed, 0 failed/skipped, 1 live-camera probe deliberately deselected; browser smoke: all 26 steps passed; Ruff unavailable; diff checks clean. No hardware, camera, paid API, secrets or real configuration changes made during resolution. Merge left uncommitted for review.
- 2026-11-04 M1 integration review fixes: polling-independent voice scene context and server-bound one-use clarifications; explicit new Find transition; inventory never homes; corrected offline property for empty snapshots; source/upload/survey changes retire stable tracks and invalidate snapshots; removed stable-ID/box-ID fallback; separate cumulative snapshot/selection provenance prevents offline authority laundering. External integration probes promoted to 36 permanent synthetic regressions; full offline suite: 172 passed, 0 failed/skipped; Ruff unavailable. Browser smoke passed all 26 steps, including M1 selection/poll/speech and explicit simulator reset/recovery. No hardware, secrets or real configuration changes.
- 2026-11-04 Independent safety review fixes: disarmed hardware boots; fsynced atomic lift marker and conservative recovery of missing/corrupt records; no xArm constructor recovery/enabling; immutable watch generations/tickets/worker identity prevent old-callback authority reuse; mandatory hardware operator authentication and authenticated audit identity; nonce-bound one-use STOP confirmations; checked SDK stop/recovery/state results without implicit restart; unavailable FK/missing limits refuse joint motion. Review probes promoted into repository regressions: 131 offline tests passed (0 failed/skipped); Ruff unavailable. No secrets/hardware settings changed.
- 2026-11-04 P0 motion fixes: independent operator-reset software STOP latch; cancelled request epochs survive reset, concurrent admission, delayed Grok, clarification/confirmation and watch callbacks. Central authorization guards home/camera/room/watch motion, Cartesian limits+home fail closed, move=false never repositions; interrupted lifts require separate staff-confirmed recovery with STOP latched. Offline regression coverage added; no hardware/configuration changes.
- 2026-11-04 Checkerboard preview, stationary hand-eye capture, and an approach proposal use the read-only pose. A matrix is stored only when the validation gates pass. Unset camera-body clearance, an unassessed keep-out, a stale pose, and a read-only reader on a simulated driver all refuse execution. No physical calibration was solved and the arm was not moved.
- 2026-11-04 Read-only xArm telemetry is a separate connection (`TROVA_ARM_READ`). It reads pose, joints, and state, and it refuses actuator calls. A live read on the controller returned error code 0 and was not stored as a calibration.
- 2026-11-04 Scan captures from a simulated pose are rejected and left out of the COLMAP export. An approach without calibration names the missing measurements and does not invent a target. A repeated skill stops the loop. The header separates camera, Grok, calibration, a proposed motion, and an executed movement. The Pi and a live Grok Vision call were exercised; the arm was not moved.
- 2026-11-04 Hand-eye acceptance requires a leave-one-out check. Inconsistent, noisy, or insufficiently diverse observations are not stored, and simulated checks do not mark a calibration valid.
- 2026-11-04 M3 skills and capture sessions: Grok names a skill, the gateway checks every approach, and hardware stays blocked until calibration and the operator unlock are both present. Scan sessions record images and robot poses and do not claim a reconstruction. The arm was not moved.
- 2026-11-04 M3 camera geometry: intrinsics, a Park/Martin hand-eye solve, and a table-plane ray. No calibration file is fabricated. Moving the camera more than 25 mm stops image-space track ids from being trusted. Tool pose reads do not enable the motors.
- 2026-11-04 M3 Pi camera client: `TROVA_CAMERA=pi` reads `/snapshot.jpg` after the request and drops a failed frame. The Pi server sources are kept in `trova-stream/` and were not installed. `trova-cam.local` did not resolve; one read-only check of the previous DHCP address returned a 1640×1232 JPEG.
- 2026-11-04 M2 motion gateway: every arm move is checked before a driver runs. Hardware stays off unless an operator sets the unlock. Spoken distances are kept when a jog is rejected. A software STOP is not a safety-rated e-stop.
- 2026-11-04 M1 scene snapshot: stable track ids, per-object uncertainty, and select by voice or click (`POST /api/select`) without moving the arm. Mock/offline recognition is not authoritative.
- 2026-11-04 Camera on the arm (the team's xArm 5 sphere camera): taught survey pose as joints, pictures only from it, gating of pixel moves / watch / overlays, joint moves checked by forward kinematics, simulated arm camera, camera finder, bring-up --camera-on-arm, preflight items. Feature 14: look around the room and face the object. Optional LED ring driver + firmware.
- 2026-11-04 xArm 5 on site: guided bring-up script (`python -m trova.arm.bringup <ip>`), `TROVA_ARM_TCP_MM` pointer offset, `/api/arm/clear_errors` + UI button; guide's arm-slot steps use the bring-up.
- 2026-11-04 Docs: README and morning guide cover features 8-13, the required safety box and home pose, preflight, TROVA_TOKEN and the browser smoke test; known gaps list what the review could not confirm.
- 2026-11-04 Polish: approving a NOT clear line needs a written reason (and decision must be approve/reject); a spoken answer right after Trova's question is treated as the answer (UI race fixed); spoken results opt-in (on with the first push-to-talk) and deferred, because headless Chrome stalled ~15 s while speaking; `tests/browser_smoke.py` (new `ui` extra) clicks through every feature and fails on JS errors, 5xx, unexpected 4xx or sideways scroll on phones; lint clean (ruff E,F,W,B).
- 2026-11-04 Feature 13: audit trail page with filters, search and CSV; tamper-evident hash chain in events.jsonl with a verify endpoint.
- 2026-11-04 Feature 12: browser speech fallback (push to talk, EN/IT command parser, stop on interim transcript) and spoken results.
- 2026-11-04 Feature 11: label, lot and expiry reading with a wrong-batch check inside line clearance; printable demo label sheet.
- 2026-11-04 Feature 10: watch the line (live guard with alarm, optional arm pointing, Grok explanation in the background).
- 2026-11-04 Feature 9: preflight checklist (endpoint, UI button, CLI).
- 2026-11-04 Review fixes, robustness: Grok drops a request field only when the error names it (an 'image format' error no longer disables the JSON schema) and answers missing required fields raise GrokError (was a KeyError/500); more than 12 change candidates skip Grok (camera probably moved) and stay NOT clear; a webcam frame older than 2 s is refused and the camera reopens itself; camera errors are 503 and /api/state keeps answering; empty uploads 400; a find or check can be judged once; voice releases the mic when the connection drops; optional TROVA_TOKEN locks the API on shared Wi-Fi; pytest no longer writes into eval/results.
- 2026-11-04 Review fixes, arm safety: STOP latches in every driver (was lost between tour targets, lift waypoints and during a find); one motion at a time with 409 when busy; a running lift is never interrupted by a new command, an interrupted lift blocks automatic moves until Arm home; UR/Elite sends serialised and STOP re-sent if it raced a move; xArm late start tolerated; real arms need a safety box and home pose. Calibration: arm points separate from table clicks, degenerate and >15 mm fits rejected, image size stored, uploaded photo blocks the arm, bad saved file no longer blocks start-up. Clarify answers reuse the question's frame and box numbers. Keyboard: Escape = STOP always; space no longer swallowed by the search box.
- 2026-11-04 Feature 8: offline fallback when Wi-Fi or xAI is down (local colour match, OFFLINE badge, 30 s circuit breaker); eval never falls back.
- 2026-11-04 Fix: UI buttons calling `form()` from inline onclick did nothing (the name resolves to `button.form`): helper renamed `sendForm`; test forbids shadowed names in inline handlers. Header wraps on phones; favicon 404 gone; already-shown errors no longer reported as uncaught.
- 2026-11-04 README and morning guide list the extra features; guide mentions /jury and AGENTS.md.
- 2026-11-04 Feature 7: experimental tray lift with the fork (waypoint skill, off by default). Fixed /api/voice/log (request model was inside create_app) and added a test that forbids it.
- 2026-11-04 Feature 6: point at all matching objects (find_all + arm tour), UI button and voice tool.
- 2026-11-04 Feature 5: failure log with automatic cause and owner per wrong find; failures report in run_eval.py; UI 'arm missed' button.
- 2026-11-04 Feature 4: full-screen jury scoreboard at /jury; median find time in state metrics.
- 2026-11-04 Feature 3: PDF line-clearance record; reportlab added to dependencies.
- 2026-11-04 Feature 2: line-clearance zones (empty / allowed items), shown on the live view, in the check result and in the record. Fake-controller test race fixed (wait_for).
- 2026-11-04 Feature 1: table inventory map; reference objects are named and the names reach the change-verification prompt.
- 2026-11-04 AGENTS.md added as the shared state file; `docs/make_guide.py` added.
- 2026-11-04 Initial skeleton pushed (commit a95e832): everything in the feature table marked Done.
