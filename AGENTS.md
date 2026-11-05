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
| Line-clearance record + sign-off (HTML) | Done | `server.py` `/api/report` | Sign-off requires the event id of the reviewed check and cannot replace a previous decision |
| Cobot drivers UR / Elite (sockets), xArm (SDK), safety box, stop-and-hold | Done, tested vs fakes | `arm/`, `server.py`, `safety.py`, `tests/test_motion_safety.py`, `tests/test_safety_review.py` | **The team has the xArm 5 (XF1203, 700 mm reach, 3 kg).** Bring-up remains staff-guided. Every physical boot is disarmed; explicit authenticated rearming is required. xArm construction never enables/recovers the controller, STOP only requests stop state and checks SDK results. `TROVA_ARM_TCP_MM` sets pointer length. Software reset uses the current STOP nonce and never moves or clears a lift interlock. Central authorization guards home/camera/room/watch motion; Cartesian limits + home are required and unavailable FK refuses joint movement. `move=false` never repositions. Only explicit built-in simulators bypass hardware startup/authentication requirements. Bring-up reads the actual fence flag and tests the same checked STOP as the driver; Ctrl+C requests STOP. Move deadlines follow distance/speed; unreadable arrival poses fail closed. xArm STOP bypasses the SDK motion lock and each move uses the current yaw. No real hardware validation yet |
| Calibration (ArUco, clicks, arm-tip clicks) | Done, tested | `calibration.py`, `server.py` | Real arm refuses to move until calibrated with the arm. Arm points (`st.arm_points`) are never mixed with table clicks; robot-frame fits require at least five points and report maximum per-point error; duplicate / collinear points and fits worse than 15 mm are rejected (the bad point is dropped); the image size is stored and an uploaded photo, recorded source, another resolution or survey pose blocks the arm. Camera/calibration changes discard pending targets; normal pending moves expire after 60 s. Corrupt taught poses fail loudly and saves are atomic |
| Grok Voice (realtime, confirm-before-move, stop word) | Done; config/token tested | `voice.py`, `static/voice.js` | Not yet heard live |
| Optional dog room search (sim) | Done, tested | `dog.py` | Go2 driver is a template |
| Jury mode + eval script | Done, tested | `server.py`, `eval/run_eval.py` | |
| 1. Table inventory map ("what's on the table?") | Done, tested (mock) | `inventory.py`, `/api/inventory`, UI button, voice tool `describe_table` | Saving a reference now also names its objects (one extra Grok call; `describe=false` to skip), so changes say which object moved |
| 2. Line-clearance zones | Done, tested (mock) | `zones.py`, `/api/zones*`, UI "Draw a zone" | Rules: `empty` or `allowed` (list). Evaluated in every change check; a violation makes the line NOT clear. Zones are in image pixels: redraw if the camera moves |
| 3. PDF clearance record | Done, tested | `record_pdf.py`, `GET /api/report.pdf`, UI "Download PDF" | Verdict, images, changes, zones, rejected candidates, sign-off, audit id. Copy saved in `data/records/`. Uses reportlab (now a dependency) |
| 4. Full-screen jury scoreboard | Done, tested | `static/jury.html`, `GET /jury`, `POST /api/metrics/reset` | Found x/N, median find time, false alarms, checks; latest request + verdict; live camera. Reset before the demo. Link "scoreboard" in the UI header |
| 5. Failure log (cause of each wrong find) | Done, tested | `failures.py`, `/api/label`, `/api/judge` (cause), `GET /api/failures`, `run_eval.py failures` | Causes: missed_box (P2), wrong_pick / said_not_found / picked_absent_object / asked_unnecessarily / slow (P3), arm_missed (P1). Stored in `data/cases/failures.jsonl`. Clicking outside every box now records missed_box instead of an error |
| 6. "Point at all the X" | Done, tested (mock) | `finder.find_all`, `POST /api/find_all`, `State.run_tour`, UI "Find all", voice tool `find_all` | Grok returns every matching box; the arm visits each (1 s pause). STOP ends the tour. With move=false, `/api/arm/point_last` starts the tour after a yes |
| 7. Tray lift with the fork (experimental) | Done, tested vs fake robot; OFF by default | `arm/skills.py`, `safety.py`, `POST /api/arm/lift_tray` | 12 straight moves; whole path checked against safety box. All waypoints are validated before admitting the request. A durable atomic in-progress marker precedes the FIRST physical waypoint. Missing/corrupt/ambiguous/incomplete records require authenticated staff recovery on restart; storage failures block dispatch/retain the interlock. Home/reset cannot clear it; `/api/arm/recover_lift` requires the current nonce, fork-freed confirmation and written note, keeps STOP latched and never moves |
| 8. Offline fallback (Wi-Fi or xAI down) | Done, tested | `grok.py` (`GrokUnavailable`, circuit breaker), `finder.py` (mode `offline`), header badge | Network error, timeout or 429/5xx after retries: the call is answered by the local handler and marked offline; for 30 s calls skip the network, then live is retried. 400/401/403/422 still raise. Change verification offline keeps every candidate as real (never a false 'clear'). `TROVA_OFFLINE_FALLBACK=0` disables; eval always disables it. `TROVA_GROK_TIMEOUT_S` (60), connect timeout 5 s |
| 9. Preflight check | Done, tested | `preflight.py`, `GET /api/preflight`, header button "preflight", `python -m trova.preflight [url]` | ok / warn / fail per item with the fix: camera (frozen, resolution), light, focus, empty table, objects / occlusion, calibration (real arm), arm state, safety box, home pose, tray lift, Wi-Fi access, Grok ping (real network attempt, latency), voice token, disk. Run it each morning and before the jury; exit code 1 on any failure |
| 10. Watch the line (live guard) | Done, tested (mock) | `watch.py`, `POST /api/watch` (on, point), `/api/watch/ack`, UI "Watch the line" + red alarm bar with beep, jury page banner, voice tool `watch_line` | Twice a second, plain code compares the live view with the reference (2-of-3 persistence) and checks 'must be empty' zones; an alarm appears in ~1.5 s, Grok explains it in the background and can close a false alarm. Hand or moving arm pauses it. Optional: the arm points at the problem (homes again on acknowledge). An alarm stays until acknowledged; the same situation does not re-alarm; a new reference resets it |
| 11. Labels, lots and batch check | Done, tested (mock) | `labels.py`, `POST /api/batch` (lot), `/api/labels`, `/api/labels/sheet`, UI batch row + "Read labels" + "Print labels", demo "+ old-lot carton", voice tool `read_labels` | QR codes and barcodes decoded locally by OpenCV (second pass per object when several codes are in view); Grok reads printed product / lot / expiry from close-ups ('?' when unreadable). Once a batch is set, every change check also reads labels: a different lot, a code that disagrees with the print, an expired date, an unreadable or missing lot make the line NOT clear. In the HTML and PDF record. Print the label sheet (3 good, 1 old lot, 1 expired, 1 mismatch) for the demo. `TROVA_BATCH` sets a default lot |
| 12. Browser speech fallback + spoken results | Done; parser tested (node) | `static/speech.js`, UI row "Push to talk (browser)" + EN/IT + "speak results" | When Grok Voice is not available (mock, token error, Wi-Fi): the browser's speech recognition (Chrome / Edge / Safari; Chrome sends audio to Google) + a small EN/IT command parser: find, find all, check, labels, inventory, watch on/off, save reference, home, and answers to Trova's question. A stop word stops the arm on the interim transcript already. Results are spoken with speechSynthesis (off while Grok Voice runs, which speaks itself) |
| 13. Audit trail page + tamper evidence | Done, tested | `history.py`, `static/history.html`, `GET /history`, `/api/history`, `/api/history.csv`, `/api/history/verify`, header link "history"; hash chain in `audit.py` | One plain line per event with its photo, newest first; filter by group (finds, line clearance, alarms, arm, setup, voice, dog, errors), search, live refresh, CSV export. Every event now stores a SHA-256 chained to the previous one: editing, inserting or deleting a line is detected by Verify (older events without a hash are counted, not checked) |
| Camera on the arm (the event's xArm 5 sphere camera) | Done, tested vs simulated arm + camera | `TROVA_CAMERA_ON_ARM=1`, `State.at_survey/clear_view`, `arm/cartesian.py` (`go_joints`, `at_home`), `arm/xarm.py` (`move_joints`, `fk_mm`, joint-planned first move), `camera.ArmCameraSim` (`TROVA_CAMERA=armsim`), network stream URLs read like a webcam, `POST /api/arm/teach_pose`, UI "Teach survey / look pose here", `python -m trova.camera --list` | The sphere camera looks sideways from the flange. Home becomes the SURVEY pose, taught by hand as joint angles (data/arm_poses.json or `TROVA_ARM_HOME_JOINTS`); every picture is taken there after `TROVA_CAMERA_SETTLE_S`. Pixel-based moves, watch mode and live-view overlays only run at the survey pose ("moving view" otherwise). Calibrate with ArUco markers recorded by touching them with the arm. Joint targets are checked against the safety box with forward kinematics. A new survey pose deletes the old calibration. Bring-up: `--camera-on-arm` |
| 14. Look around the room | Done, tested vs simulated arm + camera (mock Grok) | `room.py`, `POST /api/room/find` (poll `state.room`), UI "Look around the room", voice tool `look_around`, speech "look around for ..." | From the taught LOOK pose the arm turns one joint (`TROVA_PAN_JOINT`, default 1) over `TROVA_ROOM_PAN` (-90,90,30), one photo per step; Grok picks the photo, then boxes the object; pixels per degree are measured from neighbouring photos (ORB), else `TROVA_CAMERA_HFOV_DEG`; the arm turns to face it. STOP aborts. The Track A brief: Waldo in a real room, without the dog |
| First supervised run (find → [level] → hover → descend → measure) | Done, tested vs simulated arm + arm camera (mock Grok); OFF by default; **not yet run on hardware** | `TROVA_FIRST_RUN=1`, `State.first_run_problems/new_approach/run_approach`, `/api/arm/point_last` (stage, confirm, x_mm, y_mm, speed_mm_s), `/api/arm/measured_error`, `CartesianArm.approach`, preflight "First run", UI approach panel, `tests/test_first_run.py`, runbook `docs/first-supervised-run.md` | Find never moves, including camera preparation; explicit Arm home is required before another capture away from survey. From a tilted survey pose stage 0 levels the flange in place (J4 = -(J2+J3), controller FK must confirm, `is_level`/`level_joints`); stage 1 hovers at `TROVA_ARM_Z_MM + TROVA_FIRST_RUN_HOVER_MM` (100), stage 2 descends straight down, each after its own single-use confirmation bound to stage + target (dead after STOP, a new find or 60 s; stage 2 only at that target's hover point). All Cartesian moves ≤ 20 mm/s (env only lowers). Only approach + home may move in this mode. Refuses unless calibrated with the arm at the current survey pose (`Calibration.pose`) and image size, safety box, home, operator credentials, STOP not latched. Arrival logs target + controller pose (`arm_approach`); ruler error → `arm_accuracy`, above `TROVA_ARM_MISS_MM` (10) also `arm_missed`. Homing from pointing height is a joint move that does not rise first |
| LED ring on the arm camera (optional) | Done, tested vs fake serial; OFF by default | `light.py`, `hardware/led_ring/led_ring.ino`, `TROVA_LED=serial:/dev/tty.usbmodemXXXX` | Only if the ring is driven by a microcontroller you may flash (WS2812 needs one). White light for pictures, green found, amber not found, red flashing alarm, orange STOP. A missing or unplugged ring never stops Trova |

## Architecture (data flow)

```
camera.py ──frame──> proposals.py (numbered boxes) ──> finder.py ──Grok (grok.py)──> box
                                   │                                                  │
                                   └──> changes.py (reference vs now) ──Grok verify──>│
calibration.py: pixel -> robot mm  <──────────────────────────────────────────────────┘
arm/ (ArmDriver: sim | cartesian: ur, elite, xarm | taught_grid) points, stops, parks
server.py (FastAPI) + static/index.html (UI) + static/voice.js (Grok Voice) + audit.py (events.jsonl)
```

### Rules that keep it safe and testable
1. **Grok never moves the arm.** Grok returns structured JSON; plain code validates it, converts pixels to mm and
   calls the ArmDriver. Every move passes the safety box (`TROVA_ARM_LIMITS`). Physical taught-grid backends also require forward kinematics for their joint targets.
2. **STOP never waits for a model**: UI button, space bar, Escape, and the stop word in the live voice transcript all
   call `/api/arm/stop` directly. Its dedicated worker capacity remains available when model-bound HTTP workers are full. Stop = hold position, never torque off. The independent software STOP latch and
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
6. `data/` and `.env` are never committed. State-changing browser requests must have the same Origin/Referer as the server; CLI requests without browser headers remain supported. Uploads are capped at 10 MiB. On the venue Wi-Fi set `TROVA_TOKEN` (the API then needs `?token=` once per device). Copy the final test set to `eval/data/` before the deadline.
7. **Request models (pydantic) live at module level in `server.py`**, never inside `create_app`: FastAPI cannot
   resolve them under postponed annotations (a test enforces this).

## Module map

| File | Owner (role) | What it does |
|---|---|---|
| `trova/proposals.py` | P2 | Empty-table difference in LAB, merge boxes, occlusion check |
| `trova/camera.py` | P2 | Webcam / Continuity Camera / video / images / synthetic demo source |
| `trova/calibration.py` | P2 | Homography pixel <-> mm, ArUco, click calibration |
| `trova/finder.py` | P3 | Set-of-marks selection prompt + schema, clarification, direct fallback, mocks |
| `trova/grok.py` | P3 | xAI Responses API client (vision + JSON schema), retries, field dropping, mock mode |
| `trova/changes.py` | P4 | Diff, persistence, Grok verification, naive baseline |
| `trova/arm/` | P1 | `base.py` interface; `cartesian.py` safety + reached check; `ur.py`, `elite.py`, `xarm.py`, `rtde_reader.py`; `sim.py`; `taught_grid.py`; `xarm_config.py` (controller safety settings: read-only by default, `--apply` sets + saves, never moves) |
| `trova/voice.py`, `trova/static/voice.js` | P5 | Grok Voice session config, tools, token; browser client |
| `trova/dog.py`, `trova/dog_go2.py` | P5 | Optional dog route search (sim), Go2 template |
| `trova/server.py`, `trova/static/index.html` | all | API + UI |
| `trova/audit.py` | all | Append-only event log |
| `eval/run_eval.py` | P4 | Measurements against baselines |
| `tests/` | all | Offline tests (`fake_robot.py` = fake UR/Elite controller) |

## HTTP API

`GET /` UI · `GET /api/state` · `GET /api/frame.jpg` · `GET /api/log`
Find: `POST /api/find` (text, photo, move) · `POST /api/clarify` · `POST /api/arm/point_last`
Changes: `POST /api/reference` · `POST /api/changes` · `POST /api/signoff` (check_id, operator, decision, note; stale/repeated decisions 409) · `GET /api/report`
Setup: `POST /api/background` · `/api/source` · `/api/source/step` · `/api/upload_scene` · `/api/upload_scene/clear` · `/api/tune`
Calibration: `POST /api/calibration/aruco` · `/click` · `/arm_click` · `/marker_from_arm` · `/reset` · `/markers`
Arm: `POST /api/arm/stop` · `/home` · `/point` · `/freedrive` · `/clear_errors` · `/teach` · `/teach_home` · `/compliant`
Software reset: `POST /api/arm/reset` JSON (operator, stop_nonce, area_clear=true, controller_ready=true), no motion; `GET /api/state` includes `stop_latched`, `stop_nonce`, `physical_arm`, `safety_persistence_error`. Lift recovery: `POST /api/arm/recover_lift` same JSON + fork_freed=true + note, requires STOP latched, clears only the lift interlock. Hardware reset/recovery/clear_errors/freedrive/compliant require x-trova-operator-token with configured operator credentials (403 if unconfigured, 401 if invalid); stale nonce 409. Successful recovery/reset rotates the nonce. Software reset never recovers controller/physical stops. Watch acknowledgement dismisses normally and returns `motion_blocked` when its guarded homing is refused.
Jury: `POST /api/judge` · `/api/label`
Voice: `GET /api/voice/config` · `POST /api/voice/session` · `/api/voice/log`
Inventory: `POST /api/inventory` (names every object, labelled map image) · `POST /api/reference` takes `describe` (default true)
Zones: `GET /api/zones` · `POST /api/zones/point` · `/api/zones/finish` (name, rule, allowed) · `/api/zones/cancel` · `DELETE /api/zones/{name}`
Record: `GET /api/report.pdf` (PDF of the last check; copy in data/records/)
Jury: `GET /jury` (scoreboard page) · `POST /api/metrics/reset`
Failures: `GET /api/failures` (counts by cause, owner, next fix) · `POST /api/judge` takes `cause` (arm_missed)
Find all: `POST /api/find_all` (text, move) · `/api/arm/point_last` also starts a pending tour
Tray: `POST /api/arm/lift_tray` (place_u, place_v; pick defaults to the last find; 403 unless TROVA_ALLOW_LIFT=1)
Preflight: `GET /api/preflight` (grok=1 pings Grok) · CLI `python -m trova.preflight [url]`
Watch: `POST /api/watch` (on, point) · `/api/watch/ack` (who) · state field `watch`
Labels: `POST /api/batch` (lot) · `/api/labels` · `/api/labels/sheet` · demo `/api/demo/tamper` kind=label
History: `GET /history` · `/api/history` (groups, q, n) · `/api/history.csv` · `/api/history/verify`
First run (`TROVA_FIRST_RUN=1`): `POST /api/arm/point_last` JSON (stage 1|2, confirm, x_mm, y_mm, speed_mm_s?) from `state.approach` · `POST /api/arm/measured_error` (error_mm) · state fields `first_run`, `approach`; `/api/find` returns `approach`
Camera on arm: `POST /api/arm/teach_pose` (name=home_joints|look_joints) · Room: `POST /api/room/find` (text; poll state.room)
Dog: `POST /api/dog/search` · `/api/dog/stop` · Demo: `POST /api/demo/tamper` · `/api/demo/hand`

## Known gaps

- Every physical restart is disarmed and lift state is durable in data/arm_safety.json, but application persistence
  cannot stop an abandoned controller command after a hard process crash: hardware watchdog/controller-side
  safeguards are separately required. Software STOP is not safety-rated. Use one server process per robot with
  exclusive ownership of its data directory (no multiple Uvicorn workers). Reset/recovery still rely on staff's
  physical checks; target safety boxes do not validate swept volume. Hardware/communication stop latency and
  actual SDK stop/hold behavior require independent review and staff-supervised validation. Controller protective
  stops and physical emergency stops remain separate procedures.
- First supervised run: proven only against `cartesian:fake` + `armsim`. Real xArm speed, stop/hold, reached check,
  controller-reported pose and pixel→mm accuracy are unmeasured until the runbook (`docs/first-supervised-run.md`) is
  done. The bring-up's `zmax` (max(home z, Z) + 80) can be below the default hover (Z + 100): the mode then refuses.
  The xArm 5 moves in Cartesian lines only with roll ±180 and pitch 0 (manual p. 156); otherwise
  `XArmBackend.move_linear` uses `motion_type=2`, a curve the safety box does not check. So from a tilted survey pose,
  stage 0 first levels the flange in place: J4 = -(J2+J3), confirmed by controller FK. That formula and its sign
  convention are unverified on the real arm; if FK disagrees, stage 0 is refused. Joint moves (stage 0, Arm home)
  are capped at 5°/s in first-run mode; the controller's reduced mode and safety boundary are the backstop. The
  dome's swing during stage 0 is only end-point checked. The runbook's controller settings were checked
  against the xArm 5 user manual; its reduced-mode numbers are suggestions (the manual gives none).
- No live Grok / Grok Voice call has run yet (keys arrive at the event).
- Arm drivers are tested against fakes, not real robots or simulators. Elite is the least certain.
- Go2 dog driver is a template.
- Elite: the freedrive script (`while True:` inside `def ... end`) and the `a=`/`v=` keywords of `movel` are unverified:
  run them once on the Elite simulator or with staff. xArm start latency is handled (0.5 s grace) but not measured.
- Voice: `audio.input.transcription` (grok-transcribe + keyterms) is not in the public session docs; if the session
  rejects it, drop that block in `voice.py`. Browser speech fallback in Chrome needs internet (Google recognition).
- Label OCR (feature 11) and watch explanations (feature 10) are tested in mock only; QR decoding is real (OpenCV).
- The arm camera is a Raspberry Pi Zero 2 W + IMX219 (host `trova-cam.local`), confirmed on 2026-11-04. Its own
  service `trova-stream` serves MJPEG over Wi-Fi: `TROVA_CAMERA=http://trova-cam.local:8080/stream.mjpg`, 1640x1232,
  ~9-10 fps measured on the Mac, already rotated 180. The cable only powers it. The 16-LED WS2812 ring is driven by
  the Pi's GPIO (rpi_ws281x scripts on the Pi), so Trova's `TROVA_LED` cannot control it. The stream's end-to-end delay
  is not measured: set `TROVA_CAMERA_SETTLE_S` above it. When hot, the Pi drops to 640x480, and the calibration size
  check then blocks the arm.
  Room scan pan direction and pixels/degree are measured from the photos, but only in simulation so far.
- Change verification sends at most 12 candidates to Grok; more means "the camera probably moved" (not clear).

## Changelog (newest first)

- 2026-11-05 Review A HTTP fixes: dedicated STOP capacity, finite numeric requests and targets, bounded frame width/jury counts/uploads, complete label coordinates and byte token comparisons; same-origin browser mutations; sign-off bound to the displayed check id and single-use; clarifying questions cannot be judged as completed finds or saved as absent eval cases. Offline regressions added.

- 2026-11-05 Review A target/calibration fixes: bind pending finds/tours/lifts to view identity and expiry, clear targets on view changes, refuse first-run auto-homing and uploaded ArUco calibration; require five robot-frame points and maximum fit error, bind source/survey pose and atomically save taught poses; enforce grid safety/calibration/FK gates and validate tray paths before the durable marker. Offline regressions added.

- 2026-11-05 Review A controller fixes: correct fence layout; consistent checked bring-up/interrupt STOP; SDK list returns and sensitivity without restart; finite payload/approach validation and TCP recheck before fence; STOP bypasses paused motion lock; current yaw, distance-based move deadlines and fail-closed arrival checks. Offline regression coverage added.

- 2026-11-04 `python -m trova.arm.xarm_config <ip>`: reads the xArm controller's safety settings; with `--apply` sets collision sensitivity (3-5), self-collision, rebound off, reduced-mode speeds, safety boundary from TROVA_ARM_LIMITS (refused if the tool is outside it), optional TCP offset and payload, after a typed YES, using the core sensitivity setting command to avoid the SDK restart side effect, then save_conf and read-back. No motion calls. 150 offline tests pass.
- 2026-11-04 First run stage 0: from a tilted survey pose, the flange is levelled in place (J4 = -(J2+J3), checked by controller FK) before stage 1, so stage 1 is a straight line. Joint moves are capped at 5°/s in that mode. 149 offline tests pass.
- 2026-11-04 Arm camera is a Pi Zero 2 W streaming MJPEG over Wi-Fi: a stream URL (`http://...`) is now read by the live newest-frame reader with the 2 s stale check, not the video-file reader that repeated old frames. Runbook: network, camera-delay measurement, LED ring, Pi heat mode. 146 offline tests pass.
- 2026-11-04 First-run runbook checked against the xArm 5 user manual: mounting, payload, collision sensitivity (≥3), self-collision, rebound, reduced mode (limits joint speed), safety boundary, E-stop sag and recovery, singularity near the base, joint error codes. Documented that stage 1 from a tilted survey pose is joint-planned, not straight. Docs only; 145 tests pass.
- 2026-11-04 First supervised run mode (`TROVA_FIRST_RUN=1`, off by default): two-stage find → hover → descend with single-use, target-bound confirmations through `point_last` + `authorize_motion`, 20 mm/s cap, show-before-moving panel with the collision warning, arrival pose in the audit, ruler error → `arm_accuracy` / `arm_missed`, preflight items, calibration stores its survey pose; runbook `docs/first-supervised-run.md`. 145 offline tests pass; no hardware run.
- 2026-11-04 Independent safety review fixes: disarmed hardware boots; fsynced atomic lift marker and conservative recovery of missing/corrupt records; no xArm constructor recovery/enabling; immutable watch generations/tickets/worker identity prevent old-callback authority reuse; mandatory hardware operator authentication and authenticated audit identity; nonce-bound one-use STOP confirmations; checked SDK stop/recovery/state results without implicit restart; unavailable FK/missing limits refuse joint motion. Review probes promoted into repository regressions: 131 offline tests passed (0 failed/skipped); Ruff unavailable. No secrets/hardware settings changed.
- 2026-11-04 P0 motion fixes: independent operator-reset software STOP latch; cancelled request epochs survive reset, concurrent admission, delayed Grok, clarification/confirmation and watch callbacks. Central authorization guards home/camera/room/watch motion, Cartesian limits+home fail closed, move=false never repositions; interrupted lifts require separate staff-confirmed recovery with STOP latched. Offline regression coverage added; no hardware/configuration changes.
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
