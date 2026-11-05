# Trova

For the current Pi camera and xArm 5 workstation, follow [START_HERE.md](START_HERE.md). Start the camera/voice preview with `.venv/bin/python -m trova.station --camera-fallback 192.168.5.160`; the preview uses simulated motion. The guide covers operator calibration and physical testing.

**Tell it (type, talk, in English or Italian) or show it with a phone photo what you need. Trova finds it on the table and the robot arm points at it. Then it checks the table against a reference and reports what was added, removed or moved: line clearance you can talk to.**

Concept video (101 s, made from this code): [`media/trova_concept.mp4`](media/trova_concept.mp4). Regenerate with `python media/make_concept_video.py`.

Grok Hackathon at the Museum, Pavia, 4–5 November 2026. Track A, CTRL+F: SEA (SEA Vision).

## What Grok does, and what plain code does

| Step | Who | Why |
|---|---|---|
| Find every object on the table as a numbered box | Plain code (OpenCV, empty-table difference in LAB) | Exact, fast, no training, no GPL detector |
| Understand the request ("qualcosa per scrivere", a phone photo, "the one left of the mug") and pick the box | **Grok 4.7 vision**, structured JSON | Language, function, relations and photo matching are the hard part |
| Ask when two objects fit ("la penna blu o il pennarello nero?") | **Grok** | Never guess |
| Box missed by the proposals | **Grok** direct box (fallback) | Recall |
| Pixel → table millimetres → arm points | Plain code (homography, taught poses) | Must be exact and safe |
| Candidate changes: added / removed / moved, held over 2 of 3 frames | Plain code (matching on position, colour, shape; lighting normalised) | Deterministic |
| Reject false alarms (shadow, glare, hand), name each change, say why it matters | **Grok** | This is where false alarms die |
| Line-clearance record with operator sign-off | Plain code + Grok summary | Draft EU GMP Annex 22: AI assists, a person decides |
| Talk to it: understand, confirm before moving, report | **Grok Voice** (realtime, function calls) | Hands-free at the line, English or Italian |
| "Stop" | Plain code: the browser halts the arm on the word in the live transcript, plus STOP button and space bar | Never waits for a model turn |
| Optional: search other rooms with the robot dog | Fixed route (code) + **Grok vision** at each stop | The one case the arm cannot reach |

## Extra features (all built, see AGENTS.md for status)

| Feature | How to use it |
|---|---|
| What's on the table? (inventory map) | Button next to Find, or ask the voice. Saving a reference also names its objects, so changes say *which* object |
| Line-clearance zones | "Draw a zone", click corners, Finish: rule *must be empty* or *only these items*. Checked in every "what changed" |
| PDF clearance record | "Download PDF" after a check; a copy goes to `data/records/` |
| Jury scoreboard | Open `/jury` on the projector; "Reset score" before the demo |
| Failure log | After "wrong", click the right object (or "not on the table", or "arm missed"): cause and owner are recorded; `python eval/run_eval.py failures` |
| Point at all the X | Type "all the pens" and press "Find all", or ask the voice: the arm visits each match |
| Tray lift with the fork (experimental, off) | `TROVA_ALLOW_LIFT=1` after measuring the tray with staff; after a find, "Lift its tray" and click the destination |
| Offline fallback | Wi-Fi or xAI down: answers come from a local colour matcher, the badge says OFFLINE, live is retried every 30 s |
| Preflight check | "preflight" in the header (or `python -m trova.preflight`): camera, light, focus, empty table, calibration, arm, safety box, Grok ping, voice, disk; each with the fix |
| Watch the line | After clearance press "Watch the line": anything added, removed or moved (or an object in a *must be empty* zone) raises a red alarm with a beep in ~1.5 s; optionally the arm points at it; Grok explains it; Acknowledge |
| Labels, lots and batch | Set the current batch (e.g. A123): every check also reads QR codes and printed lot / expiry and flags old lots, expired items, code vs print mismatches. "Print labels" gives a demo sheet |
| Browser speech fallback | "Push to talk (browser)" when Grok Voice is not available: EN/IT commands, stop word works on the half-finished sentence, answers spoken |
| Look around the room | With the camera on the arm: the arm turns, takes a photo every 30°, Grok picks the one with the object and the arm turns to face it (Waldo in a real room, no dog needed) |
| Audit trail | "history" in the header: every event with its photo, filters, search, CSV; "Verify log integrity" checks the SHA-256 chain of `data/events.jsonl` |

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,robots,ui]"
cp .env.example .env          # add XAI_API_KEY; without it everything runs in MOCK mode
.venv/bin/python -m trova     # prints http://localhost:8000 and the phone URL
.venv/bin/pytest -q           # offline tests, no key or hardware needed
.venv/bin/python tests/browser_smoke.py   # clicks through every feature in Chrome
```

**Mock mode** (no key) simulates Grok with a colour-word matcher so the whole flow, UI and tests work offline. It is plumbing only; never report mock numbers.

The default `TROVA_CAMERA=demo` is a synthetic table with a "tamper" button (add / remove / move), so the change flow can be rehearsed with no camera.

## Using it

0. **Preflight** (header button) every morning and before the jury: fix every ✗.
1. **Setup** (once per arm slot): pick the camera, press **Capture empty table** with the table empty and the arm parked, then **Calibrate (ArUco)** with the 4 printed markers (or **Calibrate by clicks**).
2. **Find**: type or say a request, or tap **Photo of the object** on a phone. Grok answers, the arm points.
3. **What changed**: set the **Batch**, **Save reference** → someone changes the table → **Check now** → **Sign off** (approving a NOT clear line needs a reason) → **Open record** / **Download PDF**. **Watch the line** keeps checking on its own.
4. **Jury mode**: after each find press ✓ or ✗ (on ✗, click the right object or "not on the table"). Every judged find is saved as an eval case in `data/cases/finds.jsonl`.
5. **Voice**: on the laptop (http://localhost:8000, the browser needs it for the mic) press **Talk to Trova** and speak. With "ask before the arm moves" on, it confirms first.
6. **STOP** (top right, Escape, the space bar, or saying "stop" / "fermo") halts the arm and holds position. Software STOP stays latched until an explicit operator reset; another command, Home, alarm acknowledgement or controller error clearance cannot reset it. Staff e-stop stays primary.
7. **Optional dog**: `TROVA_DOG=sim`. When a find says "not on the table", **Send the dog** walks the route in `data/dog_route/` (photos named `01_Room 2 desk.jpg`, ...) and Grok checks each stop.

Every physical robot session starts **disarmed**, including after a clean restart, a previous STOP or a crash.
xArm connection never clears faults or enables the controller. Simulator exemptions
require an explicit `sim`, `cartesian:fake` or `grid:fake` configuration and the matching built-in fake driver.

Hardware recovery requires configured `TROVA_OPERATOR_NAME` and `TROVA_OPERATOR_TOKEN` (supply these through
the deployment environment; never commit credentials). Send `x-trova-operator-token` on reset, lift recovery,
controller error clearance and manual-mode requests. The existing venue token, if configured, is also required.
The authenticated configured identity is logged; a submitted operator name cannot impersonate staff. Missing
operator configuration returns 403; a missing/wrong operator credential returns 401. Explicit simulators retain
offline operation without operator credentials.

Staff must verify the area and separately complete any controller protective-stop / physical emergency-stop
recovery. Read `/api/state.stop_nonce` immediately before each confirmation, then `POST /api/arm/reset` with JSON
`{"operator":"Name","stop_nonce":"<current nonce>","area_clear":true,"controller_ready":true}`. Every boot,
STOP, successful recovery and reset changes the nonce: queued, replayed or stale confirmations return 409.
Reset refuses while a worker/driver stop is finishing, manual teaching mode is on, required Cartesian limits/home
are missing, or controller idle cannot be verified. It performs no motion and no controller recovery. Failed driver
STOP keeps the latch set and blocks reset until communication is restored and STOP succeeds on retry;
`/api/state.stop_error` exposes the failure. Pending finds/clarifications/point confirmations and old watch motion
stay cancelled after reset. Start a new find or watch run explicitly. Each watch worker retains its own generation,
stop event, history, ticket and identity; an old callback cannot borrow a replacement alert's authority.

`data/arm_safety.json` records lift state with fsync + atomic replacement BEFORE the first physical waypoint.
Missing, corrupt, duplicate-key, unknown-version or incomplete records require staff-confirmed lift recovery;
even a valid clear record never rearms a session. Storage failures prevent dispatch or retain the lift interlock.
An interrupted tray lift remains blocked even after reset. With staff, free and secure the fork/load manually,
leave manual teaching mode, and keep STOP latched (press STOP again if already reset). Confirm the recovery with
`POST /api/arm/recover_lift` using the reset JSON with a fresh nonce plus `"fork_freed":true` and a written `"note"`. This only clears
the lift interlock; it keeps STOP latched and never homes. Reset and any later Home are separate operator actions.

`move=false` on Find / Find all never parks or repositions the arm. With a Cartesian arm it uses the current
parked view, or returns HTTP 409 explaining that an explicit Home is needed. Homing and all camera preparation
use the same motion authorization checks as pointing, including the required Cartesian safety box and home pose.

## Hardware hookup

- **Camera**: `TROVA_CAMERA=0|1|...` for a webcam or an iPhone via Continuity Camera (turn off Center Stage / Portrait / Studio Light in macOS Control Centre → Video Effects). Mount it rigid and top-down. A video file or a folder of photos also works (`TROVA_CAMERA=station.mp4`).
- **Arm** (the event has UR3e, UR5e, UFactory xArm5, Elite CS63, CS66): `TROVA_ARM=ur:<ip>` | `xarm:<ip>` | `elite:<ip>`. UR and Elite use plain sockets (URScript / Elite script + RTDE/RTSI pose); xArm uses `pip install xarm-python-sdk`. On UR/Elite enable **Remote Control** on the pendant; on every arm set the **TCP offset** of the printed tool. Set `TROVA_ARM_Z_MM`, `TROVA_ARM_LIMITS` and `TROVA_ARM_HOME_MM` before the first move: without the last two a real arm refuses every move. Then **Freedrive on**, put the tip on the table at 4–6 spread points and click the tip each time (**Calibrate with arm**). A real arm refuses to move until this calibration exists. Rehearse everything with `TROVA_ARM=cartesian:fake`. Any other arm: `grid:` + taught poses ([`backend_template.py`](trova/arm/backend_template.py)).
- **Camera on the arm** (the event's xArm 5 has a 3D-printed sphere with a camera and an LED ring that looks sideways from the flange): `python -m trova.camera --list` to find its index, `TROVA_CAMERA=<index>`, `TROVA_CAMERA_ON_ARM=1`. Teach the **survey pose** by hand (Setup → *Teach survey pose here*: the camera sees the whole table); every picture is then taken from there. Calibrate with 4 ArUco markers touched by the arm (*Record marker from arm*, then *Calibrate (ArUco)*). Teach a **look pose** (camera facing the room) for **Look around the room**. Rehearse with `TROVA_ARM=cartesian:fake TROVA_CAMERA=armsim TROVA_CAMERA_ON_ARM=1`.
- **Venue Wi-Fi**: set `TROVA_TOKEN=<secret>`; open the printed link (`?token=...`) once on each laptop and phone. Without it anyone on the network could move the arm.
- **Markers**: Setup → **Print markers** (ArUco 4x4, ids 0–3). Put their measured centres in `data/markers.json` as `{"0": [x_mm, y_mm], ...}`; the default is the 4 corners of a `TROVA_TABLE_W_MM` × `TROVA_TABLE_H_MM` area.

## Results

Measured at the event on the station's real objects. Run `python eval/run_eval.py finds` and `python eval/run_eval.py changes --synthetic 60` / `--log`; tables land in `eval/results/`.

`data/` is not committed (it fills up with images). Before the deadline, copy the final test set into the repo so the jury has "the data used": `data/cases/finds.jsonl` plus every image it references → `eval/data/`.

| Metric | Trova | Baseline | Notes |
|---|---|---|---|
| Objects found, jury-style requests | _to measure_ | Grok with no candidate boxes; biggest object | `eval/run_eval.py finds` |
| Correct "not found" on absent objects | _to measure_ | same | |
| Median time to find (s) | _to measure_ | | |
| False alarms in "what changed" | _to measure_ | raw pixel difference | lighting drift included |

## Layout

```
trova/
  proposals.py   object boxes from the empty-table difference, occlusion check
  finder.py      Grok set-of-marks selection, clarification, direct-box fallback
  changes.py     reference vs current matching, persistence, Grok verification, naive baseline
  grok.py        xAI Responses API client (vision + JSON schema), retries, mock mode
  calibration.py pixel <-> mm homography, ArUco, click calibration
  arm/           ArmDriver interface; cartesian.py (limits, stop, reached-target check); ur.py, elite.py (sockets),
                 xarm.py (SDK), rtde_reader.py; sim.py; taught_grid.py for any other arm
  voice.py       Grok Voice session: instructions, tools, key terms, ephemeral token;  static/voice.js = browser client
  dog.py         optional dog search (sim route + Grok at each stop); dog_go2.py = Go2 template
  camera.py      webcam / Continuity Camera / video / photos / synthetic demo
  audit.py       append-only event log (data/events.jsonl) + saved images (data/runs/)
  server.py      FastAPI app;  static/index.html = the whole UI
eval/run_eval.py measurements vs baselines
experiments/grok_bbox_test.py   15-minute test: can Grok draw boxes on its own?
tests/           offline tests
AGENTS.md        shared state file: features, rules, API, changelog. Update it in every commit.
```

## Known limitations

- Software STOP and its durable lift record are application safeguards, not safety-rated emergency stops or
  controller protective stops. Every restart is disarmed, but persistence cannot stop a controller executing an
  abandoned command after a hard crash: hardware watchdogs/controller-side safeguards are separately required.
  Run one Trova process per robot with exclusive control of its data directory; do not run multiple server workers.
  Reset relies on staff's physical checks and controller-ready confirmation. Driver/network stop latency and actual
  SDK stop/hold behavior need independent review and staff-supervised hardware validation. Target safety boxes
  do not validate the full swept volume of robot links/tool; joint motion refuses unavailable FK.
- Pale objects on a pale table can be missed by the proposals; Grok's direct-box fallback covers some of them. Tune `TROVA_BLOB_THRESH`.
- Objects stacked or touching can merge into one box.
- Pointing accuracy depends on the taught poses; it is interpolation, not inverse kinematics.
- One fixed camera: objects hidden behind others are not seen.
- Mock mode understands colour words only, and has no voice.
- UR and Elite drivers are tested against a fake controller that speaks the same protocol, not yet on a real robot or simulator. xArm is tested against a stub of its SDK. Run each once with staff before the demo.
- The dog driver for the real Go2 is a template; the sim replays photos of the route.

## Prepared before the event, built at the event

To be transparent with the jury: the software skeleton (everything in this repository at the first commit) was written the night before kick-off and runs on a synthetic table and a simulated arm. At the event we connect the real arm and camera, calibrate on the station, tune with real Grok calls on SEA Vision's objects, record the evaluation and add voice. The git history shows which is which.

Licences: OpenCV (Apache-2.0), FastAPI (MIT). We avoided YOLO-World (GPL-3.0) on purpose so a company can reuse this.

### Main operator console

The console v2 from `trova-ui-console-v2` is integrated into `trova/static/` and served at `/` by both `python -m trova` and `python -m trova.station`. No separate frontend server or build is required. The package folder is an import reference; edit the integrated files in `trova/`. Self-hosted fonts are included in installed packages.

For this workstation, use the existing environment with reconstruction dependencies:

```bash
/home/simostocco/hackathon/.venv/bin/python -m trova.station --profile qualify --camera-fallback 192.168.5.160 --robot-ip 192.168.1.111 --port 8011
```

This profile reads the real camera and arm telemetry with a simulated motion driver and `dry_run`. Use a currently reachable camera address. For a local demo, `.venv/bin/python -m trova` uses the configured camera and arm settings.

The Strumenti menu connects object search/selection, calibration, scanning/reconstruction, line clearance, reports, arm controls and diagnostics to the existing backend. Finished reconstruction renders feed the Grok requests panel and voice tools; marking a flip or rotation done requires reselection in a new orientation group. `TROVA_AUTO_REVIEW=0` disables automatic review; manual review remains available. Coverage and cross-orientation mesh registration are not claimed.
