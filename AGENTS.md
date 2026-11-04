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
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,robots]"
cp .env.example .env            # XAI_API_KEY empty = MOCK mode (Grok simulated)
.venv/bin/python -m trova       # http://localhost:8000 ; phones: the printed LAN URL
.venv/bin/pytest -q             # must pass before every commit
```

## Feature status

| Feature | Status | Where | Notes |
|---|---|---|---|
| Object proposals (empty-table difference, lighting normalisation, occlusion) | Done, tested | `proposals.py` | Tune `TROVA_BLOB_THRESH` on the real table |
| Find by words / photo, clarifying question, direct-box fallback | Done, tested (mock + request format) | `finder.py` | Needs live key to verify accuracy |
| What changed (match + 2-of-3 persistence + Grok verify + naive baseline) | Done, tested | `changes.py` | |
| Line-clearance record + sign-off (HTML) | Done | `server.py` `/api/report` | |
| Cobot drivers UR / Elite (sockets), xArm (SDK), safety box, stop-and-hold | Done, tested vs fakes | `arm/` | Not yet run on real hardware. STOP latches (drivers refuse until the server calls `ready()` for a new command); one motion at a time (`State.start_arm` under `arm_lock`); STOP during a find cancels its move; a real arm needs `TROVA_ARM_LIMITS` and `TROVA_ARM_HOME_MM` or it refuses to move |
| Calibration (ArUco, clicks, arm-tip clicks) | Done, tested | `calibration.py`, `server.py` | Real arm refuses to move until calibrated with the arm. Arm points (`st.arm_points`) are never mixed with table clicks; duplicate / collinear points and fits worse than 15 mm are rejected (the bad point is dropped); the image size is stored and an uploaded photo or another resolution blocks the arm |
| Grok Voice (realtime, confirm-before-move, stop word) | Done; config/token tested | `voice.py`, `static/voice.js` | Not yet heard live |
| Optional dog room search (sim) | Done, tested | `dog.py` | Go2 driver is a template |
| Jury mode + eval script | Done, tested | `server.py`, `eval/run_eval.py` | |
| 1. Table inventory map ("what's on the table?") | Done, tested (mock) | `inventory.py`, `/api/inventory`, UI button, voice tool `describe_table` | Saving a reference now also names its objects (one extra Grok call; `describe=false` to skip), so changes say which object moved |
| 2. Line-clearance zones | Done, tested (mock) | `zones.py`, `/api/zones*`, UI "Draw a zone" | Rules: `empty` or `allowed` (list). Evaluated in every change check; a violation makes the line NOT clear. Zones are in image pixels: redraw if the camera moves |
| 3. PDF clearance record | Done, tested | `record_pdf.py`, `GET /api/report.pdf`, UI "Download PDF" | Verdict, images, changes, zones, rejected candidates, sign-off, audit id. Copy saved in `data/records/`. Uses reportlab (now a dependency) |
| 4. Full-screen jury scoreboard | Done, tested | `static/jury.html`, `GET /jury`, `POST /api/metrics/reset` | Found x/N, median find time, false alarms, checks; latest request + verdict; live camera. Reset before the demo. Link "scoreboard" in the UI header |
| 5. Failure log (cause of each wrong find) | Done, tested | `failures.py`, `/api/label`, `/api/judge` (cause), `GET /api/failures`, `run_eval.py failures` | Causes: missed_box (P2), wrong_pick / said_not_found / picked_absent_object / asked_unnecessarily / slow (P3), arm_missed (P1). Stored in `data/cases/failures.jsonl`. Clicking outside every box now records missed_box instead of an error |
| 6. "Point at all the X" | Done, tested (mock) | `finder.find_all`, `POST /api/find_all`, `State.run_tour`, UI "Find all", voice tool `find_all` | Grok returns every matching box; the arm visits each (1 s pause). STOP ends the tour. With move=false, `/api/arm/point_last` starts the tour after a yes |
| 7. Tray lift with the fork (experimental) | Done, tested vs fake robot; OFF by default | `arm/skills.py`, `run_waypoints` (Cartesian + sim), `POST /api/arm/lift_tray`, UI after a find | 12 straight moves: approach, slot height, fork in, lift, carry, set down, fork out, up. Whole path checked against the safety box before moving; STOP aborts. Enable with `TROVA_ALLOW_LIFT=1` only after measuring `TROVA_TRAY_Z_SLOT_MM` and `TROVA_TRAY_APPROACH` on the real tray with staff |
| 8. Offline fallback (Wi-Fi or xAI down) | Done, tested | `grok.py` (`GrokUnavailable`, circuit breaker), `finder.py` (mode `offline`), header badge | Network error, timeout or 429/5xx after retries: the call is answered by the local handler and marked offline; for 30 s calls skip the network, then live is retried. 400/401/403/422 still raise. Change verification offline keeps every candidate as real (never a false 'clear'). `TROVA_OFFLINE_FALLBACK=0` disables; eval always disables it. `TROVA_GROK_TIMEOUT_S` (60), connect timeout 5 s |
| 9. Preflight check | Done, tested | `preflight.py`, `GET /api/preflight`, header button "preflight", `python -m trova.preflight [url]` | ok / warn / fail per item with the fix: camera (frozen, resolution), light, focus, empty table, objects / occlusion, calibration (real arm), arm state, safety box, home pose, tray lift, Wi-Fi access, Grok ping (real network attempt, latency), voice token, disk. Run it each morning and before the jury; exit code 1 on any failure |

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
   calls the ArmDriver. Every move passes the safety box (`TROVA_ARM_LIMITS`).
2. **STOP never waits for a model**: UI button, space bar, Escape, and the stop word in the live voice transcript all
   call `/api/arm/stop` directly. Stop = hold position, never torque off. STOP latches: drivers never clear it
   themselves; only `State.start_arm` / `clear_view` call `arm.ready()` for a new command. Every motion goes through
   `State.start_arm` (one at a time; a running tray lift is never cut, 409 `ArmBusy`).
3. **Every Grok call has a mock handler** (`register_mock(name, fn)` next to the call) so the app and tests run
   without a key. New Grok call = new schema + prompt + mock + test.
4. **English and Italian**: prompts answer in the language of the request; reports follow `TROVA_LANG`.
5. **Everything is logged** to `data/events.jsonl` with saved images in `data/runs/` (traceability for the jury).
6. `data/` and `.env` are never committed. On the venue Wi-Fi set `TROVA_TOKEN` (the API then needs `?token=` once per device). Copy the final test set to `eval/data/` before the deadline.
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
| `trova/arm/` | P1 | `base.py` interface; `cartesian.py` safety + reached check; `ur.py`, `elite.py`, `xarm.py`, `rtde_reader.py`; `sim.py`; `taught_grid.py` |
| `trova/voice.py`, `trova/static/voice.js` | P5 | Grok Voice session config, tools, token; browser client |
| `trova/dog.py`, `trova/dog_go2.py` | P5 | Optional dog route search (sim), Go2 template |
| `trova/server.py`, `trova/static/index.html` | all | API + UI |
| `trova/audit.py` | all | Append-only event log |
| `eval/run_eval.py` | P4 | Measurements against baselines |
| `tests/` | all | Offline tests (`fake_robot.py` = fake UR/Elite controller) |

## HTTP API

`GET /` UI · `GET /api/state` · `GET /api/frame.jpg` · `GET /api/log`
Find: `POST /api/find` (text, photo, move) · `POST /api/clarify` · `POST /api/arm/point_last`
Changes: `POST /api/reference` · `POST /api/changes` · `POST /api/signoff` · `GET /api/report`
Setup: `POST /api/background` · `/api/source` · `/api/source/step` · `/api/upload_scene` · `/api/upload_scene/clear` · `/api/tune`
Calibration: `POST /api/calibration/aruco` · `/click` · `/arm_click` · `/marker_from_arm` · `/reset` · `/markers`
Arm: `POST /api/arm/stop` · `/home` · `/point` · `/freedrive` · `/teach` · `/teach_home` · `/compliant`
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
Dog: `POST /api/dog/search` · `/api/dog/stop` · Demo: `POST /api/demo/tamper` · `/api/demo/hand`

## Known gaps

- No live Grok / Grok Voice call has run yet (keys arrive at the event).
- Arm drivers are tested against fakes, not real robots or simulators. Elite is the least certain.
- Go2 dog driver is a template.

## Changelog (newest first)

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
