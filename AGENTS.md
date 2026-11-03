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
| Cobot drivers UR / Elite (sockets), xArm (SDK), safety box, stop-and-hold | Done, tested vs fakes | `arm/` | Not yet run on real hardware |
| Calibration (ArUco, clicks, arm-tip clicks) | Done, tested | `calibration.py`, `server.py` | Real arm refuses to move until calibrated with the arm |
| Grok Voice (realtime, confirm-before-move, stop word) | Done; config/token tested | `voice.py`, `static/voice.js` | Not yet heard live |
| Optional dog room search (sim) | Done, tested | `dog.py` | Go2 driver is a template |
| Jury mode + eval script | Done, tested | `server.py`, `eval/run_eval.py` | |
| 1. Table inventory map ("what's on the table?") | Planned | | |
| 2. Line-clearance zones | Planned | | |
| 3. PDF clearance record | Planned | | |
| 4. Full-screen jury scoreboard | Planned | | |
| 5. Failure log (cause of each wrong find) | Planned | | |
| 6. "Point at all the X" | Planned | | |
| 7. Tray lift with the fork (experimental) | Planned | | |

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
2. **STOP never waits for a model**: UI button, space bar, and the stop word in the live voice transcript all call
   `/api/arm/stop` directly. Stop = hold position, never torque off.
3. **Every Grok call has a mock handler** (`register_mock(name, fn)` next to the call) so the app and tests run
   without a key. New Grok call = new schema + prompt + mock + test.
4. **English and Italian**: prompts answer in the language of the request; reports follow `TROVA_LANG`.
5. **Everything is logged** to `data/events.jsonl` with saved images in `data/runs/` (traceability for the jury).
6. `data/` and `.env` are never committed. Copy the final test set to `eval/data/` before the deadline.

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
Dog: `POST /api/dog/search` · `/api/dog/stop` · Demo: `POST /api/demo/tamper` · `/api/demo/hand`

## Known gaps

- No live Grok / Grok Voice call has run yet (keys arrive at the event).
- Arm drivers are tested against fakes, not real robots or simulators. Elite is the least certain.
- Go2 dog driver is a template.

## Changelog (newest first)

- 2026-11-04 AGENTS.md added as the shared state file; `docs/make_guide.py` added.
- 2026-11-04 Initial skeleton pushed (commit a95e832): everything in the feature table marked Done.
