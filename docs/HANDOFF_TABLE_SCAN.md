# Handoff — voice table scan on the real xArm 5 (2026-11-05, 15:35)

For the next Claude session. Read this first, then `AGENTS.md`. The operator (Giacomo) is late for a demo: be fast,
reply in **Italian**, prefer **physical tests** over software ones, and keep the safety rules in section 9.

---

## 1. Where things are

| What | Value |
|---|---|
| Repo | `C:\Users\ferra\Desktop\Finale\Grok_Hackathon` (Windows 11, PowerShell + Git Bash) |
| Branch | `feat/m1-scene-selection` (== `origin/feat/m1-scene-selection` at `f76d677 final`) |
| Working tree | **all changes below are uncommitted**. Ask before commit/push. Never commit `.env` or `data/`. |
| Python | `.venv\Scripts\python.exe` (not `.venv/bin`) |
| Robot | xArm 5, `192.168.1.111`, firmware v2.7.0 |
| Camera | Raspberry Pi MJPEG on the arm, `192.168.5.160:8080` (1600x900, 10 fps, light on) |
| UI | http://localhost:8000 |
| Re-arm script | `scripts\rearm.ps1` (new, uncommitted; reads `.env`, sends no motion) |
| Object map on disk | `data\table_map.json` (78 objects, restored at server start) |
| Photos | `data\runs\*.jpg` (`table_view_*`, `table_over_<id>.jpg`, `table_sheet`) |
| Audit log | `data\events.jsonl` |

Someone else (another session) was restyling `trova/static/index.html` (CSS, camera OSD) at the same time. In that
file only touch the table-scan JS (`showTableScan`, `tableScan`, `tableFind`, ...), with small exact edits.

---

## 2. Start the station

PowerShell in the repo:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m trova.station --profile operate --camera-fallback 192.168.5.160 --robot-ip 192.168.1.111
```

Ready when `curl http://localhost:8000/api/state` answers 200. Startup always latches the software STOP.

### Re-arm (after every server start, every STOP, every manual/Studio move)

**Now automatic** (`TROVA_AUTO_REARM=1` in `.env`, operator decision 15:45): the next command (scan, find, confirm,
home, jog...) re-enables a controller in state 4/5 with error code 0 and clears the software STOP by itself.
Still manual: a controller error code (collision, E-stop): fix the cause, then "Cancella errori del braccio".
The script below stays as a fallback (it fails with "Latch software STOP" unless STOP is latched: press STOP first).

The operator runs it (it is their confirmation that the area is clear and the E-stop is in hand). From the Claude
prompt use **forward slashes** (the `!` prefix goes through Git Bash, which eats backslashes):

```
! powershell -ExecutionPolicy Bypass -File "C:/Users/ferra/Desktop/Finale/Grok_Hackathon/scripts/rearm.ps1"
```

Expected last line: `motion_mode=hardware stop_latched=False ...`. The trailing
`blocked=Calibrate with the arm first ...` is about the old "point" feature and does **not** block the table scan.

If the operator says in chat that the area is safe and asks you to proceed, you may run the same script yourself
with the PowerShell tool (that is what happened today).

---

## 3. Poses

| Name | Joints (deg) | TCP (mm) | Notes |
|---|---|---|---|
| **P0** rest | `0, 0, 0, 0, 0` (`TROVA_ARM_HOME_JOINTS`) | 207, 0, 112 | start of every session |
| **P1** scan pose | `2.45, -30.22, -49.65, 119.24, 221.76` | 251, 11, 433 | hand-set by the operator, stored as `table_look_joints` in `data/arm_poses.json`; camera 39 deg down, sees the table from ~53 cm out |

Teach a new P1: put the arm there, then
`curl -X POST -F name=table_look_joints http://localhost:8000/api/arm/teach_pose`.
Backup of the previous poses file: the old taught pose was `1.18, -28.15, -46.46, 112.77, -127.69`.

---

## 4. What the feature does now (all verified on the real arm today)

### Scan ("Scansiona tavolo" button, or voice "scansiona il tavolo")
P0 -> P1 (rise, 20 deg/s) -> J1 turns -330 deg in 30 deg steps, one photo every 30 deg (12 photos) -> back to P1 at
30 deg/s. Each photo is labelled by Grok while the arm keeps turning; V2 calibration maps each box to robot mm.
If the arm is not at P0 when the scan starts, it goes straight to P1 and never lower than the lower of the two heights.

Every map object stores **the arm pose of the photo it is seen best in** (`photo`: view, 5 joints, controller TCP) and
of every other photo it appears in (`photos`). The UI shows it next to each object.

Last physical scan (15:12): 12 photos, 78 objects, no controller error.

### "Trovami X" (voice, or the Trova box + "Vai sopra / punta" + "Conferma movimento")
Default `TROVA_TABLE_SCAN_APPROACH=over` -> `table_scan.plan_over_or_point`:

1. **Z never changes** (TCP height kept from the current pose, i.e. 433 after a scan).
2. Reachable (roughly <= 60-70 cm): **right above X** (camera 78-88 deg down).
3. Too far: the TCP still goes **as close as it can in X/Y** at the same Z (to the workspace box edge minus 40/80/130 mm,
   or the arm's reach) and the camera **points at X** from there.
4. Last resort: only the base (J1) turns toward X (`plan_point`).

Joint moves that would sag below the kept Z are split into waypoints lifted back to that Z (`level_path`, sampled TCP
within 2 mm), so leaning over takes 8-17 short moves while the wrist rolls (J4 119 -> ~-60, J5 222 -> ~44). J1 always
takes the short way round (`near_j1`). One confirmation ("Confermi?") per move, valid **60 s**, single use, voided by STOP.

Physical results today (all OK, Z constant 432.6, no controller error):

| Object | Distance | Result | Picture |
|---|---|---|---|
| #1 laptop ASUS | 76 cm | J1-only point (before "closest" existed) | `data/runs/152242_8fcf94_table_over_1.jpg` |
| #5 transparent bag | 79 cm | closest: TCP (420, -14, 433), centred | `data/runs/152738_e939d7_table_over_5.jpg` |
| #26 black laptop sleeve | 96 cm | closest: TCP (-460, 59, 433), in view (~10 cm off centre) | `data/runs/152909_c79758_table_over_26.jpg` |
| #59 laptop with stickers | 92 cm | closest: TCP (431, -460, 433), fills the frame | `data/runs/153044_76b58b_table_over_59.jpg` |

Good demo targets from the map: #16 plastic cup (57 cm, goes right above), #6 green paper cup, #32 white USB charger,
#2 smartphone (89 cm, closest + point).

---

## 5. `.env` (operator file, edited today with the operator's chosen values; backup was taken)

```
TROVA_ARM_LIMITS=-500,500,-500,500,100,1000      # workspace box: limits how close far objects can be reached
TROVA_ARM_SPEED_MM_S=100                          # normal moves
TROVA_ARM_JOINT_SPEED_DEG_S=20
TROVA_ARM_MAX_SPEED_MM_S=150                      # NEW: gateway cap, operator chose 150 mm/s after the 360
TROVA_ARM_MAX_JOINT_SPEED_DEG_S=30                # NEW: gateway cap, operator chose 30 deg/s
TROVA_MOTION_MODE=hardware                        # operator unlocks: never set or change these yourself
TROVA_HARDWARE_UNLOCK=MOVE
TROVA_TABLE_SCAN_V2_HARDWARE=accept-unqualified-v2
TROVA_TABLE_SCAN_PAN_DEG=-330,0,30                # 360, 12 photos (was 0,0,1)
TROVA_TABLE_SCAN_SPEED_DEG_S=20                   # the 360 turn (was 5)
TROVA_TABLE_SCAN_GO_SPEED_DEG_S=30                # NEW: moves after the 360
TROVA_TABLE_SCAN_GO_TCP_MM_S=150                  # NEW
TROVA_TABLE_SCAN_LANG=it
# TROVA_TABLE_SCAN_APPROACH=over                  # default; photo = back to the photo pose first; level; hover
# TROVA_TABLE_SCAN_AUTO_GO=1                      # operator decision: move without "Confermi?"
```

`TROVA_TOKEN` is empty and `TROVA_OPERATOR_NAME/TOKEN` are still placeholders (works on this PC). Never print `.env`
values (it holds `XAI_API_KEY`).

---

## 6. Code changes this session (uncommitted)

| Change | Where |
|---|---|
| Post-scan speed settings `go_speed_deg_s`, `go_tcp_mm_s` | `table_scan.Settings/settings()` |
| `tcp_z`, `min_z_on_path`, `_lift_to`, `level_path`, `path_low_z`, `_level_steps`: never lower Z | `table_scan.py` |
| `near_j1`: J1 the short way round (a -268 -> +108 = 376 deg turn was planned before) | `table_scan.py` |
| `plan_level_over(..., z_floor, margin_mm)`: no dip, box margin parameter | `table_scan.py` |
| `level_over_pose(..., margin_mm)` | `table_scan.py` |
| `plan_point`, `point_j1`: J1-only pointing | `table_scan.py` |
| `plan_over_or_point`: the default approach (above / closest / point) | `table_scan.py` |
| `plan_photo_over`: back to the photo pose, then over (kept as `TROVA_TABLE_SCAN_APPROACH=photo`) | `table_scan.py` |
| `plan_scan`: "rise and tilt to P1" without going lower, "back to P1" fast | `table_scan.py` |
| `merge`: `photo` / `photos` per object | `table_scan.py` |
| `_table_go(q, cfg, fast)`: scan speed vs post-scan speed, TCP-speed cap | `server.py` |
| `_table_capture`: stores the controller TCP pose of every photo | `server.py` |
| `table_propose`: approach modes + Italian messages ("mi avvicino il più possibile ... e lo punto") | `server.py` |
| `_table_run_approach`: fast moves, statuses `over_object` / `pointing` / `at_photo_pose` | `server.py` |
| `_table_save_map` / `_table_load_map`: map persisted to `data/table_map.json` | `server.py` |
| `_table_go_to` returns `expires_at` | `server.py` |
| Voice bug fixes: `find_object` routes to the table map when one exists; `find_on_table` / `choose_table_object` now return `proposal_id` and arm the spoken "sì" (before, **every** voice confirmation was refused) | `static/voice.js` |
| Tool descriptions and instructions (P1, Z never changes, closest/point) | `voice.py` |
| UI: photo pose per object, button "Vai sopra / punta", statuses | `static/index.html` (`showTableScan`) |
| New test `test_find_returns_to_the_photo_pose_then_over_it_never_lower_z` | `tests/test_table_scan.py` |
| Changelog lines; section 6b (P0/P1) | `AGENTS.md`, `docs/TABLE_SCAN_FINAL.md` |
| Operator re-arm script copied into the repo | `scripts/rearm.ps1` |

Tests: `tests/test_table_scan.py` 12 pass, but only with default scan settings, because the operator `.env` leaks into
pytest:

```bash
TROVA_TABLE_SCAN_PAN_DEG=-60,60,30 TROVA_TABLE_SCAN_SPEED_DEG_S=10 .venv/Scripts/python.exe -m pytest -q tests/test_table_scan.py
```

Full suite: 20 failures are pre-existing Windows/environment issues (see memory `grok-hackathon-test-env`).

---

## 7. Operate it from the API (what this session did when the operator said "sì" in chat)

```bash
# state (no motion)
curl -s http://localhost:8000/api/state         # stop_latched, robot.joints, robot.xyz_mm, robot.state_code
curl -s http://localhost:8000/api/table_scan    # status, objects, proposal {id, object_id, expires_at, summary}

# scan (moves the arm)
curl -s -X POST -H "Content-Type: application/json" -d '{"move": true}' http://localhost:8000/api/table_scan/start

# plan a move to object N (no motion), then confirm it within 60 s (moves the arm)
curl -s -X POST -H "Content-Type: application/json" -d '{"object_id": N}' http://localhost:8000/api/table_scan/approach
curl -s -X POST -H "Content-Type: application/json" -d '{"proposal_id": "<id>"}' http://localhost:8000/api/table_scan/confirm
```

Only confirm a proposal after the operator's explicit yes for that object. If it expired, plan the same object again
and confirm the new id. Watch progress with a Monitor loop on `/api/state` (status, step, stop_latched, error_code).

---

## 8. Known problems and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Nothing moves, voice says "STOP cancelled this pending command" | software STOP latched (server start, STOP button, or the word **"stop"** heard by the microphone: it happened at 15:12) | re-arm (section 2); in the demo don't say "stop" near the mic |
| `RuntimeError: xArm set_servo_angle returned code 9`, `state_code 5` | controller not enabled after a manual / Studio move (arm moved to P0 by hand at 15:32) | re-arm script (its `clear_errors` re-enables) or UI Strumenti -> Braccio -> "Cancella errori del braccio", then retry |
| "sì" too late, confirm refused | proposals live 60 s | ask again ("trovami X") |
| J1 far from 0 (reached 313 deg) | each find turns from where the base is; turns add up | before the demo bring it back to P1/P0 (a new scan from P0 resets it); J1 limit is +-358 |
| Far objects stop at ~46 cm from the base | workspace box +-500 mm (minus margin) | operator may widen `TROVA_ARM_LIMITS` (their decision) |
| Aim ~10 cm off on far objects | V2 not physically qualified, partial boxes at picture edges | known; objects seen whole are better |
| Server restart loses the STOP state but not the map | map in `data/table_map.json` | re-arm after restart |
| Port 8000 still busy right after stopping the server | slow shutdown | wait 1-2 s and check `Get-NetTCPConnection -LocalPort 8000` |

State when this file was written: arm at **P0** (operator moved it), controller state 5 (needs `clear_errors`), map of 78
objects loaded. The next step the operator wanted: re-enable, then **Scansiona tavolo** again from P0.

---

## 9. Rules for the next session

- Real motion only when the operator asks for it and is present; every table move needs its own yes.
- Never set or change `TROVA_MOTION_MODE`, `TROVA_HARDWARE_UNLOCK`, `TROVA_TABLE_SCAN_V2_HARDWARE`, speed caps or the
  workspace box on your own: propose, the operator decides.
- The re-arm is the operator's confirmation; run it yourself only after they say the area is safe.
- Software STOP is not a safety-rated E-stop.
- Ask before commit/push. When committing: update `AGENTS.md` (feature table, changelog), never commit `.env` or `data/`.
- Restart the server after Python changes (static JS/HTML only needs a browser reload).

## 10. Open items

1. Commit + push this work to `feat/m1-scene-selection` (operator to approve).
2. Optional: `TROVA_TABLE_SCAN_AUTO_GO=1` for a smoother voice demo (operator decision).
3. Optional: widen the workspace box to get closer to far objects (operator decision).
4. Later: distance filter (a 360 at 39 deg down maps the whole room), ruler check of V2, closed-loop re-centring with
   the picture taken at the end of the move.
