# Voice table scan with the V2 calibration — handoff

Session of 2026-11-05, branch `feat/m1-scene-selection`. **Next session = real test on the xArm 5, no simulation.**
Read §0 (physical setup), then follow §4 step by step and tick §5. Fill §11 with the results.

---

## 0. Physical setup (confirmed by the operator, 2026-11-05)

- **The table is right under the rest position.** Rest / start pose = all joints `0,0,0,0,0` (TCP X 207, Y 0,
  Z 112 mm, bearing 0° = +X). The scan is centred on that direction automatically (the bearing of the start pose);
  `TROVA_TABLE_SCAN_CENTER_DEG` is only needed if the arm ever rests pointing elsewhere.
- **At 0,0,0 the camera looks horizontally, back-right of the robot** (measured with V2 on the real pose,
  2026-11-05 afternoon): camera centre at (180, −25, 73) mm, 114 mm above the table, optical axis (−0.72, −0.69, 0)
  = bearing ≈ −136°. The bottom row of the picture lands on the table at (2, −362) … (−206, −120): the objects
  seen in the rest picture (pliers, bottle, ArUco cube) are 120–450 mm away on the −X/−Y side. The OpenCV finder
  finds 0 boxes there (everything is large and touches the picture edge → treated as an occluder); it needs the
  looking-down scan pose.
- **Real rise corridor** (V2, real rest pose): camera centre X 180 → 299 → 144, Y −37 … 0, height above the table
  114 → 593 mm; < 200 mm above the table while X is 180–290. The objects seen at rest are not on it. Keep the
  front corridor (X 80–400, |Y| < 100) free of anything tall: the arm links are not modelled.
- **Scan centre vs. where the objects are**: centre 0° photographs the +X side only. Centres −90, −120, −135 also
  plan with no problems inside ±500 mm; `TROVA_TABLE_SCAN_CENTER_DEG=-120` covers bearings −60 … −180 (the side the
  V2 table plane was fitted on). Camera at the scan pose: 593 mm above the table (the "60 cm").
- **That is why the scan first moves:** without turning (J1 stays at 0) it climbs in a short forward arc to
  **TCP Z = 600 mm** and tilts the camera **70° down** onto the table, then turns J1 left/right (±60°) to photograph
  the whole table.
- "Etichetta vista" (label the current view without moving) therefore shows no table from the rest pose: use it only
  after the arm is in a looking pose (for example after a scan).

---

## 1. What it does

You talk to Grok Voice; the arm does the rest:

```
"Scansiona il tavolo"
   │  arm rises from the start pose to TCP Z = 600 mm and tilts the camera down at the table
   │  (just enough that the whole picture is table, no room), then turns J1 through 5 views
   ▼
5 photos ──► Grok names every object in each photo ──► V2 turns each object into robot-base mm
   │                                                   (where the object touches the table plane)
   ▼
one object map (duplicates seen in two photos are merged)    "Ho visto 6 oggetti: 1 bicchiere di carta, …"
   │
"Trova il bicchiere"
   │  1 match   → "Trovato: numero 4, bicchiere di carta, alla sinistra del robot. Vuoi che porti la camera sopra?"
   │  2+ matches → "Ho trovato 2 oggetti …: il numero 3 alla sinistra del robot; il numero 5 alla destra del robot.
   │               Quale scegli? Dimmi il numero, oppure toccalo nell'interfaccia."   ──► "il numero 3" / tap "Scegli"
   ▼
"Porta la camera sopra"  → plan only: "Porto la camera a 30 cm sopra il numero 3 … Confermi?"
"Sì, procedi"            → two slow joint moves (150 mm higher first, then 300 mm above), photo from above
```

STOP (button, space bar, or saying "stop / fermo / basta") always wins and voids any pending move.

---

## 2. Status: what is verified, what is not

| Part | Verified how | Result |
|---|---|---|
| V2 file loads, SHA-256 matches the pin | live + tests | `Test 3d live video/calib/v2/camera_robot.json`, `59c1afad…` |
| Scan-pose solver (Z=600, tilt, J1 views) | tests on the real V2 model | centre ray hits the aim point within 5 mm; whole top edge on the table |
| Object position from a box | tests (synthetic projection round trip) | box foot back to the true table point < 1 mm |
| Merge across views, duplicates → question | tests | 3 objects seen in 3–4 views each → 3 map entries; 2 cups → clarify with numbers + sides |
| Camera-over-object solver | tests | object on the image-centre ray < 0.5°, distance 300 ± 10 mm, ≥ 70° looking down |
| Full flow on the **simulated** xArm 5 | live (real Pi camera + **live Grok**) | scan 5 views ≈ 1.5 min, 7 objects named, find "bicchiere" → #4 in 2.4 s, propose → confirm → sim arm moved |
| Full flow on the **real** xArm 5 | **not run** | for the next session (section 5) |
| V2 accuracy on the real table | **not measured** | V2 is not physically qualified (`docs/M62_QUALIFICATION.md`) |

Important for reading the live simulated run: in the `simulate` profile the *simulated* arm turns, but the *real*
camera stays wherever the real arm is. Today the real arm was at the all-zero start pose, where the camera looks
horizontally at the room (expected, see §0), so Grok named "door glass", "man in a jacket", a paper cup… Positions
in that mode are flagged "Posizioni simulate". On the real arm the camera will actually be tilted at the table.
The live simulated run used the old −Y centre; the scan is now centred under the rest pose (+X), covered by tests.

Test suite: **358 passed**. 20 failures are pre-existing on this Windows PC and fail identically on both pre-merge
commits (no COLMAP, directory fsync not supported on Windows, missing retained fixtures, env var).

---

## 3. Numbers from today

| Item | Value |
|---|---|
| Start pose (set by you) | joints `0,0,0,0,0` → TCP X 207, Y 0, Z 112 mm, roll −180, pitch 0, yaw 0 |
| Camera | Pi IMX219 on the flange, looks **sideways** (45° in the flange plane), 1600×900, HFOV 63°, VFOV 38° |
| Table plane (V2 fit) | z ≈ −40.7 mm in the robot base frame (arm on a 40 mm pedestal), slightly tilted |
| Scan centre | bearing 0° (+X), taken from the start pose: the table is under the rest position |
| Step 1: rise and tilt (J1 stays 0) | joints `0, −26.6, −77.6, 174.5, −135.9` → TCP (168, 0, **600**), camera **70° down** |
| What one view covers | table from 164 mm to 790 mm from the robot base, centre of the picture at 370 mm |
| Scan views (J1) | +58.3, +28.3, −1.7, −31.7, −61.7° (J2..J5 unchanged) = bearings +60 … −60°; centres land at (185, 320), (320, 185), (370, 0), (320, −185), (185, −320) |
| Rise from start | forward arc: TCP (207, 0, 112) → (311, 0, 284) → (168, 0, 600); camera head rises monotonically 49 → 566 mm above the table; lowest point = the start pose itself |
| While turning | TCP at Z 600, camera head ≥ 565 mm above the table |
| Camera over the spot under the rest TCP (207, 0) | camera at (236, −8, 257), TCP (279, −9, 289), 300 mm away, 84° looking down |
| Camera over (350, 120) / (420, −150) | camera 84° / 80° down, inside the box |
| Camera over (650, 0) | refused: TCP would be at x = 515–553 mm, outside the ±500 mm box |
| Earlier live V2 measurement of the green cup | (−16, −665, −41) mm, taken from an older −Y looking pose; outside the box |

---

## 4. Real test on the xArm 5 — step by step (next session)

No simulation is needed: go straight to the real arm. All commands in PowerShell from
`C:\Users\ferra\Desktop\Finale\Grok_Hackathon`. The Grok key is already in `.env` (`XAI_API_KEY`, git-ignored).
Open **http://localhost:8000** in Chrome/Edge on this laptop (the microphone works only on localhost or HTTPS).
Two people: one at the laptop, one at the robot with the E-stop in hand for the whole test.

### Hard rule before anything moves: keep the rest area clear

In the rest pose (0,0,0) the camera head is only **~49 mm above the table** (TCP ~154 mm above it). Every
"torna a casa" / Arm home ends there, and coming home from "camera above an object" dips ~5 mm lower. **Nothing
taller than ~4 cm may stand within ~15 cm of the spot under the rest TCP (X 207, Y 0)** — this also covers the
start of the rise, when the head is still < 90 mm above the table over X 207–260. Put the test objects
**300–450 mm from the robot base**, inside ±60° of the rest direction (for example at (350, 120), (420, −150),
(400, 0)), never under the rest pose.

### Phase A — preparation (no motion)

1. **Robot**: powered, E-stop released and tested, no controller error, collision sensitivity ≥ 3, reduced mode on
   (`docs/first-supervised-run.md`), arm in the rest pose 0,0,0. Nobody leaning over the table.
2. **`.env`** (operator decisions — you set them, not the assistant). For the first run use one view and 5°/s:
   ```
   TROVA_ARM_LIMITS=-500,500,-500,500,100,1000     # your real safe box (or station_workspace.json with confirm: true)
   TROVA_ARM_HOME_JOINTS=0,0,0,0,0                 # the rest pose
   TROVA_OPERATOR_NAME=<your name>
   TROVA_OPERATOR_TOKEN=<a secret>                 # every physical boot is disarmed; this re-arms it
   TROVA_MOTION_MODE=hardware
   TROVA_HARDWARE_UNLOCK=MOVE
   TROVA_TABLE_SCAN_V2_HARDWARE=accept-unqualified-v2   # you accept that V2 is not physically qualified
   TROVA_TABLE_SCAN_PAN_DEG=0,0,1                  # first run: one view only (remove for the full 5-view scan)
   TROVA_TABLE_SCAN_SPEED_DEG_S=5                  # first run: 5 deg/s (default 10, max 20)
   # TROVA_TABLE_SCAN_CENTER_DEG stays empty: the scan centres on the rest pose (table under it, bearing 0 = +X)
   ```
3. **Start the server**:
   ```powershell
   .\.venv\Scripts\python.exe -m trova.station --profile operate --camera-fallback 192.168.5.160 --robot-ip 192.168.1.111
   ```
   Check the mode the gateway actually resolved (the launcher prints only the raw setting):
   ```powershell
   (Invoke-RestMethod http://localhost:8000/api/state).arm.motion_mode     # must be: hardware
   ```
   `dry_run` = an unlock is missing: plans are shown, nothing moves. Fix `.env` and restart.
4. **Controller ready + software re-arm** (STOP is latched at every boot; nothing moves until both are done):
   ```powershell
   $h = @{ "x-trova-operator-token" = "<your secret>" }
   Invoke-RestMethod -Method Post http://localhost:8000/api/arm/clear_errors -Headers $h      # enables the controller
   $nonce = (Invoke-RestMethod http://localhost:8000/api/state).stop_nonce
   Invoke-RestMethod -Method Post http://localhost:8000/api/arm/reset -Headers $h -ContentType "application/json" `
     -Body (@{operator="<your name>"; stop_nonce=$nonce; area_clear=$true; controller_ready=$true} | ConvertTo-Json)
   ```
   Repeat this block after **every** STOP.
5. **Camera at rest**: the live picture shows the room, not the table. That is expected (§0).
6. **Plan check** (no motion):
   ```powershell
   Invoke-RestMethod -Method Post http://localhost:8000/api/table_scan/plan | ConvertTo-Json -Depth 6
   ```
   Go only if: `executable: true`, `problems: []`, `center_deg: 0`, `center_from: start pose`, step 1 = J1 0 with
   J2..J5 ≈ `−26.6, −77.6, 174.5, −135.9`.

### Phase B — first motion: rise, tilt, one view

7. Press **Scansiona tavolo** (or say "Scansiona il tavolo"). Expected, slowly at 5°/s: J1 stays 0 (no sideways
   turn, Y stays 0); the TCP climbs in a **forward arc** — out to X ≈ 311 mm at mid-move, then back to X ≈ 168 mm —
   up to Z ≈ 600 mm while the camera tips ~70° down. The camera head only goes up (49 → 62 → 88 → 126 … → 566 mm
   above the table); in the first fifth of the move it is still low (< 90 mm) over X 207–260. Then one photo, then
   Grok names the objects.
   - **Pass**: the picture shows only the table (no room), telemetry Z ≈ 600, a map with names and x/y.
   - **Stop immediately** if the arm swings sideways, the head goes towards the table, or anything unexpected moves.
8. **First accuracy number**: measure one test object's centre from the robot base with a ruler (X forward,
   Y to the robot's left) and compare with its `x, y` in the panel. Write it in §11.

### Phase C — full scan

9. Remove `TROVA_TABLE_SCAN_PAN_DEG` from `.env` (5 views, ±60°), optionally raise the speed to 10, restart the server
   (step 3), re-arm (step 4), scan again. **Pass**: 5 photos, J1 turning +58 … −62° at constant height, one merged map.

### Phase D — find, duplicates, camera above

10. Put **two identical objects** (e.g. two equal cups) at two places in the test area. Say "Trova il bicchiere".
    **Pass**: Grok asks "Ho trovato 2 oggetti …: il numero X alla sinistra del robot; il numero Y alla destra del
    robot. Quale scegli?" and both are highlighted in the panel. Answer "il numero X" (or tap **Scegli**).
11. Say "Porta la camera sopra". **Pass**: it only describes the move and asks "Confermi?" — nothing moves yet.
12. Say "Sì, procedi" (or **Conferma movimento**). Expected: two slow moves (150 mm higher first, then 300 mm above),
    then a photo from above. **Pass**: the object is near the centre of that photo. Note how far off it is (§11).
13. Bring it back: "Torna a casa" (rest area clear, see the hard rule).

### Phase E — STOP

14. Say "stop" (and once press STOP) during: the rise, a J1 turn, the camera-above move. **Pass** each time: the arm
    holds, the panel says stopped, any pending proposal is gone, and nothing moves again until step 4 is repeated.

---

## 5. Checklist (tick during the real test)

- [x] Table location: under the rest position (operator, 2026-11-05) → scan centre = rest-pose bearing (0°).
- [ ] Rest area clear (hard rule); test objects 300–450 mm out, within ±60°.
- [ ] Phase A: `motion_mode` = hardware, re-armed, room visible at rest, plan executable with centre 0.
- [ ] Phase B: rise + tilt to Z 600, one view, picture = table only.
- [ ] J4 reaches 174.5° in the scan pose (limit 180°, 2° margin enforced): the controller accepted it.
- [ ] Ruler vs map for one object (§11).
- [ ] Phase C: 5 views, one merged map.
- [ ] Phase D: duplicate question by voice; choice by voice and by tap; proposal → yes → camera above; photo centred.
- [ ] Phase E: STOP in each phase.
- [ ] Decide about committing and pushing (§9).

### If something is off

| Symptom | Likely cause | What to change |
|---|---|---|
| `motion_mode` is `dry_run` | an unlock missing in `.env` | `TROVA_MOTION_MODE=hardware`, `TROVA_HARDWARE_UNLOCK=MOVE`, restart |
| "accept-unqualified-v2" refusal | V2 acknowledgement missing | `TROVA_TABLE_SCAN_V2_HARDWARE=accept-unqualified-v2`, restart |
| "STOP is latched" | after boot or any STOP | step 4 (clear_errors + reset) |
| "controller error code …" | controller fault | clear it on the robot / Studio, then step 4 |
| picture still shows the room after the rise | tilt too shallow for this table | lower `TROVA_TABLE_SCAN_FAR_MM` (e.g. 650) |
| picture shows only a small part of the table | tilt too steep | raise `TROVA_TABLE_SCAN_FAR_MM` (e.g. 950) |
| plan problem "outside the safe box" | target or path outside `TROVA_ARM_LIMITS` | move the object closer, or (operator) widen the measured box |
| "camera gives 640x480" | the Pi is hot | let it cool / restart its stream |
| objects named but x/y clearly wrong | V2 off on this table | stop motion tests; run the M6.2 qualification first |

### Optional, not needed for the real test

- `--profile simulate`: simulated xArm 5 + real camera (rehearsal only; positions not physical).
- `--profile qualify`: real arm pose read-only, no motion; "Etichetta vista" labels the current view (from the rest
  pose it sees no table).

---

## 6. Voice phrases and UI

| Say (IT / EN) | Tool | Moves? |
|---|---|---|
| "Scansiona il tavolo" / "scan the table" | `scan_table` | yes (whole scan, after the plan checks) |
| "Etichetta quello che vedi senza muoverti" | `scan_table` (move=false) | no |
| "Trova il bicchiere" / "find the cup" | `find_on_table` | no |
| "Il numero 3" / "quello alla sinistra del robot" | `choose_table_object` | no |
| "Porta la camera sopra" / "bring the camera above it" | `move_camera_above` | no (proposal only) |
| "Sì, procedi" / "yes, proceed" | `confirm_camera_move` | yes (only with a live proposal AND that yes in your own transcript) |
| "Stop" / "fermo" / "basta" | stop (browser, no model in the loop) | holds |

UI (camera header): **Scansiona tavolo**, **Etichetta vista**. Results panel *Scansione tavolo*: status, summary,
search box + **Trova**, the object list (number, name, side, x/y mm) with **Scegli** and **Camera sopra**, the
candidates of a question highlighted, **Conferma movimento** for a pending proposal, the numbered close-up sheet, and
the photo taken from above.

"Left/right" in questions is **from the robot's point of view** looking at the middle of the table.

---

## 7. Safety gates (all must pass; none was loosened)

- Every move is `CartesianArm.go_joints` → MotionGateway (workspace on sampled TCP points, joint limits, STOP latch,
  reached check) → the driver's own checks (refuses a still-moving controller, ticket speed/accel caps).
- Before any move: joint limits with a 2° margin; camera head sphere never lower than 30 mm above the table (or
  than where it started); sampled TCP inside `TROVA_ARM_LIMITS`; camera head above the tallest assumed object
  (`TROVA_TABLE_SCAN_OBJECT_MAX_MM`, 120).
- Joint speed ≤ `TROVA_TABLE_SCAN_SPEED_DEG_S` (default 10, hard max 20).
- Hardware also needs `TROVA_MOTION_MODE=hardware` + `TROVA_HARDWARE_UNLOCK=MOVE` + `TROVA_TABLE_SCAN_V2_HARDWARE=
  accept-unqualified-v2`, an authenticated operator re-arm after boot, and a controller without errors.
- Camera-over: one proposal, single use, 60 s, voided by STOP, by a new find/choice, or if the arm moved since.
  Voice confirmation also needs the "sì/procedi" in the operator's own transcript.
- A pose read while moving or under a controller error is never used as a measurement.
- A real camera at a different aspect ratio (a hot Pi drops to 640×480) refuses the scan.
- Not present: collision avoidance with objects, people or the room. Software STOP is not a safety-rated E-stop.
- First-run mode (`TROVA_FIRST_RUN=1`) refuses table-scan motion.

---

## 8. Settings

| Variable | Default | Meaning |
|---|---|---|
| `TROVA_TABLE_SCAN_Z_MM` | 600 | TCP height while looking around |
| `TROVA_TABLE_SCAN_CENTER_DEG` | empty = start-pose bearing (0° = +X, the rest position) | base bearing of the middle of the table |
| `TROVA_TABLE_SCAN_PAN_DEG` | `-60,60,30` | J1 offsets from the centre: from, to, step |
| `TROVA_TABLE_SCAN_FAR_MM` | 800 | the whole top image edge must land on the table within this radius (sets the tilt) |
| `TROVA_TABLE_SCAN_HOVER_MM` | 300 | camera-to-object distance for "camera above" (150–600) |
| `TROVA_TABLE_SCAN_OBJECT_MAX_MM` | 120 | tallest object assumed under the camera |
| `TROVA_TABLE_SCAN_SPEED_DEG_S` | 10 | joint speed cap (max 20) |
| `TROVA_TABLE_SCAN_SETTLE_S` | 0.8 | wait before each photo |
| `TROVA_TABLE_SCAN_MERGE_MM` | 60 | two sightings closer than this are one object |
| `TROVA_TABLE_SCAN_LANG` | it | language of names, questions and spoken messages |
| `TROVA_TABLE_SCAN_V2_HARDWARE` | empty | `accept-unqualified-v2` to allow real motion |
| `TROVA_SIM_START_JOINTS`, `TROVA_SIM_FAST` | zeros, 1 | simulated xArm start pose and speed-up |

API: `GET /api/table_scan` · `POST /api/table_scan/plan` (no motion) · `/start` `{move}` · `/find` `{text}` ·
`/choose` `{object_id | answer}` · `/approach` `{object_id}` (no motion) · `/confirm` `{proposal_id}`.

---

## 9. Everything changed today

**Merge.** Old `main` (now `origin/main-backup`, e28f305: safety hardening, first supervised run, stage 0,
`xarm_config`, Pi MJPEG) merged into this branch as commit **ae08265** (local, not pushed). `origin/main` had been
force-pushed to this branch's previous tip.

**Not yet committed** (working tree on `feat/m1-scene-selection`):

| File | Change |
|---|---|
| `trova/table_scan.py` (new) | V2 loading, look-pose and camera-over solvers on the image-centre ray, box → mm, merge, Grok find, clarify, choice |
| `trova/arm/xarm5_sim.py` (new) | simulated xArm 5 with the controller's kinematics (`TROVA_ARM=xarm5sim`) |
| `trova/server.py` | table-scan state, worker, endpoints; xarm5sim counted as a simulator; STOP voids table proposals; capture message names the controller error |
| `trova/voice.py`, `trova/static/voice.js` | 5 table tools + instructions; half-duplex echo guard; confirmation for table moves |
| `trova/static/index.html` | Scansiona tavolo / Etichetta vista buttons, Scansione tavolo panel |
| `trova/station.py` | `--profile simulate` |
| `trova/agent.py` | the "not stationary" message now says why (e.g. controller error 21) |
| `trova/audit.py`, `trova/failures.py`, `eval/run_eval.py` | UTF-8 logs, cp1252 fallback for old Windows lines (hash chain still verifies) |
| `trova/inventory.py` | bug: `grok.offline()` called a property → live labelling of an empty view crashed |
| `tests/test_table_scan.py`, `tests/test_audit_encoding.py` (new) | 12 tests |
| `AGENTS.md` | feature row, API line, changelog |

**Bugs found and fixed today**
1. Voice echo: Trova's own speech was transcribed as the operator → half duplex while it speaks (never while the arm moves).
2. Audit log written as cp1252 on Windows (`è` → `�`; a `→` would have crashed the write).
3. `inventory._grok_mode` called the `offline` property.
4. **Distortion fold** in the first camera-over solver: a point 55° off-axis projected to the image centre through
   OpenCV's distortion polynomial, so a "centred" pose looked elsewhere. Solvers now aim the undistorted
   image-centre ray, reject wrong arm configurations (facing away, looking back at the robot), and tests check the
   real ray. The live simulated approach ran before this fix; the fixed solver is covered by tests.

**Not pushed.** `ae08265` and today's changes are local. Pushing `feat/m1-scene-selection` (and fast-forwarding
`origin/main`) is a team decision for the next session.

---

## 10. Known limits

- V2 is unqualified: its replay error on the calibration images was 1.4 mm median / 6.2 mm p95, which is not an
  independent measurement on today's table. Run the M6.2 procedure before relying on millimetres.
- Object position is the bottom-centre of the box pushed back by half its width: good for round/compact objects,
  rough for long ones (a pen lying across the view) and for objects cut by the image edge ("visto in parte").
- The scan covers bearings ±60° around the rest direction and 164–790 mm from the base; objects outside are not
  mapped. Objects closer than 164 mm to the base (right at the robot's foot) are below the bottom of the picture.
- The tilt (70° down) comes from `TROVA_TABLE_SCAN_FAR_MM` = 800: the whole top edge of the picture must land on the
  table within 800 mm. If the table's far edge is nearer, lower it (steeper); if it is farther, raise it (shallower).
- The V2 table plane was fitted on data from the −Y side; under the rest pose (+X) it is an extrapolation of a
  nearly flat plane (z ≈ −40.7 to −42.4 mm over the area). Worth one ruler check (§5).
- Grok may name background or table marks as objects; names are per view, merged by position only.
- No obstacle model: the path checks the camera head against the table plane and the TCP against the box, not
  against objects, people or the arm's own links.
- `simulate` positions are not physical (the real camera does not move with the simulated arm).

---

## 11. Real-test results (fill in during the test)

| Date | Phase | What was checked | Expected | Measured / observed | OK? |
|---|---|---|---|---|---|
| | B | TCP Z after the rise | ≈ 600 mm | | |
| | B | picture after the rise | table only | | |
| | B | object 1: ruler (x, y) vs map (x, y) | difference ≤ ~20 mm | | |
| | C | 5 views, merged map | each object once | | |
| | D | duplicate question | numbers + sides, both highlighted | | |
| | D | camera above: object offset in the photo | near the centre | | |
| | E | STOP during rise / turn / approach | holds, re-arm required | | |
