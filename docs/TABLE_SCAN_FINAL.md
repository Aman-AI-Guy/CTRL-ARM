# Table scan on the real xArm 5 — final state (2026-11-05)

Branch `feat/m1-scene-selection`, nothing committed. Supersedes the plan in `docs/TABLE_SCAN_V2.md` (kept for history:
its numbers for the folded-wrist pose are no longer used). Everything below was run on the **real arm**.

## 1. What it does now

```
rest 0,0,0,0,0
  │  "Scansiona tavolo" (button or voice)
  ▼
your taught look pose  (table_look_joints = 1.18, −28.15, −46.46, 112.77, −127.69 → TCP 263, 5, 404 mm)
  │  J1 turns −330° in 30° steps at 20°/s, one photo every 30° (12 photos)
  │  each photo is labelled while the arm keeps turning (map grows live):
  │     OpenCV boxes; if OpenCV finds < 3 boxes, Grok draws and names the boxes itself
  │     V2 turns each box into robot mm on the table plane
  ▼
back to the taught look pose            → object map (number, name, x/y mm) + photos with labelled boxes
  │
  │  "Trovami le pinze" (voice, or the Trova box)
  ▼
1 match  → move planned at once: "Numero N, pinze (x, y): mi metto sopra, alla stessa altezza Z 404 mm. Confermi?"
2+ match → "quale scegli?" → number / tap → same planned move
  │  "sì" / Conferma movimento        (or no question at all with TROVA_TABLE_SCAN_AUTO_GO=1, operator setting)
  ▼
turn J1 toward it (same height) → lean over it at the SAME TCP height (no descent) → photo from above
```

If the object is too far to get right above it (beyond ~60 cm from the base with the ±500 mm box), the arm points
the camera at it from the closest allowed spot instead ("lo indico da qui"). Far objects outside the box are refused.

## 2. Run it

PowerShell in `C:\Users\ferra\Desktop\Finale\Grok_Hackathon`:

```powershell
.\.venv\Scripts\python.exe -m trova.station --profile operate --camera-fallback 192.168.5.160 --robot-ip 192.168.1.111
```

Re-arm after every boot and every STOP (operator, area clear, E-stop in hand):

```powershell
powershell -ExecutionPolicy Bypass -File <scratchpad>\rearm.ps1     # clear_errors + reset (+ one-time lift record)
```

(`rearm.ps1` reads `TROVA_TOKEN`, `TROVA_OPERATOR_NAME`, `TROVA_OPERATOR_TOKEN` from `.env`; copy it into the repo
if you want to keep it.) Then open http://localhost:8000, press **Scansiona tavolo**, type or say what to find.

Teach a different look pose: put the arm there (manual/Studio), then
`Invoke-RestMethod -Method Post http://localhost:8000/api/arm/teach_pose -Form @{name="table_look_joints"}`.

## 3. `.env` lines for this setup

```
TROVA_ARM_LIMITS=-500,500,-500,500,100,1000
TROVA_ARM_HOME_JOINTS=0,0,0,0,0
TROVA_MOTION_MODE=hardware
TROVA_HARDWARE_UNLOCK=MOVE
TROVA_OPERATOR_NAME=<your name>          # still the placeholder: change it
TROVA_OPERATOR_TOKEN=<a secret>          # still the placeholder: change it
TROVA_TOKEN=<a secret>                   # was lost in a paste: add it back if the UI is opened from other devices
TROVA_TABLE_SCAN_V2_HARDWARE=accept-unqualified-v2
TROVA_TABLE_SCAN_PAN_DEG=-330,0,30       # 360: 12 photos   (today passed as a process env var, .env still says 0,0,1)
TROVA_TABLE_SCAN_SPEED_DEG_S=20          # 4x the first run (today passed as a process env var, .env still says 5)
TROVA_TABLE_SCAN_LANG=it
# TROVA_TABLE_SCAN_AUTO_GO=1             # optional: "trovami X" moves without "Confermi?" (operator decision)
# TROVA_TABLE_SCAN_APPROACH=hover        # optional: old behaviour, camera 30 cm above the object (descends)
```

`TROVA_TABLE_SCAN_CENTER_DEG` is ignored while a `table_look_joints` pose is taught.

## 4. Physical results today

| Time | Run | Result |
|---|---|---|
| 12:59 | first scan, solver pose with folded wrist (J4 → 174.5) | **controller C22 self-collision at J4 = 164°** (camera head against the arm). Arm held at Z 562. |
| 13:17 | rest → your taught pose → 360 at 5°/s → back | OK, no error. 12 photos, 40 objects. Front view: **0 OpenCV boxes** (cables join everything). |
| 13:31 | 360 at **20°/s** with the Grok-box fallback | OK, ~1 min of motion, no error. 84 objects, boxes on every photo (front view 0 → 13). |
| 13:33 | "trovami le pinze" | **not found**: no pliers in any of the 12 photos (Grok checked the close-ups too). |
| 13:34 | "mettiti sopra" #43 *alimentatore nero* (599, −106) | OK, ~14 s. TCP Z 404 → min 381 → 404 (dip 23 mm, as planned). Camera 76° down, 43 cm from the target. **Aim off by ~7–13 cm**: the object's box was cut by the bottom edge of the scan photo, so its foot was taken at the picture edge (too far from the robot). |

Tracked: J1 at 20°/s at TCP radius 264 mm = 92 mm/s. Every table move now also lowers its joint speed so the
sampled TCP stays ≤ `TROVA_ARM_SPEED_MM_S` (100 mm/s) — leaning out at long reach would otherwise reach ~175 mm/s.

## 5. What changed in the code (all uncommitted)

| Change | Where |
|---|---|
| Windows: lift safety record written with `MoveFileExW(WRITE_THROUGH)`; re-arm was impossible before | `trova/safety.py` |
| Folded wrist forbidden (J4 ≤ 120°) after C22; straight-wrist family preferred; picture rotated 180° when the camera is upside down, pixels mapped back for V2 | `trova/table_scan.py`, `server._table_capture` |
| Operator-taught look pose `table_look_joints` replaces the solver; scan ends back at it | `table_scan.plan_scan`, `/api/arm/teach_pose` |
| Labels during the turn (each photo labelled in the background, map published live) | `server.run_table_scan`, `_table_publish` |
| Grok draws the boxes when OpenCV finds < 3 | `table_scan.grok_boxes`, `server._table_describe` |
| Photos saved with labelled boxes (`#n name x y`), shown in the panel | `table_scan.annotate_view`, `static/index.html` |
| "Over it at the same height" (`level_over`): J1 turn, then lean over, TCP Z kept, inside the box minus 40 mm, J5 kept within ±180° (cable) | `table_scan.plan_level_over`, `server.table_propose` |
| Find/choice plans the move at once and asks one "Confermi?"; optional `TROVA_TABLE_SCAN_AUTO_GO=1` | `server._table_go_to`, `voice.py` instructions |
| Joint speed capped by TCP speed | `server._table_go` |

No new software tests were written (physical testing requested). The existing table-scan and safety tests pass with
default scan settings.

## 6. Known limits

- **Partial boxes** (cut by the picture edge) give a wrong foot point → the "over it" move misses by several cm.
  Fix options: skip partial sightings when the same object is seen whole in another view, or re-centre with a second
  photo from above (closed loop).
- From the taught pose the camera sees from ~54 cm out; at Z 404 the arm can get right above things only up to
  ~60 cm. Objects closer than 54 cm are never mapped; farther than ~60 cm are pointed at, not gone over.
- A 360 at 37° down maps the whole room (chairs, shoes, people). No distance filter yet.
- V2 is still not physically qualified: no ruler measurement was taken today.
- Grok calls take 8–56 s; one hit the 60 s timeout (offline names "grey object").
- The wrist rolls ~175° in the air when leaning over (folded → straight wrist). Camera head stays ≥ 300 mm above the
  table, but nothing models people or objects taller than ~12 cm.
- Software STOP is not a safety-rated E-stop.

## 6b. P0 / P1 and photo poses (2026-11-05 afternoon)

- **P0** = rest, all joints 0 (`TROVA_ARM_HOME_JOINTS`). **P1** = the scan pose the operator set by hand and taught as
  `table_look_joints`: `2.45, -30.22, -49.65, 119.24, 221.76` (TCP 251, 11, 433 mm, camera 39 deg down).
- Scan: P0 -> P1 -> J1 -330 deg in 30 deg steps (12 photos, 20 deg/s) -> back to P1 at the post-scan speed. Started away
  from P0, it goes straight to P1 and never below the lower of the two heights.
- Every object stores the arm pose of the photo it is seen best in (`photo`: view, 5 joints, controller TCP pose) and of
  every other photo it is in (`photos`). The UI shows it next to each object.
- "Trovami X" / **Vai (foto + sopra)**: back to the joints of X's best photo (J1 the short way round, e.g. -267.6 ->
  +92.4), one picture there (`table_photo_pose_<id>`), then lean over X at the same height. If it cannot get over X,
  it stops at the photo pose (X is in that picture).
- After the 360 the TCP never goes below the scan height: sagging joint moves are split into waypoints lifted back to
  that height (sampled TCP within 2 mm). Speed after the 360: 30 deg/s and 150 mm/s TCP (`TROVA_TABLE_SCAN_GO_*`,
  gateway caps `TROVA_ARM_MAX_JOINT_SPEED_DEG_S=30`, `TROVA_ARM_MAX_SPEED_MM_S=150`); everything else stays at
  `TROVA_ARM_SPEED_MM_S=100`.
- Leaning over from P1 rolls the wrist from J4 119 / J5 222 to J4 ~-60 / J5 ~44 in the air (8 short moves).

## 7. Open questions for the operator

1. Where should the pliers be for the next test (55–60 cm from the base, in the front view)?
2. Ruler: real position of the object under the camera after "mettiti sopra" (X forward, Y left of the base).
3. Keep "Confermi?" or set `TROVA_TABLE_SCAN_AUTO_GO=1`?
4. Speed: 20°/s fine, or raise the cap to 25°/s (TCP would reach ~115 mm/s at the J1 turn)?
5. Table size: drop objects beyond N mm from the base?
6. Commit and push these changes?
