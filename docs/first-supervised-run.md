# First supervised run on the xArm 5: find → hover → descend → measure

This is the first time Trova moves the real arm to an object. Everything below is done by people, on site, with the
staff member responsible for the arm. Trova's software STOP is **not** safety-rated, and Trova has **no collision
avoidance**. The physical E-stop is the only stop you can rely on.

What the mode does (`TROVA_FIRST_RUN=1`, off by default):

- The find never moves the arm, even with `move=true`. The UI shows the photo with the chosen box, the target in mm,
  the height, the distance from the current pose, the speed and "Collision avoidance: NOT PRESENT. Hold the physical
  E-stop."
- **Stage 1** (one confirmation) moves the tip straight to `TROVA_ARM_Z_MM + TROVA_FIRST_RUN_HOVER_MM` (default +100 mm)
  above the object. **Stage 2** (a new confirmation, issued only after stage 1 arrived) descends straight down to
  `TROVA_ARM_Z_MM`. Each confirmation can be used once, only for its own stage and target, within 60 s, and never
  after a STOP.
- Speed is at most 20 mm/s for every Cartesian move (`TROVA_FIRST_RUN_SPEED_MM_S` can only lower it).
- In this mode, only the two stages and **Arm home** can move the arm. Point, tours, tray lift, room scan and the
  watch arm are all refused.
- After each stage, the audit log (`arm_approach`) records the target and the pose the controller reports. After
  stage 2 you type the ruler error. Anything above `TROVA_ARM_MISS_MM` (default 10 mm) is also logged as `arm_missed`
  in the failure log.

## 1. Bring-up and controller safety settings (staff)

```bash
.venv/bin/python -m trova.arm.bringup <xarm ip> --camera-on-arm
```

Follow its steps with staff at the E-stop. Then, in UFACTORY Studio, **with staff**, set these and write down the
values you chose:

- [ ] **Reduced mode ON**, with a reduced TCP speed limit of ≤ 50 mm/s. This also limits the joint-space homing
      move, which Trova's mm/s cap does not cover.
- [ ] **Collision sensitivity** high (as high as staff accept without false trips).
- [ ] **Safety boundary** (controller fence) ON, at or just outside `TROVA_ARM_LIMITS`. Step 1 of the bring-up
      prints whether it is on.
- [ ] TCP offset of the pointer set (or `TROVA_ARM_TCP_MM`).
- [ ] The physical E-stop is tested: pressed, the arm stops, then released and recovered by staff.

## 2. Measure and set the environment (you)

Put these in `.env`. Measure each value on the real arm; never copy them from an example.

```
TROVA_ARM=xarm:<ip>
TROVA_ARM_LIMITS=xmin,xmax,ymin,ymax,zmin,zmax     # from the bring-up; zmax must be >= TROVA_ARM_Z_MM + hover
TROVA_ARM_Z_MM=                                     # pointing height: tip 30-50 mm above the tallest object
TROVA_ARM_TCP_MM=0,0,<pointer length>
TROVA_CAMERA_ON_ARM=1
TROVA_CAMERA=<index from python -m trova.camera --list>
TROVA_ARM_JOINT_SPEED_DEG_S=5                       # slow homing for the first run
TROVA_OPERATOR_NAME=<your name>
TROVA_OPERATOR_TOKEN=<a long random secret>
TROVA_TOKEN=<secret>                                # venue Wi-Fi
TROVA_FIRST_RUN=1
# optional: TROVA_FIRST_RUN_SPEED_MM_S=10  TROVA_FIRST_RUN_HOVER_MM=100  TROVA_ARM_MISS_MM=10
```

- [ ] Check that `zmax ≥ TROVA_ARM_Z_MM + 100`. The bring-up sets `zmax` to the higher of (home z, Z) + 80 mm, so the
      default hover can fall **outside** the box. The UI then shows "Safety box: … outside" and refuses. Either raise
      `zmax` (only if the space above is clear, with staff) or set `TROVA_FIRST_RUN_HOVER_MM` lower (e.g. 60).
- [ ] Start Trova. Every physical boot is disarmed. Rearm only after you check that the area is clear:

```bash
NONCE=$(curl -s "localhost:8000/api/state?token=$TROVA_TOKEN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["stop_nonce"])')
curl -s -X POST "localhost:8000/api/arm/reset?token=$TROVA_TOKEN" -H "x-trova-operator-token: $TROVA_OPERATOR_TOKEN" \
  -H 'content-type: application/json' \
  -d "{\"operator\":\"$TROVA_OPERATOR_NAME\",\"stop_nonce\":\"$NONCE\",\"area_clear\":true,\"controller_ready\":true}"
```

## 3. Survey pose and calibration (you, arm in manual mode)

- [ ] Move the arm by hand until the camera sees the whole table. Then Setup → **Teach survey pose here**.
- [ ] Touch the centre of each ArUco marker (0-3) with the tip. Use **Record marker from arm** for each one.
- [ ] Return to the survey pose (**Arm home**). Then **Calibrate (ArUco)**. The result must show `source
      aruco_robot`, an error ≤ 8 mm and a `pose` equal to the survey pose.
- [ ] Teaching a new survey pose deletes the calibration. If you re-teach, calibrate again.

## 4. Preflight

- [ ] Header → **preflight** (or `python -m trova.preflight`) until there are **no failures**. The "First run" item
      must say *ready*. It checks: calibration made with the arm at the current survey pose and image size, safety
      box, home pose, operator credentials, and STOP not latched.

## 5. Dry test: no motion

- [ ] **Capture empty table**. Put one object down and Find it. The arm does not move; the panel shows the target.
- [ ] Measure the object's centre in robot coordinates. Touch it with the tip in manual mode and read the pose on
      the pendant, or measure with a ruler from a marker. Compare it with the target shown in mm. If it is more than
      10 mm off, **stop here** and recalibrate.

## 6. First move: stage 1 (hover) only

- [ ] Area clear. A second person watches the arm, not the screen. Your hand is **on the E-stop**.
- [ ] Read the panel: the target, the height, the distance from the current pose and the speed. Then **Confirm
      stage 1**.
- [ ] The tip ends about 100 mm above the object. Check that the audit line `arm_approach` (stage 1) shows `reached`
      and an `offset_mm` of a few mm.

## 7. Stage 2, measure, repeat

- [ ] Read the new panel: the height is `TROVA_ARM_Z_MM`, about 100 mm down. **Confirm stage 2**.
- [ ] Measure the horizontal tip-to-object-centre error with a ruler. Type it in and press **Record error**.
- [ ] Press **Arm home**. This is a joint move from pointing height back to the survey pose. **It does not rise
      first: watch the path**, E-stop in hand.
- [ ] Repeat with **5 objects at different spots**: corners and centre of the work area. Note the errors:

| # | object | target x,y (mm) | controller reported (mm) | ruler error (mm) |
|---|--------|-----------------|--------------------------|------------------|
| 1 |        |                 |                          |                  |
| 2 |        |                 |                          |                  |
| 3 |        |                 |                          |                  |
| 4 |        |                 |                          |                  |
| 5 |        |                 |                          |                  |

The same data is in **history** (group "arm": `arm_approach`, `arm_accuracy`) and in `GET /api/failures`.

## 8. Stop criteria

**Hit the E-stop immediately if:**

- the arm starts moving in a direction you did not expect, or toward a person, object or cable;
- it moves noticeably faster than 20 mm/s;
- in stage 2 it moves sideways instead of straight down;
- anyone enters the area, or the second person says stop;
- the homing move swings toward the table or the camera cable.

**Abandon the session (no more moves today, debrief) if:**

- the E-stop was needed once because of an unexpected motion, until the cause is understood;
- a ruler error is above 20 mm, or two errors are above 10 mm (calibration, TCP offset or survey pose is wrong);
- the controller reports a pose more than 10 mm from the target (`offset_mm` in the audit), or reports collision or
  protective-stop errors that you cannot explain;
- the dry-test target and the ruler disagree by more than 10 mm after one recalibration;
- software STOP does not hold the arm (it keeps moving after STOP).

After any E-stop: staff recover the controller, you check the area, then use the software reset (step 2). Old
confirmations are dead; find again.
