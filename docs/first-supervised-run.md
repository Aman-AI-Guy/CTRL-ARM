# First supervised run on the xArm 5: find → hover → descend → measure

This is the first time Trova moves the real arm to an object. Everything below is done by people, on site, with the
staff member responsible for the arm. Trova's software STOP is **not** safety-rated, and Trova has **no collision
avoidance**. The physical E-stop is the stop of last resort. The xArm manual says it "should not be used as a risk
reduction measure": the real protection is slow speed, reduced mode, the safety boundary, collision detection and a
clear area.

Checked against the UFACTORY xArm 5 user manual (manualslib 1893450). The manual's page numbers are given below.

What the mode does (`TROVA_FIRST_RUN=1`, off by default):

- The find never moves the arm, even with `move=true`. The UI shows the photo with the chosen box, the target in mm,
  the height, the distance from the current pose, the speed and "Collision avoidance: NOT PRESENT. Hold the physical
  E-stop."
- **Stage 0** (one confirmation, only if the flange is tilted at the survey pose, as with the sideways camera) levels
  the flange **in place**. Only joint 4 turns, to J4 = −(J2 + J3) (manual p. 156), at ≤ 5°/s. The controller's own
  forward kinematics must confirm the result is level, and the end pose must be inside the safety box; otherwise
  stage 0 is refused. The camera dome and the tip swing on an arc around the wrist, and the panel shows where the tip
  ends up. Stage 0 exists because the xArm 5 moves in straight lines only with the flange level (roll ±180°,
  pitch 0). From a tilted pose, the driver would make stage 1 a joint-space curve that the safety box does not
  check.
- **Stage 1** (one confirmation, from the survey pose or from the levelled pose) moves the tip in a **straight line** to
  `TROVA_ARM_Z_MM + TROVA_FIRST_RUN_HOVER_MM` (default +100 mm) above the object.
- **Stage 2** (a new confirmation, issued only after stage 1 arrived) descends straight down to
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

**Faster, and repeatable: set them with the script.** It never moves the arm. Without `--apply` it only prints the
current settings; with `--apply` it shows the changes, asks you to type YES, saves them on the controller and reads
them back:

```bash
.venv/bin/python -m trova.arm.xarm_config 192.168.1.111                        # read only
.venv/bin/python -m trova.arm.xarm_config 192.168.1.111 --apply \
    --limits <xmin,xmax,ymin,ymax,zmin,zmax from the bring-up> --payload-kg <weighed> --payload-cog <x,y,z> \
    [--tcp <x,y,z after the yaw test>]
```

It sets collision sensitivity 4, self-collision on, rebound off, reduced mode 50 mm/s and 20°/s, and the safety
boundary = the limits. The script does not cover mounting direction (check it is Floor), the TCP offset
measurement and yaw test, or the E-stop test: do those in Studio. Otherwise, follow the bring-up's steps with staff
at the E-stop, then in xArm Studio → Settings, **with staff**, set the items below and
write down each value. The manual gives no numeric values for the reduced-mode limits, so the numbers here are
suggestions to agree with staff.

- [ ] **Mounting direction** matches the real mount (Floor is the default; manual p. 82). A wrong value makes the arm
      misjudge gravity. That causes frequent collision stops and "uncontrolled motion" in manual mode.
- [ ] **TCP payload**: the mass and centre of gravity of everything on the flange: pointer, camera sphere and LED ring
      (manual p. 71). Weigh them. Wrong payload values cause false collision alarms.
- [ ] **TCP offset**: the pointer tip from the flange centre (manual p. 71), or set `TROVA_ARM_TCP_MM`. Use **one**
      of the two, not both.
- [ ] **Collision sensitivity** 4-5. The scale is 0-5, 0 is off, and higher stops on less extra torque. The manual
      says not to go below 3 (p. 67). Lower it only for false trips, and only after payload and mounting are confirmed
      right.
- [ ] **Self-collision detection** ON (manual p. 93).
- [ ] **Collision rebound**: decide with staff. If it is ON, the arm moves backward on its own after a collision. If
      it is OFF, the arm stays where the collision was detected (p. 93). For this run, OFF is the more predictable
      choice.
- [ ] **Reduced mode** ON (manual p. 81). It limits the maximum linear speed, the maximum **joint speed** and the joint
      range. Suggested: linear ≤ 50 mm/s and joint ≤ 20°/s. It also limits joint-space moves (stage 0 and Arm home).
      Trova caps those at 5°/s itself, but its 20 mm/s cap only applies to straight-line moves.
- [ ] **Safety boundary** ON (manual p. 80), at or just outside `TROVA_ARM_LIMITS`. It must include the survey pose
      and the hover height. If the TCP leaves it, the controller stops the arm. Step 1 of the bring-up prints whether
      it is on.
- [ ] **Test the E-stop**: press it, then recover. It cuts arm power within 300 ms, and the manual says the arm "will
      slightly brake and fall" (p. 20), so the tip **sags** a little. Recover by twisting the button in the arrow
      direction (the power light comes on), then enable the arm (p. 20).

## 1b. The physical setup (from the photo of the team's xArm 5)

- [ ] **The base is bolted or clamped** to the table, not just resting on the board under it.
- [ ] **There is a tip to measure from.** The flange carries only the camera dome. Fit a thin pointer, or pick one
      fixed point on the dome as the tip. Measure it from the flange centre for the TCP offset. Calibration, the
      dry test and every ruler error are measured to this point.
- [ ] **The camera's power cable is routed and secured** along the arm, with slack for a full turn of joint 1, so it
      cannot snag during homing. The camera is a Raspberry Pi Zero 2 W (IMX219). It sends pictures over **Wi-Fi**
      (`http://trova-cam.local:8080`), so this cable only powers it.
- [ ] **Network**: the Mac is cabled to the arm's control box **and** on the same Wi-Fi as the Pi. Check both:
      `ping trova-cam.local` and open `http://trova-cam.local:8080` (the live page). If the venue Wi-Fi blocks
      devices from seeing each other, use your own router or a phone hotspot for the Mac and the Pi.
- [ ] **Measure the camera delay.** Hold a phone running a stopwatch in front of the camera. Photograph the phone and
      the live page on the Mac together, and read the difference between the two times. Set
      `TROVA_CAMERA_SETTLE_S` = delay + 0.5 s (the default 0.6 s is probably too short over Wi-Fi). Then a find right
      after homing can never use a frame taken while the arm was still moving.
- [ ] **LED ring**: Trova cannot drive it (the Pi's GPIO drives it). Leave `TROVA_LED` empty. During the run, keep
      it either off or steady white from the Pi. Changing colours change the picture.
- [ ] **Nobody uses the Pi page's "scan" or settings during the run.** They change resolution and exposure. If the
      Pi gets hot it drops to 640×480. Trova then refuses to move the arm (the calibration image size no longer
      matches). Let it cool: check `http://trova-cam.local:8080/status.json` → `temp`, `protection`.
- [ ] **The reach circle is clear** (700 mm around the base). Move out the power strip, chargers, cups, the mouse and
      the boxes, so only the test objects remain. Keep the emergency-exit walkway outside the reach.
- [ ] **The E-stop is within the operator's reach**: control box button, or a remote button on a cable.

## 2. Measure and set the environment (you)

Put these in `.env`. Measure each value on the real arm; never copy them from an example.

```
TROVA_ARM=xarm:<ip>
TROVA_ARM_LIMITS=xmin,xmax,ymin,ymax,zmin,zmax     # from the bring-up; zmax must be >= TROVA_ARM_Z_MM + hover
TROVA_ARM_Z_MM=                                     # pointing height: tip >= 50 mm above the tallest object (E-stop sag)
TROVA_ARM_TCP_MM=0,0,<pointer length>
TROVA_CAMERA_ON_ARM=1
TROVA_CAMERA=http://trova-cam.local:8080/stream.mjpg   # the Pi on the arm, over Wi-Fi (1640x1232)
TROVA_CAMERA_SETTLE_S=                              # measured camera delay + 0.5 s
TROVA_ARM_JOINT_SPEED_DEG_S=5                       # slow homing for the first run
TROVA_OPERATOR_NAME=<your name>
TROVA_OPERATOR_TOKEN=<a long random secret>
TROVA_TOKEN=<secret>                                # venue Wi-Fi
TROVA_FIRST_RUN=1
# optional: TROVA_FIRST_RUN_SPEED_MM_S=10  TROVA_FIRST_RUN_HOVER_MM=100  TROVA_ARM_MISS_MM=10
```

- [ ] Survey pose and target spots: keep them **away from the base axis**. When the wrist comes close to the J1 axis,
      J1 can turn far too fast and control can be lost (singularity, manual p. 156). The manual gives no distance:
      agree on a keep-out radius around the base with staff and put no test objects there.
- [ ] Check that `zmax ≥ TROVA_ARM_Z_MM + 100`. The bring-up sets `zmax` to the higher of (home z, Z) + 80 mm, so the
      default hover can fall **outside** the box. The UI then shows "Safety box: … outside" and refuses. Either raise
      `zmax` (only if the space above is clear, with staff) or set `TROVA_FIRST_RUN_HOVER_MM` lower (e.g. 60).
- [ ] Start Trova. Every physical boot is disarmed. Check `/api/state` for `lift_interrupted` first. A fresh
      data directory has no lift record, so its state is unknown even when tray lifting is disabled. **Arm home and
      reset cannot clear this interlock.** Staff must physically check that the fork is free and there is no load.
      While STOP remains latched, recover it with the current nonce, authenticated operator token, both safety
      confirmations, `fork_freed=true` and a written note:

```bash
NONCE=$(curl -s "localhost:8000/api/state?token=$TROVA_TOKEN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["stop_nonce"])')
curl -s -X POST "localhost:8000/api/arm/recover_lift?token=$TROVA_TOKEN" -H "x-trova-operator-token: $TROVA_OPERATOR_TOKEN" \
  -H 'content-type: application/json' \
  -d "{\"operator\":\"$TROVA_OPERATOR_NAME\",\"stop_nonce\":\"$NONCE\",\"area_clear\":true,\"controller_ready\":true,\"fork_freed\":true,\"note\":\"Staff verified fork free and no load\"}"
```

Recovery never moves the arm and keeps STOP latched. It rotates the nonce. Fetch the new nonce below and reset
separately only after staff have checked that the area and controller are ready:

```bash
NONCE=$(curl -s "localhost:8000/api/state?token=$TROVA_TOKEN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["stop_nonce"])')
curl -s -X POST "localhost:8000/api/arm/reset?token=$TROVA_TOKEN" -H "x-trova-operator-token: $TROVA_OPERATOR_TOKEN" \
  -H 'content-type: application/json' \
  -d "{\"operator\":\"$TROVA_OPERATOR_NAME\",\"stop_nonce\":\"$NONCE\",\"area_clear\":true,\"controller_ready\":true}"
```

## 3. Survey pose and calibration (you, arm in manual mode)

- [ ] Move the arm by hand until the camera sees the whole table. Then Setup → **Teach survey pose here**.
- [ ] Print the ArUco markers **flat**, taped onto the same surface the objects stand on. Don't use the raised blocks
      or cube faces: a marker above the surface, seen at an angle, adds parallax error. All 5 must be visible
      from the survey pose.
- [ ] Place markers 0-3 near the corners and marker 4 near the centre. Four points fit exactly and cannot test
      calibration error. Trova requires at least five robot-frame points and rejects a maximum error above 15 mm.
- [ ] Touch the centre of each ArUco marker (0-4) with the tip. Use **Record marker from arm** for each one.
- [ ] Return to the survey pose (**Arm home**). Then **Calibrate (ArUco)**. The result must show `source
      aruco_robot`, a maximum per-point error ≤ 8 mm and a `pose` equal to the survey pose.
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

## 6. First moves: stage 0 (level) and stage 1 (hover) only

- [ ] Area clear. A second person watches the arm, not the screen. Your hand is **on the E-stop**. Nobody's hand is
      under the tip (it sags on an E-stop).
- [ ] If the panel shows **Stage 0**: read the joint 4 change and where the tip ends up. Check that nothing is under
      the dome's swing, then **Confirm stage 0**. Only the wrist turns, slowly; the rest of the arm must not move. If
      the panel says "cannot plan the approach … does not level the flange", the levelling formula does not fit
      this arm: stop and tell the developer. Don't work around it.
- [ ] Read the panel: the target, the height, the distance from the current pose and the speed. Then **Confirm
      stage 1**.
- [ ] The tip ends about 100 mm above the object. Check that the audit line `arm_approach` (stage 1) shows `reached`
      and an `offset_mm` of a few mm.

## 7. Stage 2, measure, repeat

- [ ] Read the new panel: the height is `TROVA_ARM_Z_MM`, about 100 mm down. **Confirm stage 2**. After stage 1 the
      flange is level, so this is a straight line down.
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
- the homing move swings toward the table or the camera cable;
- in stage 0 any joint other than the wrist (joint 4) moves, or the dome swings toward something;
- stage 1 does not move in a straight line.

**Abandon the session (no more moves today, debrief) if:**

- the E-stop was needed once because of an unexpected motion, until the cause is understood;
- a ruler error is above 20 mm, or two errors are above 10 mm (calibration, TCP offset or survey pose is wrong);
- the controller reports a pose more than 10 mm from the target (`offset_mm` in the audit), or reports collision or
  protective-stop errors that you cannot explain;
- the dry-test target and the ruler disagree by more than 10 mm after one recalibration;
- software STOP does not hold the arm (it keeps moving after STOP);
- the controller reports a joint error such as S11 overcurrent, S12 overspeed or S14 position command over-limit
  (manual p. 162). Recovery is an E-stop restart, but the cause has to be understood before any more moves;
- collision stops repeat with nothing touched. Check the payload and mounting direction; don't just lower the
  sensitivity.

After any E-stop: staff recover the controller, you check the area, then use the software reset (step 2). Old
confirmations are dead; find again.
