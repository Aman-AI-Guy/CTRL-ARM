# ctrl-arm pitch: 5 minutes = 1:52 ad + 3:08 live

Video: [`media/ctrl_arm_ad.mp4`](../media/ctrl_arm_ad.mp4) (1080p, 112 s, sound on: the robot speaks Italian).
Rebuild: see "Rebuilding the video" below. Slides with the video as slide 1 and the full speaker script: the
ctrl-arm pitch deck (Claude artifact; ask Bhoumik to share it).

## Run of show

| Time | Who | What |
|---|---|---|
| 0:00–1:52 | nobody speaks | The ad plays (sound on). It ends on “Now — live.” |
| 1:52–2:10 | Presenter | The problem: line clearance by eye, one leftover = a recall, Annex 22 says AI only assists |
| 2:10–3:30 | Demo driver + arm operator | LIVE: 1 “Trovami il mouse rosso” → “Confermi?” → “Sì” → the arm goes over it · 2 what changed (add an object, NOT CLEAR, sign, PDF) · 3 say “stop” |
| 3:30–3:50 | Presenter | How it works: Grok decides what you mean, code decides where the arm goes |
| 3:50–4:10 | Presenter | What runs today: 50+ commits, 260+ tests, 84 objects in one 360° scan, 1.4 mm replay median, 3D mesh, IT/EN |
| 4:10–4:30 | Presenter | Market: $15.8B machine vision, $1.1B pharma inspection, 7,000+ SEA Vision systems |
| 4:30–4:45 | Presenter | Ask: a 30-day pilot on one SEA Vision line |
| 4:45–5:00 | Presenter | “ctrl-arm: find anything, for real.” Thank you. |

## What the video shows (and where it comes from)

| Scene | Source |
|---|---|
| ctrl + F works on a document, not on the desk | the hook |
| 360° table scan: 12 photos, 84 objects, live map | the real run of 5 Nov 13:31 (`docs/TABLE_SCAN_FINAL.md`), real Pi-camera frames |
| “Trovami il mouse rosso” → “Numero 7… Confermi?” → “Sì” → over it at the same height | the real voice flow in `trova/server.py` (`_table_go_to`), real top-down and approach photos |
| STOP: the arm freezes mid-move | `trova/safety.py`, STOP latch |
| What changed: NOT CLEAR, the black cap | real before/after frames from the Pi camera |
| 3D: 12-view 220° orbit at 350 mm, the mesh of the mouse, “Giralo, per favore” | `arm_pc/runs/*_scan_plan.json`, `docs/SCANNING_RECONSTRUCTION.md` (the flip request is real) |
| A second arm turns the object | **NEXT**: not built, labelled as such in the video |
| Ceiling-mounted arm scans whatever you put on the desk while you work | **vision**: the scale pitch, not built |
| Market numbers | MarketsandMarkets 2025, Research and Markets 2025, Marchesini / SEA Vision 2025 |

The “0.94 · 1.8 s” style numbers of the old video are gone; the only figures on screen are the real ones above.

## Rules that win hackathon demos

1. **Hook in the first 10 seconds** (the ad does it), then the problem in one sentence.
2. **The demo is the heart.** Show the 3 strongest moments, not every feature.
3. **Only run what you rehearsed today.** Reset /jury; run preflight first; re-arm after every STOP.
4. **Backup ready:** if the arm or Wi-Fi fails, say “here is the run from this morning” and play the clip. Never debug on stage.
5. **Say the jury's metrics out loud:** objects found out of 10, time to find, false alarms; keep /jury on screen.
6. **Rehearse 5 times with a timer.** Cut words, not the demo.
7. **End with the ask**, then stop talking.

## Likely questions

- *Why not YOLO?* No training; Grok understands “qualcosa per scrivere”, photos, and asks back when two objects fit.
- *Is it safe?* Grok never moves the arm. Every move passes one gate (workspace, speed at the tool, IK, clearance); STOP latches; the operator confirms each move.
- *How accurate?* Calibration replay median 1.4 mm (p95 6.2 mm); today's "over it" move missed by 7–13 cm on a partially-seen object, and we know why (box cut by the photo edge). The live scoreboard is the real number.
- *Regulation?* Draft EU GMP Annex 22: AI assists, the operator signs; every decision is in a SHA-256-chained audit log.
- *The ceiling arm?* A vision of scale: same software, a better camera, a second arm to turn the object. Not built.

## Rebuilding the video (media/ad)

`ad.html` is the whole ad as a GSAP timeline (open it in Chrome to watch it live). `voice.py` makes the robot's
Italian lines with macOS `say` (voice Alice) and processes them with ffmpeg; `audio.py` synthesises the music and
mixes the voice in, ducked; `render.py frames <k> 4` renders frame k, k+4, … at 30 fps in headless Chrome (run 4
at once), then:
`ffmpeg -framerate 30 -i frames/%05d.jpg -i soundtrack.wav -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest ctrl_arm_ad.mp4`.
The photos in `assets/` are real frames from the arm's Pi camera.
