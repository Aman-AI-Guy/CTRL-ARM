# Trova pitch: 5 minutes = 83 s ad + 3 min 37 s live

Video: [`media/trova_ad.mp4`](../media/trova_ad.mp4) (1080p, 83 s, sound on). Rebuild: `media/ad/README` below.
Slides with the video as slide 1 and the full speaker script: the Trova pitch deck (Claude artifact; ask Bhoumik to share it).

## Run of show

| Time | Who | What |
|---|---|---|
| 0:00–1:23 | nobody speaks | The ad plays (sound on). It ends on “Now — live.” |
| 1:23–1:45 | Presenter | The problem: line clearance by eye, one leftover = a recall, Annex 22 says AI only assists |
| 1:45–3:15 | Demo driver + arm operator | LIVE: 1 find the Red Bull can (aim, then approach, operator confirms each stage) · 2 what changed (add an object, NOT CLEAR, sign, PDF) · 3 say “stop” |
| 3:15–3:40 | Presenter | How it works: Grok decides what you mean, code decides where the arm goes |
| 3:40–4:05 | Presenter | What runs today: 53 commits, 260+ tests, 1.4 mm replay median, 3D mesh, 0 false alarms (synthetic), EN/IT |
| 4:05–4:30 | Presenter | Market: $15.8B machine vision, $1.1B pharma inspection, 7,000+ SEA Vision systems |
| 4:30–4:45 | Presenter | Ask: a 30-day pilot on one SEA Vision line |
| 4:45–5:00 | Presenter | “Trova: CTRL+F for the physical world.” Thank you. |

## Rules that win hackathon demos

1. **Hook in the first 10 seconds** (the ad does it), then the problem in one sentence.
2. **The demo is half the time.** Show the 3 strongest moments, not every feature.
3. **Only run what you rehearsed today.** Hard-code the demo path; reset /jury; run preflight first.
4. **Backup ready:** if the arm or Wi-Fi fails, say “here is the run from this morning” and play the clip. Never debug on stage.
5. **Say the jury's metrics out loud:** objects found out of 10, time to find, false alarms; keep /jury on screen.
6. **Rehearse 5 times with a timer.** Cut words, not the demo.
7. **End with the ask**, then stop talking.

## Likely questions

- *Why not YOLO?* No training; Grok understands “something to write with”, photos, Italian, and asks back when two objects fit.
- *Is it safe?* Grok never moves the arm. Every move passes one motion gate (workspace, speed, IK, clearance); STOP latches; the operator confirms each stage.
- *How accurate?* Calibration replay median 1.4 mm (p95 6.2 mm); the live scoreboard is today’s real number.
- *Regulation?* Draft EU GMP Annex 22: AI assists, the operator signs; every decision is in a SHA-256-chained audit log.

## Rebuilding the video (media/ad)

`ad.html` is the whole ad as a GSAP timeline (open it in Chrome to watch it live). `render.py frames <k> 4` renders
frame k, k+4, … at 30 fps in headless Chrome (run 4 at once), `audio.py` synthesises the soundtrack, then:
`ffmpeg -framerate 30 -i frames/%05d.jpg -i soundtrack.wav -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest trova_ad.mp4`.
The photos in `assets/` are real frames from the arm's Pi camera.
