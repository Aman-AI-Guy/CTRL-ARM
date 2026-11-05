# ctrl-arm pitch: speaker notes

One section per slide of `ctrl_arm_pitch.pdf`. Times are for the 5-minute slot (video 0:00–1:52).

## Slide 1 · ad

RUN OF SHOW (5:00 total). 0:00–1:52 this video plays by itself with sound — nobody speaks (the robot’s Italian voice is in the video). It ends on “Now — live.” 1:52 Presenter steps in on the next slide. 2:10–3:30 live demo (80 s). 3:30 how it works, 3:50 proof, 4:10 market, 4:30 ask, 4:45 close.

ROLES: Presenter (story, slides 2, 4–8). Demo driver at the laptop (Trova console + /jury scoreboard). Arm operator at the e-stop, confirms each stage. Backup: has this video and a recorded demo clip ready if the arm or Wi-Fi fails. Q&A lead: answers technical questions.

BEFORE WE START: run preflight, reset the /jury score, put the Red Bull can and the test objects on the table, arm at its survey pose, sound on, this slide selected so the clip starts with the slide.

## Slide 2 · problem

PRESENTER · 18 s · 1:52–2:10
“You can CTRL+F a document. You can’t CTRL+F a room — and in pharma that costs money. After every batch, an operator walks the packaging line looking for anything left over from the last batch: one wrong carton or label means a mix-up and a recall. It’s slow, it leaves no proof, and the new EU rules say AI can help, but a person must decide. So we built ctrl-arm. Let us show you — live.”
Hand over to the demo driver.

## Slide 3 · demo

DEMO DRIVER (+ arm operator) · 80 s · 2:10–3:30. Switch the projector to the console; keep /jury on a second screen if there is one.
1 · FIND (35 s): say “Trovami il mouse rosso” (or the Red Bull can, whichever you rehearsed) → point at the numbered boxes and Grok’s choice → the robot answers in Italian and asks “Confermi?” → say “Sì” → the arm goes over it at the same height. Say: “Grok decided which object; plain code decided where the arm goes.” If the 360° scan is rehearsed and under a minute, you may open with “Scansiona tavolo” instead — only if it ran clean today.
2 · WHAT CHANGED (30 s): Save reference → a judge or teammate adds an object → Check now → read the verdict and Grok’s reason aloud → operator name → Sign off → open the PDF record for 2 seconds.
3 · STOP (15 s): start a move, say “stop” (or press space / Escape) → the arm holds.
ONLY RUN WHAT WE REHEARSED TODAY. If the arm or Wi-Fi fails: say “here is the run from this morning” and play the backup clip; never debug on stage.

## Slide 4 · how

PRESENTER · 20 s · 3:30–3:50
“How does it work? A Raspberry Pi camera rides on the xArm. Plain code puts a number on every object — no training, no dataset. Grok looks at the numbered picture and decides which one you meant, asks back when two fit, and explains changes. Then plain code takes over: a hand–eye calibration turns the pixel into robot millimetres, and every move goes through one motion gate — workspace, speed, inverse kinematics, cable clearance, and a STOP that never waits for a model. Grok never moves the arm.”

## Slide 5 · proof

PRESENTER · 20 s · 3:50–4:10
“All of this was built here in 36 hours: more than fifty commits from five of us, more than 260 automated tests. This morning the real arm turned once and mapped 84 objects from 12 photos in about a minute. The calibration lands at about one and a half millimetres median on replayed station data. The same camera can orbit an object and rebuild it in 3D — this mouse came out at 104 by 67 by 39 millimetres. And you can talk to it in Italian or English.”
Be precise if asked: the 1.4 mm is a replay of recorded data, not a fresh physical qualification; the 84-object scan is from today’s log (docs/TABLE_SCAN_FINAL.md); the jury’s live scoreboard is the real test.

## Slide 6 · market

PRESENTER · 20 s · 4:10–4:30
“Machine vision is a sixteen-billion-dollar market heading to twenty-four. Pharma inspection alone is over a billion and growing up to ten percent a year. And the channel is in this room: SEA Vision has more than seven thousand inspection systems installed. Trova is software per line, on the cameras and cobots a plant already owns — and the same skill works in warehouses, labs and hospitals.”

## Slide 7 · ask

PRESENTER · 15 s · 4:30–4:45
“Our ask is simple: thirty days on one SEA Vision packaging line. Week one we mount the camera and approve a reference. Weeks two and three Trova runs next to your operator and we count what it catches and every false alarm. Week four we compare time and errors with manual clearance.”

## Slide 8 · close

PRESENTER · 15 s · 4:45–5:00
“ctrl-arm: find anything, for real. Ask in words, it finds, it points, it proves what changed — and a person always decides. Thank you.” Stop talking. Smile. Wait for questions.
LIKELY QUESTIONS: Why not YOLO? (No training; Grok understands requests like “something to write with” and asks back when two objects fit.) Is it safe? (Grok never moves the arm; one motion gate; STOP latches; operator confirms each stage.) Accuracy? (Replay median 1.4 mm; live scoreboard today.) Cost? (Pi camera plus the cobot a plant already has; software per line.) Regulation? (Annex 22: AI assists, operator signs, audit trail with a SHA-256 chain.)
