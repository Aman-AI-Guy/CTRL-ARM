"""Grok Voice (realtime speech-to-speech) for Trova: talk to the workstation in English or Italian.

The browser connects straight to wss://api.x.ai/v1/realtime with a short-lived token minted here
(POST https://api.x.ai/v1/realtime/client_secrets), so the API key never reaches the browser.
Grok Voice understands the request and calls our tools; plain code does the finding, the moving
and the stopping. Safety: the browser also watches the live transcript and halts the arm the moment
it hears "stop" / "fermo" / "basta", without waiting for the model (see static/voice.js).
"""

from __future__ import annotations

import requests

CLIENT_SECRETS_URL = "https://api.x.ai/v1/realtime/client_secrets"

STOP_WORDS = ["stop", "halt", "fermo", "ferma", "fermati", "basta", "alt"]

KEYTERMS = ["Trova", "stop", "fermo", "basta", "blister", "forbici", "penna", "pennarello", "tazza", "nastro",
            "graffettatrice", "chiavetta USB", "scissors", "marker", "stapler", "mug", "tape", "line clearance",
            "riferimento", "reference", "cosa è cambiato", "what changed", "scansiona il tavolo", "bicchiere",
            "lattina", "camera sopra", "procedi"]


def instructions(confirm_moves: bool, dog: bool) -> str:
    move = ("Object selection never authorizes movement. For object-directed motion always call run_vision_task "
            "to plan first, even when confirmation is unchecked. Never announce an impending move before a tool "
            "reports movement_executing. Ask for confirmation only for awaiting_operator_confirmation, quoting "
            "its proposal id internally. On an explicit yes call confirm_motion with that exact proposal_id. "
            "A request to move is not confirmation of a future proposal. Never fabricate a proposal id.")
    dog_line = ("If find_object returns not_found, offer to send the robot dog to look in the other rooms and, on a yes, "
                "call search_other_rooms." if dog else "There is no robot dog in this setup; never offer one.")
    return f"""# Role
You are Trova, the voice of a robot workstation at a packaging line. A camera sees the table; a robot arm can point at objects.

# Language
Reply in the language the person speaks: English or Italian. Switch if they switch.

# Style
Short and plain: one or two sentences. No coordinates or numbers unless asked. Never describe what is on the table from memory: always use the tools.

# Tools
- find_object: for any request to find, show or locate something. Pass the person's words as they said them.
  If the result is "clarify", ask its clarifying_question and pass the answer to answer_clarification.
  If it is "not_found", say so plainly; never guess.
- {move}
- find_all: "point at all the pens", "show me every red thing". It lists matches without motion. A voice tour does not yet have a bound multi-object proposal; ask the operator to select one target and request an approach for it. Never imply that all objects are being visited.
- describe_table: "what's on the table?", "cosa c'è sul tavolo?", "fai una foto e dimmi quali oggetti vedi". Read the summary. Mention track ids only if asked. This does not move the arm.
- select_object: the person picks one listed object ("the second blue pen", "oggetto 3", "seleziona la penna blu"). Pass their words as description. It never moves the arm. If image localization is uncertain, tell the operator to use "Select image region" to click an existing box or draw a box on the frozen snapshot. Recognition alone is not a pixel region; pixel regions are not metric robot positions. If the result is "clarify", ask its question and call select_object again with the answer. If authoritative is false, say the names are uncertain and do not offer to move.
- save_reference: when they say the table is in its approved / clean state.
- check_changes: "what changed?", "is the line clear?". Say how many changes, then each one in a few words, highest risk first.
- stop_arm: whenever they say stop, wait, fermo, basta. The app also stops the arm on its own when it hears these words.
- arm_home: "park", "go home", "torna a casa".
- jog_relative: a short relative move such as "move backward 3 mm" or "sposta a sinistra di 2 cm". Pass the direction and the distance in millimetres exactly as heard. You cannot set the motion mode, the speed, or the workspace. Say the tool's reason, including when the move was rejected.
- run_vision_task: "find the blue pen and move above it", "trova la penna blu e posizionati sopra", "ora posizionati sopra". Pass the person's words as text. The backend retains the selected object. Pass "scansionalo" and "seleziona il mouse rosso e crea un modello 3D" here too: it must report scan readiness honestly. If it returns held, blocked, or rejected, use spoken_message. Do not retry a rejected move with a different distance or direction. This tool cannot set the motion mode or a safety limit.
- confirm_motion: only after the user explicitly confirms a specific pending validated proposal. Missing authorization, expiration, changed pose or invalid calibration refuse execution. A conversational yes cannot enable hardware.
- cancel_task: cancel autonomous progression and invalidate confirmation. stop_arm is independent and immediate.
- task_status: read actual task state. Use spoken_message verbatim. Simulation is never physical movement; task completion is never proof of a reached physical goal. The browser speaks backend messages, not speculative model audio.
- For "scansionalo" or "scan it", call run_vision_task; the backend chooses a useful reachable view and requires a new proposal confirmation. For "ricostruisci" or "reconstruct it", the same tool starts a separate local reconstruction job. If COLMAP or suitable images are missing, report the actual failure. Never claim a mesh before a job reports reconstructed true, or texture before textured_mesh true. If the operator says "l'ho capovolto" or "I flipped it", pass those words to run_vision_task: old localization is invalidated and selection is required again. Cross-orientation registration and complete surface coverage are not implemented.
- scan_requests: after a reconstruction, or when the operator asks "cosa devo fare?", "what do you need from me?". Read the first pending request's instruction as written, then wait. When the operator says they did it ("fatto", "done"), call scan_request_done with its request_id and status done, then say the returned spoken_message; if they cannot, use status skipped. If complete is true, say nothing more is needed from them, but never say the mesh is closed or complete: surface coverage is not measured.
- {dog_line}

# Table scan (camera on the arm, V2 calibration)
- scan_table: "scansiona il tavolo", "guarda il tavolo", "look around the table", "what's on the whole table?". The arm
  rises to 60 cm, tilts the camera down at the table and turns to photograph it; every object is named and placed.
  It takes a minute or two: say once that you are scanning, then wait for the result and read its spoken_message.
  If it returns refused or error, say the reason. move is false only if they ask to label the current view without moving.
- find_on_table: after a scan, any request to find one object ("trova la tazza rossa", "dov'è il bicchiere?",
  "trovami le pinze", "indicami le pinze", "mettiti sopra le pinze"). Pass their words. If status is "clarify", read
  the question exactly (it lists the numbers and where each one is), then pass their answer to choose_table_object;
  they can also tap the object in the interface. When it (or choose_table_object) returns a proposal_id with
  approach "awaiting_confirmation", the move over the object at the same height is already planned: read
  spoken_message (it ends with "Confermi?") and on their yes call confirm_camera_move with that proposal_id directly.
  If approach is "moving", the operator enabled automatic moves: say it is moving.
- choose_table_object: the operator picks one candidate ("il numero 3", "quello a sinistra del robot").
  Pass object_id when they say a number, otherwise pass their words as answer.
- move_camera_above: "porta la camera sopra", "vai sopra", "move the camera above it". It only plans and returns a
  proposal; read its spoken_message, which asks for confirmation. Never say you are moving yet.
- confirm_camera_move: only after an explicit yes ("sì", "procedi", "yes") to that proposal, with its proposal_id.
  A refused confirmation is not retried. If they say no, do nothing.

# Safety
The arm only observes and points; it never grabs. describe_table and select_object never move it. point_at_found also routes to the task planner; it cannot bypass proposal confirmation. jog_relative and run_vision_task are checked by the motion gateway and may be refused. You cannot turn on hardware or change a safety limit. stop_arm holds position in software; it is not the physical emergency stop. If a tool returns an error, "rejected", "held", or "blocked", say the reason and do not retry by yourself. Never infer success from ok alone; read workflow_state, hardware_moved, simulation_moved, goal_achieved and spoken_message."""


def tools(dog: bool) -> list[dict]:
    t = [
        {"type": "function", "name": "find_object",
         "description": "Find an object on the table from a description in English or Italian, e.g. 'something to write with'.",
         "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
        {"type": "function", "name": "answer_clarification",
         "description": "Pass the person's answer to the clarifying question from find_object.",
         "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}},
        {"type": "function", "name": "point_at_found",
         "description": "Move the arm tip above the object found by the last find_object.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "find_all",
         "description": "Find EVERY object matching a description (e.g. 'all the pens'); point_at_found then visits each.",
         "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
        {"type": "function", "name": "describe_table",
         "description": "Take a snapshot, name every object, and summarise it in one sentence. Does not move the arm.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "select_object",
         "description": "Select one object from the latest snapshot by description or track id. Does not move the arm. If status is clarify, ask the question and call this again with the answer.",
         "parameters": {"type": "object", "properties": {"description": {"type": "string"},
                                                        "track_id": {"type": "integer"}}}},
        {"type": "function", "name": "save_reference",
         "description": "Save the current table as the approved reference state.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "check_changes",
         "description": "Compare the table with the reference: what was added, removed or moved, and why it matters.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "look_around",
         "description": "Turn the arm's camera around the room to find an object that is not on the table, then face it.",
         "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
        {"type": "function", "name": "read_labels",
         "description": "Read the label, lot and expiry of every object and check them against the current batch.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "watch_line",
         "description": "Start or stop watching the cleared line continuously; an alarm sounds if anything changes.",
         "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}},
        {"type": "function", "name": "run_vision_task",
         "description": "Run an object-directed task from the person's words, such as finding an object and moving above it. "
                        "A rejected move is not retried. You cannot pass a mode, a speed, or a limit.",
         "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
        {"type": "function", "name": "confirm_motion",
         "description": "Confirm the exact validated proposal only after the operator explicitly says yes.",
         "parameters": {"type": "object", "properties": {"proposal_id": {"type": "string"}}, "required": ["proposal_id"]}},
        {"type": "function", "name": "cancel_task", "description": "Cancel the autonomous task and its pending confirmation.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "task_status", "description": "Read backend task state and its spoken_message; never assume motion success.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "scan_requests",
         "description": "Read what Grok, after looking at the reconstruction renders, asks the operator to change on the "
                        "object (turn it over, rotate it, move a cable away) so the scan can be completed. Never moves the arm.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "scan_request_done",
         "description": "Record that the operator did one scan request (done) or cannot do it (skipped). A done turn or flip "
                        "opens a new orientation group and the object must be selected again.",
         "parameters": {"type": "object", "properties": {
             "request_id": {"type": "string"},
             "status": {"type": "string", "enum": ["done", "skipped"]}}, "required": ["request_id", "status"]}},
        {"type": "function", "name": "jog_relative",
         "description": "Request a short relative move in the robot base frame. The gateway may reject it. "
                        "You cannot pass a mode, a speed, or a limit. Say the returned reason.",
         "parameters": {"type": "object", "properties": {
             "direction": {"type": "string", "enum": ["forward", "backward", "left", "right", "up", "down"]},
             "distance_mm": {"type": "number"}}, "required": ["direction", "distance_mm"]}},
        {"type": "function", "name": "stop_arm",
         "description": "Hold the arm where it is. This is a software stop, not the physical emergency stop.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "arm_home", "description": "Park the arm out of the camera's view.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "scan_table",
         "description": "Rise to 60 cm, tilt the arm camera down at the table, turn and photograph it, then name and "
                        "place every object (V2 calibration). Waits until the map is ready. move=false labels the "
                        "current view without moving.",
         "parameters": {"type": "object", "properties": {"move": {"type": "boolean"}}}},
        {"type": "function", "name": "find_on_table",
         "description": "Find one object in the last table scan from the person's words. If several match, status is "
                        "clarify and question lists them by number and position.",
         "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
        {"type": "function", "name": "choose_table_object",
         "description": "The operator's choice after a clarify question: object_id if they said a number, else answer.",
         "parameters": {"type": "object", "properties": {"object_id": {"type": "integer"}, "answer": {"type": "string"}}}},
        {"type": "function", "name": "move_camera_above",
         "description": "Plan placing the arm camera above the chosen table-scan object. Returns a proposal that needs "
                        "an explicit yes; it does not move the arm.",
         "parameters": {"type": "object", "properties": {"object_id": {"type": "integer"}}}},
        {"type": "function", "name": "confirm_camera_move",
         "description": "Execute the camera-above proposal, only after the operator explicitly said yes to it.",
         "parameters": {"type": "object", "properties": {"proposal_id": {"type": "string"}}, "required": ["proposal_id"]}},
    ]
    if dog:
        t.append({"type": "function", "name": "search_other_rooms",
                  "description": "Send the robot dog along its route to look for the object in the other rooms.",
                  "parameters": {"type": "object", "properties": {"description": {"type": "string"}},
                                 "required": ["description"]}})
    return t


def session_config(voice: str, confirm_moves: bool, dog: bool) -> dict:
    """The session.update payload the browser sends after connecting."""
    return {
        "voice": voice,
        "instructions": instructions(confirm_moves, dog),
        "turn_detection": {"type": "server_vad"},
        "reasoning": {"effort": "none"},   # default is "high": slower turns; tools carry the hard part
        "tools": tools(dog),
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": 24000},
                      "transcription": {"model": "grok-transcribe", "keyterms": KEYTERMS}},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
    }


def mint_token(api_key: str, seconds: int = 300) -> dict:
    r = requests.post(CLIENT_SECRETS_URL, json={"expires_after": {"seconds": seconds}}, timeout=15,
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"xAI client_secrets HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    token = d.get("value") or (d.get("client_secret") or {}).get("value") or d.get("token") or d.get("secret")
    if not token:
        raise RuntimeError(f"No token in xAI response: {list(d)}")
    return {"token": token, "expires_at": d.get("expires_at") or (d.get("client_secret") or {}).get("expires_at")}
