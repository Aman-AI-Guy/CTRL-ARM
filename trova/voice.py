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
            "riferimento", "reference", "cosa è cambiato", "what changed"]


def instructions(confirm_moves: bool, dog: bool) -> str:
    move = ("Before the arm moves, ask one short yes/no question to confirm, and call point_at_found only after a "
            "clear yes (si, yes, ok, vai, go). If the person already said 'point at it' or 'show me', that counts as yes."
            if confirm_moves else "When an object is found, call point_at_found straight away.")
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
- find_all: "point at all the pens", "show me every red thing". Say how many you found; then, if they want them
  shown, call point_at_found (it visits each one in turn).
- describe_table: "what's on the table?", "cosa c'è sul tavolo?". Read the summary, then list names only if asked.
- save_reference: when they say the table is in its approved / clean state.
- check_changes: "what changed?", "is the line clear?". Say how many changes, then each one in a few words, highest risk first.
- stop_arm: whenever they say stop, wait, fermo, basta. The app also stops the arm on its own when it hears these words.
- arm_home: "park", "go home", "torna a casa".
- {dog_line}

# Safety
The arm only points; it never grabs. If a tool returns an error or "blocked", say so and do not retry by yourself."""


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
         "description": "Name every object on the table and summarise it in one sentence.",
         "parameters": {"type": "object", "properties": {}}},
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
        {"type": "function", "name": "stop_arm", "description": "Stop the arm immediately and hold position.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "arm_home", "description": "Park the arm out of the camera's view.",
         "parameters": {"type": "object", "properties": {}}},
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
