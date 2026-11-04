"""Convert typed or microphone instructions to constrained JSON for UFACTORY xArm 5.

This script NEVER actuates the robot. It emits one validated JSON object to stdout.

Install:
    pip install xai-sdk pydantic requests sounddevice

Configure:
    export XAI_API_KEY="..."   # macOS/Linux
    # PowerShell: $env:XAI_API_KEY="..."

Examples:
    python grok_to_xarm_json.py --text "Sposta il braccio a sinistra di 2 cm"
    python grok_to_xarm_json.py --mic
"""

import argparse
import io
import json
import os
import re
import sys
import wave
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RobotCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["move", "stop", "ignore"]
    direction: Literal[
        "left", "right", "up", "down", "forward", "backward", "none"
    ]
    distance_mm: int


SYSTEM_PROMPT = """You convert spoken Italian or English into ONE xArm 5 command.
Return data matching the schema exactly.
Allowed commands:
- move: one relative direction and explicit distance in millimetres
- stop: stop request; use direction=none and distance_mm=0
- ignore: unsupported, ambiguous, unsafe, or multi-step request;
  use direction=none and distance_mm=0
Directions: left, right, up, down, forward, backward.
Convert centimetres to millimetres (50 cm = 500 mm).
Keep the distance you heard, even when it is outside 1 to 500 mm.
Do not invent a distance or a reference frame, and do not replace an
out-of-range move with ignore: the motion gateway rejects it and reports
the requested millimetres. ignore is only for ambiguous or non-motion
requests (direction none, distance_mm 0).
An output is only a request for review, not permission to move hardware.
"""


def record_wav(seconds: float = 4.0) -> bytes:
    import sounddevice as sd

    sample_rate = 16000
    print(f"Recording for {seconds:g}s... speak now", file=sys.stderr)
    samples = sd.rec(
        int(sample_rate * seconds),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return audio.getvalue()


def transcribe_wav(wav_bytes: bytes, api_key: str) -> str:
    import requests

    response = requests.post(
        "https://api.x.ai/v1/stt",
        headers={"Authorization": f"Bearer {api_key}"},
        data={"model": "grok-voice-transcribe-2.0"},
        files={"file": ("command.wav", wav_bytes, "audio/wav")},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["text"].strip()


def grok_to_command(text: str, api_key: str) -> RobotCommand:
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    client = Client(api_key=api_key)
    chat = client.chat.create(
        model=os.getenv("XAI_MODEL", "grok-4.7"),
        response_format=RobotCommand,
    )
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(text))
    response = chat.sample()
    result = RobotCommand.model_validate_json(response.content)
    # Do not rewrite the distance or the action. The motion gateway rejects it in the open.
    return preserve_robot_command(result)


_DIRECTIONS = {
    "left": "left", "sinistra": "left", "right": "right", "destra": "right",
    "up": "up", "su": "up", "down": "down", "giu": "down", "giù": "down",
    "forward": "forward", "avanti": "forward", "backward": "backward",
    "back": "backward", "indietro": "backward",
}
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "uno": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5}


def preserve_robot_command(command: RobotCommand) -> RobotCommand:
    """Return the parsed command unchanged, including a distance the gateway will reject."""
    return command


def offline_robot_command(text: str) -> RobotCommand:
    """English or Italian relative move, with no network. Distances are kept even when they are too large."""
    raw = " " + (text or "").lower().replace(",", ".") + " "
    if not raw.strip():
        return RobotCommand(action="ignore", direction="none", distance_mm=0)
    dirs = "|".join(sorted(_DIRECTIONS, key=len, reverse=True))
    nums = "|".join(_NUMBERS)
    unit = r"(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|centimetr[oi])"
    match = re.search(rf"\b({dirs})\b.{{0,48}}?\b(\d+(?:\.\d+)?|{nums})\s*{unit}\b", raw)
    if match is None:
        if re.search(r"\b(stop|halt|fermo|ferma|fermati|basta)\b", raw):
            return RobotCommand(action="stop", direction="none", distance_mm=0)
        return RobotCommand(action="ignore", direction="none", distance_mm=0)
    token = match.group(2)
    value = float(_NUMBERS[token]) if token in _NUMBERS else float(token)
    if match.group(3).startswith("c"):
        value *= 10.0
    return RobotCommand(action="move", direction=_DIRECTIONS[match.group(1)], distance_mm=int(round(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice/text -> Grok -> validated JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Use typed instruction instead of a microphone")
    group.add_argument("--mic", action="store_true", help="Record microphone audio")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        parser.error("Set XAI_API_KEY in your environment first")

    if args.mic:
        if not 0.5 <= args.seconds <= 15:
            parser.error("--seconds must be between 0.5 and 15")
        text = transcribe_wav(record_wav(args.seconds), api_key)
        print(f"Transcript: {text}", file=sys.stderr)
    else:
        text = args.text if args.text is not None else input("Command: ")

    if not text.strip():
        command = RobotCommand(action="ignore", direction="none", distance_mm=0)
    else:
        command = grok_to_command(text, api_key)

    print(json.dumps(command.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
