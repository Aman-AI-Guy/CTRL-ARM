"""Ask Grok where a named object is in one picture: a tight box in pixels. Standard library only (urllib).

    from scan3d.grok_find import find_box
    r = find_box(jpeg_bytes, "red bull can", (1600, 900))   # r["box_px"] = [x0, y0, x1, y1], r["pixel"] = its centre

The key comes from XAI_API_KEY, else from the repo's .env (XAI_API_KEY=...); the model from TROVA_MODEL (default
grok-4.7). Grok answers in a 0-1000 grid on both axes (the request format of experiments/grok_bbox_test.py), which
is turned into pixels of the picture that was sent. Errors (no key, not found, two equally good matches, HTTP) raise
GrokFindError with a sentence that says what to do: e.g. --click to pick the object by hand.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

from .config import ROOT

API_URL = "https://api.x.ai/v1/responses"
DEFAULT_MODEL = "grok-4.7"
ENV_FILE = ROOT.parent / ".env"            # the Trova repo's .env, next to "Test 3d live video"

BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "label": {"type": ["string", "null"]},
        "box_2d": {"anyOf": [BOX, {"type": "null"}]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "clarifying_question": {"type": ["string", "null"]},
    },
    "required": ["found", "label", "box_2d", "confidence", "reason", "clarifying_question"],
    "additionalProperties": False,
}


class GrokFindError(RuntimeError):
    pass


def env_value(name: str, path=ENV_FILE) -> str:
    """name from the environment, else from a KEY=VALUE .env file (inline '# comments' allowed), else ''."""
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return ""
    for ln in lines:
        key, sep, val = ln.partition("=")
        if sep and key.strip() == name:
            return val.split(" #")[0].split("\t#")[0].strip().strip("'\"")
    return ""


def prompt_for(what: str) -> str:
    return (f'Find this in the image: "{what}".\n'
            "Coordinates are integers normalised to 0-1000 on both axes: box_2d = [x_min, y_min, x_max, y_max], "
            "origin at top-left, (1000, 1000) is the bottom-right corner.\n"
            "Return the single best match with a tight box around the WHOLE object (for a can: from its lid to its "
            "bottom edge). If nothing matches, set found=false and box_2d=null. Never guess a box for an object that "
            "isn't visible. If two or more objects fit equally well, set clarifying_question to a short question that "
            "would tell them apart. confidence is 0-1.")


def request_body(jpeg: bytes, what: str, model: str, effort: str | None) -> dict:
    body = {
        "model": model,
        "store": False,
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
             "detail": "high"},
            {"type": "input_text", "text": prompt_for(what)},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "result", "schema": FIND_SCHEMA, "strict": True}},
    }
    if effort:
        body["reasoning"] = {"effort": effort}
    return body


def output_text(resp: dict) -> str:
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c["text"]
    raise GrokFindError(f"no text in Grok's answer: {json.dumps(resp)[:300]}")


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except ValueError as e:
        raise GrokFindError(f"Grok's answer is not JSON: {text[:200]}") from e


def box_to_pixels(box, size) -> list[float]:
    """[x0, y0, x1, y1] on the 0-1000 grid -> pixels of a (width, height) picture, ordered and clipped."""
    w, h = size
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((min(max(x0, 0.0), 1000.0), min(max(x1, 0.0), 1000.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1000.0), min(max(y1, 0.0), 1000.0)))
    return [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]


def _post(body: dict, key: str, timeout: float) -> dict:
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def find_box(jpeg: bytes, what: str, size, model: str | None = None, effort: str | None = "low",
             key: str | None = None, timeout: float = 60.0) -> dict:
    """The best box for `what` in the JPEG picture of `size` (width, height): box_px, pixel (its centre), label,
    confidence, reason, latency_s, model. Raises GrokFindError when there is no single clear answer."""
    key = key or env_value("XAI_API_KEY")
    if not key:
        raise GrokFindError("no XAI_API_KEY (environment or the repo's .env): set it, or pick the object with --click")
    model = model or env_value("TROVA_MODEL") or DEFAULT_MODEL
    body = request_body(jpeg, what, model, effort)
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            resp = _post(body, key, timeout)
            break
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if e.code == 400 and attempt == 0 and "reasoning" in body and "reason" in text.lower():
                body.pop("reasoning")                 # this model takes no reasoning effort: ask again without
                continue
            raise GrokFindError(f"Grok HTTP {e.code}: {text[:300]}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise GrokFindError(f"Grok not reachable ({e}): check the Wi-Fi, or pick the object with --click") from e
    latency = time.perf_counter() - t0
    r = parse_json(output_text(resp))
    if r.get("clarifying_question"):
        raise GrokFindError(f"Grok is not sure which one: {r['clarifying_question']} (describe it better, or --click)")
    if not r.get("found") or not r.get("box_2d"):
        raise GrokFindError(f"Grok did not find \"{what}\": {r.get('reason') or 'no reason given'}")
    box = box_to_pixels(r["box_2d"], size)
    return {"box_px": box, "pixel": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], "label": r.get("label") or what,
            "confidence": float(r.get("confidence") or 0.0), "reason": r.get("reason", ""),
            "latency_s": round(latency, 2), "model": model}
