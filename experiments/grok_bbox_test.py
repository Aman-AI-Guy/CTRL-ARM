#!/usr/bin/env python3
"""Can Grok locate objects itself? A 15-minute test for Track A (CTRL+F: SEA).

xAI's docs don't document bounding-box output for Grok vision. This script asks
Grok for boxes anyway, draws them on the image, and times each call. Look at the
annotated image: if the boxes land on the objects, we can skip a separate
detector. If they don't, we fall back to "set-of-marks" (a local detector draws
numbered boxes and Grok picks one).

Setup:
    pip install requests pillow            # opencv-python only if you use --webcam
    export XAI_API_KEY=...

Examples:
    # List every object Grok sees, with boxes
    python grok_bbox_test.py desk.jpg

    # "Find this": one object from a description (should return found=false if absent)
    python grok_bbox_test.py desk.jpg --find "something to tell the time"

    # Compare speed/accuracy across settings; 3 calls each for latency
    python grok_bbox_test.py desk.jpg --find "the red mug" --effort high --repeats 3
    python grok_bbox_test.py desk.jpg --model grok-4.20-0309-non-reasoning --effort ""

    # Grab a frame from a webcam / Continuity Camera iPhone (index 0, 1, ...)
    python grok_bbox_test.py --webcam 1 --find "scissors"

Output: <image>_grok.jpg with boxes drawn, plus the raw JSON and latencies.
"""

import argparse
import base64
import io
import json
import os
import statistics
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = "https://api.x.ai/v1/responses"
MAX_SIDE = 1600  # downscale before sending: faster, and coordinates are normalised anyway

BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}

LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "box_2d": BOX},
                "required": ["label", "box_2d"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}

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


def coord_rules(coords, w, h):
    if coords == "pixel":
        return (f"Coordinates are integer pixels in this {w}x{h} image: "
                "box_2d = [x_min, y_min, x_max, y_max], origin at top-left.")
    return ("Coordinates are integers normalised to 0-1000 on both axes: "
            "box_2d = [x_min, y_min, x_max, y_max], origin at top-left, "
            "(1000, 1000) is the bottom-right corner.")


def build_prompt(find, coords, w, h):
    rules = coord_rules(coords, w, h)
    if find:
        return (f'Find this in the image: "{find}".\n{rules}\n'
                "Return the single best match with a tight box. If nothing matches, set "
                "found=false and box_2d=null. Never guess a box for an object that isn't "
                "visible. If two or more objects fit equally well, set clarifying_question "
                "to a short question that would tell them apart. confidence is 0-1.")
    return (f"List every distinct physical object visible in the image with a tight box.\n{rules}\n"
            "Use short, specific labels (e.g. 'red ceramic mug', not 'object').")


def load_image(args):
    if args.webcam is not None:
        try:
            import cv2
        except ImportError:
            sys.exit("--webcam needs opencv: pip install opencv-python")
        cap = cv2.VideoCapture(args.webcam)
        for _ in range(10):  # let auto-exposure settle
            ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Could not read from webcam {args.webcam}")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        path = f"webcam{args.webcam}_{int(time.time())}.jpg"
        img.save(path, quality=92)
        print(f"Saved webcam frame to {path}")
        return img, path
    if not args.image:
        sys.exit("Give an image path or --webcam N")
    return Image.open(args.image).convert("RGB"), args.image


def encode(img):
    send = img.copy()
    send.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    send.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), send.size


def call_grok(key, model, effort, prompt, b64, schema):
    body = {
        "model": model,
        "store": False,  # docs advise against server-side history with images
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                {"type": "input_text", "text": prompt},
            ],
        }],
        "text": {"format": {"type": "json_schema", "name": "result", "schema": schema, "strict": True}},
    }
    if effort:
        body["reasoning"] = {"effort": effort}

    # If the API rejects an optional field, drop it and retry so the test still runs.
    dropped = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        dt = time.perf_counter() - t0
        if r.status_code == 200:
            return r.json(), dt, dropped
        err = r.text.lower()
        if r.status_code == 400 and "reasoning" in body and "reason" in err:
            body.pop("reasoning"); dropped.append("reasoning")
        elif r.status_code == 400 and "store" in body and "store" in err:
            body.pop("store"); dropped.append("store")
        elif r.status_code == 400 and "text" in body and ("format" in err or "schema" in err):
            body.pop("text"); dropped.append("text.format")
            body["input"][0]["content"][1]["text"] += (
                "\nReply with JSON only, matching this schema:\n" + json.dumps(schema))
        else:
            sys.exit(f"HTTP {r.status_code}: {r.text[:800]}")
    sys.exit(f"Gave up after dropping {dropped}: HTTP {r.status_code}: {r.text[:800]}")


def output_text(resp):
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c["text"]
    raise ValueError(f"No output text in response: {json.dumps(resp)[:500]}")


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def to_pixels(box, coords, sent_size, orig_size):
    x0, y0, x1, y1 = box
    W, H = orig_size
    if coords == "pixel":
        sx, sy = W / sent_size[0], H / sent_size[1]
    else:
        sx, sy = W / 1000, H / 1000
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def draw(img, items, coords, sent_size, out_path):
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    lw = max(2, img.width // 300)
    try:
        font = ImageFont.truetype("Arial.ttf", max(14, img.width // 60))
    except OSError:
        font = ImageFont.load_default()
    colors = ["#ff3b30", "#34c759", "#0a84ff", "#ff9f0a", "#bf5af2", "#64d2ff", "#ffd60a"]
    for i, (label, box) in enumerate(items):
        c = colors[i % len(colors)]
        x0, y0, x1, y1 = to_pixels(box, coords, sent_size, img.size)
        d.rectangle([x0, y0, x1, y1], outline=c, width=lw)
        tb = d.textbbox((x0, y0), label, font=font)
        d.rectangle([tb[0], tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=c)
        d.text((x0 + 2, y0), label, fill="black", font=font)
    canvas.save(out_path, quality=92)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", help="path to a JPG/PNG photo")
    p.add_argument("--webcam", type=int, help="grab a frame from this camera index instead")
    p.add_argument("--find", help='description to search for, e.g. "something to cut paper"')
    p.add_argument("--model", default="grok-4.7")
    p.add_argument("--effort", default="low", help='reasoning effort (low/medium/high/xhigh); "" to omit')
    p.add_argument("--coords", choices=["norm1000", "pixel"], default="norm1000")
    p.add_argument("--repeats", type=int, default=1, help="repeat the call to measure latency")
    p.add_argument("--out", help="annotated output path (default: <image>_grok.jpg)")
    args = p.parse_args()

    key = os.environ.get("XAI_API_KEY")
    if not key:
        sys.exit("Set XAI_API_KEY first")

    img, path = load_image(args)
    b64, sent_size = encode(img)
    prompt = build_prompt(args.find, args.coords, *sent_size)
    schema = FIND_SCHEMA if args.find else LIST_SCHEMA

    latencies, result = [], None
    for i in range(args.repeats):
        resp, dt, dropped = call_grok(key, args.model, args.effort, prompt, b64, schema)
        latencies.append(dt)
        result = parse_json(output_text(resp))
        if dropped and i == 0:
            print(f"Note: API rejected {dropped}; ran without them.")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.find:
        items = [(result.get("label") or args.find, result["box_2d"])] if result.get("found") and result.get("box_2d") else []
        if result.get("clarifying_question"):
            print(f"\nGrok asks: {result['clarifying_question']}")
    else:
        items = [(o["label"], o["box_2d"]) for o in result.get("objects", [])]

    out = args.out or os.path.splitext(path)[0] + "_grok.jpg"
    draw(img, items, args.coords, sent_size, out)

    lat = ", ".join(f"{x:.2f}s" for x in latencies)
    print(f"\nModel {args.model}, effort {args.effort or 'default'}, coords {args.coords}")
    print(f"Latency: {lat}" + (f" (median {statistics.median(latencies):.2f}s)" if len(latencies) > 1 else ""))
    print(f"{len(items)} box(es) drawn -> {out}")
    usage = resp.get("usage")
    if usage:
        print(f"Tokens: {usage}")


if __name__ == "__main__":
    main()
