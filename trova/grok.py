"""Grok client for vision calls with structured JSON output.

Uses the xAI Responses API (POST https://api.x.ai/v1/responses): images as `input_image`
parts, JSON Schema output under `text.format`. If the API rejects an optional field
(`reasoning`, `store`, `text.format`), the call is retried without it and the dropped field
is reported, so the system keeps working while you fix the config.

Mock mode (no XAI_API_KEY, or TROVA_MOCK=1) answers from registered handlers, so the whole
app runs offline tonight. Every module that calls Grok registers its own mock handler.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
import requests

API_URL = "https://api.x.ai/v1/responses"
MAX_SIDE = 1600

MockHandler = Callable[[str, list, dict], dict]
_MOCK_HANDLERS: dict[str, MockHandler] = {}


def register_mock(name: str, handler: MockHandler) -> None:
    _MOCK_HANDLERS[name] = handler


class GrokError(RuntimeError):
    pass


@dataclass
class GrokResult:
    data: dict
    latency_s: float
    model: str
    mock: bool
    usage: dict | None = None
    dropped: list[str] = field(default_factory=list)
    raw_text: str = ""


def encode_image(img: Any, max_side: int = MAX_SIDE) -> str:
    """BGR ndarray or JPEG/PNG bytes -> data URL (JPEG). Downscales the long side to max_side."""
    if isinstance(img, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise GrokError("Could not decode image bytes")
        img = arr
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise GrokError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _output_text(resp: dict) -> str:
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", "")
    raise GrokError(f"No output text in response: {json.dumps(resp)[:400]}")


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise GrokError(f"Grok did not return JSON: {text[:300]}")
    return json.loads(text[start:end + 1])


class GrokClient:
    def __init__(self, api_key: str = "", model: str = "grok-4.7", effort: str = "low",
                 mock: bool = False, timeout: float = 90.0,
                 on_call: Callable[[dict], None] | None = None):
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.mock = mock or not api_key
        self.timeout = timeout
        self.on_call = on_call
        self.session = requests.Session()
        self.calls = 0
        self.total_latency = 0.0
        self._dropped: set[str] = set()   # fields the API rejected once; never resent

    def json(self, *, name: str, prompt: str, schema: dict, images: list | tuple = (),
             effort: str | None = None, model: str | None = None,
             mock_context: dict | None = None) -> GrokResult:
        """One vision call that must return an object matching `schema`."""
        model = model or self.model
        effort = self.effort if effort is None else effort
        t0 = time.perf_counter()
        if self.mock:
            handler = _MOCK_HANDLERS.get(name)
            if handler is None:
                raise GrokError(f"No mock handler registered for '{name}'")
            data = handler(prompt, list(images), mock_context or {})
            result = GrokResult(data=data, latency_s=time.perf_counter() - t0, model=f"mock:{model}", mock=True)
        else:
            result = self._call_api(name, prompt, schema, images, effort, model)
        self.calls += 1
        self.total_latency += result.latency_s
        if self.on_call:
            self.on_call({"name": name, "model": result.model, "latency_s": round(result.latency_s, 3),
                          "mock": result.mock, "dropped": result.dropped, "usage": result.usage,
                          "output": result.data})
        return result

    def _call_api(self, name, prompt, schema, images, effort, model) -> GrokResult:
        content = [{"type": "input_image", "image_url": encode_image(img), "detail": "high"} for img in images]
        content.append({"type": "input_text", "text": prompt})
        body: dict = {
            "model": model,
            "store": False,  # xAI docs advise against server-side history when sending images
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        }
        if effort:
            body["reasoning"] = {"effort": effort}
        for f in self._dropped:
            self._drop(body, f, schema)

        dropped: list[str] = []
        backoff = 1.0
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                r = self.session.post(API_URL, json=body, timeout=self.timeout,
                                      headers={"Authorization": f"Bearer {self.api_key}"})
            except requests.RequestException as e:
                if attempt >= 2:
                    raise GrokError(f"Network error calling Grok: {e}") from e
                time.sleep(backoff)
                backoff *= 2
                continue
            latency = time.perf_counter() - t0
            if r.status_code == 200:
                resp = r.json()
                text = _output_text(resp)
                return GrokResult(data=_parse_json(text), latency_s=latency, model=model, mock=False,
                                  usage=resp.get("usage"), dropped=dropped, raw_text=text)
            err = r.text.lower()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (400, 422):
                for f, hint in (("reasoning", "reason"), ("store", "store"), ("text.format", "format")):
                    if f not in self._dropped and hint in err:
                        self._dropped.add(f)
                        dropped.append(f)
                        self._drop(body, f, schema)
                        break
                else:
                    raise GrokError(f"Grok HTTP {r.status_code}: {r.text[:600]}")
                continue
            raise GrokError(f"Grok HTTP {r.status_code}: {r.text[:600]}")
        raise GrokError(f"Grok call '{name}' failed after retries (dropped {dropped})")

    @staticmethod
    def _drop(body: dict, f: str, schema: dict) -> None:
        if f == "reasoning":
            body.pop("reasoning", None)
        elif f == "store":
            body.pop("store", None)
        elif f == "text.format" and "text" in body:
            body.pop("text", None)
            body["input"][0]["content"][-1]["text"] += (
                "\n\nReply with one JSON object only, no prose, matching this JSON Schema:\n" + json.dumps(schema))

    def stats(self) -> dict:
        return {"mock": self.mock, "model": self.model, "effort": self.effort, "calls": self.calls,
                "avg_latency_s": round(self.total_latency / self.calls, 3) if self.calls else None,
                "dropped_fields": sorted(self._dropped)}
