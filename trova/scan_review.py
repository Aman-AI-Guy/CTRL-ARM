"""Grok reviews a finished reconstruction and asks the operator for what only a hand can do.

The arm orbits the object, but it cannot turn it over, move a cable that hides one side, or change the light.
After a reconstruction, Grok looks at its renders together with the coverage counts and the pipeline warnings
and returns a short list of requests to the operator ("Capovolgi il mouse: la base non è mai stata ripresa").
Plain code numbers the requests, keeps their state, and never moves the arm.

Honest limits, carried in every review:
- a request is a judgement from pictures, not a measured surface coverage (`surface_coverage_measured` false);
- a flipped object starts a new orientation group, and groups are not merged into one mesh
  (`cross_orientation_registration` false), so `complete` means "nothing more to ask the operator",
  never "the mesh is closed".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .grok import GrokClient, register_mock

ACTIONS = ["flip", "rotate", "reposition", "uncover", "lighting", "clean", "other"]
SIDES = ["top", "bottom", "left", "right", "front", "back", "none"]
REORIENTING = {"flip", "rotate", "reposition"}     # the object's pose changes: old localization is invalid
STATUSES = {"pending", "done", "skipped"}
MAX_REQUESTS = 4
MAX_RENDERS = 4

SCHEMA = {
    "type": "object",
    "properties": {
        "complete": {"type": "boolean"},
        "summary": {"type": "string"},
        "requests": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ACTIONS},
                "side": {"type": "string", "enum": SIDES},
                "instruction": {"type": "string"},
                "reason": {"type": "string"},
                "render": {"type": "integer"},
            },
            "required": ["action", "side", "instruction", "reason", "render"],
            "additionalProperties": False}},
    },
    "required": ["complete", "summary", "requests"],
    "additionalProperties": False,
}

PROMPT = """You are reviewing a 3D scan of one object made by a robot arm that carries a camera. The arm can only orbit
what faces it. It cannot turn the object over, move other things, or change the light. A person standing next to it can.

Images 1 to {n} are renders of the reconstructed mesh: {renders}.
Scan facts from plain code (trust them, do not contradict them):
- object: {name}
- orientation group of this reconstruction: {orientation}; groups already reconstructed for this scan: {groups}
- registered photos: {registered} of {inputs}
- estimated object size: {dims}
- pipeline warnings: {warnings}

Decide what the PERSON must do so the next acquisition covers what this mesh is missing. Look for a side that was never
seen (holes, flat cut-offs, the face resting on the table), parts hidden by another object or a cable, shiny or very dark
areas that came out as noise, an object too close to the edge of the table.
Return at most {max_requests} requests, the most important first. For each one:
- action: what kind of change it is;
- side: the side of the object it concerns;
- instruction: one imperative sentence the operator can carry out with their hands, naming the object and the side;
- reason: one sentence saying what is missing and in which render;
- render: the number of the image that shows it, or 0.
Never ask the person to move the robot or to choose camera positions: the arm plans its own views.
Never state measurements beyond the facts above. If nothing more is needed from the person, return complete true and no
requests. summary: one sentence a person can hear.
{lang}"""

LANG = {"it": "Write instruction, reason and summary in Italian.",
        "en": "Write instruction, reason and summary in English."}


def _facts(job: dict, render_names: list[str], name: str, groups: int) -> dict:
    q = job.get("quality") or {}
    dims = q.get("estimated_object_dimensions_mm") or q.get("mesh_bbox_extent_mm")
    if isinstance(dims, dict):
        dims = " x ".join(f"{float(dims[k]):.1f}" for k in ("L", "W", "H") if k in dims) + " mm"
    elif isinstance(dims, (list, tuple)):
        dims = " x ".join(f"{float(v):.1f}" for v in dims) + " mm"
    return {"name": name or "the selected object", "orientation": job.get("orientation") or "unknown",
            "groups": max(1, int(groups)), "registered": q.get("registered_images", "unknown"),
            "inputs": q.get("input_images", "unknown"), "dims": dims or "not measured",
            "warnings": list(q.get("warnings") or []), "renders": render_names}


def parse(data: dict, render_names: list[str]) -> list[dict]:
    """Model answer -> numbered pending requests. Unknown enums become other/none; empty instructions are dropped."""
    out = []
    for raw in (data.get("requests") or [])[:MAX_REQUESTS]:
        instruction = str(raw.get("instruction") or "").strip()
        if not instruction:
            continue
        action = raw.get("action") if raw.get("action") in ACTIONS else "other"
        side = raw.get("side") if raw.get("side") in SIDES else "none"
        try:
            index = int(raw.get("render") or 0)
        except (TypeError, ValueError):
            index = 0
        out.append({"id": f"R{len(out) + 1}", "action": action, "side": side, "instruction": instruction,
                    "reason": str(raw.get("reason") or "").strip(),
                    "render": render_names[index - 1] if 1 <= index <= len(render_names) else None,
                    "reorients": action in REORIENTING, "status": "pending", "updated": None})
    return out


def review(grok: GrokClient, job: dict, renders: list[tuple[str, np.ndarray]], *, name: str = "",
           groups: int = 1, lang: str = "it") -> dict:
    """One Grok call over the renders of a completed job. Never moves the arm."""
    if job.get("status") != "completed":
        raise ValueError("Only a completed reconstruction can be reviewed.")
    renders = [(n, img) for n, img in renders if img is not None][:MAX_RENDERS]
    if not renders:
        raise ValueError("The reconstruction has no renders to review.")
    names = [n for n, _ in renders]
    facts = _facts(job, names, name, groups)
    prompt = PROMPT.format(n=len(renders), renders=", ".join(names), name=facts["name"],
                           orientation=facts["orientation"], groups=facts["groups"], registered=facts["registered"],
                           inputs=facts["inputs"], dims=facts["dims"],
                           warnings="; ".join(facts["warnings"]) or "none", max_requests=MAX_REQUESTS,
                           lang=LANG.get(lang, LANG["it"]))
    r = grok.json(name="trova_scan_review", prompt=prompt, schema=SCHEMA, images=[img for _, img in renders],
                  mock_context={**facts, "lang": lang})
    requests = parse(r.data, names)
    mode = "offline" if r.offline else "mock" if r.mock else "live"
    return {"job_id": job.get("id"), "session_id": job.get("session_id"), "orientation": job.get("orientation"),
            "time": time.strftime("%H:%M:%S"), "at": time.time(), "grok_mode": mode,
            "latency_s": round(r.latency_s, 2), "summary": str(r.data.get("summary") or "").strip(),
            "complete": bool(r.data.get("complete")) and not requests, "requests": requests, "renders": names,
            "groups": facts["groups"], "moves_arm": False, "surface_coverage_measured": False,
            "cross_orientation_registration": False}


def pending(rev: dict | None) -> list[dict]:
    return [q for q in (rev or {}).get("requests", []) if q.get("status") == "pending"]


def set_status(rev: dict, request_id: str, status: str) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    for q in rev.get("requests", []):
        if q["id"] == request_id:
            q["status"], q["updated"] = status, time.time()
            return q
    raise KeyError(f"Unknown request {request_id}")


def save(path: Path, rev: dict | None) -> None:
    if rev is not None:
        Path(path).write_text(json.dumps(rev, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("requests"), list) else None
    except (OSError, ValueError):
        return None


def _mock(prompt: str, images: list, ctx: dict) -> dict:
    """Deterministic and honest: one orientation group never shows the face resting on the table."""
    it = ctx.get("lang", "it") != "en"
    name = ctx.get("name") or ("l'oggetto" if it else "the object")
    last = len(images)
    reqs = []
    if int(ctx.get("groups") or 1) < 2:
        reqs.append({"action": "flip", "side": "bottom", "render": last,
                     "instruction": (f"Capovolgi {name} così la faccia appoggiata al tavolo sia rivolta verso l'alto."
                                     if it else f"Turn {name} over so the face resting on the table points up."),
                     "reason": ("Mock: con un solo gruppo di orientamento la faccia sul tavolo non è mai stata ripresa."
                                if it else "Mock: with one orientation group the face on the table was never seen.")})
    if ctx.get("warnings"):
        reqs.append({"action": "reposition", "side": "none", "render": 1 if last else 0,
                     "instruction": (f"Sposta {name} al centro del tavolo, lontano da cavi e altri oggetti."
                                     if it else f"Move {name} to the middle of the table, away from cables and other objects."),
                     "reason": ("Mock: la ricostruzione ha segnalato " if it else "Mock: the reconstruction reported ")
                               + str(ctx["warnings"][0])})
    if reqs:
        summary = (f"Mock: {len(reqs)} richieste per completare la scansione." if it
                   else f"Mock: {len(reqs)} requests to complete the scan.")
    else:
        summary = "Mock: nessuna altra richiesta." if it else "Mock: nothing more to ask."
    return {"complete": not reqs, "summary": summary, "requests": reqs}


register_mock("trova_scan_review", _mock)
