"""OPTIONAL dog scenario: "it's not on the table? Send the dog to look in the other room."

The dog walks a fixed route of named spots (no autonomous navigation: out of scope in the brief),
grabs a camera frame at each spot, and Grok checks whether the requested object is visible.
It stops at the first sighting and reports the spot, a box and the photo.

Drivers (TROVA_DOG):
  off        disabled (default): the voice agent never offers it
  sim        simulated dog. Each spot shows a photo from data/dog_route/<NN>_<spot name>.jpg if present
             (photograph the museum rooms on Wednesday and the sim replays them), else a synthetic scene.
  go2:<ip>   Unitree Go2 (template in trova/dog_go2.py; confirm the model, firmware and SDK with staff first).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .boxes import Box
from .finder import BOX4
from .grok import GrokClient, register_mock
from .marks import color_name, crop
from .proposals import find_blobs
from .synthetic import SyntheticTable


@dataclass
class Spot:
    name: str
    frame_path: Path | None = None
    seed: int = 0


class DogDriver:
    name = "dog"

    def go_to(self, spot: Spot, stop: threading.Event) -> bool:
        raise NotImplementedError

    def frame(self, spot: Spot) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class SimDog(DogDriver):
    name = "sim-dog"

    def __init__(self, seconds_per_spot: float = 1.2):
        self.t = seconds_per_spot

    def go_to(self, spot, stop):
        t0 = time.time()
        while time.time() - t0 < self.t:
            if stop.is_set():
                return False
            time.sleep(0.05)
        return True

    def frame(self, spot):
        if spot.frame_path and spot.frame_path.exists():
            img = cv2.imread(str(spot.frame_path))
            if img is not None:
                return img
        return SyntheticTable(seed=spot.seed, n_objects=6).render()


def load_route(data_dir: Path) -> list[Spot]:
    folder = Path(data_dir) / "dog_route"
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")) if folder.exists() else []
    if files:
        return [Spot(re.sub(r"^\d+[_-]", "", f.stem).replace("_", " "), f) for f in files]
    return [Spot("Room 2, desk", seed=211), Spot("Room 2, shelf", seed=212), Spot("Room 3, workbench", seed=213)]


def make_dog(spec: str) -> DogDriver | None:
    if spec in ("", "off", None):
        return None
    if spec == "sim":
        return SimDog()
    if spec.startswith("go2:"):
        from .dog_go2 import Go2Dog
        return Go2Dog(spec[4:])
    raise ValueError(f"Unknown TROVA_DOG '{spec}'")


LOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "box_2d": {"anyOf": [BOX4, {"type": "null"}]},
        "label": {"type": "string"},
        "where": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["found", "box_2d", "label", "where", "confidence"],
    "additionalProperties": False,
}

LOOK_PROMPT = """This is a photo from a robot dog's camera at the spot "{spot}".
Is this object visible: "{request}"?
If yes: found = true, a tight box_2d as integers normalised to 0-1000 ([x_min, y_min, x_max, y_max], origin top-left),
a short label, and "where": a few words a person can use to find it (e.g. "on the left of the desk").
If not: found = false and box_2d = null. Never guess. Answer in the language of the request."""


@dataclass
class DogSearch:
    request: str
    status: str = "running"             # running | found | not_found | stopped | error
    spot: str | None = None
    label: str = ""
    where: str = ""
    box: list[int] | None = None
    photo: str | None = None
    visited: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    total_s: float = 0.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("request", "status", "spot", "label", "where", "box", "photo",
                                                "visited", "total_s")}


def run_search(dog: DogDriver, grok: GrokClient, route: list[Spot], request: str, stop: threading.Event,
               save_image, search: DogSearch) -> DogSearch:
    for spot in route:
        if not dog.go_to(spot, stop):
            search.status = "stopped"
            break
        img = dog.frame(spot)
        search.visited.append(spot.name)
        boxes = find_blobs(img, None)
        ctx = {"request": request, "colors": [color_name(crop(img, b, pad=0)) for b in boxes], "boxes": boxes}
        r = grok.json(name="trova_dog_look", prompt=LOOK_PROMPT.format(spot=spot.name, request=request),
                      schema=LOOK_SCHEMA, images=[img], mock_context=ctx)
        d = r.data
        if d.get("found") and d.get("box_2d"):
            H, W = img.shape[:2]
            b = Box.from_norm1000(d["box_2d"], W, H)
            marked = img.copy()
            cv2.rectangle(marked, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (60, 220, 60), 5)
            search.status, search.spot, search.label, search.where = "found", spot.name, d["label"], d["where"]
            search.box, search.photo = b.to_list(), save_image("dog_found", marked)
            break
    else:
        search.status = "not_found"
    search.total_s = round(time.time() - search.started, 2)
    return search


_COLOR_WORDS = {"red": ("red", "rosso", "rossa"), "blue": ("blue", "blu"), "green": ("green", "verde"),
                "yellow": ("yellow", "giallo", "gialla"), "black": ("black", "nero", "nera"),
                "orange": ("orange", "arancione"), "purple": ("purple", "viola")}


def _mock_look(prompt: str, images: list, ctx: dict) -> dict:
    words = set(re.findall(r"[a-zà-ù]+", ctx["request"].lower()))
    wanted = [c for c, ws in _COLOR_WORDS.items() if words & set(ws)]
    for c, b in zip(ctx["colors"], ctx["boxes"]):
        if c in wanted:
            H, W = images[0].shape[:2]
            bx = [int(b.x0 * 1000 / W), int(b.y0 * 1000 / H), int(b.x1 * 1000 / W), int(b.y1 * 1000 / H)]
            return {"found": True, "box_2d": bx, "label": f"{c} object", "where": "in view of the dog", "confidence": 0.7}
    return {"found": False, "box_2d": None, "label": "", "where": "", "confidence": 0.0}


register_mock("trova_dog_look", _mock_look)
