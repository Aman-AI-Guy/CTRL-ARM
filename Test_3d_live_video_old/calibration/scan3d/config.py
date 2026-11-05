"""Load scan3d/config.json, the single home of every tunable value, with per-run overrides.

    from scan3d.config import load_config
    cfg = load_config(["capture.fps=10", "aruco.marker_mm=40"])

Command-line tools add `--config PATH` and `--set section.key=value` (repeatable) via add_config_args().
Values given with --set are parsed as JSON when possible (numbers, true/false, lists), else kept as strings.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "config.json"
ROOT = HERE.parent          # "Test 3d live video": data/ and results live here


def _parse_value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Return a copy of cfg with each 'a.b.c=value' applied. Unknown keys are allowed (new settings)."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {dotted}: {k} is not a section")
        node[keys[-1]] = _parse_value(raw.strip())
    return out


def load_config(overrides: list[str] | None = None, path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return apply_overrides(cfg, overrides)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_PATH), help="config file (default scan3d/config.json)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override one config value for this run (repeatable)")


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.overrides, args.config)
