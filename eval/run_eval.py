"""Measure Trova against baselines. Writes a Markdown table to eval/results/ for the README.

Finds (the jury metric: objects found out of N requests, time to find):
    python eval/run_eval.py finds                       # cases recorded in jury mode (data/cases/finds.jsonl)
    python eval/run_eval.py finds --synthetic 30        # generated table + requests (no camera needed)
  modes compared:  som      = our pipeline (candidate boxes + Grok choice, direct fallback)
                   direct   = Grok asked for a box with no candidates (model-only baseline)
                   largest  = always the biggest object (floor)

What changed (the jury metric: false alarms):
    python eval/run_eval.py changes --synthetic 60      # add/remove/move + lighting drift, auto ground truth
    python eval/run_eval.py changes --log               # checks judged in jury mode (data/events.jsonl)
  modes compared:  naive    = raw pixel difference (baseline)
                   matched  = our object matching, no Grok
                   grok     = our matching + Grok verification (live only when XAI_API_KEY is set)

Without XAI_API_KEY the Grok parts run in mock mode: the numbers then test the plumbing only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trova import changes as chg  # noqa: E402
from trova.boxes import Box  # noqa: E402
from trova.config import load_settings  # noqa: E402
from trova.finder import _direct, FindResult, find  # noqa: E402
from trova.grok import GrokClient  # noqa: E402
from trova.proposals import drop_occluders, find_blobs  # noqa: E402
from trova.synthetic import COLORS, SyntheticTable  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–"


def lat(xs):
    if not xs:
        return "–"
    xs = sorted(xs)
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return f"{statistics.mean(xs):.2f} / {statistics.median(xs):.2f} / {p95:.2f}"


def hit(box: Box | None, expected) -> bool:
    if box is None or expected is None:
        return False
    e = Box.from_list(expected)
    return e.contains(*box.center) or box.iou(e) >= 0.3


# ------------------------------------------------------------------ finds
def load_find_cases(data_dir: Path) -> list[dict]:
    p = data_dir / "cases" / "finds.jsonl"
    if not p.exists():
        sys.exit(f"No recorded cases at {p}. Use jury mode (✓/✗ after each find) or --synthetic N.")
    cases = []
    for line in p.read_text().splitlines():
        c = json.loads(line)
        c["frame_img"] = cv2.imread(str(data_dir / c["frame"]))
        c["query_img"] = cv2.imread(str(data_dir / c["query"])) if c.get("query") else None
        c["bg_img"] = cv2.imread(str(data_dir / c["background"])) if c.get("background") else None
        cases.append(c)
    return cases


def synthetic_find_cases(n: int) -> list[dict]:
    cases, seed = [], 100
    while len(cases) < n:
        t = SyntheticTable(seed=seed)
        seed += 1
        f, bg = t.render(), t.empty()
        colors = [o.color for o in t.objects]
        for o, box in zip(t.objects, t.truth()):
            if colors.count(o.color) == 1:   # unambiguous request
                cases.append({"request": f"the {o.color} {o.name}", "frame_img": f, "bg_img": bg, "query_img": None,
                              "expected": box.to_list(), "absent": False})
        missing = [c for c in COLORS if c not in colors]
        if missing:
            cases.append({"request": f"the {missing[0]} object", "frame_img": f, "bg_img": bg, "query_img": None,
                          "expected": None, "absent": True})
    return cases[:n]


def run_finds(grok: GrokClient, cases: list[dict], modes: list[str]) -> str:
    rows = []
    for mode in modes:
        found = present = rejected = absent = false_pos = 0
        times = []
        for c in cases:
            f = c["frame_img"]
            boxes = drop_occluders(find_blobs(f, c.get("bg_img")), f.shape)
            t0 = time.perf_counter()
            if mode == "som":
                r = find(grok, f, boxes, c["request"], c.get("query_img"))
            elif mode == "direct":
                r = FindResult(status="not_found")
                _direct(grok, f, c["request"], c.get("query_img"), "", r)
            else:  # largest
                b = max(boxes, key=lambda b: b.area) if boxes else None
                r = FindResult(status="found" if b else "not_found", box=b)
            times.append(time.perf_counter() - t0)
            got = r.box if r.status == "found" else None
            if c.get("absent") or c.get("expected") is None:
                absent += 1
                rejected += got is None
                false_pos += got is not None
            else:
                present += 1
                found += hit(got, c["expected"])
        rows.append(f"| {mode} | {pct(found, present)} | {pct(rejected, absent)} | {false_pos} | {lat(times)} |")
    head = ("| Mode | Found (object on table) | Correct \"not found\" | Wrong picks on absent objects | "
            "Time per request, mean / median / p95 (s) |\n|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


# ------------------------------------------------------------------ changes
def run_changes_synthetic(grok: GrokClient, n: int) -> str:
    stats = {m: {"right": 0, "fa": 0, "missed": 0, "t": []} for m in ("naive", "matched", "grok")}
    trials = 0
    for i in range(n):
        t = SyntheticTable(seed=500 + i)
        empty = t.empty()
        ref = t.render()
        rb = find_blobs(ref, empty)
        scenario = ["add", "remove", "move", "none", "drift"][i % 5]
        truth = []
        if scenario in ("add", "remove", "move"):
            truth = [t.tamper(scenario)["kind"]]
        frames = [t.render() for _ in range(3)]
        if scenario == "drift":   # exposure / white balance shift, nothing moved
            frames = [np.clip(f.astype(np.float32) * np.array([1.1, 1.18, 1.22]), 0, 255).astype(np.uint8)
                      for f in frames]
        per = [find_blobs(f, empty) for f in frames]
        trials += 1
        # naive baseline: every changed pixel region counts as a change
        t0 = time.perf_counter()
        naive = chg.naive_pixel_diff(ref, frames[-1])
        stats["naive"]["t"].append(time.perf_counter() - t0)
        _score(stats["naive"], len(naive), truth)
        for mode in ("matched", "grok"):
            t0 = time.perf_counter()
            res = chg.check(grok if mode == "grok" else None, ref, rb, frames, per, verify=(mode == "grok"))
            stats[mode]["t"].append(time.perf_counter() - t0)
            kinds = [c.kind for c in res.changes]
            stats[mode]["right"] += kinds == truth
            stats[mode]["fa"] += max(0, len(kinds) - len(truth)) if kinds != truth or not truth else 0
            stats[mode]["missed"] += sum(1 for k in truth if k not in kinds)
    head = (f"{trials} scenes: add / remove / move / nothing changed / lighting drift (one fifth each)\n\n"
            "| Mode | Scenes fully right | False alarms | Missed changes | Time per check, mean / median / p95 (s) |\n"
            "|---|---|---|---|---|")
    rows = [f"| {m} | {pct(s['right'], trials)} | {s['fa']} | {s['missed']} | {lat(s['t'])} |" for m, s in stats.items()]
    return head + "\n" + "\n".join(rows)


def _score(s: dict, n_reported: int, truth: list[str]) -> None:
    # a moved object shows as 1-2 regions in a pixel diff; count one region per true change as right
    s["right"] += (n_reported == len(truth)) or (truth == ["moved"] and n_reported == 2)
    expected = 2 if truth == ["moved"] else len(truth)
    s["fa"] += max(0, n_reported - expected)
    s["missed"] += max(0, len(truth) - n_reported)


def run_changes_log(data_dir: Path) -> str:
    p = data_dir / "events.jsonl"
    events = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    judged = [e for e in events if e["kind"] == "judge_check"]
    if not judged:
        return "No judged checks yet: after each check, use the jury buttons (✓ or false alarms / missed)."
    ok = sum(e["correct"] for e in judged)
    return ("| Checks judged | Fully right | False alarms | Missed |\n|---|---|---|---|\n"
            f"| {len(judged)} | {pct(ok, len(judged))} | {sum(e['false_alarms'] for e in judged)} | "
            f"{sum(e['missed'] for e in judged)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["finds", "changes"])
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic cases instead of recorded ones")
    ap.add_argument("--log", action="store_true", help="changes: summarise jury-judged checks")
    ap.add_argument("--modes", default="", help="finds: comma list of som,direct,largest")
    args = ap.parse_args()
    s = load_settings()
    grok = GrokClient(s.xai_api_key, s.model, s.effort, s.mock)
    label = "MOCK Grok (plumbing only)" if grok.mock else f"live {s.model}, effort {s.effort or 'default'}"
    if args.what == "finds":
        cases = synthetic_find_cases(args.synthetic) if args.synthetic else load_find_cases(s.data_dir)
        modes = [m for m in (args.modes or "som,direct,largest").split(",") if m]
        title = f"Finds: {len(cases)} {'synthetic' if args.synthetic else 'recorded'} requests, {label}"
        table = run_finds(grok, cases, modes)
    else:
        if args.log:
            title, table = "What changed: checks judged in jury mode", run_changes_log(s.data_dir)
        else:
            title, table = f"What changed: synthetic, {label}", run_changes_synthetic(grok, args.synthetic or 60)
    out = f"## {title}\n\n_{time.strftime('%Y-%m-%d %H:%M')}_\n\n{table}\n"
    print(out)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.what}.md"
    path.write_text(out)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
