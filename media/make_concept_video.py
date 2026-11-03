#!/usr/bin/env python3
"""Render the Trova concept video: about 80 s, 1280x720, H.264 (needs ffmpeg on PATH).

What is real: the numbered object boxes and the change detection are computed by Trova's own
code (trova.proposals, trova.changes) on a drawn table, and the arm motion is a straight-line
move solved with inverse kinematics. What is illustrative: Grok's replies and the latencies.

    .venv/bin/python media/make_concept_video.py           # -> media/trova_concept.mp4
    .venv/bin/python media/make_concept_video.py --stills  # also writes key frames as PNG
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from scene import (PARK, SCISSORS_NEW, TH, TW, USB, draw_arm, draw_hand, initial_objects,  # noqa: E402
                   make_table, render)
from trova import changes as chg  # noqa: E402
from trova.proposals import find_blobs, is_occluded  # noqa: E402

W, H, FPS, DUR = 1280, 720, 30, 101.0
VX, VY, VW, VH = 24, 72, 800, 500
SC = VW / TW
MM = 600 / TW   # default calibration: 600 mm across the camera's view

BG, PANEL, LINE = (13, 15, 20), (25, 28, 35), (48, 52, 62)
INK, MUTED = (238, 240, 244), (150, 156, 168)
ACCENT, GREEN, ORANGE, BLUE, RED, CYAN = (255, 84, 222), (72, 214, 124), (255, 152, 48), (86, 172, 255), \
    (255, 98, 88), (90, 230, 255)

_FONT_CACHE: dict = {}


def F(size: int, w: str = "reg") -> ImageFont.FreeTypeFont:
    idx = {"bold": 0, "demi": 2, "med": 5, "reg": 7}[w]
    key = (size, idx)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size, index=idx)
        except OSError:
            _FONT_CACHE[key] = ImageFont.load_default(size)
    return _FONT_CACHE[key]


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def ramp(t: float, a: float, b: float) -> float:
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def lerp(p, q, k):
    return (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)


def wrap(text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def typed(text: str, t: float, a: float, cps: float = 24) -> str:
    return text[: max(0, int((t - a) * cps))]


# ------------------------------------------------------------------ precomputed scene states
TABLE = make_table()
OBJS = initial_objects()
OBJ = {o.key: o for o in OBJS}
F0 = render(TABLE, OBJS)
BOXES = find_blobs(F0, TABLE)
BID = {o.key: next(b.id for b in BOXES if b.contains(o.x, o.y)) for o in OBJS}
BOX_OF = {b.id: b for b in BOXES}
OBJS_AFTER = [o.at(*SCISSORS_NEW[:2], SCISSORS_NEW[2]) if o.key == "scissors" else o for o in OBJS] + [USB]
DRIFT = (0.96, 1.0, 1.07)
F_AFTER = render(TABLE, OBJS_AFTER, gain=DRIFT)
BOXES_AFTER = find_blobs(F_AFTER, TABLE)
CHANGES = chg.diff_objects(F0, BOXES, F_AFTER, BOXES_AFTER)
NAIVE = chg.naive_pixel_diff(F0, F_AFTER)
assert sorted(c.kind for c in CHANGES) == ["added", "moved"], [c.kind for c in CHANGES]
MOVED = next(c for c in CHANGES if c.kind == "moved")
ADDED = next(c for c in CHANGES if c.kind == "added")
MOVE_CM = math.dist(MOVED.ref_box.center, MOVED.cur_box.center) * MM / 10

PEN, MARKER, SCIS = BID["pen"], BID["marker"], BID["scissors"]


def mm(p):
    return f"x {p[0] * MM:.0f} mm · y {p[1] * MM:.0f} mm"


# ------------------------------------------------------------------ timeline of the table scene
def arm_tip(t: float):
    legs = [(16.4, 18.2, PARK, (OBJ["pen"].x, OBJ["pen"].y)),
            (22.0, 23.2, (OBJ["pen"].x, OBJ["pen"].y), PARK),
            (25.8, 27.4, PARK, (OBJ["scissors"].x, OBJ["scissors"].y)),
            (31.0, 32.4, (OBJ["scissors"].x, OBJ["scissors"].y), PARK)]
    pos = PARK
    for a, b, p, q in legs:
        if t >= a:
            pos = lerp(p, q, ramp(t, a, b))
    return pos


def hand_state(t: float):
    """(hand position or None, scissors pose, usb visible, usb pose)."""
    s0 = (OBJ["scissors"].x, OBJ["scissors"].y)
    s1 = SCISSORS_NEW[:2]
    top = lambda p: (p[0] + 40, -140)  # noqa: E731
    scis, ang, usb, usbp, hand = s0, OBJ["scissors"].a, False, (USB.x, USB.y), None
    if 34.0 <= t < 35.2:
        hand = lerp(top(s0), s0, ramp(t, 34.0, 35.2))
    elif 35.2 <= t < 36.8:
        k = ramp(t, 35.2, 36.8)
        hand = scis = lerp(s0, s1, k)
        ang = OBJ["scissors"].a + (SCISSORS_NEW[2] - OBJ["scissors"].a) * k
    elif 36.8 <= t < 37.6:
        hand = lerp(s1, top(s1), ramp(t, 36.8, 37.6))
    elif 38.0 <= t < 39.3:
        hand = usbp = lerp(top(usbp), usbp, ramp(t, 38.0, 39.3))
        usb = True
    elif 39.3 <= t < 40.3:
        hand = lerp((USB.x, USB.y), top((USB.x, USB.y)), ramp(t, 39.3, 40.3))
    if t >= 35.2:
        scis, ang = (s1 if t >= 36.8 else scis), (SCISSORS_NEW[2] if t >= 36.8 else ang)
    if t >= 39.3:
        usb = True
        usbp = (USB.x, USB.y)
    return hand, scis, ang, usb, usbp


def table_frame(t: float) -> np.ndarray:
    if t < 34.0:
        img = F0.copy()
    elif t >= 40.3:
        k = ramp(t, 40.3, 40.9)
        img = render(TABLE, OBJS_AFTER, gain=tuple(1 + (g - 1) * k for g in DRIFT)) if k < 1 else F_AFTER.copy()
    else:
        hand, scis, ang, usb, usbp = hand_state(t)
        objs = [o.at(scis[0], scis[1], ang) if o.key == "scissors" else o for o in OBJS]
        if usb:
            objs.append(USB.at(*usbp))
        img = render(TABLE, objs)
        if hand:
            draw_hand(img, *hand)
    tip = arm_tip(t)
    if tip != PARK:
        draw_arm(img, tip)
    return img


_HAND_OCC: dict = {}


def hand_in_view(t: float) -> bool:
    k = round(t * 5) / 5
    if k not in _HAND_OCC:
        f = table_frame(k)
        _HAND_OCC[k] = is_occluded(find_blobs(f, TABLE), f.shape)
    return _HAND_OCC[k]


# ------------------------------------------------------------------ drawing helpers
def card(d: ImageDraw.ImageDraw, x, y, w, head: str, body: str, gloss: str = "", head_color=MUTED,
         alpha: float = 1.0, body_size: int = 21, accent=None) -> int:
    """Draws a card, returns its bottom y."""
    fb, fg = F(body_size, "demi"), F(15, "reg")
    bl = wrap(body, fb, w - 32)
    gl = wrap(gloss, fg, w - 32) if gloss else []
    h = 16 + 20 + 8 + len(bl) * (body_size + 8) + (len(gl) * 21 + 6 if gl else 0) + 14
    a = int(255 * alpha)
    d.rounded_rectangle([x, y, x + w, y + h], 12, fill=PANEL + (a,), outline=(accent or LINE) + (a,),
                        width=2 if accent else 1)
    d.text((x + 16, y + 14), head, font=F(14, "demi"), fill=head_color + (a,))
    yy = y + 44
    for line in bl:
        d.text((x + 16, yy), line, font=fb, fill=INK + (a,))
        yy += body_size + 8
    if gl:
        yy += 4
        for line in gl:
            d.text((x + 16, yy), line, font=fg, fill=MUTED + (a,))
            yy += 21
    return y + h



def dashed_rect(d, r, color, dash=9):
    x0, y0, x1, y1 = r
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(math.dist(a, b) / dash))
        for i in range(0, n, 2):
            d.line([lerp(a, b, i / n), lerp(a, b, min(1, (i + 1) / n))], fill=color + (255,), width=2)


def arrow(d, p, q, color, a=255, width=3):
    d.line([p, q], fill=color + (a,), width=width)
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    pts = [q, (q[0] - 13 * math.cos(ang - 0.45), q[1] - 13 * math.sin(ang - 0.45)),
           (q[0] - 13 * math.cos(ang + 0.45), q[1] - 13 * math.sin(ang + 0.45))]
    d.polygon(pts, fill=color + (a,))

def vbox(b, pad=0):
    return [VX + b.x0 * SC - pad, VY + b.y0 * SC - pad, VX + b.x1 * SC + pad, VY + b.y1 * SC + pad]


def tag(d, x, y, text, color, a=255):
    f = F(15, "bold")
    w = f.getlength(text) + 10
    d.rectangle([x, y - 20, x + w, y], fill=color + (a,))
    d.text((x + 5, y - 20), text, font=f, fill=(255, 255, 255, a))


def draw_boxes(d, t, appear_at, highlight=None, dim=False):
    for i, b in enumerate(BOXES):
        k = ramp(t, appear_at + i * 0.12, appear_at + i * 0.12 + 0.25)
        if k <= 0:
            continue
        hit = highlight == b.id
        if dim and not hit:
            k *= 0.35
        col = GREEN if hit else ACCENT
        d.rectangle(vbox(b), outline=col + (int(255 * k),), width=4 if hit else 2)
        tag(d, VX + b.x0 * SC, VY + b.y0 * SC, str(b.id), col, int(255 * k))


def reticle(d, p, t):
    x, y = VX + p[0] * SC, VY + p[1] * SC
    r = 16 + 4 * math.sin(t * 6)
    d.ellipse([x - r, y - r, x + r, y + r], outline=CYAN + (255,), width=3)
    d.line([x - r - 10, y, x - r + 4, y], fill=CYAN + (255,), width=3)
    d.line([x + r - 4, y, x + r + 10, y], fill=CYAN + (255,), width=3)


def thinking(d, x, y, w, t, label="Grok 4.7 is looking"):
    dots = "." * (1 + int(t * 4) % 3)
    d.rounded_rectangle([x, y, x + w, y + 52], 12, fill=PANEL + (255,), outline=LINE + (255,))
    d.text((x + 16, y + 14), label + dots, font=F(18, "med"), fill=MUTED + (255,))


def phone(canvas: Image.Image, d, x, y, k, photo: Image.Image, t):
    w, h = 168, 320
    x = x + (1 - k) * 260
    d.rounded_rectangle([x, y, x + w, y + h], 26, fill=(8, 9, 12, 255), outline=(90, 94, 104, 255), width=3)
    d.rounded_rectangle([x + 10, y + 34, x + w - 10, y + h - 40], 8, fill=(30, 33, 40, 255))
    ph = photo.resize((w - 28, int((w - 28) * photo.height / photo.width)))
    canvas.paste(ph, (int(x + 14), int(y + 60)))
    d.text((x + 16, y + 12), "Trova", font=F(14, "demi"), fill=INK + (255,))
    d.rounded_rectangle([x + 24, y + h - 34, x + w - 24, y + h - 10], 10, fill=ACCENT + (255,))
    d.text((x + w / 2, y + h - 22), "Find", font=F(14, "bold"), fill=(255, 255, 255, 255), anchor="mm")


def query_photo() -> Image.Image:
    """The judge's photo: the same kind of scissors, other angle, on a grey desk, a bit blurred."""
    import scene
    img = np.full((300, 300, 3), (120, 118, 112), np.uint8)
    o = OBJ["scissors"].at(150, 150, 55)
    scene.draw_obj(img, o)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


PHOTO = query_photo()


# ------------------------------------------------------------------ captions and chapters
CAPTIONS = [
    (4.0, 9.0, "", "Before every new batch, an operator checks the line by eye and signs a paper checklist."),
    (9.0, 22.0, "1 / 5  Find by words", "The jury asks in plain English or Italian. When two objects fit, Grok asks back instead of guessing."),
    (22.0, 32.0, "2 / 5  Find by photo", "A judge photographs an object on their own phone. The arm points at the match."),
    (32.0, 41.6, "4 / 5  What changed", "Save the approved state. Then someone tampers with the table, and the light changes."),
    (41.6, 52.0, "4 / 5  What changed",
     f"A raw pixel difference only says {len(NAIVE)} regions differ. Trova says what changed, and why it matters."),
    (52.0, 58.0, "5 / 5  The record", "Trova assists, the operator decides and signs. That is what the draft EU GMP Annex 22 asks."),
    (58.0, 66.0, "Under the hood", "Plain code finds, measures and moves. Grok decides what the person means."),
    (66.0, 76.0, "The business", "A line-clearance copilot for regulated packaging lines."),
]


def chrome(d, t):
    d.text((24, 20), "Trova", font=F(26, "bold"), fill=INK + (255,))
    d.text((112, 29), "Grok Hackathon Pavia · Track A · CTRL+F: SEA", font=F(15, "med"), fill=MUTED + (255,))
    b = "Concept demo · Grok replies are illustrative"
    f = F(13, "demi")
    bw = f.getlength(b) + 24
    d.rounded_rectangle([W - 24 - bw, 22, W - 24, 48], 13, outline=(224, 176, 76, 255), width=1)
    d.text((W - 24 - bw / 2, 35), b, font=f, fill=(224, 176, 76, 255), anchor="mm")
    for a, b2, chap, text in CAPTIONS:
        if a <= t < b2:
            k = min(ramp(t, a, a + 0.4), 1 - ramp(t, b2 - 0.3, b2))
            al = int(255 * k)
            d.rounded_rectangle([24, 596, W - 24, 696], 14, fill=(20, 22, 28, al))
            x = 44
            if chap:
                fc = F(15, "bold")
                cw = fc.getlength(chap) + 20
                d.rounded_rectangle([x, 612, x + cw, 638], 13, fill=ACCENT + (al,))
                d.text((x + cw / 2, 625), chap, font=fc, fill=(255, 255, 255, al), anchor="mm")
            lines = wrap(text, F(24, "med"), W - 100)
            for i, line in enumerate(lines[:2]):
                d.text((44, 646 + i * 30 - (0 if chap else 22)), line, font=F(24, "med"), fill=INK + (al,))


# ------------------------------------------------------------------ the frames
RX, RW = 848, 408


def frame_old(t: float) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas, "RGBA")

    if t < 4.0:          # title
        k = ramp(t, 0.2, 1.2)
        d.text((W / 2, 290), "Trova", font=F(96, "bold"), fill=INK + (int(255 * k),), anchor="mm")
        d.text((W / 2, 370), "Find anything on the line. Know what changed. Sign it off.", font=F(30, "med"),
               fill=MUTED + (int(255 * ramp(t, 0.8, 1.8)),), anchor="mm")
        d.text((W / 2, 430), "Grok 4.7 vision + a cobot · Grok Hackathon Pavia · Track A, proposed by SEA Vision",
               font=F(18, "reg"), fill=MUTED + (int(200 * ramp(t, 1.4, 2.4)),), anchor="mm")
        return canvas

    if 4.0 <= t < 52.0:  # live table scenes
        img = cv2.cvtColor(table_frame(t), cv2.COLOR_BGR2RGB)
        view = Image.fromarray(cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA))
        mask = Image.new("L", (VW, VH), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, VW, VH], 14, fill=255)
        canvas.paste(view, (VX, VY), mask)
        d.text((VX + 12, VY + VH - 26), "camera 2 · top-down", font=F(13, "demi"), fill=(60, 60, 60, 200))
        scene_overlays(canvas, d, t)

    if 52.0 <= t < 58.0:
        record_slide(d, t)
    if 58.0 <= t < 66.0:
        hood_slide(d, t)
    if 66.0 <= t < 76.0:
        market_slide(d, t)
    if t >= 76.0:
        k = ramp(t, 76.2, 77.2)
        d.text((W / 2, 300), "Trova", font=F(80, "bold"), fill=INK + (int(255 * k),), anchor="mm")
        d.text((W / 2, 370), "Find it. Check it. Sign it.", font=F(30, "med"), fill=MUTED + (int(255 * k),),
               anchor="mm")
        d.text((W / 2, 430), "Runs on the event's cobots: UR3e · UR5e · xArm5 · Elite CS63 / CS66",
               font=F(18, "reg"), fill=MUTED + (int(220 * ramp(t, 77, 78)),), anchor="mm")
    chrome(d, t)
    return canvas


def scene_overlays(canvas, d, t):
    y = VY
    # ---- 1: find by words, with a clarifying question
    if 9.0 <= t < 22.0:
        draw_boxes(d, t, 10.4, highlight=PEN if t >= 16.0 else None, dim=t >= 16.0)
        if t < 14.6:
            y = card(d, RX, y, RW, "JURY", typed("Find something to write with", t, 9.0) or " ",
                     "Italian works too: «trova qualcosa per scrivere»")
        if 10.4 <= t < 14.6:
            d.text((RX, y + 10), f"{len(BOXES)} objects boxed by plain code (OpenCV, no model)", font=F(14, "med"),
                   fill=MUTED + (int(255 * ramp(t, 10.4, 10.8)),))
            y += 34
        if 11.6 <= t < 12.3:
            thinking(d, RX, y, RW, t)
        if t >= 12.3:
            y = card(d, RX, y, RW, "GROK 4.7 · 1.9 s", f"I see two things to write with: the blue pen ({PEN}) and the black marker ({MARKER}). Which one?",
                     "It asks instead of guessing.",
                     head_color=ACCENT, alpha=ramp(t, 12.3, 12.7), accent=ACCENT if t < 14.6 else None,
                     body_size=21 if t < 14.6 else 18) + 10
        if t >= 14.6:
            y = card(d, RX, y, RW, "JURY", typed("The pen", t, 14.6, 14) or " ") + 10
        if 15.4 <= t < 16.0:
            thinking(d, RX, y, RW, t)
        if t >= 16.0:
            y = card(d, RX, y, RW, "GROK 4.7 · 1.4 s", f"{PEN} · blue pen", "The only pen on the table.",
                     head_color=GREEN, alpha=ramp(t, 16.0, 16.3), accent=GREEN, body_size=24) + 10
        if t >= 18.2:
            reticle(d, (OBJ["pen"].x, OBJ["pen"].y), t)
            d.text((RX, y + 4), f"Arm to {mm((OBJ['pen'].x, OBJ['pen'].y))}", font=F(17, "demi"), fill=CYAN + (255,))
            d.text((RX, y + 28), "Request to pointing: 4.2 s", font=F(15, "med"), fill=MUTED + (255,))
    # ---- 2: find by photo
    if 22.0 <= t < 32.0:
        draw_boxes(d, t, 22.0, highlight=SCIS if t >= 25.3 else None, dim=t >= 25.3)
        phone(canvas, d, RX + RW - 170, VY + 150, ramp(t, 22.5, 23.4), PHOTO, t)
        cw = RW - 190
        y = card(d, RX, y, cw, "JURY'S PHONE", typed("[photo] Find this", t, 23.5, 18) or " ",
                 alpha=ramp(t, 23.4, 23.7))
        if 24.5 <= t < 25.3:
            thinking(d, RX, y + 10, cw, t, "Grok is comparing")
        if t >= 25.3:
            y = card(d, RX, y + 10, cw, "GROK 4.7 · 2.1 s", f"{SCIS} · scissors",
                     "Same shape and orange handles as the photo, seen from another angle.",
                     head_color=GREEN, alpha=ramp(t, 25.3, 25.6), accent=GREEN, body_size=24)
        if 27.4 <= t < 31.0:
            reticle(d, (OBJ["scissors"].x, OBJ["scissors"].y), t)
            d.text((RX, y + 14), f"Arm to {mm((OBJ['scissors'].x, OBJ['scissors'].y))}", font=F(16, "demi"),
                   fill=CYAN + (255,))
    # ---- 3: what changed
    if 32.0 <= t < 52.0:
        if t < 34.0:
            draw_boxes(d, t, 32.2)
        if t >= 33.0:
            y = card(d, RX, y, RW, "REFERENCE SAVED · 16:02:11", f"{len(BOXES)} objects in the approved state",
                     alpha=ramp(t, 33.0, 33.3)) + 10
        if 34.0 <= t < 41.6:
            occ = hand_in_view(t)
            col = ORANGE if occ else MUTED
            msg = "Hand in view: waiting, nothing is judged" if occ else "View clear"
            d.rounded_rectangle([RX, y, RX + RW, y + 44], 12, fill=PANEL + (255,), outline=col + (255,))
            d.text((RX + 16, y + 11), msg, font=F(17, "demi"), fill=col + (255,))
            y += 54
            if t >= 40.3:
                d.text((RX, y), "Light got warmer: normalised against the reference", font=F(15, "med"),
                       fill=MUTED + (int(255 * ramp(t, 40.3, 40.8)),))
                y += 30
        if 41.6 <= t < 42.6:
            dots = int((t - 41.6) * 3) + 1
            d.text((RX, y), f"Checking: frame {min(dots, 3)} of 3", font=F(17, "demi"), fill=INK + (255,))
        if 42.6 <= t < 44.4:
            for b in NAIVE:
                d.rectangle(vbox(b, 3), outline=RED + (230,), width=2)
            d.text((VX + 14, VY + 12), f"Raw pixel difference: {len(NAIVE)} regions, no meaning", font=F(20, "bold"),
                   fill=RED + (255,))
        if t >= 44.4:
            dashed_rect(d, vbox(MOVED.ref_box, 3), BLUE)          # where the scissors were
            p, q = MOVED.ref_box.center, MOVED.cur_box.center
            arrow(d, (VX + p[0] * SC, VY + p[1] * SC), (VX + q[0] * SC, VY + q[1] * SC), BLUE)
            d.rectangle(vbox(MOVED.cur_box, 3), outline=BLUE + (255,), width=4)
            tag(d, VX + MOVED.cur_box.x0 * SC, VY + MOVED.cur_box.y0 * SC - 3, "C2 moved", BLUE)
            d.rectangle(vbox(ADDED.cur_box, 3), outline=ORANGE + (255,), width=4)
            tag(d, VX + ADDED.cur_box.x0 * SC, VY + ADDED.cur_box.y0 * SC - 3, "C1 added", ORANGE)
            d.text((VX + 14, VY + 12), "Trova: 2 changes", font=F(20, "bold"), fill=GREEN + (255,))
        if t >= 44.8:
            y = card(d, RX, y, RW, "GROK 4.7 · 2.6 s · C1 ADDED · RISK HIGH",
                     "A USB key next to the blister pack: a foreign object, mix-up risk.",
                     "IT: chiavetta USB accanto al blister, oggetto estraneo.", head_color=ORANGE,
                     alpha=ramp(t, 44.8, 45.1), body_size=19) + 10
        if t >= 45.6:
            y = card(d, RX, y, RW, "C2 MOVED · RISK MEDIUM", f"Scissors moved {MOVE_CM:.0f} cm.",
                     f"IT: forbici spostate di {MOVE_CM:.0f} cm.", head_color=BLUE, alpha=ramp(t, 45.6, 45.9), body_size=19) + 10
        if t >= 46.6:
            k = ramp(t, 46.6, 47.0)
            d.rounded_rectangle([RX, y, RX + RW, y + 50], 12, fill=RED + (int(60 * k),), outline=RED + (int(255 * k),), width=2)
            d.text((RX + RW / 2, y + 25), "LINE NOT CLEAR · linea non libera", font=F(20, "bold"),
                   fill=RED + (int(255 * k),), anchor="mm")


def record_slide(d, t):
    k = ramp(t, 52.0, 52.6)
    x0, y0, w = 190, 76, 900
    d.rounded_rectangle([x0, y0, x0 + w, y0 + 440], 16, fill=(246, 246, 243, int(255 * k)))
    ink, muted = (20, 22, 26, int(255 * k)), (90, 95, 105, int(255 * k))
    d.text((x0 + 32, y0 + 26), "Line clearance record", font=F(30, "bold"), fill=ink)
    d.text((x0 + 32, y0 + 70), "Check 0f96 · 2026-11-05 16:04 · camera 2 · grok-4.7", font=F(16, "med"), fill=muted)
    d.text((x0 + 32, y0 + 108), "NOT CLEAR", font=F(28, "bold"), fill=(179, 38, 30, int(255 * k)))
    rows = [("C1", "added", "USB key", "high", "Foreign object next to the blister pack: mix-up risk."),
            ("C2", "moved", "scissors", "medium", f"Moved {MOVE_CM:.0f} cm from the approved position.")]
    yy = y0 + 160
    for c, kind, obj, risk, why in rows:
        if t < 52.6 + (0.4 if c == "C2" else 0):
            continue
        d.line([x0 + 32, yy - 8, x0 + w - 32, yy - 8], fill=(210, 210, 205, int(255 * k)), width=1)
        d.text((x0 + 32, yy), f"{c}  {kind}", font=F(18, "demi"), fill=ink)
        d.text((x0 + 170, yy), obj, font=F(18, "demi"), fill=ink)
        d.text((x0 + 300, yy), risk, font=F(18, "demi"), fill=(179, 38, 30, int(255 * k)) if risk == "high" else (154, 103, 0, int(255 * k)))
        d.text((x0 + 410, yy), why, font=F(17, "reg"), fill=ink)
        yy += 44
    d.text((x0 + 32, yy + 10), "Rejected as false alarms: light change across the whole table", font=F(16, "reg"), fill=muted)
    if t >= 54.0:
        kk = ramp(t, 54.0, 54.5)
        d.rounded_rectangle([x0 + 32, yy + 60, x0 + w - 32, yy + 150], 12, outline=(19, 115, 51, int(255 * kk)), width=2)
        d.text((x0 + 52, yy + 76), "Signed off by: M. Rossi, operator · REJECTED, line re-cleared",
               font=F(19, "demi"), fill=(19, 115, 51, int(255 * kk)))
        d.text((x0 + 52, yy + 108), "Every request, Grok decision and sign-off is in the audit log.", font=F(16, "reg"),
               fill=muted[:3] + (int(255 * kk),))


def hood_slide(d, t):
    boxes = [("Camera", "phone or webcam", 60, 150), ("Object boxes", "plain code, OpenCV", 330, 150),
             ("Grok 4.7 vision", "picks, asks, verifies", 600, 150), ("Calibration", "pixels to millimetres", 870, 150),
             ("Cobot", "UR · xArm · Elite", 870, 330), ("What changed", "match + 2 of 3 frames", 330, 330),
             ("Record", "operator signs", 600, 330), ("Grok Voice", "talk · confirm · stop", 60, 330)]
    for i, (name, sub, x, y) in enumerate(boxes):
        k = ramp(t, 58.2 + i * 0.25, 58.6 + i * 0.25)
        grok = name.startswith("Grok")
        col = ACCENT if grok else LINE
        d.rounded_rectangle([x, y, x + 230, y + 96], 14, fill=(PANEL if not grok else (52, 24, 50)) + (int(255 * k),),
                            outline=col + (int(255 * k),), width=2 if grok else 1)
        d.text((x + 115, y + 36), name, font=F(22, "bold"), fill=INK + (int(255 * k),), anchor="mm")
        d.text((x + 115, y + 66), sub, font=F(16, "med"), fill=MUTED + (int(255 * k),), anchor="mm")
    k = ramp(t, 60.0, 60.6)
    a = int(255 * k)
    for (x0, y0, x1, y1) in [(290, 198, 330, 198), (560, 198, 600, 198), (830, 198, 870, 198), (985, 246, 985, 330),
                             (445, 246, 445, 330), (560, 378, 600, 378), (715, 246, 715, 330), (290, 378, 330, 378)]:
        arrow(d, (x0, y0), (x1, y1), MUTED, a)
    d.text((W / 2, 490), "Grok is essential, and boxed in: it never moves the arm, and every decision is checked by code.",
           font=F(20, "med"), fill=INK + (int(255 * ramp(t, 61, 61.6)),), anchor="mm")
    d.text((W / 2, 524), "Optional: a robot dog searches the other rooms. Measured against baselines: found /10, false alarms, time.",
           font=F(18, "reg"), fill=MUTED + (int(255 * ramp(t, 61.5, 62.1)),), anchor="mm")


def market_slide(d, t):
    stats = [("$1.0–1.2B", "pharma inspection machines, 2025", "analyst estimates"),
             ("Every changeover", "line clearance is checked by eye, on paper", "a missed leftover can mean a recall"),
             ("7,000+", "SEA Vision inspection systems installed", "the channel, in Pavia")]
    for i, (big, line, note) in enumerate(stats):
        k = ramp(t, 66.3 + i * 0.4, 66.8 + i * 0.4)
        a = int(255 * k)
        x = 60 + i * 400
        d.rounded_rectangle([x, 100, x + 370, 330], 16, fill=PANEL + (a,), outline=LINE + (a,))
        d.text((x + 24, 128), big, font=F(38 if len(big) < 10 else 30, "bold"), fill=ACCENT + (a,))
        for j, ln in enumerate(wrap(line, F(20, "demi"), 320)):
            d.text((x + 24, 196 + j * 28), ln, font=F(20, "demi"), fill=INK + (a,))
        d.text((x + 24, 290), note, font=F(15, "reg"), fill=MUTED + (a,))
    k = ramp(t, 68.2, 68.8)
    d.rounded_rectangle([60, 360, 1230, 560], 16, fill=PANEL + (int(255 * k),), outline=GREEN + (int(255 * k),), width=2)
    d.text((88, 384), "The pilot we propose", font=F(16, "demi"), fill=GREEN + (int(255 * k),))
    d.text((88, 414), "Two weeks on one SEA Vision demo line.", font=F(26, "bold"), fill=INK + (int(255 * k),))
    d.text((88, 456), "Trova saves the reference at each changeover, flags leftovers, and the operator signs off.",
           font=F(19, "med"), fill=INK + (int(255 * ramp(t, 68.8, 69.3)),))
    d.text((88, 490), "We measure clearance time and missed items against the paper checklist.", font=F(19, "med"),
           fill=INK + (int(255 * ramp(t, 69.2, 69.7)),))
    d.text((88, 524), "Market figures: analyst and vendor estimates, ranges differ by source.", font=F(14, "reg"),
           fill=MUTED + (int(255 * ramp(t, 69.6, 70)),))


# ------------------------------------------------------------------ new scenes: voice, optional dog
SEGMENTS = [(0.0, 32.0, 0.0), (44.0, 64.0, -12.0), (73.0, 101.0, -21.0)]   # (start, end, shift to old time)


def frame(t: float) -> Image.Image:
    for a, b, shift in SEGMENTS:
        if a <= t < b:
            return frame_old(t + shift)
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas, "RGBA")
    if 32.0 <= t < 44.0:
        voice_scene(canvas, d, t - 32.0)
    else:
        dog_scene(canvas, d, t - 64.0)
    return canvas


def top_bar(d):
    chrome(d, -1)   # bar only: no caption at old time -1


def caption(d, chap, text, k=1.0):
    al = int(255 * k)
    d.rounded_rectangle([24, 596, W - 24, 696], 14, fill=(20, 22, 28, al))
    fc = F(15, "bold")
    cw = fc.getlength(chap) + 20
    d.rounded_rectangle([44, 612, 44 + cw, 638], 13, fill=ACCENT + (al,))
    d.text((44 + cw / 2, 625), chap, font=fc, fill=(255, 255, 255, al), anchor="mm")
    for i, ln in enumerate(wrap(text, F(24, "med"), W - 100)[:2]):
        d.text((44, 646 + i * 30), ln, font=F(24, "med"), fill=INK + (al,))


def paste_view(canvas, d, img_bgr, label):
    view = Image.fromarray(cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (VW, VH), interpolation=cv2.INTER_AREA))
    mask = Image.new("L", (VW, VH), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, VW, VH], 14, fill=255)
    canvas.paste(view, (VX, VY), mask)
    d.text((VX + 12, VY + VH - 26), label, font=F(13, "demi"), fill=(60, 60, 60, 200))


def mic(d, x, y, t, on=True):
    r = 14 + (3 * math.sin(t * 8) if on else 0)
    d.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], outline=ACCENT + (120,), width=2)
    d.rounded_rectangle([x - 6, y - 12, x + 6, y + 6], 6, fill=ACCENT + (255,))
    d.arc([x - 11, y - 6, x + 11, y + 12], 0, 180, fill=ACCENT + (255,), width=2)
    d.line([x, y + 12, x, y + 17], fill=ACCENT + (255,), width=2)


MUG = (OBJ["mug_red"].x, OBJ["mug_red"].y)


def voice_tip(tl):
    if tl < 4.8:
        return PARK
    if tl < 9.6:
        k = 0.55 * min(1.0, (tl - 4.8) / 1.8)          # moving ... frozen by "stop" at 6.6
        return lerp(PARK, MUG, k)
    return lerp(PARK, MUG, 0.55 + 0.45 * ramp(tl, 9.6, 11.0))


def voice_scene(canvas, d, tl):
    img = F0.copy()
    tip = voice_tip(tl)
    if tip != PARK:
        draw_arm(img, tip)
    paste_view(canvas, d, img, "camera 2 · top-down")
    top_bar(d)
    mug_id = BID["mug_red"]
    if tl >= 2.4:
        draw_boxes(d, 99, 0, highlight=mug_id, dim=True)
    y = VY
    d.rounded_rectangle([RX, y, RX + RW, y + 52], 12, fill=PANEL + (255,), outline=ACCENT + (255,))
    mic(d, RX + 30, y + 26, tl)
    d.text((RX + 60, y + 14), "Grok Voice · listening (English or Italian)", font=F(17, "demi"), fill=INK + (255,))
    y += 62
    if tl >= 0.6:
        y = card(d, RX, y, RW, "YOU (SPOKEN)", typed("Trova la tazza rossa", tl, 0.6, 16) or " ", "Find the red mug") + 8
    if tl >= 2.4:
        y = card(d, RX, y, RW, "GROK VOICE", "Ho trovato la tazza rossa. La indico?", "I found the red mug. Shall I point at it?",
                 head_color=ACCENT, alpha=ramp(tl, 2.4, 2.7), body_size=19) + 8
    if 4.2 <= tl < 7.6:
        y = card(d, RX, y, RW, "YOU", "Sì, vai", "Yes, go", body_size=19) + 8
    if 6.4 <= tl < 9.0:
        d.rounded_rectangle([RX, y, RX + RW, y + 64], 12, fill=RED + (50,), outline=RED + (255,), width=2)
        d.text((RX + 16, y + 8), "YOU: «Stop!»", font=F(20, "bold"), fill=RED + (255,))
        if tl >= 6.6:
            d.text((RX + 16, y + 36), "Arm halted on the word itself, no model in the loop", font=F(15, "demi"),
                   fill=INK + (255,))
        y += 74
    if 7.6 <= tl < 9.0:
        y = card(d, RX, y, RW, "GROK VOICE", "Fermo. Dimmi quando riprendere.", "Stopped. Tell me when to continue.",
                 head_color=ACCENT, body_size=19) + 8
    if tl >= 9.0:
        y = card(d, RX, y, RW, "YOU (SWITCHES TO ENGLISH)", "Okay, go on", body_size=19) + 8
        y = card(d, RX, y, RW, "GROK VOICE", "Pointing at the red mug.", head_color=ACCENT, body_size=19,
                 alpha=ramp(tl, 9.3, 9.6)) + 8
    if tl >= 11.0:
        reticle(d, MUG, tl)
    caption(d, "3 / 5  Talk to it", "Speak English or Italian. It asks before the arm moves, and \"stop\" halts it at once.",
            k=min(ramp(tl, 0, 0.4), 1 - ramp(tl, 11.7, 12.0)))


ROOMS = [("Lab · Trova table", 24, 40, 270, 452), ("Room 2", 290, 40, 530, 452), ("Room 3", 550, 40, 776, 452)]
SPOTS = {"start": (150, 250), "desk": (410, 120), "shelf": (410, 380), "bench": (660, 250)}
DOG_PATH = [("start", 3.3), ("desk", 4.6), ("shelf", 6.3)]


def dog_photo():
    import scene as sc
    t2 = sc.make_table(seed=9)
    t2 = np.clip(t2.astype(np.float32) * np.array([0.82, 0.86, 0.9]), 0, 255).astype(np.uint8)
    objs = [sc.Obj("cup", "mug", 520, 380, color=(70, 160, 70)), sc.Obj("box", "stapler", 860, 300, 70),
            sc.Obj("bl", "blister", 300, 560, 15)]
    img = sc.render(t2, objs)
    cv2.rectangle(img, (440, 300), (630, 460), (60, 220, 60), 6)
    return Image.fromarray(cv2.cvtColor(cv2.resize(img, (384, 240)), cv2.COLOR_BGR2RGB))


DOG_PHOTO = dog_photo()


def dog_pos(tl):
    if tl < DOG_PATH[0][1]:
        return SPOTS["start"]
    for (a, ta), (b, tb) in zip(DOG_PATH, DOG_PATH[1:]):
        if tl < tb:
            return lerp(SPOTS[a], SPOTS[b], ramp(tl, ta, tb))
    return SPOTS["shelf"]


def V(p):
    return VX + p[0], VY + p[1]


def dog_scene(canvas, d, tl):
    d.rounded_rectangle([VX, VY, VX + VW, VY + VH], 14, fill=(22, 25, 31, 255), outline=LINE + (255,))
    for name, x0, y0, x1, y1 in ROOMS:
        d.rounded_rectangle([VX + x0, VY + y0, VX + x1, VY + y1], 10, outline=MUTED + (150,), width=2)
        d.text((VX + x0 + 14, VY + y0 + 10), name, font=F(15, "demi"), fill=MUTED + (255,))
    tx, ty = V(SPOTS["start"])
    d.rounded_rectangle([tx - 60, ty - 36, tx + 60, ty + 36], 8, outline=ACCENT + (180,), width=2)
    d.text((tx, ty + 52), "Trova table + arm", font=F(13, "med"), fill=MUTED + (255,), anchor="mm")
    route = ["start", "desk", "shelf", "bench"]
    for a, b in zip(route, route[1:]):
        col = (90, 96, 110, 255) if b != "bench" else (60, 64, 74, 255)
        d.line([V(SPOTS[a]), V(SPOTS[b])], fill=col, width=2)
    for key in ("desk", "shelf", "bench"):
        x, y = V(SPOTS[key])
        done = (key == "desk" and tl >= 4.6) or (key == "shelf" and tl >= 6.3)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(GREEN if done else MUTED) + (230,), width=2)
        d.text((x + 16, y - 9), key, font=F(14, "med"), fill=MUTED + (230,))
    x, y = V(dog_pos(tl))
    d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=ORANGE + (255,))
    d.text((x, y + 26), "dog (sim)", font=F(13, "demi"), fill=ORANGE + (255,), anchor="mm")
    d.text((VX + 24, VY + VH - 30), "Optional scenario · fixed route, no autonomous navigation", font=F(13, "demi"),
           fill=MUTED + (220,))
    top_bar(d)
    yy = card(d, RX, VY, RW, "YOU", typed("Find the green mug", tl, 0.2, 18) or " ") + 8
    if 1.4 <= tl < 6.6:
        yy = card(d, RX, yy, RW, "GROK", "It's not on the table. Shall I send the dog to look in the other rooms?",
                  head_color=ACCENT, alpha=ramp(tl, 1.4, 1.7), body_size=19) + 8
    if 2.8 <= tl < 6.6:
        yy = card(d, RX, yy, RW, "YOU", "Yes", body_size=19) + 8
    if 3.3 <= tl < 6.6:
        spot = "Room 2, desk" if tl < 4.6 else "Room 2, shelf"
        d.text((RX, yy + 6), f"Dog walking to {spot}: Grok checks its camera", font=F(16, "demi"), fill=ORANGE + (255,))
        if tl >= 4.8:
            d.text((RX, yy + 32), "Room 2, desk: not there", font=F(15, "med"), fill=MUTED + (255,))
    if tl >= 6.6:
        k = ramp(tl, 6.6, 7.0)
        a = int(255 * k)
        d.rounded_rectangle([RX, yy, RX + RW, yy + 316], 12, fill=PANEL + (a,), outline=GREEN + (a,), width=2)
        d.text((RX + 16, yy + 12), "FOUND · DOG CAMERA · ROOM 2, SHELF", font=F(14, "demi"), fill=GREEN + (a,))
        canvas.paste(DOG_PHOTO.resize((376, 235)), (RX + 16, yy + 40))
        d.text((RX + 16, yy + 284), "Green mug, on the shelf left of the black box.", font=F(16, "demi"), fill=INK + (a,))
    caption(d, "Optional  Robot dog", "Not on the table? The dog walks a fixed route and Grok checks its camera at each stop.",
            k=min(ramp(tl, 0, 0.4), 1 - ramp(tl, 8.7, 9.0)))


def main():
    out = HERE / "trova_concept.mp4"
    stills = "--stills" in sys.argv
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
           "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    keys = {int(s * FPS) for s in (13.5, 19, 35, 39, 43.5, 51, 59.5, 68, 72, 84, 99)}
    for i in range(n):
        img = frame(i / FPS)
        p.stdin.write(img.tobytes())
        if stills and i in keys:
            (HERE / "stills").mkdir(exist_ok=True)
            img.save(HERE / "stills" / f"t{i / FPS:05.1f}.png")
        if i % 300 == 0:
            print(f"  {i / FPS:4.0f}s / {DUR:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) · naive alarms {len(NAIVE)} · moved {MOVE_CM:.1f} cm")


if __name__ == "__main__":
    main()
