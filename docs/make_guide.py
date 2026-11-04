"""Build the Trova morning guide PDF for the team."""
import sys

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/Trova_Morning_Guide.pdf"
SUP = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Arial", SUP + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", SUP + "Arial Bold.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

INK = colors.HexColor("#16181D")
MUTED = colors.HexColor("#5D6370")
ACCENT = colors.HexColor("#B4009E")
ACCENT_BG = colors.HexColor("#FBEAF8")
LINE = colors.HexColor("#D9D9D3")
HEAD_BG = colors.HexColor("#F3F2EE")
GREEN = colors.HexColor("#137333")
GREEN_BG = colors.HexColor("#E8F4EC")
WARN_BG = colors.HexColor("#FFF5E0")
WARN = colors.HexColor("#8A5A00")

base = ParagraphStyle("base", fontName="Arial", fontSize=9.6, leading=13.2, textColor=INK, alignment=TA_LEFT)
small = ParagraphStyle("small", parent=base, fontSize=8.6, leading=11.4)
muted = ParagraphStyle("muted", parent=small, textColor=MUTED)
title = ParagraphStyle("title", parent=base, fontName="Arial-Bold", fontSize=24, leading=28, spaceAfter=2)
subtitle = ParagraphStyle("subtitle", parent=base, fontSize=10.5, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=14.5, leading=18, spaceBefore=10, spaceAfter=6,
                    textColor=INK)
h2 = ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=10.6, leading=14, spaceBefore=6, spaceAfter=3)
cell = ParagraphStyle("cell", parent=base, fontSize=8.8, leading=11.6)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Arial-Bold")
code = ParagraphStyle("code", fontName="Courier", fontSize=8.3, leading=11, textColor=INK, backColor=HEAD_BG,
                      borderPadding=(6, 6, 6, 6), leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=8)


def P(t, s=base):
    return Paragraph(t, s)


def bullets(items, s=base):
    return [Paragraph(f"<bullet>&bull;</bullet>{t}", ParagraphStyle("b", parent=s, leftIndent=12, bulletIndent=2,
                                                                     spaceAfter=2)) for t in items]


def numbered(items, s=base):
    return [Paragraph(f"<bullet>{i}.</bullet>{t}", ParagraphStyle("n", parent=s, leftIndent=14, bulletIndent=0,
                                                                   spaceAfter=3)) for i, t in enumerate(items, 1)]


def box(flowables, bg=ACCENT_BG, border=ACCENT, width=None):
    t = Table([[flowables]], colWidths=[width or 174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, border),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def table(rows, widths, head=True):
    data = [[P(c, cellb if (head and i == 0) else cell) if isinstance(c, str) else c for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    t.setStyle(TableStyle(st))
    return t


def flow_diagram():
    import math
    W, H = 174 * mm, 46 * mm
    d = Drawing(W, H)
    bw, bh = 29 * mm, 14 * mm
    gap = (W - 5 * bw) / 4
    top_y, low_y = H - bh - 1 * mm, 1 * mm
    xs = [i * (bw + gap) for i in range(5)]
    grey = colors.HexColor("#7A808C")

    def node(x, y, name, sub, hi=False):
        d.add(Rect(x, y, bw, bh, rx=6, ry=6, fillColor=ACCENT_BG if hi else colors.white,
                   strokeColor=ACCENT if hi else colors.HexColor("#9AA0AB"), strokeWidth=1.4 if hi else 0.9))
        d.add(String(x + bw / 2, y + bh - 5.4 * mm, name, fontName="Arial-Bold", fontSize=8.4, fillColor=INK,
                     textAnchor="middle"))
        d.add(String(x + bw / 2, y + 3.2 * mm, sub, fontName="Arial", fontSize=7.2, fillColor=MUTED,
                     textAnchor="middle"))

    def arrow(x0, y0, x1, y1):
        d.add(Line(x0, y0, x1, y1, strokeColor=grey, strokeWidth=1))
        a = math.atan2(y1 - y0, x1 - x0)
        d.add(Polygon([x1, y1, x1 - 5 * math.cos(a - 0.45), y1 - 5 * math.sin(a - 0.45),
                       x1 - 5 * math.cos(a + 0.45), y1 - 5 * math.sin(a + 0.45)], fillColor=grey, strokeColor=None))

    for i, (n, sub) in enumerate([("Camera", "fixed, top-down"), ("Object boxes", "plain code, no model"),
                                  ("Grok 4.7", "picks a box, or asks"), ("Calibration", "pixels to robot mm"),
                                  ("Cobot", "points, then parks")]):
        node(xs[i], top_y, n, sub, hi=(i == 2))
    for i in range(4):
        arrow(xs[i] + bw, top_y + bh / 2, xs[i + 1] - 1, top_y + bh / 2)
    for i, (n, sub) in zip((1, 2, 3), [("Reference", "approved table"), ("What changed", "Grok checks it"),
                                        ("Record", "operator signs")]):
        node(xs[i], low_y, n, sub, hi=(i == 2))
    arrow(xs[1] + bw / 2, top_y - 1, xs[1] + bw / 2, low_y + bh + 1)
    arrow(xs[2] + bw / 2, top_y - 1, xs[2] + bw / 2, low_y + bh + 1)
    arrow(xs[1] + bw, low_y + bh / 2, xs[2] - 1, low_y + bh / 2)
    arrow(xs[2] + bw, low_y + bh / 2, xs[3] - 1, low_y + bh / 2)
    return d


def on_page(c, doc):
    c.saveState()
    c.setFont("Arial", 7.6)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Trova · morning start guide · Grok Hackathon Pavia, 4-5 Nov 2026")
    c.drawRightString(192 * mm, 10 * mm, f"page {doc.page}")
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(18 * mm, 287 * mm, 192 * mm, 287 * mm)
    c.restoreState()


s = []
# ------------------------------------------------------------------ page 1: the essentials
s += [P("Trova: morning start guide", title),
      P("Grok Hackathon Pavia · Wed 4 Nov 2026 · Track A: CTRL+F: SEA (SEA Vision)",
        subtitle)]
s.append(box([P("<b>What we are building, in 30 seconds</b>", h2),
              P("You ask for an object, typed, spoken (English or Italian) or as a phone photo. Trova finds it on the "
                "table and the cobot points at it. Then it compares the table with an approved state and reports what "
                "was added, removed or moved. That is <b>line clearance</b>, the check SEA Vision sells to pharma "
                "packaging lines. <b>Nothing is trained:</b> Grok already knows what objects are.")]))
s.append(Spacer(1, 7))
s.append(box([P("<b>The code is already written.</b> It runs today in demo mode on a simulated table and arm "
                "(25 tests pass). Your job is to connect the real camera and arm, switch on live Grok, tune on the "
                "real objects, measure, and rehearse.", base)], bg=GREEN_BG, border=GREEN))
s.append(Spacer(1, 4))
s.append(P("The first three things at 09:00", h1))
s += numbered(["<b>Claim Track A</b> with the organisers. Only 2-3 teams can take it.",
               "<b>Ask for the arm:</b> UFactory xArm5 first, UR5e second. Get its IP address and the slot calendar.",
               "<b>Get the Grok API key</b> into the file <font face='Courier'>.env</font>. The badge in the app "
               "turns from MOCK to LIVE."])
s.append(P("Who does what (write the names in)", h1))
s.append(table([["Role", "Owns", "Name"],
                ["P1 Arm", "Connecting and calibrating the cobot; STOP; drives the arm in the demo", ""],
                ["P2 Camera and vision", "Mounting the camera, empty-table capture, object boxes, markers", ""],
                ["P3 Grok", "API key, prompts, speed of each find", ""],
                ["P4 What changed and tests", "Test set, numbers against baselines, README results", ""],
                ["P5 Demo, voice, pitch", "Questions for SEA Vision, Grok Voice test, slides, backup video", ""]],
               [42, 102, 30]))
s.append(Spacer(1, 8))
s.append(box([P("<b>Golden rules.</b> Staff present for every arm move. STOP always within reach (button, space bar, "
                "or say \"stop\" / \"fermo\"). Grok never moves the arm by itself: every move goes through code with a "
                "safety box. Never commit <font face='Courier'>.env</font>. Feature freeze 02:00, code freeze Thursday "
                "12:00, demo Thursday 16:00.", base)], bg=WARN_BG, border=WARN))

s.append(P("Before you leave home: bring", h1))
s += bullets(["Laptop with its charger, and the repo already installed (see section 3).",
              "<b>USB-C to Ethernet adapter and an Ethernet cable.</b> The arms connect by cable and MacBooks have no "
              "Ethernet port. Without this we cannot talk to the arm.",
              "A phone, a clamp or small tripod, and a USB cable: a second, sharper camera (iPhone works as a webcam).",
              "4 printed ArUco markers, 5 cm wide, and tape. Print them from the app: Setup, Print markers.",
              "A phone hotspot (venue Wi-Fi carries 100 people) and an extension lead."])

# ------------------------------------------------------------------ 1. hardware
s.append(PageBreak())
s.append(P("1. The hardware", h1))
s.append(P("Ask the organisers for", h2))
s.append(table([["Hardware", "Choice", "Why"],
                ["UFactory xArm5", "<b>1st</b>", "Official Python SDK, no pendant setting to change. 700 mm reach covers "
                 "the table."],
                ["Universal Robots UR5e", "<b>2nd</b>", "Best reach (850 mm). Staff must enable <i>Remote Control</i> on "
                 "its pendant, which may need an admin password."],
                ["Elite Robots CS66", "3rd", "914 mm reach, but our least-tested driver."],
                ["UR3e, Elite CS63", "Avoid", "Reach too short for a full table (500 / 624 mm)."],
                ["Camera 2 (the station's)", "Yes", "The top-down camera of the Track A station."],
                ["Unitree Go2 / Go1 dog", "Skip for now", "Optional scenario. Photos of the rooms are enough for the "
                 "simulated dog; ask for a Go2 slot only if the core demo already works."]],
               [42, 24, 108]))
# ------------------------------------------------------------------ 2. how it works
s.append(P("2. How the setup works", h1))
s.append(flow_diagram())
s.append(P("Top row: a find. Bottom row: a line-clearance check. Typing, a phone photo or Grok Voice all start "
           "the same flows.", muted))
s.append(Spacer(1, 5))
s += bullets([
    "<b>Camera.</b> Fixed above the table, looking down. Once per arm slot we photograph the empty table. Anything that "
    "differs from that photo is an object and gets a numbered box. Shadows and light changes are ignored.",
    "<b>Grok.</b> It receives the photo with numbered boxes, close-ups of each object and the request. It answers with "
    "a box number, \"not found\", or a question when two objects fit (\"the blue pen or the black marker?\").",
    "<b>Calibration.</b> The arm tip is moved by hand (freedrive) onto 4-6 spots on the table, and each time you click "
    "the tip in the live view. Trova then knows which camera pixel is which robot position, in millimetres.",
    "<b>The arm.</b> It receives one target: x, y, about 15 cm above the table, pointing down. The robot works out its "
    "own joint angles. It moves in a straight line, slowly, inside a safety box, and parks out of the camera's view "
    "before each new photo. We never program left, right or forward: calibration takes care of directions.",
    "<b>What changed.</b> Save a reference of the approved table. Later, Trova compares object by object (position, "
    "colour, shape), Grok confirms each change and says why it matters, and the operator signs a record.",
    "<b>Voice.</b> On the laptop, press <i>Talk to Trova</i> and speak English or Italian. It asks before the arm "
    "moves; the word \"stop\" halts the arm immediately, without waiting for Grok."])
s.append(Spacer(1, 4))
s.append(box([P("<b>The objects must:</b> look different from the table (pale-on-pale and shiny metal can be missed), "
                "not touch each other (touching objects merge into one box), and sit inside both the camera view and the "
                "arm's reach. The station's 20-30 objects come from SEA Vision; we do not need to bring any.", small)]))

# ------------------------------------------------------------------ 3. how to make it
s.append(PageBreak())
s.append(P("3. How to make it: the morning, step by step", h1))
s.append(P("Before 09:00 (on the train): install and try the demo", h2))
s.append(Preformatted("git clone https://github.com/Aman-AI-Guy/Grok_Hackathon && cd Grok_Hackathon\n"
                      "python3 -m venv .venv && .venv/bin/pip install -e \".[dev,robots,ui]\"\n"
                      "cp .env.example .env\n"
                      ".venv/bin/python -m trova          # then open http://localhost:8000", code))
s.append(table([["Time", "Who", "What"],
                ["09:00", "P1 + P5", "Claim Track A. Ask for the xArm5 (or UR5e), its IP, and the arm slot calendar."],
                ["09:00", "P3", "Put the API key in <font face='Courier'>.env</font>, restart the app: badge shows LIVE. "
                 "Press <i>preflight</i> (header): fix every red item."],
                ["09:00", "P5", "Ask Claudio Peroni the questions in section 4."],
                ["09:15", "P2", "Mount the camera top-down; set <font face='Courier'>TROVA_CAMERA</font>; press "
                 "<i>Capture empty table</i>; tune until every object has exactly one box; record 10 minutes of station "
                 "video for offline work."],
                ["09:15", "P3", "Run <font face='Courier'>experiments/grok_bbox_test.py</font> on station photos; compare "
                 "the speed of grok-4.7 and the faster model."],
                ["09:30", "P4", "Start recording test cases: ask for an object, then press correct or wrong."],
                ["First arm slot", "P1 (+ P2)", "Connect, first slow move, test STOP, calibrate (steps below)."],
                ["13:00 target", "Everyone", "<b>First real find:</b> real camera, live Grok, real arm pointing."],
                ["Afternoon", "Everyone", "Measure, tune, live voice test, rehearse. Record the backup video tonight. "
                 "For the demo: <i>preflight</i>, set the batch, stick the <i>Print labels</i> sheet on cartons, open "
                 "<font face='Courier'>/jury</font> on the projector, reset the score, and show <i>Watch the line</i> "
                 "and <font face='Courier'>/history</font>."]],
               [24, 20, 130]))
s.append(P("The first arm slot (P1, with museum staff)", h2))
s += numbered([
    "Plug in the Ethernet adapter. In the Mac's network settings give it a manual IP on the arm's subnet, for example "
    "192.168.1.50, mask 255.255.255.0, and <b>no router</b>, so the internet stays on Wi-Fi. Ping the arm.",
    "UR or Elite only: staff enable <i>Remote Control</i> on the pendant. On every arm: set the TCP offset of the "
    "printed tool, otherwise positions refer to the flange, not the tip.",
    "Fill in <font face='Courier'>.env</font> (below), restart the app, make the first move at low speed, and press STOP "
    "in the middle of a move to check that it halts and holds.",
    "<i>Freedrive on</i>, put the tip on the table at 4-6 spread-out spots and click the tip each time "
    "(<i>Calibrate with arm</i>). Then touch each ArUco marker and record it, so later recalibration takes seconds."])
s.append(Preformatted("TROVA_ARM=xarm:192.168.1.xxx        # or ur:<ip>  or  elite:<ip>\n"
                      "TROVA_ARM_SPEED_MM_S=50\n"
                      "TROVA_ARM_Z_MM=150                  # tip height: table surface + 3-5 cm, measure it\n"
                      "TROVA_ARM_LIMITS=xmin,xmax,ymin,ymax,zmin,zmax   # REQUIRED: moves outside are refused\n"
                      "TROVA_ARM_HOME_MM=x,y,z             # REQUIRED: parked pose, out of the camera's view\n"
                      "TROVA_TOKEN=some-secret             # venue Wi-Fi: open the printed ?token= link once per device", code))
s.append(P("A real arm refuses to move without a safety box, a home pose and a calibration made with the arm itself; "
           "<i>preflight</i> lists what is missing. STOP (button, Escape, space, \"stop\"/\"fermo\") stays latched until "
           "the next command. If the arm misbehaves, keep working with <font face='Courier'>TROVA_ARM=sim</font> and come "
           "back in the next slot.", muted))

# ------------------------------------------------------------------ 4. who to talk to
s.append(PageBreak())
s.append(P("4. Who to talk to", h1))
s.append(table([["Who", "Ask them about", "When"],
                ["<b>Organisers</b>: Luca Bianchi, Federico Pellegrini (hosts)",
                 "Claim Track A. Prizes per track or overall? The common judging rubric. Pitch length. API credits.",
                 "09:00 kick-off"],
                ["<b>Museum hardware staff</b>",
                 "Which arm is arm 2 and its IP. Remote Control and TCP offset. Arm slot calendar, overnight too. "
                 "May we mount our camera and tape markers on the table? Dog access, later.",
                 "09:00, then every slot: staff must be present for every arm move"],
                ["<b>Claudio Peroni</b>, SEA Vision (mentor and juror)",
                 "The object list. Five requests like the jury's. How close the tip must be to count as found. How "
                 "find time is measured. Is the lighting fixed? How far must an object move to count as moved?",
                 "Wednesday and Thursday"],
                ["<b>Michele Antolini</b>, SEA Vision (mentor)", "A technical look at the demo before the pitch.",
                 "Thursday"],
                ["<b>Ralph Johnson</b>, SpaceXAI",
                 "Grok API and Grok Voice: speed settings, voice latency, structured outputs.",
                 "Thursday: 09:00 talk, then at the tables"],
                ["<b>Inside the team</b>", "Arm: P1 · Camera: P2 · Grok: P3 · Tests and numbers: P4 · Voice and pitch: P5",
                 "Always"]],
               [48, 88, 38]))
s.append(Spacer(1, 4))
s.append(P("One person (P5) collects everyone's questions for Claudio Peroni, so he is asked once, not five times. "
           "He is also on the jury: show him progress, not problems.", muted))

# ------------------------------------------------------------------ 5. understand everything
s.append(P("5. How to understand everything in 20 minutes", h1))
s += numbered([
    "<b>Watch the 101-second concept video</b>: <font face='Courier'>media/trova_concept.mp4</font> in the repo. It "
    "shows every moment of the demo: find by words, by photo, by voice, what changed, the record and the dog.",
    "<b>Play with the demo</b> on your laptop (demo mode is the default): type \"the red one\" and press Find; press "
    "<i>+ add</i>, <i>Save reference</i>, <i>Check now</i>; open the record.",
    "<b>Read the README</b>, sections \"What Grok does\" and \"Layout\": they map every file to what it does.",
    "<b>AGENTS.md</b> in the repo lists every feature, its status, the safety rules and the API. Update it in the "
    "same commit whenever you change code.",
    "<b>The full team plan</b> (roles, schedule, risks, market): <link href='https://claude.ai/artifact/Xc5nfd5Q41HmeHHwScSxmb' color='#B4009E'>claude.ai/artifact/Xc5nfd5Q41HmeHHwScSxmb</link>"])
s.append(P("Words you will hear", h2))
s.append(table([["Term", "Meaning"],
                ["Mock mode", "No API key: Grok answers are simulated (colour words only). The badge says MOCK."],
                ["Empty table", "Photo of the table without objects. Everything different from it is an object."],
                ["Object boxes", "Numbered rectangles that plain code draws around each object. Grok picks a number."],
                ["Reference", "The approved state of the table that \"what changed\" compares against."],
                ["Calibration", "4-6 clicks that map camera pixels to robot millimetres."],
                ["TCP", "The tool tip of the arm. Its offset must be set, or positions refer to the flange."],
                ["Remote Control", "Pendant setting on UR and Elite that lets a laptop send moves."],
                ["Freedrive", "The arm goes limp so a person can move it by hand. Calibration only, staff present."],
                ["Jury mode", "Correct or wrong after each find. It saves test cases and keeps the score."]],
               [32, 142]))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=18 * mm, title="Trova: morning start guide", author="Trova team")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
