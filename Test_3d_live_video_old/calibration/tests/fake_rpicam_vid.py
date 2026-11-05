"""Stand-in for rpicam-vid on a PC, so stream.py can be tested without the Pi.

Writes MJPEG frames (a real Pi frame with a random-size comment segment, so frame sizes vary like real ones) to
stdout at --framerate, and per-frame JSON metadata to --metadata like rpicam-vid 1.13 does. Auto exposure reports
30 ms x2.75; with --shutter / --gain / --awbgains it reports the locked values (gain above 10.667 is split into
analogue 10.667 x digital, like the IMX219 pipeline).
"""
import argparse
import json
import os
import random
import signal
import sys
import time

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "frame.jpg")
MAX_AG = 10.667


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=1640)
    p.add_argument("--height", type=int, default=1232)
    p.add_argument("--framerate", type=float, default=30)
    p.add_argument("-q", type=int, default=50)
    p.add_argument("--shutter", type=float, default=0)
    p.add_argument("--gain", type=float, default=0)
    p.add_argument("--awbgains", default="")
    p.add_argument("--metadata", default="")
    p.add_argument("--metadata-format", default="json")
    p.add_argument("-o", default="-")
    args, _ = p.parse_known_args()

    with open(FIXTURE, "rb") as f:
        frame = f.read()
    meta = open(args.metadata, "w") if args.metadata else None
    if meta:
        meta.write("[\n")
        meta.flush()

    def bye(*_):
        if meta:
            meta.write("\n]\n")
            meta.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    exp, ag, dg = 30000.0, 2.75, 1.0
    if args.shutter:
        exp = args.shutter
    if args.gain:
        ag = min(args.gain, MAX_AG)
        dg = args.gain / ag
    gains = [float(v) for v in args.awbgains.split(",")] if args.awbgains else [1.27, 1.96]

    time.sleep(float(os.environ.get("FAKE_START_DELAY", "0.3")))
    out = sys.stdout.buffer
    period = 1.0 / args.framerate
    t_next = time.monotonic()
    i = 0
    rng = random.Random(1)
    while True:
        n = rng.randint(0, 30000)
        com = b"\xff\xfe" + (n + 2).to_bytes(2, "big") + bytes(n)
        try:
            out.write(frame[:2] + com + frame[2:])
            out.flush()
        except (BrokenPipeError, OSError):
            bye()
        if meta:
            obj = {"ExposureTime": int(exp), "AnalogueGain": ag, "DigitalGain": dg, "ColourGains": gains,
                   "Lux": 345.0, "ColourTemperature": 3300, "FocusFoM": 3000 + n // 100,
                   "SensorTimestamp": time.monotonic_ns(), "FrameWallClock": time.time_ns()}
            meta.write(("" if i == 0 else ",\n") + json.dumps(obj, indent=4))
            meta.flush()
        i += 1
        t_next += period
        time.sleep(max(0.0, t_next - time.monotonic()))


if __name__ == "__main__":
    main()
