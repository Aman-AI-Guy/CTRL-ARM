"""Soundtrack for the ctrl-arm ad: 120 BPM, A minor, synthesised with numpy; the robot's Italian voice lines
(voice/*.wav, macOS Alice processed) are mixed in and the music ducks under them."""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = 44100
T = dict(hook=0, logo=7.5, scan=13, find=27, arm=43, chg=51, d3=62, ceil=75, mkt=96, close=103, end=112)
DUR = float(T['end'])
BEAT = 0.5
N = int(SR * DUR)
rng = np.random.default_rng(3)
mix = np.zeros((N, 2), np.float32)
voice = np.zeros((N, 2), np.float32)


def add(sig, t0, gain=1.0, pan=0.0, to=None):
    dst = mix if to is None else to
    i = int(t0 * SR)
    if i >= N or len(sig) == 0:
        return
    sig = np.asarray(sig, np.float64)[: N - i] * gain
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    dst[i:i + len(sig), 0] += sig * l
    dst[i:i + len(sig), 1] += sig * r


def tt(d):
    return np.arange(int(d * SR)) / SR


def lowpass(x, cut):
    x = np.asarray(x, np.float64)
    if np.ndim(cut) == 0:
        a = np.exp(-2 * np.pi * float(cut) / SR)
        n = 1 << int(np.ceil(np.log2(len(x) + SR // 10)))
        w = np.exp(-2j * np.pi * np.fft.rfftfreq(n))
        H = (1 - a) / (1 - a * w)
        return np.fft.irfft(np.fft.rfft(x, n) * H, n)[:len(x)]
    a = np.exp(-2 * np.pi * np.asarray(cut, np.float64) / SR).tolist()
    xs, y, acc = x.tolist(), [0.0] * len(x), 0.0
    for i in range(len(xs)):
        acc = (1 - a[i]) * xs[i] + a[i] * acc
        y[i] = acc
    return np.array(y)


def env(d, a=0.005, r=None):
    t = tt(d)
    r = d - a if r is None else r
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / max(r / 4, 1e-4))


def kick(gain=1.0):
    t = tt(0.45)
    f = 50 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * gain + 0.15 * rng.standard_normal(len(t)) * np.exp(-t * 200)


def clap():
    t = tt(0.25)
    n = rng.standard_normal(len(t))
    e = sum(np.exp(-np.maximum(0, t - d) * 60) * (t >= d) for d in (0, 0.012, 0.024))
    return (n - lowpass(n, 1200)) * e * 0.5


def hat(open_=False):
    d = 0.18 if open_ else 0.05
    t = tt(d)
    n = rng.standard_normal(len(t))
    return (n - lowpass(n, 7000)) * np.exp(-t * (18 if open_ else 80)) * 0.35


def saw(f, d, detune=0.0):
    t = tt(d)
    out = 0
    for k in (-detune, 0, detune):
        out = out + 2 * ((t * f * (1 + k)) % 1) - 1
    return out / 3


def note(freq, d, cut=900, gain=0.3, detune=0.004):
    return lowpass(saw(freq, d, detune), cut) * env(d, 0.004, d * 0.9) * gain


def noise_sweep(d, f0, f1, gain=0.4):
    n = rng.standard_normal(int(d * SR))
    cut = np.geomspace(f0, f1, len(n))
    hp = n - lowpass(n, cut * 0.3)
    return lowpass(hp, cut) * np.sin(np.linspace(0, np.pi, len(n))) * gain


def blip(f0, f1, d=0.06, gain=0.3):
    t = tt(d)
    f = np.geomspace(f0, f1, len(t))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 40) * gain


def boom(d=1.8, gain=1.0):
    t = tt(d)
    f = 30 + 70 * np.exp(-t * 6)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    n = rng.standard_normal(len(t))
    return (s + 0.4 * lowpass(n, 900) * np.exp(-t * 5)) * gain


def pad(freqs, d, gain=0.12):
    s = sum(saw(f, d, 0.006) for f in freqs) / len(freqs)
    s = lowpass(s, 1400)
    a = np.minimum(1, tt(d) / 0.6) * np.minimum(1, (d - tt(d)) / 0.6)
    return s * a * gain


def keyclick(gain=0.5):
    t = tt(0.06)
    n = rng.standard_normal(len(t))
    return (n - lowpass(n, 2500)) * np.exp(-t * 120) * gain + np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 90) * 0.3 * gain


def servo(d, f0, f1, gain=0.14):
    t = tt(d)
    f = np.linspace(f0, f1, len(t))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    n = lowpass(rng.standard_normal(len(t)), 1500) * 0.6
    return (s * 0.5 + n) * np.sin(np.linspace(0, np.pi, len(t))) * gain


def shutter(gain=0.35):
    n = rng.standard_normal(int(0.04 * SR))
    return (n - lowpass(n, 3000)) * np.exp(-tt(0.04) * 90) * gain


A1, C2, E2, F1, G1 = 55.0, 65.41, 82.41, 43.65, 49.0
CHORDS = [(A1, [220, 261.6, 329.6]), (F1, [174.6, 220, 261.6]), (C2, [196, 261.6, 329.6]), (G1, [196, 246.9, 293.7])]


def groove(t0, t1, kick_on=True, arp=True, hats=True, bass=True, level=1.0):
    nb = int(round((t1 - t0) / BEAT))
    for b in range(nb):
        tb = t0 + b * BEAT
        bar = int((tb - T['logo']) // 2) % 4
        root, triad = CHORDS[bar]
        if kick_on:
            add(kick(0.95 * level), tb)
            if b % 2 == 1:
                add(clap(), tb, 0.8 * level)
        if hats:
            add(hat(), tb + 0.25, 0.7 * level, pan=-0.2)
            add(hat(), tb, 0.35 * level, pan=0.2)
        if bass:
            for k in (0, 1):
                add(note(root * (2 if k else 1), 0.24, cut=500 + 400 * k, gain=0.42 * level, detune=0.002), tb + k * 0.25)
        if arp:
            for k in range(4):
                f = triad[(b * 4 + k) % 3] * 2
                add(note(f, 0.11, cut=3000, gain=0.07 * level, detune=0.003), tb + k * 0.125, pan=0.4 * np.sin(b + k))


# ---------------------------------------------------------------- hook (0-7.5): pad, keycaps, typing, error, riser
add(pad([110, 130.8, 164.8], 7.5, 0.3), 0)
add(pad([55, 110], 7.5, 0.22), 0)
for tk in (0.5, 0.8):                       # keys land
    add(boom(0.5, 0.5), tk)
for tk in (1.3, 1.5, 7.0, 7.1):             # key presses
    add(keyclick(0.7), tk)
for i in range(9):
    add(blip(2600, 2400, 0.02, 0.22), 2.0 + i * 0.09)
add(blip(1800, 2400, 0.08, 0.25), 2.95)
for i in range(12):
    add(blip(2600, 2400, 0.02, 0.2), 4.9 + i * 0.07)
t = tt(0.35)
add(np.sign(np.sin(2 * np.pi * 70 * t)) * np.exp(-t * 6) * 0.25, 5.75)   # "0 results"
add(noise_sweep(1.6, 300, 9000, 0.5), 5.9)
add(blip(400, 1600, 0.25, 0.3), 6.4)        # key flips
add(boom(2.2, 1.0), T['logo'])
add(noise_sweep(1.2, 9000, 2000, 0.35), T['logo'])

# ---------------------------------------------------------------- the groove, with the stop and the breakdown
groove(T['logo'], T['scan'], arp=False)
for i in range(3):
    add(pad([c * 0.5 for c in CHORDS[i][1]], 2.0, 0.08), T['logo'] + 2 * i)
groove(T['scan'], T['find'])
groove(T['find'], T['arm'])
groove(T['arm'], T['arm'] + 5.0)                     # STOP: music stops dead ...
groove(T['arm'] + 6.2, T['chg'], arp=False)          # ... and comes back
groove(T['chg'], T['d3'])
groove(T['d3'], T['ceil'], kick_on=False, bass=False, level=1.3)   # breakdown under the 3D
for i in range(6):
    add(pad([c * 0.5 for c in CHORDS[i % 4][1]], 2.0, 0.34), T['d3'] + 2 * i)
    add(note(CHORDS[i % 4][0] * 2, 1.9, cut=300, gain=0.35, detune=0.002), T['d3'] + 2 * i)
groove(T['ceil'], T['ceil'] + 3.0, kick_on=False, bass=False, level=0.9)   # quiet while the object lands
add(boom(1.4, 0.8), T['ceil'] + 2.4)                 # the object lands
groove(T['ceil'] + 3.0, T['mkt'])                    # full groove for the scan
groove(T['mkt'], T['close'])
groove(T['close'], T['close'] + 6.0)
for i in range(3):
    add(pad([c * 0.5 for c in CHORDS[0][1]], 2.0, 0.12 - 0.03 * i), T['close'] + 6.0 + 2 * i)

# ---------------------------------------------------------------- transitions and scene effects
for s in ('scan', 'find', 'arm', 'chg', 'd3', 'ceil', 'mkt', 'close'):
    add(noise_sweep(0.45, 800, 8000, 0.45), T[s] - 0.3)
for i in range(4):                                   # scan: 4 photo cuts, each with a shutter and a few label pops
    t0 = T['scan'] + 0.8 + i * 2.6
    add(shutter(0.4), t0)
    for k in range(5):
        add(blip(1200 + 80 * k, 1500 + 80 * k, 0.05, 0.16), t0 + 0.9 + k * 0.22, pan=-0.5 + 0.2 * k)
t = tt(10.4)                                         # scan sweep tone rising with the map
add(np.sin(2 * np.pi * np.cumsum(np.linspace(180, 420, len(t))) / SR) * 0.05, T['scan'] + 0.8)
for i in range(9):                                   # find: boxes pop
    add(blip(900 + 90 * i, 1500 + 90 * i, 0.07, 0.28), T['find'] + 3.4 + i * 0.12, pan=-0.6 + 0.15 * i)
for i in range(6):
    add(blip(500, 520, 0.12, 0.12), T['find'] + 4.8 + i * 0.25)
t = tt(0.45)
add(np.sin(2 * np.pi * np.cumsum(np.geomspace(600, 2400, len(t))) / SR) * np.exp(-t * 4) * 0.25, T['find'] + 6.4)
add(blip(1760, 1760, 0.5, 0.22), T['find'] + 10.6)   # "Sì"
add(noise_sweep(1.2, 400, 6000, 0.3), T['find'] + 11.8)
add(servo(1.5, 160, 230), T['arm'] + 0.5)
add(servo(1.4, 180, 260), T['arm'] + 2.6)
add(servo(0.9, 220, 200), T['arm'] + 4.4)
add(boom(1.0, 0.9), T['arm'] + 5.0)                  # STOP
t = tt(0.5)
add(np.sign(np.sin(2 * np.pi * 110 * t)) * np.exp(-t * 5) * 0.2, T['arm'] + 5.0)
add(noise_sweep(1.1, 600, 5000, 0.3), T['chg'] + 1.4)
for i in range(3):
    add(blip(880, 880, 0.14, 0.22), T['chg'] + 2.6 + i * 0.32)
    add(blip(660, 660, 0.14, 0.22), T['chg'] + 2.76 + i * 0.32)
add(boom(1.0, 0.9), T['chg'] + 4.0)
for i in range(12):                                  # 3D: a shutter per view
    add(shutter(0.35), T['d3'] + 0.5 + (i + 1) * 4.0 / 12 - 0.02, pan=-0.8 + 0.15 * i)
add(servo(1.6, 140, 190, 0.12), T['d3'] + 6.4)       # the second arm turns it
for i in range(6):
    add(shutter(0.3), T['d3'] + 8.2 + i * 0.25, pan=0.3)
add(blip(1320, 1760, 0.3, 0.2), T['d3'] + 9.6)       # mesh ready
add(noise_sweep(1.0, 200, 4000, 0.35), T['ceil'] + 1.0)      # toss
add(servo(2.0, 120, 200, 0.12), T['ceil'] + 3.0)     # arm swoops
for i in range(12):
    add(shutter(0.3), T['ceil'] + 3.4 + (i + 1) * 8.0 / 12 - 0.02, pan=-0.6 + 0.11 * i)
add(blip(1320, 1760, 0.3, 0.2), T['ceil'] + 11.6)    # mesh
add(blip(2093, 2637, 0.18, 0.3), T['ceil'] + 15.3)   # phone notification: two-note chime
add(blip(2637, 2637, 0.25, 0.3), T['ceil'] + 15.5)
add(boom(1.6, 0.9), T['ceil'] + 19.0)                # "Any arm. Any camera."
for c in range(3):
    for i in range(14):
        add(blip(2000, 2000, 0.02, 0.12), T['mkt'] + 0.5 + c * 0.6 + i * 0.1)
for i in range(5):
    add(kick(0.6), T['close'] + 0.3 + i * 0.45)
add(boom(2.6, 1.0), T['close'] + 3.0)
add(noise_sweep(2.0, 9000, 1200, 0.35), T['close'] + 3.0)
for i in range(4):
    add(kick(0.5), T['close'] + 5.4 + i * 0.8)
    add(kick(0.35), T['close'] + 5.62 + i * 0.8)

# ---------------------------------------------------------------- the robot's voice (Italian), music ducked under it
LINES = {"scan_start": T['scan'] + 0.7, "scan_done": T['scan'] + 11.4, "find": T['find'] + 7.1, "stop": T['arm'] + 5.8,
         "changed": T['chg'] + 4.7, "turn": T['d3'] + 4.7, "mesh": T['d3'] + 9.6, "sent": T['ceil'] + 15.4}
duck = np.ones(N)
dur = json.load(open(HERE / "voice" / "durations.json"))
for name, t0 in LINES.items():
    with wave.open(str(HERE / "voice" / f"{name}.wav"), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    add(x, t0, 0.95, to=voice)
    a, b = int((t0 - 0.15) * SR), int((t0 + dur[name] + 0.25) * SR)
    duck[max(0, a):min(N, b)] = 0.32
k = int(0.12 * SR)                                   # smooth the ducking edges
duck = np.convolve(duck, np.ones(k) / k, mode="same")
mix *= duck[:, None]
mix += voice

# ---------------------------------------------------------------- master: fade, soft clip, normalise
fade = np.ones(N)
fade[-int(2.0 * SR):] = np.linspace(1, 0, int(2.0 * SR))
mix *= fade[:, None]
mix = np.tanh(mix * 1.2)
mix /= np.abs(mix).max() / 0.89
out = HERE / "soundtrack.wav"
with wave.open(str(out), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out, f"{DUR:.0f}s")
