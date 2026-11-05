"""Soundtrack for the Trova ad: 120 BPM, A minor, synthesised with numpy, cut to the scene timings in ad.html."""
import wave
from pathlib import Path

import numpy as np

SR = 44100
DUR = 83.0
BEAT = 0.5
N = int(SR * DUR)
rng = np.random.default_rng(3)
mix = np.zeros((N, 2), np.float32)
T = dict(hook=0, logo=7, find=12, arm=25, chg=34.5, d3=45, cases=53, mkt=63, close=73, end=83)


def add(sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N or len(sig) == 0:
        return
    sig = sig[: N - i] * gain
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    mix[i:i + len(sig), 0] += sig * l
    mix[i:i + len(sig), 1] += sig * r


def tt(d):
    return np.arange(int(d * SR)) / SR


def lowpass(x, cut):
    """One-pole low-pass. Constant cut: exact frequency response via FFT (fast). Array cut (sweeps): sample loop."""
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
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / max(r / 4, 1e-4))
    return e


def kick(gain=1.0):
    t = tt(0.45)
    f = 50 + 110 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 7) * gain + 0.15 * rng.standard_normal(len(t)) * np.exp(-t * 200)


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
    s = lowpass(saw(freq, d, detune), cut)
    return s * env(d, 0.004, d * 0.9) * gain


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


A1, C2, E2, F1, G1 = 55.0, 65.41, 82.41, 43.65, 49.0
CHORDS = [(A1, [220, 261.6, 329.6]), (F1, [174.6, 220, 261.6]), (C2, [196, 261.6, 329.6]), (G1, [196, 246.9, 293.7])]

# ---------------------------------------------------------------- intro (0-7): pad, ticks, typing, error, riser
add(pad([110, 130.8, 164.8], 7.0, 0.3), 0)
add(pad([55, 110], 7.0, 0.22), 0)
for i in range(int(6.5 / 0.25)):
    add(hat(), 1.0 + i * 0.25, 0.35 if i % 2 else 0.6, pan=0.3)
for i in range(9):
    add(blip(2600, 2400, 0.02, 0.25), 1.0 + i * 0.11)
add(blip(1800, 2400, 0.08, 0.25), 2.1)                      # 1 of 1
for i in range(9):
    add(blip(2600, 2400, 0.02, 0.2), 4.6 + i * 0.012)
t = tt(0.35)
add(np.sign(np.sin(2 * np.pi * 70 * t)) * np.exp(-t * 6) * 0.25, 4.7)   # "0 results" buzz
add(noise_sweep(2.0, 300, 9000, 0.5), 5.0)
t = tt(2.0)
add(np.sin(2 * np.pi * np.cumsum(np.geomspace(200, 1200, len(t))) / SR) * (t / 2) * 0.12, 5.0)

# ---------------------------------------------------------------- the groove
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


add(boom(2.2, 1.0), T['logo'])
add(noise_sweep(1.2, 9000, 2000, 0.35), T['logo'])
groove(T['logo'], T['find'], arp=False)
for i in range(4):
    add(pad([c * 0.5 for c in CHORDS[i][1]], 2.0, 0.08), T['logo'] + 2 * i) if T['logo'] + 2 * i < T['find'] else None
groove(T['find'], T['arm'])
groove(T['arm'], 31.4)                                      # the music stops dead at STOP ...
groove(32.4, T['chg'], arp=False)                           # ... and comes back after the hold
groove(T['chg'], T['d3'])
groove(T['d3'], T['cases'], kick_on=False, bass=False, level=1.3)   # breakdown under the 3D orbit
for i in range(4):
    add(pad([c * 0.5 for c in CHORDS[i][1]], 2.0, 0.34), T['d3'] + 2 * i)
    add(note(CHORDS[i][0] * 2, 1.9, cut=300, gain=0.35, detune=0.002), T['d3'] + 2 * i)
groove(T['cases'], T['mkt'])
groove(T['mkt'], T['close'])
groove(T['close'], 79.0)
for i in range(3):
    add(pad([c * 0.5 for c in CHORDS[0][1]], 2.0, 0.12 - 0.03 * i), 79.0 + 2 * i)

# ---------------------------------------------------------------- scene transitions and effects
for s in ('find', 'arm', 'chg', 'd3', 'cases', 'mkt', 'close'):
    add(noise_sweep(0.45, 800, 8000, 0.45), T[s] - 0.3)
for i in range(9):                                          # numbered boxes pop
    add(blip(900 + 90 * i, 1500 + 90 * i, 0.07, 0.28), T['find'] + 3.2 + i * 0.13, pan=-0.6 + 0.15 * i)
for i in range(6):                                          # Grok thinking
    add(blip(500, 520, 0.12, 0.12), T['find'] + 4.8 + i * 0.25)
t = tt(0.45)                                                # lock-on
add(np.sin(2 * np.pi * np.cumsum(np.geomspace(600, 2400, len(t))) / SR) * np.exp(-t * 4) * 0.25, T['find'] + 6.2)
add(blip(1760, 1760, 0.6, 0.25), T['find'] + 6.6)
add(noise_sweep(1.0, 400, 6000, 0.3), T['find'] + 8.2)      # dive into the close-up


def servo(d, f0, f1, gain=0.14):
    t = tt(d)
    f = np.linspace(f0, f1, len(t))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    n = lowpass(rng.standard_normal(len(t)), 1500) * 0.6
    return (s * 0.5 + n) * np.sin(np.linspace(0, np.pi, len(t))) * gain


add(servo(1.6, 160, 230), T['arm'] + 0.6)
add(blip(1200, 1200, 0.25, 0.2), T['arm'] + 2.5)            # confirm
add(blip(1600, 1600, 0.12, 0.2), T['arm'] + 3.4)
add(servo(1.2, 180, 260), T['arm'] + 3.5)
add(servo(0.8, 220, 200), T['arm'] + 5.6)
add(boom(1.0, 0.9), T['arm'] + 6.4)                         # STOP
t = tt(0.5)
add(np.sign(np.sin(2 * np.pi * 110 * t)) * np.exp(-t * 5) * 0.2, T['arm'] + 6.4)
add(noise_sweep(1.1, 600, 5000, 0.3), T['chg'] + 1.4)       # the wipe
for i in range(3):                                          # alarm
    add(blip(880, 880, 0.14, 0.22), T['chg'] + 2.6 + i * 0.32)
    add(blip(660, 660, 0.14, 0.22), T['chg'] + 2.76 + i * 0.32)
add(boom(1.0, 0.9), T['chg'] + 4.0)                         # stamp
for i in range(12):                                         # camera shutters, one per 3D view
    n = rng.standard_normal(int(0.04 * SR))
    add((n - lowpass(n, 3000)) * np.exp(-tt(0.04) * 90) * 0.35, T['d3'] + 0.6 + (i + 1) * 4.4 / 12 - 0.02, pan=-0.8 + 0.15 * i)
add(boom(1.6, 0.8), T['cases'])
add(noise_sweep(1.5, 9000, 1500, 0.3), T['cases'])
for c in range(3):                                          # counters
    for i in range(14):
        add(blip(2000, 2000, 0.02, 0.12), T['mkt'] + 0.6 + c * 0.7 + i * 0.11)
for i in range(5):                                          # the five pills
    add(kick(0.6), T['close'] + 0.3 + i * 0.5)
add(boom(2.6, 1.0), T['close'] + 3.2)                       # logo
add(noise_sweep(2.0, 9000, 1200, 0.35), T['close'] + 3.2)
for i in range(4):                                          # "Now - live." heartbeat
    add(kick(0.5), T['close'] + 5.2 + i * 0.8)
    add(kick(0.35), T['close'] + 5.42 + i * 0.8)

# ---------------------------------------------------------------- master: fade, soft clip, normalise
fade = np.ones(N)
fade[-int(2.0 * SR):] = np.linspace(1, 0, int(2.0 * SR))
mix *= fade[:, None]
mix = np.tanh(mix * 1.2)
mix /= np.abs(mix).max() / 0.89
out = Path(__file__).resolve().parent / "soundtrack.wav"
with wave.open(str(out), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out, f"{DUR:.0f}s")
