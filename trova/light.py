"""Optional: the LED ring around the camera on the arm (it looks like a 16-pixel WS2812 / NeoPixel ring).

A WS2812 ring needs a microcontroller to drive it; the xArm tool outputs cannot. IF the ring is wired to an Arduino or
ESP32 that you can flash, load hardware/led_ring/led_ring.ino and set TROVA_LED=serial:/dev/tty.usbmodemXXXX
(ls /dev/tty.usb* to find it). Trova then shows its state on the ring:

  idle  white (lights the scene for the camera)   found      green         not_found  amber
  alarm red, flashing (watch the line)             stop       orange        busy       blue

Protocol (one line each, 115200 baud): "C r g b" fill colour, "F r g b n" flash n times then back to the fill colour.
Off by default (TROVA_LED empty): every call is then a no-op. A light problem never stops Trova.
"""

from __future__ import annotations

import os
import threading

COLORS = {"idle": (90, 90, 90), "found": (0, 180, 0), "not_found": (200, 110, 0), "alarm": (220, 0, 0),
          "stop": (230, 70, 0), "busy": (0, 60, 200), "off": (0, 0, 0)}


class NoLight:
    name = "off"

    def show(self, state: str) -> None:
        pass

    def status(self) -> dict:
        return {"name": self.name}


class SerialRing(NoLight):
    def __init__(self, port: str, baud: int = 115200, serial_mod=None):
        if serial_mod is None:
            try:
                import serial as serial_mod   # pyserial
            except ImportError as e:
                raise RuntimeError("pip install pyserial (or: pip install -e '.[robots]')") from e
        self.name, self.port = "serial", port
        self.ser = serial_mod.Serial(port, baud, timeout=0.2, write_timeout=0.2)
        self.lock = threading.Lock()
        self.last, self.error = None, None
        self.show("idle")

    def _send(self, line: str) -> None:
        with self.lock:
            try:
                self.ser.write((line + "\n").encode())
                self.error = None
            except Exception as e:  # noqa: BLE001 - a light problem must never stop Trova
                self.error = str(e)

    def show(self, state: str) -> None:
        r, g, b = COLORS.get(state, COLORS["idle"])
        if state in ("alarm", "found", "not_found", "stop"):
            self._send(f"F {r} {g} {b} {6 if state == 'alarm' else 2}")
        self._send(f"C {r} {g} {b}" if state in ("alarm", "stop", "busy") else "C %d %d %d" % COLORS["idle"])
        self.last = state

    def status(self) -> dict:
        return {"name": self.name, "port": self.port, "last": self.last, "error": self.error}


def make_light(spec: str | None = None):
    spec = (spec if spec is not None else os.getenv("TROVA_LED", "")).strip()
    if not spec or spec == "off":
        return NoLight()
    kind, _, rest = spec.partition(":")
    if kind == "serial":
        port, _, baud = rest.partition("@")
        return SerialRing(port, int(baud or 115200))
    raise ValueError(f"Unknown TROVA_LED '{spec}' (use serial:/dev/tty.usbmodemXXXX)")
