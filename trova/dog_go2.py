"""TEMPLATE: Unitree Go2 for the optional dog scenario. NOT TESTED: we have no dog tonight.

Before writing any code, confirm with museum staff at 09:00:
  - model and variant (Go2 Air/Pro/EDU; there is also a Go1 at the event),
  - firmware version, and whether they can share the app account (newer firmware needs a per-device key),
  - Wi-Fi mode (the dog's own AP or the venue network), and when the dog slots are (always with staff).
Research notes from tonight point to `unitree_webrtc_connect` (Python, no ROS) for Air/Pro and to
`unitree_sdk2_python` (DDS over Ethernet) for EDU. Check its README for the exact calls.

The flow only needs three things from the dog: walk a taught route spot by spot, stop, and give a frame.
Simplest safe route: short, pre-tested moves (e.g. "walk 2 m forward, turn 90 degrees") per spot, staff
next to the dog, low speed. Until this works, TROVA_DOG=sim replays photos taken along the same route.
"""

from __future__ import annotations

import threading

import numpy as np

from .dog import DogDriver, Spot


class Go2Dog(DogDriver):
    name = "go2"

    def __init__(self, ip: str):
        self.ip = ip
        raise NotImplementedError("Go2 driver not written: confirm the model and SDK with staff, then fill this in "
                                  "(see the docstring). Use TROVA_DOG=sim meanwhile.")

    def go_to(self, spot: Spot, stop: threading.Event) -> bool:
        raise NotImplementedError

    def frame(self, spot: Spot) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
