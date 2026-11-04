"""Grab one frame from the trova-cam Pi. The client lives in trova.pi_cam.

    python trova_cam.py [host]

Prints /status.json and writes trova-frame.jpg in the current directory.
The picture is captured after the request. It is already upright.
"""

import sys
import time

from trova.pi_cam import CameraError, PiCam, TrovaCam

__all__ = ["CameraError", "PiCam", "TrovaCam"]


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else ""
    cam = PiCam(host)
    try:
        status = cam.status()
        print(f"camera {status.get('camera')}, {status.get('width')}x{status.get('height')}, "
              f"{status.get('temp')} C, protection {status.get('protection')}")
        for warning in status.get("warnings") or []:
            print("warning:", warning)
        started = time.time()
        jpeg = cam.get_jpeg()
        with open("trova-frame.jpg", "wb") as handle:
            handle.write(jpeg)
        print(f"saved trova-frame.jpg ({len(jpeg) // 1024} KB) in {time.time() - started:.2f} s")
    except CameraError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
