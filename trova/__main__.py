"""python -m trova  ->  starts the web app and prints the URLs for this laptop and for phones."""

import os
import socket

import uvicorn

from .config import load_settings
from .server import create_app


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    s = load_settings()
    app = create_app(s)
    mode = "MOCK (no XAI_API_KEY: Grok answers are simulated)" if s.mock else f"LIVE Grok: {s.model}, effort {s.effort or 'default'}"
    token = os.getenv("TROVA_TOKEN", "").strip()
    q = f"/?token={token}" if token else ""
    print(f"\n  Trova  |  {mode}\n  camera={s.camera}  arm={s.arm}\n"
          f"  this laptop: http://localhost:{s.port}{q}\n  phones (same Wi-Fi): http://{lan_ip()}:{s.port}{q}\n"
          + ("" if token else "  (anyone on this Wi-Fi can use it: set TROVA_TOKEN=<secret> to lock it)\n"))
    uvicorn.run(app, host=s.host, port=s.port, log_level="warning")


if __name__ == "__main__":
    main()
