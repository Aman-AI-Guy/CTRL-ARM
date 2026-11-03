"""Settings, read from environment variables and an optional .env file (see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].split("\t#", 1)[0]   # allow inline comments
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    xai_api_key: str
    model: str            # vision model for finds and change checks
    effort: str           # reasoning effort; "" sends none
    mock: bool            # True when there is no key, or TROVA_MOCK=1
    camera: str           # "demo" | webcam index "0" | video, image or folder path
    arm: str              # "sim" | "grid:<module>:<BackendClass>"
    data_dir: Path
    table_w_mm: float     # used by the default calibration and the simulated arm
    table_h_mm: float
    blob_thresh: float    # colour-difference threshold for object proposals
    min_area_frac: float  # smallest object, as a fraction of the frame area
    host: str
    port: int
    lang: str             # language for change reports: en | it | both (finds follow the request)
    dog: str              # off | sim | go2:<ip>
    voice_model: str
    voice: str


def load_settings() -> Settings:
    key = os.getenv("XAI_API_KEY", "").strip()
    mock_env = os.getenv("TROVA_MOCK", "auto").lower()
    mock = mock_env in ("1", "true", "yes") or (mock_env == "auto" and not key)
    data_dir = Path(os.getenv("TROVA_DATA", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        xai_api_key=key,
        model=os.getenv("TROVA_MODEL", "grok-4.7"),
        effort=os.getenv("TROVA_EFFORT", "low"),
        mock=mock,
        camera=os.getenv("TROVA_CAMERA", "demo"),
        arm=os.getenv("TROVA_ARM", "sim"),
        data_dir=data_dir,
        table_w_mm=float(os.getenv("TROVA_TABLE_W_MM", "600")),
        table_h_mm=float(os.getenv("TROVA_TABLE_H_MM", "400")),
        blob_thresh=float(os.getenv("TROVA_BLOB_THRESH", "30")),
        min_area_frac=float(os.getenv("TROVA_MIN_AREA_FRAC", "0.0008")),
        host=os.getenv("TROVA_HOST", "0.0.0.0"),
        port=int(os.getenv("TROVA_PORT", "8000")),
        lang=os.getenv("TROVA_LANG", "en"),
        dog=os.getenv("TROVA_DOG", "off"),
        voice_model=os.getenv("TROVA_VOICE_MODEL", "grok-voice-latest"),
        voice=os.getenv("TROVA_VOICE", "eve"),
    )
