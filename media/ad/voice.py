"""The robot's Italian lines for the ad: macOS `say` (voice Alice), then a light robot timbre with ffmpeg.
Run from media/ad:  python voice.py   -> voice/<name>.wav + voice/durations.json (audio.py reads them)."""
import json
import subprocess
from pathlib import Path

LINES = {"scan_start": "Scansione del tavolo in corso.", "scan_done": "Fatto. Ottantaquattro oggetti mappati.",
         "find": "Numero sette, mouse rosso. Mi metto sopra, alla stessa altezza. Confermi?", "stop": "Fermo.",
         "changed": "Linea non libera. Un oggetto aggiunto: un cappellino nero.", "turn": "Giralo, per favore.",
         "mesh": "Ricostruzione pronta.", "sent": "Modello tre D inviato sul tuo telefono."}
out = Path(__file__).resolve().parent / "voice"
out.mkdir(exist_ok=True)
durations = {}
for name, text in LINES.items():
    subprocess.run(["say", "-v", "Alice", "-r", "168", "-o", str(out / f"{name}.aiff"), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / f"{name}.aiff"), "-af",
                    "asetrate=44100*1.06,aresample=44100,highpass=f=120,lowpass=f=9000,volume=1.5",
                    "-ar", "44100", "-ac", "1", str(out / f"{name}.wav")], check=True)
    (out / f"{name}.aiff").unlink()
    durations[name] = round(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                                  "csv=p=0", str(out / f"{name}.wav")], capture_output=True, text=True).stdout), 2)
json.dump(durations, open(out / "durations.json", "w"))
print(durations)
