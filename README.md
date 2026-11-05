# Trova

Ctrl+Arm, for the Grok Hackathon in Pavia, 4–5 November 2026. Track A, CTRL+F: SEA (SEA Vision).

You ask for an object, typed, spoken in English or Italian, or shown as a phone photo. Trova finds it on the table and the robot arm points at it. It also compares the table with an approved reference and reports what was added, removed, or moved. That line-clearance record is signed by a person.

Grok returns structured JSON. Plain code checks it, converts pixels to millimetres, and moves the arm. STOP does not wait for a model: the button, Escape, the space bar, and the stop word halt the arm and hold position.

## Demo

- Scan, then the 3D result: [hackathon media/Ctrl+Arm xArm5 scan.mp4](<hackathon media/Ctrl+Arm xArm5 scan.mp4>)
- Pitch slides: [hackathon media/ctrl-arm pitch.pdf](<hackathon media/ctrl-arm pitch.pdf>)
- Pitch video: [media/ctrl_arm_ad.mp4](media/ctrl_arm_ad.mp4)

## 3D results

The scan in the video is the pliers (pinze). The preview is the fast mesh. The full render is the finished mesh. Same four views, top first.

### Top

| Preview | Full render |
|---|---|
| ![Pliers preview, top](<hackathon media/RENDER_PREVIEW/top.png>) | ![Pliers full render, top](<hackathon media/RENDER_FULL/top.png>) |

### Front

| Preview | Full render |
|---|---|
| ![Pliers preview, front](<hackathon media/RENDER_PREVIEW/front.png>) | ![Pliers full render, front](<hackathon media/RENDER_FULL/front.png>) |

### Side

| Preview | Full render |
|---|---|
| ![Pliers preview, side](<hackathon media/RENDER_PREVIEW/side.png>) | ![Pliers full render, side](<hackathon media/RENDER_FULL/side.png>) |

### Back

| Preview | Full render |
|---|---|
| ![Pliers preview, back](<hackathon media/RENDER_PREVIEW/back.png>) | ![Pliers full render, back](<hackathon media/RENDER_FULL/back.png>) |

Meshes in `hackathon media/3D scan/`:

| Object | Full | Also |
|---|---|---|
| Pliers (wire cutter) | [pinze_wire_cutter.stl](<hackathon media/3D scan/pinze_wire_cutter.stl>) | [pinze_wire_cutter_preview.stl](<hackathon media/3D scan/pinze_wire_cutter_preview.stl>) |
| Mouse | [mouse.stl](<hackathon media/3D scan/mouse.stl>) | [mouse_preview.stl](<hackathon media/3D scan/mouse_preview.stl>) |
| Green cup | [tazza_verde.stl](<hackathon media/3D scan/tazza_verde.stl>) | [tazza_verde_arm.stl](<hackathon media/3D scan/tazza_verde_arm.stl>) |

The arm pose photo and the orbit plan are in `hackathon media/BRACCIO/`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,robots,ui]"
cp .env.example .env
.venv/bin/python -m trova
.venv/bin/pytest -q
```

Leave `XAI_API_KEY` empty and Trova runs in mock mode: Grok is simulated, so the UI and tests work offline. Add the key when you want live answers. The default camera is a synthetic table, so the find and change flow can be rehearsed with no hardware.

`python -m trova.station` is the workstation entry. Operator steps, the real arm, and the camera addresses are in [docs/HANDOFF_TABLE_SCAN.md](docs/HANDOFF_TABLE_SCAN.md). Current status is in [AGENTS.md](AGENTS.md).

## Where things live

| Path | What it is |
|---|---|
| `trova/` | The app: camera, proposals, Grok, calibration, arm, voice, server, UI |
| `tests/` | Offline tests. No key and no robot |
| `docs/` | Handoff, table scan, and the pitch notes |
| `eval/` | Measurements against baselines |
| `scripts/rearm.ps1` | Windows helper that re-arms a stopped controller. It sends no motion |
| `test_xarm.py` | Prints firmware, state, pose, and joints for `XARM_IP`, then disconnects |
| `trova_cam.py` | Grabs one frame from the Pi camera through `trova.pi_cam` |
| `speech_to_text/` | Turns a typed or spoken line into JSON for the xArm. It does not move the arm |
| `xarm_voice_preview.py` | Read-only path: text or microphone, Grok, JSON, then telemetry |
| `xarm_voice_execute.py` | A relative move through the motion gateway. Dry-run unless the operator unlocks hardware |
| `media/` | The pitch video and its sources |
| `hackathon media/` | Scan video, photos, renders, and STL meshes |
| `Test 3d live video/` | The scan experiment used at the table |

## Licence

This project is [MIT](LICENSE). OpenCV is Apache-2.0 and FastAPI is MIT. Files under `trova/scan3d/LICENSE-UFACTORY` stay under UFACTORY's own licence. YOLO-World (GPL-3.0) is not used, so a company can reuse this.
