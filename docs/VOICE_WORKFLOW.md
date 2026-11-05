# Voice task audit and acceptance

The previous realtime client asked for browser confirmation before sending a task, so the operator was confirming an intention with no calculated trajectory. The server then started a new task with no selected object. Selection lived only in the snapshot/UI response. A short follow-up could lose its referent. Browser fallback treated “yes” and “scan it” as ordinary find requests. Legacy voice point/home wrappers inferred success from an HTTP response, and find tools reported “waiting for yes” without a validated proposal. Realtime speech could play before tool results arrived. Agent `hardware_moved` counted simulated executions too.

Both voice clients and the typed task box now call `/api/agent/task`. `/api/select` retains selected-object and snapshot context on the workstation. Follow-up “Ora posizionati sopra” uses that selection; a changed scene or unreliable association requires selection again. An explicit new object request is selected afresh. Clarification answers return to the selection endpoint rather than the legacy find endpoint when appropriate.

The task localizes, plans and checks geometry and the gateway. A blocked task has no confirmation button. A dry-run trajectory reports hardware authorization missing and cannot be confirmed into a hardware move. An executable simulation or qualified hardware proposal produces a single-use receipt valid for 60 seconds. Its id, object and target are visible in the UI. `/api/agent/confirm` accepts only that id; it checks the original pose, calibration, camera, safety settings, current controller state and cancellation, then revalidates through the gateway before executing the same proposal. A model cannot confirm a future plan or use a generic yes to enable hardware. Spoken confirmation also requires a matching user transcript after the proposal arrived.

The backend returns `workflow_state`, `workflow_events`, `spoken_message`, `blocker`, `hardware_moved` and `simulation_moved`. The UI displays the actual reason. The realtime model still interprets speech and calls tools, but its generated audio/transcript are not played as operator feedback: the browser speaks the tool result via speech synthesis. This deliberately trades the realtime model's voice for deterministic status wording. Live microphone quality and browser speech voices still need operator testing.

Example without physical calibration:

1. “Seleziona il mouse rosso.” → object selected, no movement.
2. “Ora posizionati sopra.” → localization blocked: “Ho selezionato mouse rosso, ma non posso ancora muovermi sopra di esso perché manca la calibrazione che permette di determinarne la posizione fisica.”
3. “Sì, procedi.” → refused: no validated proposal exists. No hardware unlock changes.

Example in a mathematically calibrated test fixture using a fake Cartesian arm:

1. Select → localize → plan → gateway validation → proposal with id and expiry.
2. Explicit confirmation of that id → gateway revalidation → simulated movement → fresh image and evaluation.
3. Speech explicitly reports simulation. Physical movement and physical qualification are not claimed.

Expiry, translation or orientation changes, invalid calibration, changed speed/clearance/keep-outs, STOP, cancellation, new selection and a new snapshot all refuse the old confirmation. Completion requires geometric position plus fresh visual identity checks; image centering alone is insufficient. The physical emergency stop remains independent and full robot collision avoidance is not implemented.

Run `.venv/bin/pytest -q --ignore=tests/test_live_camera.py` and `PLAYWRIGHT_BROWSERS_PATH=/tmp/trova-playwright .venv/bin/python tests/browser_smoke.py`. The workflow regressions use synthetic calibration and simulated robot backends; no physical qualification is implied.
