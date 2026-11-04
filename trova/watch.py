"""Feature 10: watch the line. After clearance, Trova keeps comparing the live view with the approved reference
(plain code, about twice a second, no Grok in the loop) and raises an alarm within ~1.5 s when something is
added, removed or moved, or an object enters a "must be empty" zone. Optionally the arm points at the problem.

- Persistence: a change must be in 2 of the last 3 clean frames (same rule as "Check now").
- A hand or the arm in view pauses the watch (no alarm on people working).
- Each new alarm is explained by Grok in the background (what it is, why it matters); if Grok calls every
  candidate a false alarm, the alarm closes itself. The alarm appears before Grok answers.
- An alarm stays until the operator acknowledges it; the same situation does not alarm again after that.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from . import changes as chg
from .arm import SimArm
from .grok import GrokError
from .marks import color_name, crop, draw_changes
from .zones import draw_zones, evaluate as eval_zones

GRID_PX = 40     # changes within this many pixels count as the same situation


@dataclass(frozen=True)
class WatchRun:
    generation: int
    cancel: threading.Event
    ticket: threading.Event
    worker: threading.Thread
    point: bool
    history: deque


def _signature(changes: list, zone_problems: list[str]) -> tuple:
    def where(c):
        b = c.cur_box or c.ref_box
        cx, cy = b.center
        return c.kind, round(cx / GRID_PX), round(cy / GRID_PX)
    return tuple(sorted(where(c) for c in changes)) + tuple(sorted(zone_problems))


class Watch:
    def __init__(self, st, interval_s: float = 0.5):
        self.st = st
        self.interval = interval_s
        self.on = False
        self.point = False
        self.status = "off"
        self.alert: dict | None = None
        self.alerts = 0
        self.checks = 0
        self._hist: deque = deque(maxlen=3)
        self._acked: tuple | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.motion_ticket = None
        self._alert_motion = None            # (alert identity, ticket): never mix concurrent alarms' authority
        self._control = threading.RLock()
        self._generation = 0
        self._run: WatchRun | None = None

    def valid_run(self, run: WatchRun, owner: threading.Thread) -> bool:
        return (self._run is run and run.generation == self._generation and run.worker is owner
                and not run.cancel.is_set())

    def motion_ticket_for(self, alert):
        pair = self._alert_motion
        if pair is not None and pair[0] is alert and self.valid_run(pair[2], pair[2].worker):
            return pair[1]
        cancelled = threading.Event()       # unknown/stale alert: fail closed
        cancelled.set()
        return cancelled

    def motion_guard_for(self, alert):
        pair = self._alert_motion
        if pair is None or pair[0] is not alert:
            return lambda: False
        run = pair[2]
        return lambda: self._alert_motion is pair and self.valid_run(run, run.worker) and run.point

    # ---------- control
    def start(self, point: bool = False) -> None:
        if self.st.reference is None:
            raise ValueError("Save a reference first: the watch compares the table with it")
        with self._control:
            self.stop()                  # old run is permanently invalid before a replacement is published
            self.on, self.point, self.status = True, point, "starting"
            self.motion_ticket = self.st.motion_ticket()
            self._hist = deque(maxlen=3)  # a resumed old worker must not populate the new run's history
            self._stop = threading.Event()
            worker = threading.Thread(target=lambda: self._loop(run), daemon=True)
            run = WatchRun(self._generation, self._stop, self.motion_ticket, worker, point, self._hist)
            self._run, self._thread = run, worker
            worker.start()

    def stop(self) -> None:
        with self._control:
            self._stop.set()
            if self._run:
                self._run.cancel.set()
            self._generation += 1
            self._run = None
            worker = self._thread
            self.on, self.status = False, "off"
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2)  # never join a replacement worker or overwrite its status after waiting

    def ack(self, who: str = "") -> dict | None:
        a = self.alert
        if a:
            self._acked = a["signature"]
            a["acknowledged_by"], a["acknowledged_at"] = who, time.strftime("%H:%M:%S")
            self.st.audit.event("watch_ack", alert=a["id"], who=who)
            self.st.light.show("idle")
        self.alert = None
        return a

    def as_dict(self) -> dict:
        a = self.alert
        return {"on": self.on, "point": self.point, "status": self.status, "checks": self.checks,
                "alerts": self.alerts,
                "alert": None if not a else {k: v for k, v in a.items() if k not in ("signature",)}}

    # ---------- the loop
    def _loop(self, run: WatchRun) -> None:
        while not run.cancel.wait(self.interval):
            if not self.valid_run(run, threading.current_thread()):
                return
            try:
                self.step(run)
            except Exception as e:  # noqa: BLE001 - the watch must never die silently
                self.status = f"error: {type(e).__name__}: {e}"

    def step(self, run: WatchRun | None = None) -> None:
        """One look at the table. Public so tests can drive it frame by frame."""
        st = self.st
        if run is None:                    # deterministic offline stepping, without a live background worker
            with self._control:
                if self._thread and self._thread.is_alive():
                    return
                if self._run is None:
                    ticket = self.motion_ticket if self.motion_ticket is not None else st.motion_ticket()
                    self._run = WatchRun(self._generation, threading.Event(), ticket,
                                         threading.current_thread(), self.point, self._hist)
                run = self._run
        if not self.valid_run(run, threading.current_thread()):
            return
        ref = st.reference
        if ref is None:
            self.status = "no reference"
            return
        if not isinstance(st.arm, SimArm) and st.arm.status().get("state") == "moving":
            self._pause("paused: the arm is moving", run)
            return
        if not st.at_survey():                       # camera on the arm, looking somewhere else
            self._pause("paused: the arm camera is away from the survey pose", run)
            return
        if not st.lock.acquire(blocking=False):      # a find or check is running: do not compete with it
            return
        try:
            f = st.frame()
            boxes, occluded = st.proposals(f)
        finally:
            st.lock.release()
        self.checks += 1
        if occluded:
            self._pause("paused: hand or arm in view", run)
            return
        if not self.valid_run(run, threading.current_thread()):
            return
        run.history.append((f, boxes))
        if len(run.history) < 3:
            self.status = "watching"
            return
        per = [chg.diff_objects(ref["frame"], ref["boxes"], fr, b) for fr, b in run.history]
        cands = chg.persistent_changes(per)
        empty_zones = [z for z in st.zones if z.rule == "empty"]
        zres = eval_zones(empty_zones, boxes) if empty_zones else []
        problems = [p for z in zres for p in z["problems"]]
        if not cands and not problems:
            self._acked = None
            self.status = "watching: line clear" + (" again (alarm waiting for acknowledgement)" if self.alert else "")
            return
        sig = _signature(cands, problems)
        if self.alert and self.alert["signature"] == sig:
            return
        if sig == self._acked:
            self.status = "watching: acknowledged change still there"
            return
        self._raise(f, cands, problems, zres, sig, run=run)

    def _raise(self, f, cands: list, problems: list[str], zres: list, sig: tuple, *, run: WatchRun) -> None:
        st = self.st
        if not self.valid_run(run, threading.current_thread()):
            return
        ref_names = st.reference.get("names") or {}
        for c in cands:
            src = f if c.cur_box is not None else st.reference["frame"]
            c.label = (ref_names.get(c.ref_box.id) if c.ref_box is not None else None) or \
                f"{color_name(crop(src, c.cur_box or c.ref_box, pad=0))} object"
            c.risk = "high" if c.kind == "added" else "medium"
        img = draw_changes(f, cands, "cur")
        if st.zones:
            img = draw_zones(img, st.zones, zres)
        self.alerts += 1
        alert = {"id": self.alerts, "time": time.strftime("%H:%M:%S"), "signature": sig,
                 "changes": [c.as_dict() for c in cands], "zone_problems": problems,
                 "summary": self._summary(cands, problems), "image": st.audit.save_image("watch_alert", img),
                 "pointed": False, "grok": "pending" if cands else "not needed"}
        if not self.valid_run(run, threading.current_thread()):
            return
        self._alert_motion = (alert, run.ticket, run)
        self.alert = alert
        self.status = "ALARM"
        st.light.show("alarm")
        ev = st.audit.event("watch_alert", **{k: v for k, v in alert.items() if k != "signature"})
        alert["event_id"] = ev["id"]
        if run.point:
            self._point_at(f, cands, problems, zres, alert=alert, run=run)
        if cands:
            threading.Thread(target=self._explain, args=(alert, f, cands, run), daemon=True).start()

    def _pause(self, why: str, run: WatchRun) -> None:
        self.status = why
        run.history.clear()             # persistence restarts from clean frames only

    @staticmethod
    def _summary(cands: list, problems: list[str]) -> str:
        parts = [f"{c.label} {c.kind}" for c in cands] + problems
        return "Line changed after clearance: " + "; ".join(parts) + "."

    def _point_at(self, f, cands, problems, zres, *, alert: dict, run: WatchRun) -> None:
        st = self.st
        owner = threading.current_thread()
        if not run.point or not self.valid_run(run, owner) or self.alert is not alert:
            return
        if st.arm_blocked() or not (st.calib and st.calib.ok):
            return
        target = next((c.cur_box for c in cands if c.cur_box is not None), None)
        if target is None and cands:
            target = cands[0].ref_box                 # removed: point where it used to be
        if target is None:
            return
        x, y = st.calib.pixel_to_table(*target.center)
        try:
            if st.start_arm(st.arm.point_at, x, y, job="point", ticket=run.ticket,
                            guard=lambda: self.valid_run(run, owner) and self.alert is alert):
                alert["pointed"] = [round(x, 1), round(y, 1)]
        except RuntimeError:                           # the arm is busy (e.g. a lift): the alarm still stands
            pass

    def _explain(self, alert: dict, f, cands: list, run: WatchRun) -> None:
        st = self.st
        if not self.valid_run(run, run.worker):
            return
        try:
            real, filtered, _, summary, latency = chg.verify_changes(
                st.grok, st.reference["frame"], f, cands, st.s.lang, st.reference.get("names"))
        except GrokError as e:
            alert["grok"] = f"unavailable: {e}"[:200]
            return
        if self.alert is not alert or not self.valid_run(run, run.worker):
            return
        alert["changes"] = [c.as_dict() for c in real]
        alert["filtered"] = [c.as_dict() for c in filtered]
        alert["grok"] = f"explained in {latency:.1f} s"
        if summary:
            alert["summary"] = summary
        if not real and not alert["zone_problems"]:     # Grok: every candidate was a false alarm
            alert["grok"] = "Grok: false alarm (" + "; ".join(c.explanation for c in filtered)[:200] + ")"
            st.audit.event("watch_false_alarm", alert=alert["id"])
            self.alert, self.status, self._acked = None, "watching: last alarm was a false alarm", alert["signature"]
