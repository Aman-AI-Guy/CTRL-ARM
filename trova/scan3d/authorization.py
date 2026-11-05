"""Operator-only bounded scan scope; each move still uses its own gateway/receipt checks."""

from dataclasses import dataclass
import time
import uuid

from ..workflow import fingerprint


@dataclass
class OrbitPermit:
    id: str
    operator: str
    session_id: str
    object_id: int
    remaining: int
    expires_at: float
    binding: str
    started: bool = False
    reason: str = ""

    @classmethod
    def approve(cls, host, task, session_id, *, operator, max_views, ttl_s, confirm):
        if not confirm or not operator.strip():
            raise ValueError(
                "An operator must explicitly approve the bounded scan scope."
            )
        if not 1 <= max_views <= 12 or not 30 <= ttl_s <= 600:
            raise ValueError("Bounded scans permit 1..12 moves for 30..600 seconds.")
        if task.get("joint_operation") != "scan" or not task.get("confirmation"):
            raise ValueError(
                "First plan one scan view and review its specific proposal."
            )
        if (
            not task.get("proposal", {}).get("allow_motion")
            or host.gateway.mode == "dry_run"
        ):
            raise ValueError(
                "This proposal has no execution authorization; a scope cannot unlock hardware."
            )
        qualification = host.orbits.settings.motion_qualification or {}
        if not qualification.get("supervised_orbit_verified") or not qualification.get(
            "verification_note"
        ):
            raise ValueError(
                "Complete and record the separate supervised orbit qualification before bounded autonomy."
            )
        return cls(
            uuid.uuid4().hex,
            operator.strip(),
            session_id,
            task["selected"]["track_id"],
            max_views,
            time.time() + ttl_s,
            fingerprint(host),
        )

    def invalid_reason(self, host, task, session_id):
        if time.time() >= self.expires_at or self.remaining <= 0:
            return "Bounded orbit time or movement budget exhausted; further moves need operator review."
        if host.cancel.is_set():
            return "STOP or cancellation ended the orbit scope."
        if self.session_id != session_id or self.object_id != (
            task.get("selected") or {}
        ).get("track_id"):
            return "The scan session or object identity changed."
        if task.get("joint_operation") != "scan":
            return (
                "This scope authorizes scan viewpoints only, never object approaches."
            )
        if self.binding != fingerprint(host):
            return "Calibration, model, optical profile or measured safety settings changed."
        if not task.get("confirmation"):
            return "No fresh validated per-view proposal is available."
        return None

    def consume(self):
        self.remaining -= 1

    def public(self):
        return {
            "id": self.id,
            "operator": self.operator,
            "session_id": self.session_id,
            "object_id": self.object_id,
            "remaining_moves": self.remaining,
            "expires_at": self.expires_at,
            "started": self.started,
            "reason": self.reason,
            "unlocks_hardware": False,
        }
