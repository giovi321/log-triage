from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Notification:
    level: str
    title: str
    detail: str
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "title": self.title,
            "detail": self.detail,
            "created_at": self.created_at,
        }


_MAX_NOTIFICATIONS = 200
_notifications: List[Notification] = []


def add_notification(level: str, title: str, detail: str) -> Notification:
    entry = Notification(level=level, title=title, detail=detail)
    _notifications.append(entry)
    if len(_notifications) > _MAX_NOTIFICATIONS:
        del _notifications[0 : len(_notifications) - _MAX_NOTIFICATIONS]
    return entry


def list_notifications() -> List[Notification]:
    return sorted(_notifications, key=lambda n: n.created_at, reverse=True)


def latest_notification_level() -> Optional[str]:
    for level in ("error", "warning", "info"):
        if any(n.level == level for n in _notifications):
            return level
    return None


def notification_summary() -> dict:
    notes = list_notifications()
    if not notes:
        return {
            "state_class": "ok",
            "message": "All systems operational",
            "count": 0,
            "latest": None,
        }

    latest = notes[0]
    has_error = any(n.level == "error" for n in notes)
    has_warning = any(n.level == "warning" for n in notes)

    if has_error:
        state_class = "error"
        message = f"Attention needed: {latest.title}"
    elif has_warning:
        state_class = "warn"
        message = f"Warnings present: {latest.title}"
    else:
        state_class = "ok"
        message = f"Info: {latest.title}"

    return {
        "state_class": state_class,
        "message": message,
        "count": len(notes),
        "latest": latest,
    }
