"""Telegram notifications (CLAUDE.md §7.8). Token / chat id come from .env, never from configs.

Uses the Bot API `sendMessage` method over HTTPS. Network failures are logged, never raised:
a notification problem must not stop the analytics loop.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from src.common.logging import get_logger
from src.common.schemas import AlertEvent

log = get_logger(__name__)

LEVEL_ICON = {1: "ℹ️", 2: "⚠️", 3: "🚨"}


def format_alert(a: AlertEvent) -> str:
    who = f"người #{a.global_id}" if a.global_id is not None else "khu vực"
    zone = f" tại {a.zone}" if a.zone else ""
    cams = f" (cam: {', '.join(a.cams)})" if a.cams else ""
    clip = f"\nClip: {a.evidence.clip}" if a.evidence.clip else ""
    return f"{LEVEL_ICON.get(a.level, '')} L{a.level} {a.type.value}: {who}{zone}{cams}{clip}"


class TelegramNotifier:
    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        min_level: int = 2,
        post: Callable[..., object] | None = None,
    ) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.min_level = min_level
        self._post = post

    @classmethod
    def from_config(cls, alerts_cfg: dict) -> TelegramNotifier | None:
        t = alerts_cfg.get("telegram", {})
        if not t.get("enabled"):
            return None
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        n = cls(min_level=t.get("min_level", 2))
        if not n.token or not n.chat_id:
            log.warning("telegram enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
            return None
        return n

    def send(self, a: AlertEvent) -> bool:
        if a.level < self.min_level:
            return False
        post = self._post
        if post is None:
            import httpx

            post = httpx.post
        try:
            r = post(
                self.API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": format_alert(a)},
                timeout=5.0,
            )
            ok = getattr(r, "status_code", 200) == 200
            if not ok:
                log.warning("telegram HTTP %s", getattr(r, "status_code", "?"))
            return ok
        except Exception as e:  # noqa: BLE001 — never break the analytics loop
            log.warning("telegram send failed: %s", e)
            return False
