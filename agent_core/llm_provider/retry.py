from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping, Optional


_BODY_RETRY_PATTERNS = [
    re.compile(r"retry\s+in\s+(?P<seconds>\d+)\s*s", re.IGNORECASE),
    re.compile(r"try\s+again\s+in\s+(?P<seconds>\d+)\s*s", re.IGNORECASE),
]


def extract_retry_delay_ms(
    message: str = "",
    headers: Optional[Mapping[str, str]] = None,
    *,
    now: Optional[datetime] = None,
) -> Optional[int]:
    normalized_headers = {str(key).lower(): str(value) for key, value in (headers or {}).items()}
    current_time = now or datetime.now(timezone.utc)

    retry_after = normalized_headers.get("retry-after")
    if retry_after:
        retry_after = retry_after.strip()
        if retry_after.isdigit():
            return (int(retry_after) + 1) * 1000
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delta_ms = int((retry_at - current_time).total_seconds() * 1000)
            return max(delta_ms, 0) + 1000
        except Exception:
            pass

    reset_after = normalized_headers.get("x-ratelimit-reset-after")
    if reset_after:
        try:
            return int(float(reset_after) * 1000) + 1000
        except Exception:
            pass

    reset_at = normalized_headers.get("x-ratelimit-reset")
    if reset_at:
        try:
            reset_seconds = int(float(reset_at))
            delta_ms = (reset_seconds * 1000) - int(current_time.timestamp() * 1000)
            return max(delta_ms, 0) + 1000
        except Exception:
            pass

    for pattern in _BODY_RETRY_PATTERNS:
        match = pattern.search(message or "")
        if match:
            return (int(match.group("seconds")) + 1) * 1000

    return None


def ensure_retry_delay_within_cap(delay_ms: int, max_retry_delay_ms: Optional[int]) -> None:
    if max_retry_delay_ms is None or max_retry_delay_ms <= 0:
        return
    if delay_ms > max_retry_delay_ms:
        raise ValueError(
            f"Server requested retry delay {delay_ms}ms, which exceeds max_retry_delay_ms={max_retry_delay_ms}."
        )


__all__ = ["ensure_retry_delay_within_cap", "extract_retry_delay_ms"]
