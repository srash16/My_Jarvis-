"""Fast local JSONL audit log for system tool calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).resolve().parent / "jarvis_audit.log"

_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "app_password",
    "api_key",
    "apikey",
    "secret",
    "token",
    "credential",
    "jarvis_db_key",
)


def _redact_args(args: dict) -> dict:
    redacted = {}
    for key, value in (args or {}).items():
        key_l = str(key).lower()
        if any(frag in key_l for frag in _SENSITIVE_KEY_FRAGMENTS):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


def log_tool_call(
    tool_name: str,
    args: dict,
    result: str,
    confirmed: bool | None = None,
) -> None:
    """Append one JSON line to jarvis_audit.log (non-blocking local write)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "args": _redact_args(args if isinstance(args, dict) else {}),
        "result": (result or "")[:300],
        "confirmed": confirmed,
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Never break the voice pipeline over audit logging
        print(f"[Audit] Failed to write log: {e}")
