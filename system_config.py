"""System control settings loaded from environment variables."""

import os
from pathlib import Path


def _parse_pairs(raw: str) -> dict[str, str]:
    """Parse 'key=value;key2=value2' into a dict."""
    result = {}
    if not raw:
        return result
    for part in raw.split(";"):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip().lower()] = value.strip()
    return result


# Custom app shortcuts: JARVIS_CUSTOM_APPS=obs=C:\path\obs.exe;steam=steam
CUSTOM_APPS = _parse_pairs(os.getenv("JARVIS_CUSTOM_APPS", ""))

# Chrome profile nicknames: JARVIS_CHROME_NICKNAMES=work=mmcoe.edu.in;college=mmcoe.edu.in
CHROME_NICKNAMES = _parse_pairs(os.getenv("JARVIS_CHROME_NICKNAMES", ""))

# Seconds to wait before shutdown/restart after confirmation
POWER_DELAY_SECONDS = int(os.getenv("JARVIS_POWER_DELAY", "30"))
