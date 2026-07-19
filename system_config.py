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

# Gmail accounts for sending emails: JARVIS_GMAIL_ACCOUNTS=email1=app_password1;email2=app_password2
# Default account to use if not specified: JARVIS_GMAIL_DEFAULT=email1
def _parse_gmail_accounts(raw: str) -> dict[str, str]:
    """Parse 'email1=app_password1;email2=app_password2' into a dict."""
    return _parse_pairs(raw)

GMAIL_ACCOUNTS = _parse_gmail_accounts(os.getenv("JARVIS_GMAIL_ACCOUNTS", ""))
GMAIL_DEFAULT = os.getenv("JARVIS_GMAIL_DEFAULT", "")

# Gemini API key (falls back to GOOGLE_API_KEY used elsewhere in the project)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Vision system: set to false to disable webcam-based object/action detection
# On-demand camera vision (voice-triggered capture only; no always-on watcher)
VISION_ENABLED = os.getenv("JARVIS_VISION_ENABLED", "true").lower() == "true"
