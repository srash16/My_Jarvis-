"""Handle common commands locally — no Gemini API call needed."""

import re

from system_config import CHROME_NICKNAMES
from system_control import TOOL_EXECUTORS

# Google web apps → URL (longest names first for matching)
GOOGLE_SERVICES = [
    ("google classroom", "https://classroom.google.com"),
    ("classroom", "https://classroom.google.com"),
    ("google drive", "https://drive.google.com"),
    ("drive", "https://drive.google.com"),
    ("google docs", "https://docs.google.com"),
    ("docs", "https://docs.google.com"),
    ("google sheets", "https://sheets.google.com"),
    ("sheets", "https://sheets.google.com"),
    ("google slides", "https://slides.google.com"),
    ("slides", "https://slides.google.com"),
    ("gmail", "https://mail.google.com"),
    ("google mail", "https://mail.google.com"),
    ("google meet", "https://meet.google.com"),
    ("meet", "https://meet.google.com"),
    ("google calendar", "https://calendar.google.com"),
    ("calendar", "https://calendar.google.com"),
    ("google photos", "https://photos.google.com"),
    ("photos", "https://photos.google.com"),
    ("google maps", "https://maps.google.com"),
    ("maps", "https://maps.google.com"),
    ("youtube", "https://youtube.com"),
    ("google", "https://google.com"),
]

KNOWN_APPS = {
    "chrome", "notepad", "calculator", "calc", "vscode", "cursor", "spotify",
    "explorer", "file explorer", "terminal", "settings", "discord", "paint",
    "word", "excel", "teams", "firefox", "edge", "cmd", "powershell",
}


def _normalize_profile(raw: str) -> str:
    """Clean whisper typos like 'mmcoe profit' → 'mmcoe'."""
    p = raw.lower().strip()
    p = re.sub(r"\b(profile|profit|account|email|chrome)\b", "", p).strip()
    for word in p.split():
        if word in CHROME_NICKNAMES:
            return word
    return p.split()[0] if p.split() else raw.strip()


def _extract_profile(text: str) -> str | None:
    m = re.search(
        r"(?: on| with| using| in| from) (?:my )?(.+?)(?:'s)?(?: profile| profit| account| email)?\.?$",
        text,
    )
    if m:
        return _normalize_profile(m.group(1))
    return None


def _strip_open_prefix(text: str) -> str:
    t = re.sub(r"^[,.\!\?\s]+", "", text.lower().strip())
    t = re.sub(r"^(?:hey |okay |please )", "", t)
    t = re.sub(r"^(?:open|launch|start|run)\s+(?:the\s+)?", "", t)
    return t.strip()


def _try_google(text: str) -> str | None:
    """Open Google websites in Chrome (optionally with a specific account)."""
    if not re.search(
        r"\bgoogle\b|classroom|gmail|drive|youtube|docs|sheets|slides|meet|calendar|photos|maps",
        text,
    ):
        return None

    profile = _extract_profile(text)
    subject = _strip_open_prefix(text)
    if profile:
        subject = re.sub(
            r"(?: on| with| using| in| from) (?:my )?.+$", "", subject
        ).strip()

    for name, url in GOOGLE_SERVICES:
        if subject == name or subject.startswith(name + " ") or name in subject:
            if profile:
                return TOOL_EXECUTORS["open_chrome"](email=profile, url=url)
            return TOOL_EXECUTORS["open_website"](url=url)

    return None


def _try_local(text: str) -> str | None:
    t = text.lower().strip()
    t = re.sub(r"^[,.\!\?\s]+", "", t)

    if any(w in t for w in ("goodbye", "quit", "exit", "bye", "stop")):
        return "__EXIT__"

    # Google services BEFORE generic "open X" (prevents "start google" Windows error)
    google_result = _try_google(t)
    if google_result:
        return google_result

    # open chrome with [email/nickname]
    m = re.search(
        r"open chrome(?: with| as| using)?(?: my)? (.+?)(?: account| email| profile| profit)?$",
        t,
    )
    if m:
        return TOOL_EXECUTORS["open_chrome"](email=_normalize_profile(m.group(1)))

    if re.search(r"open chrome|launch chrome|start chrome", t):
        return TOOL_EXECUTORS["open_application"](app_name="chrome")

    # open known desktop apps only (not arbitrary text)
    m = re.search(r"(?:open|launch|start|run)\s+(?:the\s+)?(.+?)(?: app| application)?$", t)
    if m:
        app = m.group(1).strip()
        # first word or full alias match
        if app in KNOWN_APPS:
            return TOOL_EXECUTORS["open_application"](app_name=app)
        first = app.split()[0]
        if first in KNOWN_APPS and len(app.split()) == 1:
            return TOOL_EXECUTORS["open_application"](app_name=first)

    # open website with domain
    m = re.search(r"(?:open|go to|visit|launch)\s+(?:the\s+)?(?:website\s+)?([\w\.\-]+\.\w{2,})", t)
    if m:
        return TOOL_EXECUTORS["open_website"](url=m.group(1))

    # volume
    m = re.search(r"(?:set volume to|volume)\s+(\d+)", t)
    if m:
        return TOOL_EXECUTORS["set_volume"](level=int(m.group(1)))

    if re.search(r"\b(mute|silence)\b", t):
        return TOOL_EXECUTORS["mute_volume"](mute=True)
    if re.search(r"\bunmute\b", t):
        return TOOL_EXECUTORS["mute_volume"](mute=False)
    if re.search(r"(?:what(?:'s| is) (?:my |the )?volume|get volume)", t):
        return TOOL_EXECUTORS["get_volume"]()

    # brightness
    m = re.search(r"(?:set brightness to|brightness)\s+(\d+)", t)
    if m:
        return TOOL_EXECUTORS["set_brightness"](level=int(m.group(1)))
    if re.search(r"(?:reduce|lower|decrease) (?:the )?brightness", t):
        return TOOL_EXECUTORS["set_brightness"](level=40)
    if re.search(r"(?:increase|raise) (?:the )?brightness", t):
        return TOOL_EXECUTORS["set_brightness"](level=80)

    # power
    if re.search(r"lock (?:my |the )?(?:pc|computer|screen)", t):
        return TOOL_EXECUTORS["lock_computer"]()
    if re.search(r"(?:list|show) (?:open )?windows", t):
        return TOOL_EXECUTORS["list_open_windows"]()
    if re.search(r"list chrome profiles", t):
        return TOOL_EXECUTORS["list_chrome_profiles"]()
    if re.search(r"(?:what(?:'s| is) on (?:my )?screen|see (?:my )?screen|look at (?:my )?screen)", t):
        return TOOL_EXECUTORS["see_screen"](question="Describe what is visible on the screen.")

    # list folder
    m = re.search(r"list(?: files in| folder| directory)? (.+)", t)
    if m and any(w in t for w in ("list", "files", "folder", "directory", "contents")):
        return TOOL_EXECUTORS["list_directory"](folder_path=m.group(1).strip())

    return None


def handle_locally(text: str) -> tuple[bool, str]:
    """Return (handled, result). handled=True means skip Gemini."""
    result = _try_local(text)
    if result == "__EXIT__":
        return True, "__EXIT__"
    if result is not None:
        return True, result
    return False, ""
