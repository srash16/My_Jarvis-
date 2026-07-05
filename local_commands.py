"""Handle common commands locally — no Gemini API call needed."""

import re

from system_control import TOOL_EXECUTORS


def _try_local(text: str) -> str | None:
    t = text.lower().strip()
    t = re.sub(r"^[,.\!\?\s]+", "", t)

    if any(w in t for w in ("goodbye", "quit", "exit", "bye", "stop")):
        return "__EXIT__"

    # open chrome with [email/nickname]
    m = re.search(r"open chrome(?: with| as| using)?(?: my)? (.+?)(?: account| email| profile)?$", t)
    if m:
        return TOOL_EXECUTORS["open_chrome"](email=m.group(1).strip())

    if re.search(r"open chrome|launch chrome|start chrome", t):
        return TOOL_EXECUTORS["open_application"](app_name="chrome")

    # open [app]
    m = re.search(r"(?:open|launch|start|run)\s+(?:the\s+)?(.+?)(?: app| application)?$", t)
    if m:
        app = m.group(1).strip()
        skip = {"chrome", "google chrome", "a website", "the website", "this website"}
        if app not in skip:
            return TOOL_EXECUTORS["open_application"](app_name=app)

    # open website
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
