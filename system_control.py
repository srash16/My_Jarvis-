"""
Windows system control tools for JARVIS.

File operations are restricted to the user's home directory.
Destructive actions require confirmed=True after verbal user confirmation.
PyAutoGUI failsafe: move mouse to top-left corner to abort automation.
"""

import io
import json
import os
import shutil
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path

import pyautogui
import pygetwindow as gw
from pycaw.pycaw import AudioUtilities

from system_config import CHROME_NICKNAMES, CUSTOM_APPS, POWER_DELAY_SECONDS

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.15

CHROME_USER_DATA = Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data"
SCREENSHOT_DIR = Path.home() / ".jarvis_screenshots"
_gemini_client = None

APP_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "vscode": "code",
    "visual studio code": "code",
    "cursor": "cursor",
    "spotify": "spotify",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "cmd": "cmd",
    "command prompt": "cmd",
    "terminal": "wt",
    "windows terminal": "wt",
    "powershell": "powershell",
    "settings": "ms-settings:",
    "paint": "mspaint",
    "word": "winword",
    "excel": "excel",
    "teams": "msteams",
    "discord": "discord",
    "task manager": "taskmgr",
}


def set_gemini_client(client):
    """Inject Gemini client for screen vision (see_screen tool)."""
    global _gemini_client
    _gemini_client = client


def _resolve_path(path: str) -> Path:
    p = Path(path.strip().strip('"')).expanduser()
    if not p.is_absolute():
        p = Path.home() / p
    return p.resolve()


def _is_allowed_path(path: Path) -> bool:
    try:
        path.relative_to(Path.home().resolve())
        return True
    except ValueError:
        return False


def _require_confirmation(action: str, target: str, confirmed: bool) -> str | None:
    if confirmed:
        return None
    return (
        f"CONFIRMATION REQUIRED: {action} '{target}'. "
        "Ask the user to confirm verbally, then call this tool again with confirmed=True."
    )


def _run_start(command: str) -> None:
    subprocess.Popen(
        f'start "" {command}',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _launch_path(path: str) -> None:
    subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _find_chrome_exe() -> Path | None:
    for base in (
        os.environ.get("PROGRAMFILES", ""),
        os.environ.get("PROGRAMFILES(X86)", ""),
        os.environ.get("LOCALAPPDATA", ""),
    ):
        candidate = Path(base) / "Google/Chrome/Application/chrome.exe"
        if candidate.exists():
            return candidate
    return None


def _load_chrome_profiles() -> list[dict]:
    state_file = CHROME_USER_DATA / "Local State"
    if not state_file.exists():
        return []
    data = json.loads(state_file.read_text(encoding="utf-8"))
    info_cache = data.get("profile", {}).get("info_cache", {})
    return [
        {
            "profile_id": pid,
            "email": (info.get("user_name") or "").strip(),
            "name": (info.get("name") or "").strip(),
        }
        for pid, info in info_cache.items()
    ]


def _match_chrome_profile(email_or_name: str) -> dict | None:
    query = email_or_name.lower().strip()
    if query in CHROME_NICKNAMES:
        query = CHROME_NICKNAMES[query].lower()

    profiles = _load_chrome_profiles()
    for profile in profiles:
        email = profile["email"].lower()
        name = profile["name"].lower()
        if query and (query == email or query in email or query == name or query in name):
            return profile
    return None


def _get_volume_interface():
    return AudioUtilities.GetSpeakers().EndpointVolume


def list_chrome_profiles() -> str:
    """List all Google Chrome profiles on this PC with their linked emails.

    Returns:
        A formatted list of Chrome profiles and configured nicknames.
    """
    profiles = _load_chrome_profiles()
    if not profiles:
        return "No Chrome profiles found. Is Google Chrome installed?"

    lines = ["Chrome profiles on this PC:"]
    for p in profiles:
        email = p["email"] or "(no email signed in)"
        label = p["name"] or p["profile_id"]
        lines.append(f'  - {label}: {email} [{p["profile_id"]}]')

    if CHROME_NICKNAMES:
        lines.append("\nConfigured nicknames (say these instead of full email):")
        for nick, email in CHROME_NICKNAMES.items():
            lines.append(f'  - "{nick}" → {email}')
    return "\n".join(lines)


def open_chrome(email: str, url: str = "") -> str:
    """Open Google Chrome with a specific Google account profile.

    Args:
        email: Google account email, partial email, profile name, or nickname
            from JARVIS_CHROME_NICKNAMES (e.g. "work", "mmcoe.edu.in").
        url: Optional website to open in that profile.

    Returns:
        A short status message.
    """
    profile = _match_chrome_profile(email)
    if not profile:
        return (
            f'No Chrome profile found for "{email}". '
            "Use list_chrome_profiles to see available accounts and nicknames."
        )

    chrome_exe = _find_chrome_exe()
    if not chrome_exe:
        return "Google Chrome executable not found on this PC."

    target_url = url.strip()
    if target_url and not target_url.startswith(("http://", "https://")):
        target_url = f"https://{target_url}"

    args = [str(chrome_exe), f'--profile-directory={profile["profile_id"]}']
    if target_url:
        args.append(target_url)

    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        label = profile["email"] or profile["name"] or profile["profile_id"]
        if target_url:
            return f"Opened Chrome as {label} and navigated to {target_url}."
        return f"Opened Chrome as {label}."
    except Exception as e:
        return f"Could not open Chrome profile: {e}"


def open_application(app_name: str) -> str:
    """Open a desktop application on Windows.

    Args:
        app_name: App name or alias (chrome, notepad, vscode, cursor, etc.).
            Custom apps from JARVIS_CUSTOM_APPS in .env are also supported.

    Returns:
        A short status message.
    """
    key = app_name.lower().strip()

    if key in CUSTOM_APPS:
        try:
            _launch_path(CUSTOM_APPS[key])
            return f"Opened {app_name}."
        except Exception as e:
            return f"Could not open custom app {app_name}: {e}"

    command = APP_ALIASES.get(key, app_name)
    try:
        _run_start(command)
        return f"Opened {app_name}."
    except Exception as e:
        return f"Could not open {app_name}: {e}"


def open_website(url: str) -> str:
    """Open a website in the default browser.

    Args:
        url: Website URL. https:// is added automatically if missing.

    Returns:
        A short status message.
    """
    target = url.strip()
    if not target.startswith(("http://", "https://")):
        target = f"https://{target}"
    try:
        webbrowser.open(target)
        return f"Opened {target} in your browser."
    except Exception as e:
        return f"Could not open website: {e}"


def list_directory(folder_path: str = ".") -> str:
    """List files and folders in a directory inside the user's home folder.

    Args:
        folder_path: Path to list. Use "." for home directory.

    Returns:
        A formatted list of entries.
    """
    path = _resolve_path(folder_path if folder_path != "." else str(Path.home()))
    if not _is_allowed_path(path):
        return "Access denied. I can only manage files inside your home folder."
    if not path.exists():
        return f"Path not found: {path}"
    if not path.is_dir():
        return f"Not a folder: {path}"

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    if not entries:
        return f"{path} is empty."

    lines = [f"Contents of {path}:"]
    for entry in entries[:50]:
        kind = "folder" if entry.is_dir() else "file"
        lines.append(f"  [{kind}] {entry.name}")
    if len(entries) > 50:
        lines.append(f"  ... and {len(entries) - 50} more")
    return "\n".join(lines)


def open_file_or_folder(path: str) -> str:
    """Open a file or folder with its default Windows application.

    Args:
        path: File or folder path inside the user's home directory.

    Returns:
        A short status message.
    """
    target = _resolve_path(path)
    if not _is_allowed_path(target):
        return "Access denied. I can only open paths inside your home folder."
    if not target.exists():
        return f"Path not found: {target}"
    try:
        os.startfile(target)
        return f"Opened {target.name}."
    except Exception as e:
        return f"Could not open {target}: {e}"


def create_folder(folder_path: str) -> str:
    """Create a new folder inside the user's home directory.

    Args:
        folder_path: Folder path to create.

    Returns:
        A short status message.
    """
    target = _resolve_path(folder_path)
    if not _is_allowed_path(target):
        return "Access denied. I can only create folders inside your home folder."
    if target.exists():
        return f"Already exists: {target}"
    try:
        target.mkdir(parents=True, exist_ok=False)
        return f"Created folder {target}."
    except Exception as e:
        return f"Could not create folder: {e}"


def delete_file(path: str, confirmed: bool = False) -> str:
    """Delete a file or empty folder. REQUIRES confirmed=True after user verbally confirms.

    Args:
        path: File or empty folder path inside the user's home directory.
        confirmed: Must be True only after the user explicitly confirms deletion.

    Returns:
        A short status message or confirmation request.
    """
    target = _resolve_path(path)
    if not _is_allowed_path(target):
        return "Access denied. I can only delete files inside your home folder."
    if not target.exists():
        return f"Path not found: {target}"

    msg = _require_confirmation("delete", str(target), confirmed)
    if msg:
        return msg

    try:
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()
        return f"Deleted {target.name}."
    except OSError as e:
        return f"Could not delete {target.name}: {e}. Folder must be empty."


def move_file(source: str, destination: str, confirmed: bool = False) -> str:
    """Move or rename a file/folder. REQUIRES confirmed=True if overwriting an existing file.

    Args:
        source: Source path inside home directory.
        destination: Destination path inside home directory.
        confirmed: Must be True if destination already exists and user confirmed overwrite.

    Returns:
        A short status message.
    """
    src = _resolve_path(source)
    dst = _resolve_path(destination)
    if not _is_allowed_path(src) or not _is_allowed_path(dst):
        return "Access denied. Paths must be inside your home folder."
    if not src.exists():
        return f"Source not found: {src}"
    if dst.exists() and not confirmed:
        return _require_confirmation("overwrite/move to", str(dst), False)

    try:
        shutil.move(str(src), str(dst))
        return f"Moved {src.name} to {dst}."
    except Exception as e:
        return f"Could not move file: {e}"


def see_screen(question: str = "Describe what is visible on the screen in detail.") -> str:
    """Capture a screenshot and analyze it with vision AI.

    Args:
        question: What to look for on screen, e.g. "what error is shown?" or
            "what application is open?".

    Returns:
        A description of what's on screen.
    """
    if _gemini_client is None:
        return "Screen vision is not initialized."

    from google.genai import types

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = SCREENSHOT_DIR / f"screen_{timestamp}.png"

    try:
        img = pyautogui.screenshot()
        img.save(screenshot_path)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=question),
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            ])],
        )
        return response.text or "I couldn't analyze the screen."
    except Exception as e:
        return f"Could not analyze screen: {e}"


def get_volume() -> str:
    """Get the current system master volume percentage.

    Returns:
        Current volume level 0-100.
    """
    try:
        volume = _get_volume_interface()
        level = int(round(volume.GetMasterVolumeLevelScalar() * 100))
        muted = volume.GetMute()
        state = "muted" if muted else f"{level}%"
        return f"System volume is {state}."
    except Exception as e:
        return f"Could not read volume: {e}"


def set_volume(level: int) -> str:
    """Set system master volume.

    Args:
        level: Volume percentage from 0 (mute) to 100 (max).

    Returns:
        A short status message.
    """
    level = max(0, min(100, level))
    try:
        volume = _get_volume_interface()
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        if level == 0:
            volume.SetMute(1, None)
        else:
            volume.SetMute(0, None)
        return f"Volume set to {level}%."
    except Exception as e:
        return f"Could not set volume: {e}"


def mute_volume(mute: bool = True) -> str:
    """Mute or unmute system volume.

    Args:
        mute: True to mute, False to unmute.

    Returns:
        A short status message.
    """
    try:
        volume = _get_volume_interface()
        volume.SetMute(1 if mute else 0, None)
        return "Volume muted." if mute else "Volume unmuted."
    except Exception as e:
        return f"Could not change mute state: {e}"


def set_brightness(level: int) -> str:
    """Set screen brightness on supported laptops.

    Args:
        level: Brightness percentage from 0 to 100.

    Returns:
        A short status message.
    """
    level = max(0, min(100, level))
    try:
        subprocess.run(
            [
                "powershell", "-Command",
                f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
                f".WmiSetBrightness(1, {level})",
            ],
            check=True,
            capture_output=True,
        )
        return f"Brightness set to {level}%."
    except subprocess.CalledProcessError:
        return "Brightness control is not supported on this display."
    except Exception as e:
        return f"Could not set brightness: {e}"


def lock_computer() -> str:
    """Lock the Windows workstation immediately.

    Returns:
        A short status message.
    """
    try:
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
        return "Computer locked."
    except Exception as e:
        return f"Could not lock computer: {e}"


def sleep_computer() -> str:
    """Put the computer to sleep.

    Returns:
        A short status message.
    """
    try:
        subprocess.run(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            check=True,
        )
        return "Going to sleep."
    except Exception as e:
        return f"Could not sleep: {e}"


def shutdown_computer(confirmed: bool = False, restart: bool = False) -> str:
    """Shutdown or restart the PC. REQUIRES confirmed=True after user verbally confirms.

    Args:
        confirmed: Must be True only after explicit user confirmation.
        restart: If True, restart instead of shutdown.

    Returns:
        A short status message or confirmation request.
    """
    action = "restart" if restart else "shutdown"
    msg = _require_confirmation(action, f"PC in {POWER_DELAY_SECONDS}s", confirmed)
    if msg:
        return msg

    flag = "/r" if restart else "/s"
    try:
        subprocess.run(
            ["shutdown", flag, "/t", str(POWER_DELAY_SECONDS),
             "/c", f"JARVIS initiated {action}. Cancel with: shutdown /a"],
            check=True,
        )
        return (
            f"{'Restarting' if restart else 'Shutting down'} in {POWER_DELAY_SECONDS} seconds. "
            "Say 'cancel shutdown' to abort."
        )
    except Exception as e:
        return f"Could not {action}: {e}"


def cancel_shutdown() -> str:
    """Cancel a pending shutdown or restart initiated by JARVIS.

    Returns:
        A short status message.
    """
    try:
        subprocess.run(["shutdown", "/a"], check=True, capture_output=True)
        return "Pending shutdown cancelled."
    except subprocess.CalledProcessError:
        return "No pending shutdown to cancel."
    except Exception as e:
        return f"Could not cancel shutdown: {e}"


def type_text(text: str) -> str:
    """Type text at the current cursor position using the keyboard.

    Args:
        text: Text to type into the active window.

    Returns:
        A short status message.
    """
    try:
        pyautogui.write(text, interval=0.02)
        return f'Typed "{text[:60]}{"..." if len(text) > 60 else ""}".'
    except Exception as e:
        return f"Could not type text: {e}"


def press_hotkey(*keys: str) -> str:
    """Press a keyboard shortcut.

    Args:
        *keys: Keys to press together, e.g. "ctrl", "c" or "alt", "tab".

    Returns:
        A short status message.
    """
    if not keys:
        return "No keys specified."
    try:
        pyautogui.hotkey(*keys)
        return f"Pressed {' + '.join(keys)}."
    except Exception as e:
        return f"Could not press hotkey: {e}"


def click_screen(x: int, y: int, button: str = "left") -> str:
    """Click at a specific screen coordinate.

    Args:
        x: Horizontal pixel position from the left edge of the screen.
        y: Vertical pixel position from the top edge of the screen.
        button: Mouse button: left, right, or middle.

    Returns:
        A short status message.
    """
    try:
        pyautogui.click(x=x, y=y, button=button)
        return f"Clicked {button} button at ({x}, {y})."
    except Exception as e:
        return f"Could not click: {e}"


def list_open_windows() -> str:
    """List visible open window titles on the desktop.

    Returns:
        A formatted list of window titles.
    """
    titles = [t for t in gw.getAllTitles() if t.strip()]
    if not titles:
        return "No open windows found."
    lines = ["Open windows:"]
    for title in titles[:30]:
        lines.append(f"  - {title}")
    if len(titles) > 30:
        lines.append(f"  ... and {len(titles) - 30} more")
    return "\n".join(lines)


def focus_window(window_title: str) -> str:
    """Bring a window to the front by matching part of its title.

    Args:
        window_title: Text that appears in the window title bar.

    Returns:
        A short status message.
    """
    matches = [w for w in gw.getAllWindows() if window_title.lower() in w.title.lower()]
    if not matches:
        return f'No window found matching "{window_title}".'
    try:
        win = matches[0]
        if win.isMinimized:
            win.restore()
        win.activate()
        return f'Focused window "{win.title}".'
    except Exception as e:
        return f"Could not focus window: {e}"


SYSTEM_TOOLS = [
    open_application,
    open_chrome,
    list_chrome_profiles,
    open_website,
    list_directory,
    open_file_or_folder,
    create_folder,
    delete_file,
    move_file,
    see_screen,
    get_volume,
    set_volume,
    mute_volume,
    set_brightness,
    lock_computer,
    sleep_computer,
    shutdown_computer,
    cancel_shutdown,
    type_text,
    press_hotkey,
    click_screen,
    list_open_windows,
    focus_window,
]

TOOL_EXECUTORS = {fn.__name__: fn for fn in SYSTEM_TOOLS}

SYSTEM_CONTROL_PROMPT = (
    "\n\nYou control the user's Windows PC via tools. Use tools for actions; "
    "answer normally for questions.\n"
    "Chrome with a specific account: open_chrome(email=...) — supports nicknames "
    "from .env (e.g. 'work'). Use list_chrome_profiles to see accounts.\n"
    "Screen vision: see_screen(question=...) to analyze what's on screen.\n"
    "Volume/brightness: set_volume, mute_volume, set_brightness, get_volume.\n"
    "Power: lock_computer, sleep_computer, shutdown_computer(restart=True/False).\n"
    "DESTRUCTIVE actions (delete_file, move_file overwrite, shutdown/restart) "
    "REQUIRE confirmed=True ONLY after the user explicitly says yes/confirm. "
    "First call with confirmed=False to ask; second call with confirmed=True to execute.\n"
    "cancel_shutdown aborts a pending shutdown."
)
