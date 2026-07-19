"""Startup security checks that do not block JARVIS from running."""

import re
import subprocess


def check_bitlocker_status(drive: str = "C:") -> None:
    """
    Check BitLocker protection on the given drive via manage-bde.
    Prints a one-time console warning if encryption is off / incomplete.
    Fails silently if manage-bde is unavailable (e.g. Windows Home).
    """
    try:
        result = subprocess.run(
            ["manage-bde", "-status", drive],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return

        output = (result.stdout or "") + (result.stderr or "")
        if not output.strip():
            return

        protection = None
        percent = None
        for line in output.splitlines():
            lower = line.lower()
            if "protection status" in lower:
                protection = line.split(":", 1)[-1].strip()
            elif "percentage encrypted" in lower:
                raw = line.split(":", 1)[-1].strip()
                match = re.search(r"([\d.]+)", raw)
                if match:
                    percent = float(match.group(1))

        unprotected = protection and "off" in protection.lower()
        incomplete = percent is not None and percent < 100.0

        if unprotected or incomplete:
            print(
                "⚠️  BitLocker is not enabled on C:. Conversation history "
                "and memory files are stored unencrypted. Consider enabling BitLocker in "
                "Windows Settings > Privacy & Security > Device Encryption."
            )
    except Exception:
        # Windows Home / missing manage-bde / permission issues — skip quietly
        return
