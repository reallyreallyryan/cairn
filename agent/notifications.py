"""Simple notification system for cairn daemon."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


def notify(title: str, message: str):
    """Send a notification via the configured method."""
    if settings.notification_method == "macos":
        _notify_macos(title, message)
    _notify_log(title, message)  # always log to file


def _notify_macos(title: str, message: str):
    """Send macOS desktop notification via osascript."""
    try:
        # Escape quotes for AppleScript
        safe_title = title.replace('"', '\\"')
        safe_msg = message.replace('"', '\\"')[:200]
        script = f'display notification "{safe_msg}" with title "{safe_title}"'
        subprocess.run(
            ["osascript", "-e", script],
            timeout=5,
            capture_output=True,
        )
    except Exception as e:
        logger.debug("macOS notification failed: %s", e)


def _notify_log(title: str, message: str):
    """Append notification to log file."""
    try:
        log_path = Path(settings.notification_log_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {title}: {message}\n")
    except Exception as e:
        logger.debug("Notification log write failed: %s", e)
