"""Local file reader tool with directory restrictions."""

import logging
from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings

logger = logging.getLogger(__name__)

MAX_CHARS = 10_000


def _is_path_allowed(path: Path) -> bool:
    """Check if a path is within the allowed directories."""
    resolved = path.resolve()
    for allowed in settings.allowed_directories:
        allowed_path = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_path)
            return True
        except ValueError:
            continue
    return False


@tool
def file_reader(path: str) -> str:
    """Read a local file and return its contents.

    Restricted to allowed directories (~/Documents, ~/Projects by default).

    Args:
        path: Path to the file to read.
    """
    logger.info("Reading file: %s", path)
    file_path = Path(path).expanduser()

    if not _is_path_allowed(file_path):
        return f"Access denied: {path} is outside allowed directories ({', '.join(settings.allowed_directories)})"

    if not file_path.exists():
        return f"File not found: {path}"

    if not file_path.is_file():
        return f"Not a file: {path}"

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + f"\n\n[Truncated at {MAX_CHARS} chars — file is {len(content)} chars total]"
        return content
    except Exception as e:
        logger.error("File read failed: %s", e)
        return f"Error reading file: {e}"
