"""Local file writer tool with directory restrictions."""

import logging
from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings
from agent.tools.file_reader import _is_path_allowed

logger = logging.getLogger(__name__)


@tool
def file_writer(path: str, content: str, mode: str = "write") -> str:
    """Write content to a local file.

    Restricted to allowed directories (~/Documents, ~/Projects by default).

    Args:
        path: Path to the file to write.
        content: The content to write.
        mode: "write" to overwrite, "append" to add to end.
    """
    logger.info("Writing file: %s (mode=%s)", path, mode)
    file_path = Path(path).expanduser()

    if not _is_path_allowed(file_path):
        return f"Access denied: {path} is outside allowed directories ({', '.join(settings.allowed_directories)})"

    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append":
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            file_path.write_text(content, encoding="utf-8")

        return f"Successfully wrote {len(content)} chars to {file_path}"
    except Exception as e:
        logger.error("File write failed: %s", e)
        return f"Error writing file: {e}"
