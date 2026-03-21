"""Note-taking tool — saves markdown files and SCMS entries."""

import logging
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to a URL/filename-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:60]


@tool
def note_taker(title: str, content: str, project: str = "", tags: str = "") -> str:
    """Create a structured note as a markdown file and store a summary in the SCMS.

    Args:
        title: Title of the note.
        content: The note content (markdown supported).
        project: Optional project name to associate with.
        tags: Comma-separated tags.
    """
    logger.info("Taking note: %s (project=%s)", title, project)

    # Build the markdown file
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(title)
    filename = f"{date_str}_{slug}.md"

    notes_dir = Path(settings.notes_directory).expanduser()
    notes_dir.mkdir(parents=True, exist_ok=True)
    filepath = notes_dir / filename

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    md_content = f"""# {title}

**Date**: {date_str}
**Project**: {project or 'None'}
**Tags**: {', '.join(tag_list) if tag_list else 'None'}

---

{content}
"""
    try:
        filepath.write_text(md_content, encoding="utf-8")
        logger.info("Note saved to: %s", filepath)
    except Exception as e:
        logger.error("Failed to write note file: %s", e)
        return f"Error writing note file: {e}"

    # Also store summary in SCMS
    try:
        from scms.client import SCMSClient
        client = SCMSClient()

        summary = content[:500] if len(content) > 500 else content
        client.store_memory(
            content=f"Note: {title}\n{summary}",
            memory_type="reference",
            project_name=project or None,
            tags=tag_list,
            source="note_taker",
        )
        logger.info("Note summary stored in SCMS")
    except Exception as e:
        logger.warning("Failed to store note in SCMS: %s", e)
        return f"Note saved to {filepath} but SCMS storage failed: {e}"

    return f"Note saved to {filepath} and stored in SCMS"
