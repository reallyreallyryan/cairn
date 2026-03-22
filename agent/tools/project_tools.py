"""Project CRUD tools for the cairn agent."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def create_project(
    name: str,
    description: str = "",
    status: str = "active",
    metadata: str = "",
) -> str:
    """Create a new project in the SCMS knowledge base.

    Args:
        name: Project name (must be unique, e.g. 'ridgeline').
        description: What the project is about.
        status: Initial status — 'active', 'idea', 'paused', or 'completed'.
        metadata: JSON string with keys like stack, goals, etc. Example: '{"stack": ["flowise"]}'
    """
    import json

    from scms.client import SCMSClient

    logger.info("Creating project: %s", name)
    try:
        meta_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        return f"Error: invalid JSON in metadata: {metadata}"

    try:
        with SCMSClient() as client:
            result = client.create_project(
                name=name, description=description, status=status, metadata=meta_dict,
            )
        return f"Created project '{result['name']}' (id: {result['id']}, status: {result['status']})"
    except Exception as e:
        error_msg = str(e)
        if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
            return f"Error: project '{name}' already exists"
        return f"Error creating project: {e}"


@tool
def update_project(
    name: str,
    description: str = "",
    status: str = "",
    metadata: str = "",
) -> str:
    """Update an existing project's fields. Only non-empty fields are changed.

    Args:
        name: Project name to update.
        description: New description (empty string to skip).
        status: New status — 'active', 'idea', 'paused', 'completed', 'archived' (empty to skip).
        metadata: JSON string for new metadata (replaces existing). Empty to skip.
    """
    import json

    from scms.client import SCMSClient

    logger.info("Updating project: %s", name)
    try:
        meta_dict = json.loads(metadata) if metadata else None
    except json.JSONDecodeError:
        return f"Error: invalid JSON in metadata: {metadata}"

    try:
        with SCMSClient() as client:
            result = client.update_project(
                name=name,
                description=description or None,
                status=status or None,
                metadata=meta_dict,
            )
        if "error" in result:
            return result["error"]
        return f"Updated project '{result['name']}' (status: {result['status']})"
    except Exception as e:
        return f"Error updating project: {e}"


@tool
def archive_project(name: str) -> str:
    """Archive (soft-delete) a project. Sets status to 'archived'.

    The project and all linked memories/decisions are preserved.
    Reversible via update_project(name, status='active').

    Args:
        name: Project name to archive.
    """
    from scms.client import SCMSClient

    logger.info("Archiving project: %s", name)
    try:
        with SCMSClient() as client:
            result = client.archive_project(name=name)
        if "error" in result:
            return result["error"]
        return f"Archived project '{result['name']}'"
    except Exception as e:
        return f"Error archiving project: {e}"
