"""SCMS tools for the LangGraph agent."""

import logging

from langchain_core.tools import tool

from scms.client import SCMSClient

logger = logging.getLogger(__name__)


@tool
def scms_search(query: str, project: str = "", memory_type: str = "") -> str:
    """Search the Shared Context Memory Store for relevant memories and context.

    Args:
        query: What to search for in the knowledge store.
        project: Optional project name to filter by (e.g., 'cairn', 'Research').
        memory_type: Optional type filter (concept, pattern, decision, learning, etc.).
    """
    logger.info("SCMS search: query='%s', project='%s'", query, project)
    try:
        client = SCMSClient()
        results = client.search_memories(
            query=query,
            limit=5,
            project_name=project or None,
            memory_type=memory_type or None,
        )

        if not results:
            # Fall back to listing projects if no memories match
            projects = client.list_projects()
            if projects:
                lines = ["No matching memories found. Here are the current projects:"]
                for p in projects:
                    meta = p.get("metadata", {})
                    goals = meta.get("goals", [])
                    lines.append(
                        f"- **{p['name']}** ({p['status']}): {p['description']}"
                    )
                    if goals:
                        lines.append(f"  Goals: {', '.join(goals)}")
                return "\n".join(lines)
            return "No memories or projects found in the SCMS."

        lines = [f"Found {len(results)} relevant memories:\n"]
        for r in results:
            sim = r.get("similarity", 0)
            lines.append(f"[{sim:.2f}] ({r['memory_type']}) {r['content']}")
            if r.get("tags"):
                lines.append(f"  Tags: {', '.join(r['tags'])}")
        return "\n".join(lines)

    except Exception as e:
        logger.error("SCMS search failed: %s", e)
        return f"Error searching SCMS: {e}"


@tool
def scms_store(
    content: str,
    memory_type: str = "learning",
    project: str = "",
    tags: str = "",
    source: str = "agent",
) -> str:
    """Store a new memory in the Shared Context Memory Store.

    Args:
        content: The knowledge, fact, or insight to store.
        memory_type: Type of memory: concept, pattern, decision, learning, reference, etc.
        project: Optional project name to associate with.
        tags: Comma-separated tags for this memory.
        source: Where this knowledge came from (agent, web_search, user, etc.).
    """
    # Validate and normalize memory_type to match DB constraint
    VALID_TYPES = {
        "concept", "pattern", "decision", "tool_eval",
        "problem_solution", "reference", "context", "learning",
    }
    if memory_type not in VALID_TYPES:
        logger.info("Mapping invalid memory_type '%s' -> 'concept'", memory_type)
        memory_type = "concept"

    logger.info("SCMS store: type=%s, project='%s'", memory_type, project)
    try:
        client = SCMSClient()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        result = client.store_memory(
            content=content,
            memory_type=memory_type,
            project_name=project or None,
            tags=tag_list,
            source=source,
        )
        return f"Successfully stored memory (id: {result['id']})"

    except Exception as e:
        logger.error("SCMS store failed: %s", e)
        return f"Error storing to SCMS: {e}"
