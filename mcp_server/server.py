"""cairn MCP Server — exposes SCMS as tools via Streamable HTTP."""

import logging

from fastmcp import FastMCP
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from mcp.server.auth.settings import ClientRegistrationOptions

from mcp_server.config import mcp_settings
from scms.client import SCMSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Use OAuth when base_url is configured (production); no auth for local dev.
_auth: InMemoryOAuthProvider | None = None
if mcp_settings.mcp_base_url:
    _auth = InMemoryOAuthProvider(
        base_url=mcp_settings.mcp_base_url,
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["mcp"],
            default_scopes=["mcp"],
        ),
    )

mcp = FastMCP("cairn SCMS", auth=_auth)


def _get_client() -> SCMSClient:
    return SCMSClient()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": True})
def scms_search(
    query: str,
    limit: int = 5,
    project: str | None = None,
    memory_type: str | None = None,
) -> str:
    """Search memories in the SCMS knowledge base using semantic similarity."""
    logger.info("MCP scms_search: query=%r project=%s", query[:50], project)
    try:
        results = _get_client().search_memories(
            query=query, limit=limit, project_name=project, memory_type=memory_type
        )
        if not results:
            return "No matching memories found."
        lines = []
        for r in results:
            sim = f"{r.get('similarity', 0):.2f}"
            lines.append(f"[{sim}] ({r.get('memory_type', 'unknown')}) {r['content'][:200]}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("scms_search error: %s", e)
        return f"Error searching memories: {e}"


@mcp.tool(annotations={"destructiveHint": False})
def scms_store(
    content: str,
    category: str = "learning",
    project: str | None = None,
    tags: list[str] | None = None,
    source: str = "mcp",
) -> str:
    """Store a new memory in the SCMS knowledge base."""
    logger.info("MCP scms_store: category=%s project=%s", category, project)
    try:
        result = _get_client().store_memory(
            content=content,
            memory_type=category,
            project_name=project,
            tags=tags,
            source=source,
        )
        return f"Stored memory {result['id']} (type: {category})"
    except Exception as e:
        logger.error("scms_store error: %s", e)
        return f"Error storing memory: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def get_project_context(project_name: str) -> str:
    """Get full context for a project: metadata, recent memories, and decisions."""
    logger.info("MCP get_project_context: %s", project_name)
    try:
        ctx = _get_client().get_project_context(project_name)
        if "error" in ctx:
            return ctx["error"]
        project = ctx["project"]
        lines = [f"# {project['name']}", f"Status: {project.get('status', 'unknown')}"]
        if project.get("description"):
            lines.append(f"Description: {project['description']}")
        if ctx["recent_memories"]:
            lines.append(f"\n## Recent Memories ({len(ctx['recent_memories'])})")
            for m in ctx["recent_memories"][:10]:
                lines.append(f"- [{m['memory_type']}] {m['content'][:120]}")
        if ctx["recent_decisions"]:
            lines.append(f"\n## Recent Decisions ({len(ctx['recent_decisions'])})")
            for d in ctx["recent_decisions"][:5]:
                lines.append(f"- {d['decision'][:120]}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("get_project_context error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def list_projects() -> str:
    """List all projects in the SCMS."""
    logger.info("MCP list_projects")
    try:
        projects = _get_client().list_projects()
        if not projects:
            return "No projects found."
        lines = []
        for p in projects:
            lines.append(f"- {p['name']} (status: {p.get('status', 'unknown')})")
        return "\n".join(lines)
    except Exception as e:
        logger.error("list_projects error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"destructiveHint": False})
def create_project(
    name: str,
    description: str = "",
    status: str = "active",
    metadata: dict | None = None,
) -> str:
    """Create a new project in the SCMS.

    Args:
        name: Project name (must be unique).
        description: What the project is about.
        status: Initial status — 'active', 'idea', 'paused', or 'completed'.
        metadata: Optional dict with keys like 'stack', 'goals', etc.
    """
    logger.info("MCP create_project: %s", name)
    try:
        result = _get_client().create_project(
            name=name, description=description, status=status, metadata=metadata,
        )
        return f"Created project '{result['name']}' (id: {result['id']}, status: {result['status']})"
    except Exception as e:
        error_msg = str(e)
        if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
            return f"Error: project '{name}' already exists"
        logger.error("create_project error: %s", e)
        return f"Error creating project: {e}"


@mcp.tool(annotations={"destructiveHint": False})
def update_project(
    name: str,
    description: str | None = None,
    status: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Update an existing project's fields. Only provided fields are changed.

    Args:
        name: Project name to update.
        description: New description (or None to keep current).
        status: New status — 'active', 'idea', 'paused', 'completed', 'archived'.
        metadata: New metadata dict (replaces existing metadata).
    """
    logger.info("MCP update_project: %s", name)
    try:
        result = _get_client().update_project(
            name=name, description=description, status=status, metadata=metadata,
        )
        if "error" in result:
            return result["error"]
        return f"Updated project '{result['name']}' (status: {result['status']})"
    except Exception as e:
        logger.error("update_project error: %s", e)
        return f"Error updating project: {e}"


@mcp.tool(annotations={"destructiveHint": False})
def archive_project(name: str) -> str:
    """Archive (soft-delete) a project. Sets status to 'archived'.

    The project and all linked memories/decisions are preserved but hidden
    from active views. Reversible via update_project(name, status='active').

    Args:
        name: Project name to archive.
    """
    logger.info("MCP archive_project: %s", name)
    try:
        result = _get_client().archive_project(name=name)
        if "error" in result:
            return result["error"]
        return f"Archived project '{result['name']}'"
    except Exception as e:
        logger.error("archive_project error: %s", e)
        return f"Error archiving project: {e}"


@mcp.tool(annotations={"destructiveHint": False})
def queue_task(task: str, priority: int = 5, project: str | None = None) -> str:
    """Add a task to the cairn agent's task queue."""
    logger.info("MCP queue_task: priority=%d project=%s", priority, project)
    try:
        result = _get_client().enqueue_task(task=task, priority=priority, project=project)
        return f"Queued task {result['id']} (priority: {result['priority']}, status: {result['status']})"
    except Exception as e:
        logger.error("queue_task error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def check_queue(status_filter: str | None = None, limit: int = 10) -> str:
    """Check the task queue status and list tasks.

    Args:
        status_filter: Filter by status ("pending", "completed", "failed", "running").
                       If None, shows both pending and completed tasks.
        limit: Maximum number of tasks to list per section.
    """
    logger.info("MCP check_queue: status=%s limit=%d", status_filter, limit)
    try:
        client = _get_client()
        counts = client.get_queue_status()
        lines = ["## Queue Status"]
        for status, count in sorted(counts.items()):
            lines.append(f"- {status}: {count}")
        if status_filter in ("pending", None):
            tasks = client.get_pending_tasks(limit=limit)
            if tasks:
                lines.append("\n## Pending Tasks")
                for t in tasks:
                    lines.append(
                        f"- [{t['priority']}] {t['task'][:100]} (id: {t['id']})"
                    )
        if status_filter in ("completed", None):
            completed = client.get_completed_tasks(limit=limit)
            if completed:
                lines.append("\n## Completed Tasks")
                for t in completed:
                    completed_at = t.get("completed_at", "")[:19] if t.get("completed_at") else ""
                    lines.append(
                        f"- {t['task'][:100]} (id: {t['id']}) — {completed_at}"
                    )
        return "\n".join(lines)
    except Exception as e:
        logger.error("check_queue error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def get_task_result(task_id: str) -> str:
    """Get the result of a specific task by ID."""
    logger.info("MCP get_task_result: %s", task_id)
    try:
        client = _get_client()
        result = (
            client._client.table("task_queue")
            .select("*")
            .eq("id", task_id)
            .execute()
        )
        if not result.data:
            return f"Task {task_id} not found."
        task = result.data[0]
        lines = [
            f"Task: {task['task']}",
            f"Status: {task['status']}",
            f"Priority: {task['priority']}",
            f"Created: {task['created_at']}",
        ]
        if task.get("result"):
            lines.append(f"Result: {task['result']}")
        if task.get("error"):
            lines.append(f"Error: {task['error']}")
        if task.get("model_used"):
            lines.append(f"Model: {task['model_used']}")
        if task.get("cost_usd"):
            lines.append(f"Cost: ${task['cost_usd']:.4f}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("get_task_result error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def get_decisions(project: str | None = None, limit: int = 10) -> str:
    """Get recent architectural and design decisions from the decision log."""
    logger.info("MCP get_decisions: project=%s limit=%d", project, limit)
    try:
        decisions = _get_client().get_decisions(project_name=project, limit=limit)
        if not decisions:
            return "No decisions found."
        lines = []
        for d in decisions:
            lines.append(f"### {d['decision'][:120]}")
            if d.get("reasoning"):
                lines.append(f"Reasoning: {d['reasoning'][:200]}")
            if d.get("alternatives"):
                lines.append(f"Alternatives: {', '.join(d['alternatives'])}")
            lines.append(f"Date: {d['created_at']}\n")
        return "\n".join(lines)
    except Exception as e:
        logger.error("get_decisions error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"destructiveHint": False})
def log_decision(
    decision: str,
    reasoning: str = "",
    alternatives: list[str] | None = None,
    project: str | None = None,
) -> str:
    """Log an architectural or design decision to the SCMS decision log."""
    logger.info("MCP log_decision: %s", decision[:50])
    try:
        result = _get_client().log_decision(
            decision=decision,
            reasoning=reasoning,
            alternatives=alternatives,
            project_name=project,
        )
        return f"Logged decision {result['id']}: {decision[:80]}"
    except Exception as e:
        logger.error("log_decision error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def agent_status() -> str:
    """Get cairn agent status: queue counts and daily spend."""
    logger.info("MCP agent_status")
    try:
        client = _get_client()
        counts = client.get_queue_status()
        spend = client.get_daily_spend()
        lines = ["## Agent Status", "### Queue"]
        for status, count in sorted(counts.items()):
            lines.append(f"- {status}: {count}")
        lines.append(f"\n### Budget")
        lines.append(f"- Today's spend: ${spend:.4f}")
        lines.append(f"- Daily limit: ${mcp_settings.daily_budget_usd:.2f}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("agent_status error: %s", e)
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Digest tools
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": True})
def review_digest(limit: int = 10) -> str:
    """List pending digest items awaiting human review."""
    logger.info("MCP review_digest: limit=%d", limit)
    try:
        client = _get_client()
        pending = client.get_pending_tasks(limit=50)
        items = [t for t in pending if t.get("project") == "_digest_review"]
        if not items:
            return "No digest items pending review."
        lines = [f"## Pending Digest Items ({len(items)})"]
        for item in items[:limit]:
            lines.append(f"- [{item['id'][:8]}] {item['task'][:200]}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("review_digest error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def digest_status() -> str:
    """Get digest pipeline status: recent runs and pending review count."""
    logger.info("MCP digest_status")
    try:
        client = _get_client()

        # Recent digest runs
        completed = client.get_completed_tasks(limit=30)
        digest_runs = [
            t for t in completed
            if "digest" in t.get("task", "").lower()
            and t.get("project") != "_digest_review"
        ]

        # Pending reviews
        pending = client.get_pending_tasks(limit=50)
        review_count = sum(1 for t in pending if t.get("project") == "_digest_review")

        lines = ["## Digest Status", f"Pending review items: {review_count}"]

        if digest_runs:
            lines.append("\n## Recent Runs")
            for run in digest_runs[:5]:
                run_date = (run.get("completed_at") or "")[:10]
                result = (run.get("result") or "")[:80]
                lines.append(f"- {run_date}: {result}")
        else:
            lines.append("\nNo digest runs found yet.")

        return "\n".join(lines)
    except Exception as e:
        logger.error("digest_status error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
def digest_eval() -> str:
    """Run evaluation on digest approval/rejection history and return a metrics report."""
    logger.info("MCP digest_eval")
    try:
        from agent.evaluation import run_evaluation
        from pathlib import Path

        result = run_evaluation()
        if result["total_reviewed"] == 0:
            return "No reviewed digest items found. Approve or reject some items first."

        # Return the generated report
        report_path = Path(result["report_path"])
        if report_path.exists():
            return report_path.read_text()

        # Fallback: return summary
        lines = [
            "## Digest Evaluation",
            f"Total reviewed: {result['total_reviewed']}",
            f"Approval rate: {result['approval_rate']:.1%}",
        ]
        if result["suggestions"]:
            lines.append("\n## Recommendations")
            for s in result["suggestions"]:
                lines.append(f"- {s}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("digest_eval error: %s", e)
        return f"Error: {e}"


@mcp.tool(annotations={"readOnlyHint": False})
def compile_digest(since: str | None = None) -> str:
    """Compile approved digest items into deep-dive and briefing summaries.

    Fetches full article content for approved items, generates two markdown
    documents: a technical deep dive and an accessible briefing.

    Args:
        since: ISO date string (YYYY-MM-DD) to look back from. Default: last 24 hours.
    """
    logger.info("MCP compile_digest: since=%s", since)
    try:
        from agent.compile_digest import run_compile_digest
        from pathlib import Path

        result = run_compile_digest(since=since)
        if result["articles_compiled"] == 0:
            return "No approved digest items found for this period."

        # Return the briefing content inline (useful for chat context)
        briefing_path = Path(result["briefing_path"])
        if briefing_path.exists():
            content = briefing_path.read_text()
            return (
                f"{content}\n\n---\n"
                f"Deep dive saved to: {result['deep_path']}\n"
                f"Articles: {result['articles_compiled']} "
                f"({result['articles_with_full_content']} with full content)"
            )

        return (
            f"Compiled {result['articles_compiled']} articles.\n"
            f"Deep dive: {result['deep_path']}\n"
            f"Briefing: {result['briefing_path']}"
        )
    except Exception as e:
        logger.error("compile_digest error: %s", e)
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import uvicorn

    app = mcp.http_app(transport="streamable-http")

    # Railway injects PORT at runtime; fall back to MCP_PORT for local dev
    port = int(os.environ.get("PORT", mcp_settings.mcp_port))

    logger.info(
        "Starting cairn MCP server on %s:%d (OAuth: %s)",
        mcp_settings.mcp_host,
        port,
        "enabled" if _auth else "disabled",
    )
    uvicorn.run(app, host=mcp_settings.mcp_host, port=port)
