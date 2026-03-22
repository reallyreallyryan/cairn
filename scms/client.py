"""SCMS Client — interface to the Shared Context Memory Store.

Wraps Supabase operations for memories, projects, tools, and decisions.
"""

import logging
from typing import Any

import httpx
from supabase import create_client, Client

from config.settings import settings
from scms.embeddings import get_embedding

logger = logging.getLogger(__name__)


class SCMSClient:
    """Client for the Shared Context Memory Store (Supabase + pgvector)."""

    def __init__(self, url: str | None = None, key: str | None = None):
        self._url = (url or settings.supabase_url).strip()
        # Remove embedded newlines/whitespace that can creep in from env var UIs
        self._key = "".join((key or settings.supabase_key).split())
        if not self._url or not self._key:
            raise ValueError(
                "Supabase URL and key are required. "
                "Set SUPABASE_URL and SUPABASE_KEY in your .env file."
            )
        self._client: Client = create_client(self._url, self._key)
        # Force HTTP/1.1 on the postgrest client to avoid HTTP/2 StreamReset
        # errors when running behind Railway's reverse proxy.
        pg = self._client.postgrest
        old = pg.session
        old.close()
        pg.session = httpx.Client(
            base_url=str(old.base_url),
            headers=dict(old.headers),
            timeout=old.timeout,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        """Close the underlying httpx session to release sockets."""
        try:
            self._client.postgrest.session.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def store_memory(
        self,
        content: str,
        memory_type: str = "learning",
        project_name: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Store a new memory with its embedding."""
        logger.info("Storing memory: type=%s, project=%s", memory_type, project_name)

        embedding = get_embedding(content)
        project_id = self._resolve_project_id(project_name) if project_name else None

        record = {
            "content": content,
            "memory_type": memory_type,
            "embedding": embedding,
            "tags": tags or [],
            "source": source,
            "metadata": metadata or {},
        }
        if project_id:
            record["project_id"] = project_id

        result = self._client.table("memories").insert(record).execute()
        logger.info("Stored memory: %s", result.data[0]["id"])
        return result.data[0]

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.3,
        project_name: str | None = None,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Semantic search for memories using pgvector cosine similarity."""
        logger.info("Searching memories: query='%s', limit=%d", query[:50], limit)

        query_embedding = get_embedding(query)
        project_id = self._resolve_project_id(project_name) if project_name else None

        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": limit,
        }
        if project_id:
            params["filter_project_id"] = project_id
        if memory_type:
            params["filter_memory_type"] = memory_type

        result = self._client.rpc("match_memories", params).execute()
        logger.info("Found %d memories", len(result.data))
        return result.data

    def search_memories_by_embedding(
        self,
        embedding: list[float],
        limit: int = 5,
        threshold: float = 0.0,
        project_id: str | None = None,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Semantic search using a pre-computed embedding vector.

        Like search_memories(), but accepts an embedding directly instead of
        generating one from text. Also accepts a project UUID directly instead
        of a project name. Useful for batch operations where embeddings and
        project IDs are resolved up front.
        """
        params: dict[str, Any] = {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": limit,
        }
        if project_id:
            params["filter_project_id"] = project_id
        if memory_type:
            params["filter_memory_type"] = memory_type

        result = self._client.rpc("match_memories", params).execute()
        return result.data

    def update_memory(self, memory_id: str, **updates: Any) -> dict:
        """Update fields on an existing memory."""
        # Re-generate embedding if content changed
        if "content" in updates:
            updates["embedding"] = get_embedding(updates["content"])

        result = (
            self._client.table("memories")
            .update(updates)
            .eq("id", memory_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        result = (
            self._client.table("memories")
            .delete()
            .eq("id", memory_id)
            .execute()
        )
        return len(result.data) > 0

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def get_project_context(self, project_name: str) -> dict:
        """Get full context for a project: metadata + recent memories + decisions."""
        logger.info("Getting context for project: %s", project_name)

        # Get project record
        project_result = (
            self._client.table("projects")
            .select("*")
            .eq("name", project_name)
            .execute()
        )
        if not project_result.data:
            return {"error": f"Project '{project_name}' not found"}

        project = project_result.data[0]

        # Get recent memories for this project
        memories_result = (
            self._client.table("memories")
            .select("id, content, memory_type, tags, source, created_at")
            .eq("project_id", project["id"])
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )

        # Get recent decisions
        decisions_result = (
            self._client.table("decision_log")
            .select("id, decision, reasoning, alternatives, outcome, created_at")
            .eq("project_id", project["id"])
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )

        return {
            "project": project,
            "recent_memories": memories_result.data,
            "recent_decisions": decisions_result.data,
        }

    def list_projects(self, status: str | None = None) -> list[dict]:
        """List all projects, optionally filtered by status."""
        query = self._client.table("projects").select("*")
        if status:
            query = query.eq("status", status)
        result = query.order("name").execute()
        return result.data

    # ------------------------------------------------------------------
    # Decision Log
    # ------------------------------------------------------------------

    def log_decision(
        self,
        decision: str,
        reasoning: str = "",
        alternatives: list[str] | None = None,
        project_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict:
        """Record an architectural or design decision."""
        logger.info("Logging decision: %s", decision[:50])

        project_id = self._resolve_project_id(project_name) if project_name else None

        record = {
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "context": context or {},
        }
        if project_id:
            record["project_id"] = project_id

        result = self._client.table("decision_log").insert(record).execute()
        return result.data[0]

    def get_decisions(
        self,
        project_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get recent decisions, optionally filtered by project."""
        logger.info("Getting decisions: project=%s, limit=%d", project_name, limit)
        query = self._client.table("decision_log").select(
            "id, decision, reasoning, alternatives, outcome, context, created_at"
        )
        if project_name:
            project_id = self._resolve_project_id(project_name)
            if project_id:
                query = query.eq("project_id", project_id)
            else:
                return []
        result = query.order("created_at", desc=True).limit(limit).execute()
        return result.data

    # ------------------------------------------------------------------
    # Tool Registry
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        tool_type: str = "builtin",
        function_name: str | None = None,
        config: dict[str, Any] | None = None,
        approval_status: str = "approved",
    ) -> dict:
        """Register or update a tool in the registry."""
        logger.info("Registering tool: %s (approval=%s)", name, approval_status)

        record = {
            "name": name,
            "description": description,
            "tool_type": tool_type,
            "function_name": function_name,
            "config": config or {},
            "enabled": approval_status == "approved",
            "approval_status": approval_status,
        }

        result = (
            self._client.table("tool_registry")
            .upsert(record, on_conflict="name")
            .execute()
        )
        return result.data[0]

    def list_tools(self, enabled_only: bool = True) -> list[dict]:
        """List registered tools."""
        query = self._client.table("tool_registry").select("*")
        if enabled_only:
            query = query.eq("enabled", True)
        result = query.order("name").execute()
        return result.data

    def list_pending_tools(self) -> list[dict]:
        """List tools awaiting human approval."""
        result = (
            self._client.table("tool_registry")
            .select("*")
            .eq("approval_status", "pending")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    def get_tool(self, tool_id: str) -> dict | None:
        """Get a single tool by ID."""
        result = (
            self._client.table("tool_registry")
            .select("*")
            .eq("id", tool_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def approve_tool(self, tool_id: str, approved_by: str = "human") -> dict | None:
        """Approve a pending tool, making it available to the agent."""
        logger.info("Approving tool: %s", tool_id)
        result = (
            self._client.table("tool_registry")
            .update({
                "approval_status": "approved",
                "approved_by": approved_by,
                "enabled": True,
            })
            .eq("id", tool_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def reject_tool(self, tool_id: str, reason: str = "") -> dict | None:
        """Reject a pending tool."""
        logger.info("Rejecting tool: %s (reason: %s)", tool_id, reason)
        result = (
            self._client.table("tool_registry")
            .update({
                "approval_status": "rejected",
                "enabled": False,
                "config": {"rejection_reason": reason},
            })
            .eq("id", tool_id)
            .execute()
        )
        return result.data[0] if result.data else None

    # ------------------------------------------------------------------
    # Task Queue
    # ------------------------------------------------------------------

    def enqueue_task(
        self,
        task: str,
        priority: int = 5,
        project: str | None = None,
        recurring: str | None = None,
    ) -> dict:
        """Add a task to the queue."""
        logger.info("Enqueuing task (priority=%d): %s", priority, task[:60])
        record: dict[str, Any] = {
            "task": task,
            "priority": max(1, min(10, priority)),
            "status": "pending",
        }
        if project:
            record["project"] = project
        if recurring:
            record["recurring"] = recurring
        result = self._client.table("task_queue").insert(record).execute()
        return result.data[0]

    def get_pending_tasks(self, limit: int = 10) -> list[dict]:
        """Get pending tasks ordered by priority (highest first)."""
        result = (
            self._client.table("task_queue")
            .select("*")
            .eq("status", "pending")
            .order("priority", desc=False)  # 1 = highest priority
            .order("created_at")
            .limit(limit)
            .execute()
        )
        return result.data

    def get_recurring_tasks(self) -> list[dict]:
        """Get all tasks with a recurring cron expression."""
        result = (
            self._client.table("task_queue")
            .select("*")
            .neq("recurring", None)
            .execute()
        )
        return result.data

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
        error: str | None = None,
        model_used: str | None = None,
        cost_usd: float | None = None,
    ) -> dict:
        """Update task status and results."""
        from datetime import datetime, timezone

        updates: dict[str, Any] = {"status": status}
        now = datetime.now(timezone.utc).isoformat()
        if status == "running":
            updates["started_at"] = now
        if status in ("completed", "failed"):
            updates["completed_at"] = now
        if result is not None:
            updates["result"] = result
        if error is not None:
            updates["error"] = error
        if model_used is not None:
            updates["model_used"] = model_used
        if cost_usd is not None:
            updates["cost_usd"] = cost_usd

        res = (
            self._client.table("task_queue")
            .update(updates)
            .eq("id", task_id)
            .execute()
        )
        return res.data[0] if res.data else {}

    def get_completed_tasks(self, limit: int = 20) -> list[dict]:
        """Get recently completed tasks."""
        result = (
            self._client.table("task_queue")
            .select("*")
            .eq("status", "completed")
            .order("completed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    def get_queue_status(self) -> dict[str, int]:
        """Get counts by status for the task queue."""
        result = self._client.table("task_queue").select("status").execute()
        counts: dict[str, int] = {}
        for row in result.data:
            s = row["status"]
            counts[s] = counts.get(s, 0) + 1
        return counts

    def get_daily_spend(self) -> float:
        """Get total cloud spend for today."""
        from datetime import date
        today = date.today().isoformat()
        result = (
            self._client.table("task_queue")
            .select("cost_usd")
            .gte("created_at", today)
            .execute()
        )
        return sum(row.get("cost_usd", 0) or 0 for row in result.data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_project_id(self, project_name: str) -> str | None:
        """Look up a project ID by name."""
        result = (
            self._client.table("projects")
            .select("id")
            .eq("name", project_name)
            .execute()
        )
        if result.data:
            return result.data[0]["id"]
        logger.warning("Project not found: %s", project_name)
        return None
