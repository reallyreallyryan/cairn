"""cairn daemon — persistent autonomous task execution."""

import logging
import signal
import time

from rich.console import Console

from agent.notifications import notify
from config.settings import settings

logger = logging.getLogger(__name__)
console = Console()

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    console.print("\n[yellow]Shutdown requested, finishing current task...[/yellow]")


def _poll_and_execute():
    """Poll task_queue for the highest-priority pending task and execute it."""
    if _shutdown_requested:
        return

    from scms.client import SCMSClient

    try:
        client = SCMSClient()
    except Exception as e:
        logger.error("Cannot connect to SCMS: %s", e)
        return

    with client:
        # Get highest priority pending task
        pending = client.get_pending_tasks(limit=1)
        if not pending:
            return

        task_row = pending[0]
        task_id = task_row["id"]
        task_text = task_row["task"]
        project = task_row.get("project", "")

        logger.info("=== Executing task %s (priority %d): %s ===",
                    task_id[:8], task_row["priority"], task_text[:80])
        client.update_task_status(task_id, "running")

        # Digest tasks bypass the graph and run the digest pipeline directly
        task_lower = task_text.lower()
        if any(kw in task_lower for kw in ["run digest", "daily digest", "research digest"]):
            try:
                from agent.digest import run_digest

                frequency = "weekly" if "weekly" in task_lower else "daily"
                result_data = run_digest(frequency=frequency)
                result_text = (
                    f"Digest completed: {result_data['items_found']} items from "
                    f"{result_data['sources_processed']} sources, "
                    f"{result_data['items_queued']} queued for review."
                )
                if result_data["digest_path"]:
                    result_text += f"\nSaved to: {result_data['digest_path']}"
                if result_data["errors"]:
                    result_text += f"\nErrors: {', '.join(result_data['errors'])}"

                client.update_task_status(
                    task_id, "completed",
                    result=result_text, model_used="local", cost_usd=0.0,
                )
                console.print(f"[green]Digest complete:[/green] {result_data['items_found']} items")
                notify("Digest Complete", f"{result_data['items_found']} items found")
            except Exception as e:
                logger.error("Digest task %s failed: %s", task_id[:8], e)
                client.update_task_status(task_id, "failed", error=str(e)[:2000])
                notify("Digest Failed", str(e)[:60])
            return

        try:
            from langchain_core.messages import SystemMessage

            from agent.graph import build_graph

            graph = build_graph()
            initial_state = {
                "messages": [
                    SystemMessage(
                        content="You are executing a queued task. Use your tools — "
                        "call them directly rather than describing what you would do."
                    )
                ],
                "task": task_text,
                "task_type": "",
                "project": project,
                "available_tools": [],
                "context": "",
                "plan": "",
                "plan_steps": [],
                "current_step": 0,
                "step_results": {},
                "tools_used": [],
                "decisions": [],
                "pending_tools": [],
                "sandbox_logs": [],
                "result": "",
                "should_continue": True,
                "model_override": "cloud",
                "iteration": 0,
                "model_used": "",
                "cost_estimate": 0.0,
            }

            result = graph.invoke(initial_state)

            result_text = result.get("result", "No result")
            model_used = result.get("model_used", "unknown")
            cost = result.get("cost_estimate", 0.0)

            client.update_task_status(
                task_id,
                "completed",
                result=result_text[:10000],
                model_used=model_used,
                cost_usd=cost,
            )

            # Auto-store result in SCMS so downstream tasks can find it
            try:
                client.store_memory(
                    content=f"Task result: {task_text}\n\n{result_text[:3000]}",
                    memory_type="learning",
                    source="daemon",
                    project_name=project or None,
                    tags=["daemon-result", f"task-{task_id[:8]}"],
                )
            except Exception as store_err:
                logger.warning("Failed to auto-store task result: %s", store_err)

            console.print(f"[green]Completed:[/green] {task_text[:60]}...")
            notify("Task Complete", f"{task_text[:60]}")
            logger.info("Task %s completed (model=%s, cost=$%.4f)", task_id[:8], model_used, cost)

        except Exception as e:
            logger.error("Task %s failed: %s", task_id[:8], e)
            client.update_task_status(task_id, "failed", error=str(e)[:2000])
            notify("Task Failed", f"{task_text[:40]}... error: {str(e)[:60]}")


def _check_recurring_tasks():
    """Check recurring tasks and re-enqueue if due."""
    if _shutdown_requested:
        return

    try:
        from datetime import datetime, timezone
        from croniter import croniter
        from scms.client import SCMSClient

        with SCMSClient() as client:
            all_tasks = client.get_recurring_tasks()

            for task_row in all_tasks:
                cron_expr = task_row.get("recurring")
                if not cron_expr:
                    continue

                # Only re-enqueue if this instance is completed or failed
                if task_row["status"] not in ("completed", "failed"):
                    continue

                # Check if it's time for the next run
                completed_at = task_row.get("completed_at")
                if completed_at:
                    try:
                        base_time = datetime.fromisoformat(completed_at)
                        cron = croniter(cron_expr, base_time)
                        next_run = cron.get_next(datetime)
                        now = datetime.now(timezone.utc)
                        if now < next_run:
                            continue
                    except Exception:
                        pass

                # Check no pending duplicate exists
                pending = client.get_pending_tasks(limit=50)
                already_pending = any(
                    p["task"] == task_row["task"] and p.get("recurring") == cron_expr
                    for p in pending
                )
                if already_pending:
                    continue

                # Re-enqueue
                client.enqueue_task(
                    task=task_row["task"],
                    priority=task_row.get("priority", 5),
                    project=task_row.get("project"),
                    recurring=cron_expr,
                )
                logger.info("Re-enqueued recurring task: %s", task_row["task"][:60])

    except Exception as e:
        logger.warning("Recurring task check failed: %s", e)


def run_daemon():
    """Start the persistent daemon loop."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    poll_interval = settings.daemon_poll_interval

    console.print(
        f"[bold green]cairn Daemon[/bold green] started "
        f"(polling every {poll_interval}s, Ctrl+C to stop)"
    )
    notify("Daemon Started", "cairn daemon is now running")

    iteration = 0
    try:
        while not _shutdown_requested:
            _poll_and_execute()

            # Check recurring tasks every 5th poll cycle
            iteration += 1
            if iteration % 10 == 0:
                _check_recurring_tasks()

            # Sleep in small increments so shutdown is responsive
            for _ in range(poll_interval):
                if _shutdown_requested:
                    break
                time.sleep(1)

    except KeyboardInterrupt:
        pass

    console.print("[yellow]Daemon stopped.[/yellow]")
    notify("Daemon Stopped", "cairn daemon has been shut down")
