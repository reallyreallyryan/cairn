"""cairn — Personal AI Agent CLI."""

import argparse
import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config.settings import settings

console = Console()


def setup_logging():
    """Configure logging based on settings."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def check_services():
    """Check that required services are reachable, auto-starting Ollama if needed."""
    import httpx
    import subprocess
    import time

    # Check Ollama — always needed (digest pipeline uses local model directly)
    ollama_ok = False
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        r.raise_for_status()
        ollama_ok = True
    except Exception:
        # Try to start Ollama automatically
        console.print("[yellow]Ollama not running — starting it...[/yellow]")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
            # Retry health check
            r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            ollama_ok = True
        except Exception:
            pass

    if ollama_ok:
        console.print("[green]Ollama[/green] connected", style="dim")
    else:
        console.print(
            "[yellow]Warning:[/yellow] Cannot reach Ollama at "
            f"{settings.ollama_base_url}. Run: [bold]ollama serve[/bold]",
        )

    # Check Supabase
    if settings.supabase_url:
        try:
            r = httpx.get(
                f"{settings.supabase_url}/rest/v1/",
                headers={"apikey": settings.supabase_key},
                timeout=5.0,
            )
            console.print("[green]Supabase[/green] connected", style="dim")
        except Exception:
            console.print(
                "[yellow]Warning:[/yellow] Cannot reach Supabase. "
                "Check SUPABASE_URL and SUPABASE_KEY in .env",
            )
    else:
        console.print(
            "[yellow]Warning:[/yellow] No SUPABASE_URL configured. "
            "SCMS features will not work.",
        )


def run_task(task: str, model: str | None = None):
    """Run a single task through the agent."""
    from agent.graph import build_graph

    # Override model if specified
    if model:
        settings.agent_model = model

    graph = build_graph()

    initial_state = {
        "messages": [],
        "task": task,
        "task_type": "",
        "project": "",
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
        "model_override": model or "",
        "model_used": "",
        "cost_estimate": 0.0,
        "iteration": 0,
    }

    console.print(Panel(task, title="Task", border_style="blue"))

    with console.status("[bold green]Thinking...", spinner="dots"):
        result = graph.invoke(initial_state)

    # Display result
    console.print()
    if result.get("result"):
        console.print(Panel(Markdown(result["result"]), title="Result", border_style="green"))
    else:
        # Fall back to last AI message
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                console.print(Panel(Markdown(str(msg.content)), title="Result", border_style="green"))
                break

    # Show metadata
    meta_parts = []
    if result.get("task_type"):
        meta_parts.append(f"Type: {result['task_type']}")
    if result.get("project"):
        meta_parts.append(f"Project: {result['project']}")
    if result.get("tools_used"):
        meta_parts.append(f"Tools: {', '.join(result['tools_used'])}")
    if result.get("plan_steps"):
        done = sum(1 for s in result["plan_steps"] if s.get("status") == "completed")
        meta_parts.append(f"Steps: {done}/{len(result['plan_steps'])}")
    if meta_parts:
        console.print(f"[dim]{' | '.join(meta_parts)}[/dim]")


def interactive_mode(model: str | None = None):
    """Run the agent in interactive REPL mode."""
    console.print(
        Panel(
            "[bold]cairn[/bold] — Personal AI Agent\n"
            "Type your task or question. Commands: [bold]quit[/bold], [bold]exit[/bold], [bold]/q[/bold]",
            border_style="blue",
        )
    )

    while True:
        try:
            task = console.input("\n[bold green]> [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "/q"):
            console.print("Goodbye!")
            break

        try:
            run_task(task, model=model)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="cairn — Personal AI Agent with SCMS",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task or question for the agent",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive REPL mode",
    )
    parser.add_argument(
        "--model",
        choices=["local", "cloud"],
        default=None,
        help="Model to use: 'local' (Ollama) or 'cloud' (Anthropic Claude)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Tool approval commands
    parser.add_argument(
        "--pending-tools",
        action="store_true",
        help="List tools awaiting approval",
    )
    parser.add_argument(
        "--review-tool",
        metavar="ID",
        help="Show details of a pending tool",
    )
    parser.add_argument(
        "--approve-tool",
        metavar="ID",
        help="Approve a pending tool",
    )
    parser.add_argument(
        "--reject-tool",
        metavar="ID",
        help="Reject a pending tool",
    )

    # Task queue & daemon commands
    parser.add_argument(
        "--queue",
        metavar="TASK",
        help="Add a task to the queue",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="Priority for queued task (1=highest, 10=lowest, default: 5)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project name for queued task",
    )
    parser.add_argument(
        "--recurring",
        type=str,
        help="Cron expression for recurring tasks (e.g., '0 9 * * MON')",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show task queue status",
    )
    parser.add_argument(
        "--completed",
        action="store_true",
        help="Show recently completed tasks",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Start persistent daemon mode",
    )

    # Code execution
    parser.add_argument(
        "--allow-subprocess",
        action="store_true",
        help="Allow code execution via subprocess when Docker is unavailable (use with caution)",
    )

    # Digest commands
    parser.add_argument(
        "--digest",
        nargs="?",
        const="daily",
        choices=["daily", "weekly"],
        help="Run the research digest pipeline (default: daily)",
    )
    parser.add_argument(
        "--review-digest",
        action="store_true",
        help="Review pending digest items for approval/storage",
    )
    parser.add_argument(
        "--digest-status",
        action="store_true",
        help="Show digest pipeline status and recent runs",
    )
    parser.add_argument(
        "--digest-eval",
        action="store_true",
        help="Run evaluation on digest approval/rejection history",
    )
    parser.add_argument(
        "--compile-digest",
        action="store_true",
        help="Compile approved digest items into deep-dive and briefing documents",
    )
    parser.add_argument(
        "--compile-since",
        metavar="YYYY-MM-DD",
        help="Date to look back from for --compile-digest (default: last 24 hours)",
    )
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="Also generate audio when compiling digest (use with --compile-digest)",
    )
    parser.add_argument(
        "--audio-digest",
        action="store_true",
        help="Generate audio from the most recent briefing digest",
    )
    parser.add_argument(
        "--audio-from",
        metavar="PATH",
        help="Generate audio from a specific briefing markdown file",
    )

    args = parser.parse_args()

    if args.debug:
        settings.log_level = "DEBUG"

    if args.allow_subprocess:
        settings.allow_subprocess = True

    setup_logging()
    logger = logging.getLogger(__name__)

    # Warn if subprocess fallback is enabled via env var (sticky setting)
    if settings.allow_subprocess and not args.allow_subprocess:
        logger.warning("Subprocess fallback enabled via ALLOW_SUBPROCESS environment variable")

    # Handle tool approval commands
    if args.pending_tools:
        _handle_pending_tools()
        return
    if args.review_tool:
        _handle_review_tool(args.review_tool)
        return
    if args.approve_tool:
        _handle_approve_tool(args.approve_tool)
        return
    if args.reject_tool:
        _handle_reject_tool(args.reject_tool)
        return

    # Handle queue commands
    if args.queue:
        _handle_queue(args.queue, args.priority, args.project, args.recurring)
        return
    if args.status:
        _handle_status()
        return
    if args.completed:
        _handle_completed()
        return
    if args.daemon:
        check_services()
        _handle_daemon()
        return

    # Handle digest commands
    if args.digest:
        check_services()
        _handle_digest(args.digest)
        return
    if args.review_digest:
        _handle_review_digest()
        return
    if args.digest_status:
        _handle_digest_status()
        return
    if args.digest_eval:
        _handle_digest_eval()
        return
    if args.compile_digest:
        check_services()
        briefing_path = _handle_compile_digest(args.compile_since)
        if args.with_audio and briefing_path:
            _handle_audio_digest(briefing_path)
        return
    if args.audio_digest or args.audio_from:
        check_services()
        _handle_audio_digest(args.audio_from)
        return

    check_services()

    if args.interactive:
        interactive_mode(model=args.model)
    elif args.task:
        try:
            run_task(args.task, model=args.model)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def _handle_pending_tools():
    from scms.client import SCMSClient
    client = SCMSClient()
    tools = client.list_pending_tools()
    if not tools:
        console.print("[dim]No pending tools.[/dim]")
        return
    for t in tools:
        config = t.get("config", {})
        test_results = config.get("test_results", [])
        passed = sum(1 for r in test_results if r.get("passed"))
        console.print(Panel(
            f"[bold]{t['name']}[/bold]\n"
            f"Description: {t.get('description', 'N/A')}\n"
            f"Type: {t['tool_type']}\n"
            f"Tests: {passed}/{len(test_results)} passed\n"
            f"Created: {t.get('created_at', 'N/A')}\n\n"
            f"[dim]Approve: python main.py --approve-tool {t['id']}[/dim]\n"
            f"[dim]Review:  python main.py --review-tool {t['id']}[/dim]",
            title=f"Pending: {t['id'][:8]}...",
            border_style="yellow",
        ))


def _handle_review_tool(tool_id: str):
    from scms.client import SCMSClient
    from pathlib import Path
    client = SCMSClient()
    t = client.get_tool(tool_id)
    if not t:
        console.print(f"[red]Tool not found: {tool_id}[/red]")
        return

    # Show metadata
    console.print(Panel(
        f"Name: {t['name']}\n"
        f"Description: {t.get('description', 'N/A')}\n"
        f"Type: {t['tool_type']}\n"
        f"Status: {t.get('approval_status', 'unknown')}\n"
        f"Created: {t.get('created_at', 'N/A')}",
        title="Tool Details",
        border_style="blue",
    ))

    # Show source code
    config = t.get("config", {})
    source_file = config.get("source_file")
    if source_file and Path(source_file).exists():
        code = Path(source_file).read_text()
        console.print(Panel(code, title="Source Code", border_style="green"))
    elif t.get("source_code"):
        console.print(Panel(t["source_code"], title="Source Code", border_style="green"))

    # Show test results
    test_results = config.get("test_results", [])
    if test_results:
        for i, tr in enumerate(test_results):
            status = "[green]PASSED[/green]" if tr.get("passed") else "[red]FAILED[/red]"
            console.print(f"Test {i+1}: {status}")
            if tr.get("output"):
                console.print(f"  Output: {tr['output'][:200]}")


def _handle_approve_tool(tool_id: str):
    from scms.client import SCMSClient
    client = SCMSClient()
    result = client.approve_tool(tool_id)
    if result:
        console.print(f"[green]Approved:[/green] {result['name']}")
        console.print("[dim]Tool will be available on next agent run.[/dim]")
    else:
        console.print(f"[red]Failed to approve tool: {tool_id}[/red]")


def _handle_reject_tool(tool_id: str):
    from scms.client import SCMSClient
    reason = console.input("[yellow]Rejection reason:[/yellow] ")
    client = SCMSClient()
    result = client.reject_tool(tool_id, reason=reason)
    if result:
        console.print(f"[red]Rejected:[/red] {result['name']}")
    else:
        console.print(f"[red]Failed to reject tool: {tool_id}[/red]")


def _handle_queue(task: str, priority: int, project: str | None, recurring: str | None):
    from scms.client import SCMSClient
    client = SCMSClient()
    result = client.enqueue_task(task, priority=priority, project=project, recurring=recurring)
    console.print(f"[green]Queued:[/green] {task[:60]}")
    console.print(f"  ID: {result['id']}")
    console.print(f"  Priority: {priority}")
    if project:
        console.print(f"  Project: {project}")
    if recurring:
        console.print(f"  Recurring: {recurring}")


def _handle_status():
    from scms.client import SCMSClient
    from rich.table import Table

    client = SCMSClient()
    counts = client.get_queue_status()

    # Summary
    console.print(Panel(
        f"Pending: {counts.get('pending', 0)} | "
        f"Running: {counts.get('running', 0)} | "
        f"Completed: {counts.get('completed', 0)} | "
        f"Failed: {counts.get('failed', 0)}",
        title="Queue Status",
        border_style="blue",
    ))

    # Pending tasks table
    pending = client.get_pending_tasks(limit=10)
    if pending:
        table = Table(title="Pending Tasks")
        table.add_column("ID", style="dim", width=8)
        table.add_column("Pri", width=3)
        table.add_column("Task", width=50)
        table.add_column("Project", width=15)
        for t in pending:
            table.add_row(
                t["id"][:8],
                str(t["priority"]),
                t["task"][:50],
                t.get("project", "") or "",
            )
        console.print(table)

    # Budget
    try:
        spend = client.get_daily_spend()
        budget = settings.daily_budget_usd
        console.print(f"[dim]Today's spend: ${spend:.4f} / ${budget:.2f}[/dim]")
    except Exception:
        pass


def _handle_completed():
    from scms.client import SCMSClient
    from rich.table import Table

    client = SCMSClient()
    completed = client.get_completed_tasks(limit=10)

    if not completed:
        console.print("[dim]No completed tasks.[/dim]")
        return

    table = Table(title="Recently Completed Tasks")
    table.add_column("ID", style="dim", width=8)
    table.add_column("Task", width=40)
    table.add_column("Model", width=15)
    table.add_column("Cost", width=8)
    table.add_column("Result", width=40)
    for t in completed:
        table.add_row(
            t["id"][:8],
            t["task"][:40],
            t.get("model_used", "?"),
            f"${t.get('cost_usd', 0) or 0:.4f}",
            (t.get("result", "") or "")[:40],
        )
    console.print(table)


def _handle_daemon():
    from agent.daemon import run_daemon
    run_daemon()


def _handle_digest(frequency: str):
    from agent.digest import run_digest

    console.print(f"[bold green]Running {frequency} digest...[/bold green]")
    with console.status("[bold green]Fetching sources and summarizing...", spinner="dots"):
        result = run_digest(frequency=frequency)

    if not result["digest_path"]:
        console.print(f"[dim]No {frequency} sources to process.[/dim]")
        return

    console.print(Panel(
        f"Sources processed: {result['sources_processed']}\n"
        f"Items found: {result['items_found']}\n"
        f"Items queued for review: {result['items_queued']}\n"
        f"Digest saved to: {result['digest_path']}",
        title="Digest Complete",
        border_style="green",
    ))
    if result["errors"]:
        for err in result["errors"]:
            console.print(f"[yellow]Warning:[/yellow] {err}")


def _handle_review_digest():
    from scms.client import SCMSClient

    client = SCMSClient()
    pending = client.get_pending_tasks(limit=50)
    review_items = [t for t in pending if t.get("project") == "_digest_review"]

    if not review_items:
        console.print("[dim]No digest items pending review.[/dim]")
        return

    console.print(f"[bold]{len(review_items)} items pending review[/bold]\n")

    approved = 0
    rejected = 0
    for item in review_items:
        console.print(Panel(item["task"], border_style="yellow"))
        try:
            action = console.input(
                "[bold]([green]a[/green])pprove / ([red]r[/red])eject / "
                "([yellow]s[/yellow])kip / ([dim]q[/dim])uit: [/bold]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if action == "a":
            # Extract title and summary from the task text
            task_text = item["task"]
            # Store the full digest item text as a learning memory
            client.store_memory(
                content=task_text,
                memory_type="learning",
                source="digest",
                tags=["digest-approved"],
            )
            client.update_task_status(
                item["id"], "completed", result="quality=1"
            )
            console.print("[green]Approved and stored in SCMS[/green]\n")
            approved += 1
        elif action == "r":
            client.update_task_status(
                item["id"], "cancelled", result="quality=0"
            )
            console.print("[red]Rejected[/red]\n")
            rejected += 1
        elif action == "q":
            break
        else:
            console.print("[yellow]Skipped[/yellow]\n")

    console.print(f"\n[dim]Approved: {approved} | Rejected: {rejected}[/dim]")


def _handle_digest_status():
    from scms.client import SCMSClient
    from rich.table import Table

    client = SCMSClient()

    # Recent digest runs — look for completed tasks that mention "digest"
    completed = client.get_completed_tasks(limit=30)
    digest_runs = [
        t for t in completed
        if "digest" in t.get("task", "").lower()
        and t.get("project") != "_digest_review"  # exclude review items
    ]

    if digest_runs:
        table = Table(title="Recent Digest Runs")
        table.add_column("Date", width=12)
        table.add_column("Result", width=60)
        for run in digest_runs[:5]:
            run_date = (run.get("completed_at") or "")[:10]
            result = (run.get("result") or "")[:60]
            table.add_row(run_date, result)
        console.print(table)
    else:
        console.print("[dim]No digest runs found.[/dim]")

    # Pending review count
    pending = client.get_pending_tasks(limit=50)
    review_count = sum(1 for t in pending if t.get("project") == "_digest_review")
    console.print(f"\n[bold]Pending review items:[/bold] {review_count}")


def _handle_digest_eval():
    from agent.evaluation import run_evaluation

    console.print("[bold green]Running digest evaluation...[/bold green]")
    result = run_evaluation()

    if result["total_reviewed"] == 0:
        console.print("[dim]No reviewed digest items found. "
                      "Approve or reject some items first.[/dim]")
        return

    console.print(Panel(
        f"Total reviewed: {result['total_reviewed']}\n"
        f"Approval rate: {result['approval_rate']:.1%}\n"
        f"Report saved to: {result['report_path']}",
        title="Digest Evaluation",
        border_style="green",
    ))

    if result["suggestions"]:
        console.print("\n[bold]Recommendations:[/bold]")
        for s in result["suggestions"]:
            console.print(f"  [dim]-[/dim] {s}")


def _handle_compile_digest(since: str | None) -> str | None:
    """Compile digest and return the briefing path (for --with-audio chaining)."""
    from agent.compile_digest import run_compile_digest

    since_label = since or "last 24 hours"
    console.print(f"[bold green]Compiling digest ({since_label})...[/bold green]")
    with console.status("[bold green]Fetching articles and summarizing...", spinner="dots"):
        result = run_compile_digest(since=since)

    if result["articles_compiled"] == 0:
        console.print("[dim]No approved digest items found for this period.[/dim]")
        return None

    console.print(Panel(
        f"Articles compiled: {result['articles_compiled']}\n"
        f"With full content: {result['articles_with_full_content']}\n"
        f"Deep dive: {result['deep_path']}\n"
        f"Briefing: {result['briefing_path']}",
        title="Daily Digest Complete",
        border_style="green",
    ))
    if result["errors"]:
        for err in result["errors"]:
            console.print(f"  [yellow]Warning:[/yellow] {err}")

    return result.get("briefing_path") or None


def _handle_audio_digest(briefing_path: str | None):
    from agent.audio_digest import run_audio_digest

    console.print("[bold green]Generating audio digest...[/bold green]")
    with console.status("[bold green]Synthesizing speech...", spinner="dots"):
        result = run_audio_digest(briefing_path=briefing_path)

    if not result["audio_path"]:
        console.print("[dim]No audio generated.[/dim]")
        if result["errors"]:
            for err in result["errors"]:
                console.print(f"  [yellow]Warning:[/yellow] {err}")
        return

    duration_min = result["duration_seconds"] / 60
    console.print(Panel(
        f"Audio saved to: {result['audio_path']}\n"
        f"Duration: {duration_min:.1f} minutes\n"
        f"Provider: {result['provider_used']}\n"
        f"Characters: {result['char_count']:,}",
        title="Audio Digest Complete",
        border_style="green",
    ))
    if result["errors"]:
        for err in result["errors"]:
            console.print(f"  [yellow]Warning:[/yellow] {err}")


if __name__ == "__main__":
    main()
