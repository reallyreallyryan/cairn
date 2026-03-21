"""CLASSIFY node — task classification and context loading."""

import logging

from agent.state import AgentState
from agent.classifier import classify_task, detect_project

logger = logging.getLogger(__name__)


def classify_node(state: AgentState) -> dict:
    """Classify the task type, detect project, and load relevant context."""
    task = state["task"]
    logger.info("=== CLASSIFY node ===")
    logger.info("Task: %s", task)

    # 1. Classify task type and get available tools
    task_type, available_tools = classify_task(task)

    # 2. Detect project from task text
    project = ""
    try:
        from scms.client import SCMSClient
        client = SCMSClient()
        projects = client.list_projects()
        project = detect_project(task, projects)
    except Exception as e:
        logger.warning("Could not detect project: %s", e)
        projects = []

    # 3. Load context from SCMS
    context_parts = []
    try:
        from scms.client import SCMSClient
        client = SCMSClient()

        # Load project list
        if projects:
            project_lines = ["Known projects:"]
            for p in projects:
                meta = p.get("metadata", {})
                goals = meta.get("goals", [])
                stack = meta.get("stack", [])
                project_lines.append(f"- {p['name']} ({p['status']}): {p['description']}")
                if stack:
                    project_lines.append(f"  Stack: {', '.join(stack)}")
                if goals:
                    project_lines.append(f"  Goals: {', '.join(goals)}")
            context_parts.append("\n".join(project_lines))

        # If project detected, load its full context
        if project:
            proj_ctx = client.get_project_context(project)
            if proj_ctx.get("recent_memories"):
                mem_lines = [f"\nRecent memories for {project}:"]
                for m in proj_ctx["recent_memories"][:10]:
                    mem_lines.append(f"- ({m['memory_type']}) {m['content'][:150]}")
                context_parts.append("\n".join(mem_lines))
            if proj_ctx.get("recent_decisions"):
                dec_lines = [f"\nRecent decisions for {project}:"]
                for d in proj_ctx["recent_decisions"][:5]:
                    dec_lines.append(f"- {d['decision'][:150]}")
                context_parts.append("\n".join(dec_lines))

        # Also search for task-relevant memories
        results = client.search_memories(task, limit=5)
        if results:
            memory_lines = ["\nRelevant memories:"]
            for r in results:
                memory_lines.append(f"- ({r['memory_type']}) {r['content'][:150]}")
            context_parts.append("\n".join(memory_lines))

    except Exception as e:
        logger.warning("Could not load SCMS context: %s", e)

    context = "\n\n".join(context_parts) if context_parts else "No context available."

    # 4. Log classification decision
    decisions = list(state.get("decisions", []))
    decisions.append({
        "node": "classify",
        "decision": f"Classified as '{task_type}', project='{project}'",
        "reasoning": f"Keyword matching against task text",
        "alternatives": list(set(["research", "knowledge_management", "productivity", "technical"]) - {task_type}),
    })

    logger.info("Classification: type=%s, project=%s, tools=%d", task_type, project or "none", len(available_tools))

    return {
        "task_type": task_type,
        "project": project,
        "available_tools": available_tools,
        "context": context,
        "decisions": decisions,
    }
