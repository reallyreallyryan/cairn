"""PLAN node — task planning and step parsing."""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.tools import TOOL_REGISTRY
from agent.model_router import route_and_get_llm

logger = logging.getLogger(__name__)


def parse_plan_steps(plan_text: str) -> list[dict]:
    """Parse a numbered list from LLM output into plan steps.

    Only parses lines that look like action instructions, not data/descriptions.
    Stops after the first "answer from context" type step (the rest is the answer itself).
    Cap at 3 steps max.
    """
    steps = []
    # Action verbs that indicate a real plan step
    action_words = {
        "search", "find", "look", "research", "answer", "check", "get",
        "retrieve", "store", "save", "read", "write", "create", "run",
        "execute", "summarize", "analyze", "review", "list", "fetch",
        "use", "call", "query", "open", "take",
    }

    for line in plan_text.split("\n"):
        line = line.strip()
        match = re.match(r"^(\d+)[\.\)]\s+(.+)", line)
        if match:
            step_text = match.group(2).strip()

            # Skip lines that are data/descriptions, not actions:
            # - Start with ** (bold project names, titles)
            # - Start with a capital letter followed by description (e.g. "Research - Exploring...")
            if step_text.startswith("**"):
                continue

            # Check if this line contains an action word
            first_words = step_text.lower().split()[:4]
            has_action = any(w in action_words for w in first_words)
            if not has_action and len(step_text) > 15:
                # Probably a description, not a step
                continue

            if len(step_text) > 10:
                tool_hint = None
                for tool_name in TOOL_REGISTRY:
                    if tool_name in step_text.lower() or tool_name.replace("_", " ") in step_text.lower():
                        tool_hint = tool_name
                        break
                steps.append({"step": step_text, "tool_hint": tool_hint, "status": "pending"})

                # If this step says "answer from context", it's the only step needed
                if any(phrase in step_text.lower() for phrase in
                       ["answer directly", "answer from context", "respond from context",
                        "already available", "use the context"]):
                    break

    # Cap at 3 steps
    steps = steps[:3]

    # If no steps parsed, wrap entire plan as single step
    if not steps:
        steps = [{"step": plan_text[:500], "tool_hint": None, "status": "pending"}]

    return steps


def plan_node(state: AgentState) -> dict:
    """Create a plan for the task using available tools and context."""
    task = state["task"]
    iteration = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    plan_steps = state.get("plan_steps", [])
    logger.info("=== PLAN node (iteration %d) ===", iteration)

    # On subsequent iterations, don't regenerate plan — just continue
    if iteration > 0 and plan_steps:
        logger.info("Continuing existing plan at step %d/%d", current_step + 1, len(plan_steps))
        return {}

    # Build tool description from available tools
    available = state.get("available_tools", list(TOOL_REGISTRY.keys()))
    tool_descs = []
    for name in available:
        if name in TOOL_REGISTRY:
            tool = TOOL_REGISTRY[name]["tool"]
            desc = tool.description.split("\n")[0] if tool.description else name
            tool_descs.append(f"- {name}: {desc}")

    context = state.get("context", "")
    project = state.get("project", "")

    system_prompt = (
        "You are cairn, a personal AI agent with a Shared Context Memory Store.\n\n"
        f"{'Project focus: ' + project + chr(10) if project else ''}"
        "Available tools: " + ", ".join(available) + "\n\n"
        "The context below was already retrieved from your memory store.\n\n"
        "Create a SHORT numbered plan (1-3 steps max). Each step must describe an ACTION to take.\n"
        "GOOD steps: '1. Search the web for LangGraph tutorials' or '1. Answer from the context below'\n"
        "BAD steps: '1. web_search' or '1. scms_search - Search memory store'\n\n"
        "If the context already answers the question, just write: '1. Answer directly from context'\n\n"
        f"=== CONTEXT ===\n{context}\n=== END CONTEXT ==="
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task),
    ]

    llm, _tier, _cost = route_and_get_llm(state.get("task", ""), state.get("task_type", ""), override=state.get("model_override"))
    response = llm.invoke(messages)
    plan_text = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("Plan: %s", plan_text[:200])

    # Parse into structured steps
    steps = parse_plan_steps(plan_text)
    logger.info("Parsed %d plan steps", len(steps))

    return {
        "plan": plan_text,
        "plan_steps": steps,
        "current_step": 0,
        "step_results": {},
        "messages": [HumanMessage(content=task), response],
    }
