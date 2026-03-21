"""ACT node — tool execution with structured calls and keyword fallback."""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.state import AgentState
from agent.tools import TOOL_REGISTRY
from agent.model_router import route_and_get_llm
from config.settings import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fallback dispatch helpers
# ------------------------------------------------------------------


def _extract_url(text: str) -> str:
    """Extract first URL from text."""
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else text


def _extract_path(text: str) -> str:
    """Extract a file path from text."""
    match = re.search(r"[~/]\S+\.\w+", text)
    return match.group(0) if match else text


def _extract_code(text: str) -> str:
    """Extract code from text (between triple backticks or after 'code:')."""
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    for prefix in ["code:", "execute:", "run:"]:
        if prefix in text.lower():
            idx = text.lower().index(prefix)
            return text[idx + len(prefix):].strip()
    return text


def _strip_prefixes(task: str) -> str:
    """Strip common command prefixes from task text."""
    task_lower = task.lower()
    for prefix in [
        "save a note about ", "save a note: ", "store a note about ",
        "store a note: ", "store that ", "save that ", "remember that ",
        "remember: ", "add a note about ", "add a note: ",
    ]:
        if task_lower.startswith(prefix):
            return task[len(prefix):]
    return task


# Dispatch table for keyword-based tool execution fallback
FALLBACK_DISPATCH = [
    {
        "keywords": ["arxiv", "paper", "academic paper"],
        "tool": "arxiv_search",
        "arg_builder": lambda task, _ctx: {"query": task},
    },
    {
        "keywords": ["github", "repo", "repository"],
        "tool": "github_search",
        "arg_builder": lambda task, _ctx: {"query": task},
    },
    {
        "keywords": ["http://", "https://"],
        "tool": "url_reader",
        "arg_builder": lambda task, _ctx: {"url": _extract_url(task)},
    },
    {
        "keywords": ["take note", "create note", "write note", "note about", "jot down"],
        "tool": "note_taker",
        "arg_builder": lambda task, _ctx: {
            "title": task[:80],
            "content": task,
        },
    },
    {
        "keywords": ["read file", "open file", "show file", "view file"],
        "tool": "file_reader",
        "arg_builder": lambda task, _ctx: {"path": _extract_path(task)},
    },
    {
        "keywords": ["run code", "execute code", "calculate"],
        "tool": "code_executor",
        "arg_builder": lambda task, _ctx: {"code": _extract_code(task)},
    },
    {
        "keywords": ["save", "store", "remember"],
        "tool": "scms_store",
        "arg_builder": lambda task, ctx: {
            "content": _strip_prefixes(task),
            "memory_type": "concept",
            "project": ctx.get("project", ""),
            "source": "user",
        },
    },
    {
        "keywords": ["research", "search", "look up", "find out", "learn about",
                     "find recent", "latest", "updates on", "news about", "what is"],
        "tool": "web_search",
        "arg_builder": lambda task, _ctx: {
            "query": task.replace("Research ", "").replace("and save what you learn", "").strip()
        },
    },
]


# ------------------------------------------------------------------
# ACT node
# ------------------------------------------------------------------


def act_node(state: AgentState) -> dict:
    """Execute the current plan step by invoking tools."""
    current_step = state.get("current_step", 0)
    plan_steps = state.get("plan_steps", [])
    step_text = plan_steps[current_step]["step"] if current_step < len(plan_steps) else state.get("task", "")

    logger.info("=== ACT node (step %d/%d) ===", current_step + 1, len(plan_steps) or 1)
    logger.info("Step: %s", step_text[:100])

    # Get available tools for this task type
    available_tool_names = state.get("available_tools", list(TOOL_REGISTRY.keys()))
    available_tools = [TOOL_REGISTRY[n]["tool"] for n in available_tool_names if n in TOOL_REGISTRY]
    tool_map = {t.name: t for t in available_tools}

    llm, _tier, _cost = route_and_get_llm(state.get("task", ""), state.get("task_type", ""), override=state.get("model_override"))
    llm_with_tools = llm.bind_tools(available_tools)

    task = state.get("task", "")
    plan = state.get("plan", "")
    context = state.get("context", "")
    project = state.get("project", "")

    # Build tool descriptions
    tool_desc_lines = []
    for name in available_tool_names:
        if name in TOOL_REGISTRY:
            t = TOOL_REGISTRY[name]["tool"]
            desc = t.description.split("\n")[0] if t.description else name
            tool_desc_lines.append(f"- {name}: {desc}")

    system_prompt = (
        "You are cairn, an AI agent. Execute the current step by calling the appropriate tools.\n\n"
        "Available tools:\n" + "\n".join(tool_desc_lines) + "\n\n"
        "You MUST call tools using the function calling interface. Do NOT describe tool calls in text."
    )

    user_prompt = (
        f"Task: {task}\n"
        f"Current step: {step_text}\n\n"
        f"Context:\n{context[:2000]}\n\n"
        "Now execute this step by calling the appropriate tool(s)."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm_with_tools.invoke(messages)
    tools_used = list(state.get("tools_used", []))
    decisions = list(state.get("decisions", []))
    step_results = dict(state.get("step_results", {}))

    # Execute structured tool calls if the LLM produced them
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info("Tool calls: %s", [tc["name"] for tc in response.tool_calls])

        result_messages = [response]
        step_output_parts = []

        for tc in response.tool_calls[:settings.max_tool_calls_per_step]:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tools_used.append(tool_name)

            decisions.append({
                "node": "act",
                "decision": f"Called {tool_name}",
                "reasoning": f"LLM structured tool call for step: {step_text[:80]}",
                "alternatives": [n for n in available_tool_names if n != tool_name],
            })

            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    result_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                    step_output_parts.append(str(result))
                    logger.info("Tool result: %s", str(result)[:200])
                except Exception as e:
                    logger.error("Tool %s failed: %s", tool_name, e)
                    result_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))
                    step_output_parts.append(f"Error: {e}")
            else:
                result_messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tc["id"]))

        step_results[current_step] = "\n".join(step_output_parts)
        return {
            "messages": result_messages,
            "tools_used": tools_used,
            "decisions": decisions,
            "step_results": step_results,
        }

    # No structured tool calls — use keyword-based fallback dispatch
    logger.info("No structured tool calls, using keyword dispatch")
    response_text = response.content if isinstance(response.content, str) else str(response.content)

    # Check the step text AND the task text for keywords
    search_text = f"{step_text} {task}".lower()
    fallback_ctx = {"project": project}

    for entry in FALLBACK_DISPATCH:
        if any(kw in search_text for kw in entry["keywords"]):
            tool_name = entry["tool"]
            if tool_name not in tool_map:
                continue
            try:
                args = entry["arg_builder"](task, fallback_ctx)
                logger.info("Fallback dispatch: %s(%s)", tool_name, {k: str(v)[:50] for k, v in args.items()})
                result = tool_map[tool_name].invoke(args)
                tools_used.append(tool_name)
                response_text = str(result)

                decisions.append({
                    "node": "act",
                    "decision": f"Fallback: called {tool_name}",
                    "reasoning": f"Keyword match in: {search_text[:80]}",
                    "alternatives": [],
                })

                # If task asks to save/store, follow up with scms_store
                search_tools = {"web_search", "arxiv_search", "github_search", "url_reader"}
                if tool_name in search_tools and any(w in search_text for w in ("save", "store", "remember")):
                    store_content = f"Research ({tool_name}): {task}\n\n{result}"[:2000]
                    if "scms_store" in tool_map:
                        tool_map["scms_store"].invoke({
                            "content": store_content,
                            "memory_type": "learning",
                            "source": tool_name,
                            "project": project,
                        })
                        tools_used.append("scms_store")
                        response_text += "\n\n[Results saved to SCMS]"

                break
            except Exception as e:
                logger.error("Fallback %s failed: %s", tool_name, e)
                response_text = f"Error: {e}"
                break

    step_results[current_step] = response_text
    return {
        "messages": [AIMessage(content=response_text)],
        "tools_used": tools_used,
        "decisions": decisions,
        "step_results": step_results,
    }
