"""Node implementations for the Classify-Plan-Act-Reflect agent loop."""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent.state import AgentState
from agent.tools import ALL_TOOLS, TOOL_REGISTRY, get_tools_for_category
from agent.classifier import classify_task, detect_project
from agent.model_router import route_and_get_llm
from config.settings import settings

logger = logging.getLogger(__name__)


def _clean_output(text: str) -> str:
    """Strip narrated tool call XML and other artifacts from LLM output."""
    # Remove <function_calls>...</function_calls> blocks
    text = re.sub(r"<function_calls>.*?</function_calls>", "", text, flags=re.DOTALL)
    # Remove <function_result>...</function_result> blocks
    text = re.sub(r"<function_result>.*?</function_result>", "", text, flags=re.DOTALL)
    # Remove any remaining XML-ish tags (but keep markdown)
    text = re.sub(r"</?(?:tool_call|tool_result|search_results?|results?)>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_llm(model: str | None = None):
    """Create the appropriate LLM based on configuration."""
    use_cloud = (model == "cloud") or (
        model is None and settings.agent_model == "cloud"
    )

    if use_cloud:
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            logger.warning("No ANTHROPIC_API_KEY set, falling back to local Ollama model")
        else:
            logger.info("Using cloud model: claude-sonnet-4-20250514")
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                api_key=settings.anthropic_api_key,
            )

    if model == "local_light":
        logger.info("Using local-light model: %s", settings.ollama_model_light)
        return ChatOllama(
            model=settings.ollama_model_light,
            base_url=settings.ollama_base_url,
        )

    logger.info("Using local model: %s", settings.ollama_model)
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )


# ------------------------------------------------------------------
# CLASSIFY node
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# PLAN node
# ------------------------------------------------------------------


def _parse_plan_steps(plan_text: str) -> list[dict]:
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
    steps = _parse_plan_steps(plan_text)
    logger.info("Parsed %d plan steps", len(steps))

    return {
        "plan": plan_text,
        "plan_steps": steps,
        "current_step": 0,
        "step_results": {},
        "messages": [HumanMessage(content=task), response],
    }


# ------------------------------------------------------------------
# ACT node
# ------------------------------------------------------------------

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


def _extract_url(text: str) -> str:
    """Extract first URL from text."""
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else text


def _extract_path(text: str) -> str:
    """Extract a file path from text."""
    # Look for ~/... or /... or common file patterns
    match = re.search(r"[~/]\S+\.\w+", text)
    return match.group(0) if match else text


def _extract_code(text: str) -> str:
    """Extract code from text (between triple backticks or after 'code:')."""
    # Try triple backtick blocks
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try after "code:" or "execute:"
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


# ------------------------------------------------------------------
# REFLECT node
# ------------------------------------------------------------------


def reflect_node(state: AgentState) -> dict:
    """Evaluate results, advance steps, and decide whether to continue."""
    iteration = state.get("iteration", 0) + 1
    current_step = state.get("current_step", 0)
    plan_steps = state.get("plan_steps", [])
    max_iters = settings.max_iterations

    logger.info("=== REFLECT node (iteration %d/%d, step %d/%d) ===",
                iteration, max_iters, current_step + 1, len(plan_steps) or 1)

    # Mark current step as completed
    if plan_steps and current_step < len(plan_steps):
        plan_steps = [dict(s) for s in plan_steps]  # copy
        plan_steps[current_step]["status"] = "completed"

    # Advance to next step
    next_step = current_step + 1
    has_more_steps = next_step < len(plan_steps)

    # Check for redundant results — detect when we're not making progress
    step_results = state.get("step_results", {})
    tools_used = state.get("tools_used", [])
    redundant = False

    if current_step >= 1:
        prev_result = str(step_results.get(current_step - 1, ""))
        curr_result = str(step_results.get(current_step, ""))

        # Check 1: Same result text (exact or prefix match)
        same_result = (
            prev_result and curr_result and (
                prev_result == curr_result
                or (len(prev_result) > 50 and prev_result[:200] == curr_result[:200])
            )
        )

        # Check 2: Same tool called repeatedly (e.g., scms_search called 3+ times)
        recent_tools = tools_used[-3:] if len(tools_used) >= 3 else []
        same_tool_spam = len(recent_tools) >= 3 and len(set(recent_tools)) == 1

        if same_result or same_tool_spam:
            redundant = True
            reason = "same results" if same_result else f"same tool ({recent_tools[0]}) called 3+ times"
            logger.info("Redundant: %s — skipping remaining steps", reason)
            has_more_steps = False
            for i in range(next_step, len(plan_steps)):
                plan_steps[i]["status"] = "completed"

    llm, _tier, _cost = route_and_get_llm(state.get("task", ""), state.get("task_type", ""), override=state.get("model_override"))

    system_prompt = (
        "You are cairn. Write the FINAL answer the user will see.\n\n"
        f"Original task: {state.get('task', '')}\n\n"
        "CRITICAL RULES:\n"
        "- Include ALL specific details: names, descriptions, statuses, goals, etc.\n"
        "- Do NOT summarize — list every item found. The user wants the actual data.\n"
        "- Use bullet points or numbered lists for multiple items.\n"
        "- Start with COMPLETE: if done, or CONTINUE: if more work is genuinely needed.\n"
        "- After the prefix, write the FULL detailed response."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages)
    result_text = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("Reflect: %s", result_text[:200])

    # Determine continuation — respect LLM decision and redundancy detection
    should_continue = False
    if redundant:
        logger.info("Stopping due to redundant results")
    elif result_text.upper().startswith("COMPLETE:"):
        logger.info("LLM says task is complete")
    elif has_more_steps and iteration < max_iters:
        should_continue = True
        logger.info("More steps remain, continuing to step %d", next_step + 1)
    elif result_text.upper().startswith("CONTINUE:") and iteration < max_iters:
        should_continue = True
    else:
        if iteration >= max_iters:
            logger.info("Max iterations reached, finishing")
        else:
            logger.info("Task marked complete")

    # Clean result text — strip prefix and narrated XML
    clean_result = result_text
    for prefix in ("COMPLETE:", "CONTINUE:"):
        if clean_result.upper().startswith(prefix):
            clean_result = clean_result[len(prefix):].strip()
    clean_result = _clean_output(clean_result)

    # Fallback for brief responses — use best conversation content if reflect is too terse
    if len(clean_result) < 200 and not should_continue:
        best_content = clean_result
        for msg in state.get("messages", []):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and len(content) > len(best_content):
                msg_type = getattr(msg, "type", "")
                if msg_type not in ("tool", "system"):
                    best_content = _clean_output(content)
        if len(best_content) > len(clean_result):
            logger.info("Reflect too brief, using best conversation content")
            clean_result = best_content

    # Log decisions to SCMS when done
    decisions = list(state.get("decisions", []))
    if not should_continue:
        try:
            from scms.client import SCMSClient
            client = SCMSClient()
            decision_summary = f"Task: {state.get('task', '')}\n"
            decision_summary += f"Type: {state.get('task_type', 'unknown')}\n"
            decision_summary += f"Tools used: {', '.join(state.get('tools_used', []))}\n"
            decision_summary += f"Steps: {len(plan_steps)}, Iterations: {iteration}"

            client.log_decision(
                decision=decision_summary,
                reasoning=f"Completed in {iteration} iterations with {len(state.get('tools_used', []))} tool calls",
                alternatives=[d.get("decision", "") for d in decisions[:5]],
                project_name=state.get("project") or None,
            )
            logger.info("Decision logged to SCMS")
        except Exception as e:
            logger.warning("Failed to log decision: %s", e)

    return {
        "messages": [response],
        "result": clean_result,
        "should_continue": should_continue,
        "iteration": iteration,
        "current_step": next_step,
        "plan_steps": plan_steps,
        "decisions": decisions,
    }
