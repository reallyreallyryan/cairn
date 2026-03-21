"""REFLECT node — result evaluation, continuation decisions, and SCMS logging."""

import logging

from langchain_core.messages import SystemMessage

from agent.state import AgentState
from agent.utils import clean_output
from agent.model_router import route_and_get_llm
from config.settings import settings

logger = logging.getLogger(__name__)


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
    clean_result = clean_output(clean_result)

    # Fallback for brief responses — use best conversation content if reflect is too terse
    if len(clean_result) < 200 and not should_continue:
        best_content = clean_result
        for msg in state.get("messages", []):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and len(content) > len(best_content):
                msg_type = getattr(msg, "type", "")
                if msg_type not in ("tool", "system"):
                    best_content = clean_output(content)
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
