"""Agent state schema for the LangGraph agent."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class PlanStep(TypedDict):
    """A single step in a multi-step plan."""
    step: str
    tool_hint: str | None
    status: str  # "pending", "in_progress", "completed", "failed"


class AgentState(TypedDict):
    """State that flows through the Classify-Plan-Act-Reflect agent loop."""

    # Conversation messages (LangGraph manages appending)
    messages: Annotated[list, add_messages]

    # The user's original task/question
    task: str

    # Classification results (from classify node)
    task_type: str  # "research", "knowledge_management", "productivity", "technical", "multi"
    project: str  # detected project name (empty if none)
    available_tools: list[str]  # tool names filtered by task type

    # Context retrieved from SCMS
    context: str

    # Planning (supports multi-step)
    plan: str  # free-text plan summary (backward compat)
    plan_steps: list[PlanStep]  # structured steps
    current_step: int  # index into plan_steps
    step_results: dict[int, str]  # step index -> result

    # Tools invoked during this run
    tools_used: list[str]

    # Decision logging
    decisions: list[dict]  # accumulated decision entries for this run

    # Sandbox / metatool
    pending_tools: list[dict]  # tools created this session awaiting approval
    sandbox_logs: list[dict]  # execution logs from sandbox runs

    # Final result to return to the user
    result: str

    # Whether to continue the plan-act-reflect loop
    should_continue: bool

    # Model routing
    model_override: str  # "cloud" or "local" to bypass routing rules (daemon sets "cloud")
    model_used: str  # tier name used for this run
    cost_estimate: float  # accumulated estimated cost

    # Loop iteration counter (safety bound, max 10)
    iteration: int
