"""LangGraph agent definition — Classify-Plan-Act-Reflect loop."""

import logging

from langgraph.graph import END, START, StateGraph

from agent.nodes import classify_node, plan_node, act_node, reflect_node
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _should_continue(state: AgentState) -> str:
    """Route from reflect node: continue the loop or finish."""
    if state.get("should_continue", False):
        return "continue"
    return "end"


def build_graph():
    """Build and compile the Classify-Plan-Act-Reflect agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("reflect", reflect_node)

    # Wire edges: START → classify → plan → act → reflect
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "reflect")

    # Conditional: reflect → plan (continue) or END
    graph.add_conditional_edges(
        "reflect",
        _should_continue,
        {"continue": "plan", "end": END},
    )

    return graph.compile()
