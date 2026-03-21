"""Integration test — runs the full LangGraph classify→plan→act→reflect loop.

Uses mocked LLM and SCMS to verify graph wiring, state flow, and node imports
work end-to-end after the nodes.py split.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage


class FakeLLM:
    """Minimal fake LLM that returns canned responses without tool calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0

    def invoke(self, messages, **kwargs):
        text = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        msg = AIMessage(content=text)
        msg.tool_calls = []
        return msg

    def bind_tools(self, tools):
        """Return self — fake LLM ignores tool binding."""
        return self


@pytest.fixture
def mock_scms():
    """Mock all SCMS client interactions via the scms.client module."""
    instance = MagicMock()
    instance.list_projects.return_value = []
    instance.search_memories.return_value = []
    instance.log_decision.return_value = {"id": "test-decision"}

    with patch("scms.client.SCMSClient", return_value=instance):
        yield instance


@pytest.fixture
def mock_llm():
    """Fake LLM that returns a plan, then a text answer, then COMPLETE."""
    return FakeLLM([
        # plan_node response
        "1. Search the web for LangGraph tutorials",
        # act_node response (no tool calls, triggers fallback)
        "Here are some LangGraph tutorials I found: tutorial1, tutorial2",
        # reflect_node response
        "COMPLETE: Found LangGraph tutorials: tutorial1, tutorial2",
    ])


class TestGraphIntegration:
    """End-to-end graph tests with mocked LLM and SCMS."""

    @patch("agent.plan.route_and_get_llm")
    @patch("agent.act.route_and_get_llm")
    @patch("agent.reflect.route_and_get_llm")
    def test_full_loop_reaches_end(
        self, mock_reflect_route, mock_act_route, mock_plan_route, mock_scms
    ):
        """Graph should run classify→plan→act→reflect→END with mocked deps."""
        from agent.graph import build_graph

        fake = FakeLLM([
            "1. Answer directly from context below",
            "The answer is 42",
            "COMPLETE: The answer is 42",
        ])

        for mock_route in [mock_plan_route, mock_act_route, mock_reflect_route]:
            mock_route.return_value = (fake, "local", 0.0)

        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "task": "What is the answer?",
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
            "model_override": "",
            "model_used": "",
            "cost_estimate": 0.0,
            "iteration": 0,
        })

        assert result["result"], "Graph should produce a non-empty result"
        assert result["task_type"] == "research"  # "What is" → research
        assert result["should_continue"] is False

    @patch("agent.plan.route_and_get_llm")
    @patch("agent.act.route_and_get_llm")
    @patch("agent.reflect.route_and_get_llm")
    def test_graph_respects_max_iterations(
        self, mock_reflect_route, mock_act_route, mock_plan_route, mock_scms
    ):
        """Graph should stop at max_iterations even if LLM says CONTINUE."""
        from agent.graph import build_graph

        fake = FakeLLM([
            "1. Search for info\n2. Analyze results\n3. Save findings",
            "Still searching...",
            "CONTINUE: Need more data",
        ])

        for mock_route in [mock_plan_route, mock_act_route, mock_reflect_route]:
            mock_route.return_value = (fake, "local", 0.0)

        with patch("agent.reflect.settings") as mock_settings:
            mock_settings.max_iterations = 2

            graph = build_graph()
            result = graph.invoke({
                "messages": [],
                "task": "Deep research on quantum computing",
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
                "model_override": "",
                "model_used": "",
                "cost_estimate": 0.0,
                "iteration": 0,
            })

            # Should have stopped, not looped forever
            assert result["iteration"] <= 3

    @patch("agent.plan.route_and_get_llm")
    @patch("agent.act.route_and_get_llm")
    @patch("agent.reflect.route_and_get_llm")
    def test_classification_feeds_into_plan(
        self, mock_reflect_route, mock_act_route, mock_plan_route, mock_scms
    ):
        """Classifier output should be visible to subsequent nodes."""
        from agent.graph import build_graph

        fake = FakeLLM([
            "1. Answer directly from context",
            "Code executor is available",
            "COMPLETE: Done",
        ])

        for mock_route in [mock_plan_route, mock_act_route, mock_reflect_route]:
            mock_route.return_value = (fake, "local", 0.0)

        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "task": "run this Python script to calculate pi",
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
            "model_override": "",
            "model_used": "",
            "cost_estimate": 0.0,
            "iteration": 0,
        })

        # "run" and "Python" and "script" → technical
        assert result["task_type"] == "technical"
        assert "code_executor" in result["available_tools"]
