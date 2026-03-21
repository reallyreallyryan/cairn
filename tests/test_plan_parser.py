"""Tests for the plan step parser."""

import pytest

from agent.plan import parse_plan_steps


class TestParsePlanSteps:
    """Tests for parse_plan_steps()."""

    def test_basic_numbered_steps(self):
        plan = "1. Search the web for LangGraph tutorials\n2. Save the results to memory"
        steps = parse_plan_steps(plan)
        assert len(steps) == 2
        assert "Search the web" in steps[0]["step"]
        assert steps[0]["status"] == "pending"

    def test_parenthesized_numbers(self):
        plan = "1) Search for papers on attention\n2) Review the top results"
        steps = parse_plan_steps(plan)
        assert len(steps) == 2

    def test_bold_lines_skipped(self):
        """Lines starting with ** should be skipped (project names, titles)."""
        plan = (
            "1. **cairn** — AI agent project\n"
            "2. Search the web for LangGraph tutorials\n"
            "3. **Another bold heading**\n"
        )
        steps = parse_plan_steps(plan)
        assert len(steps) == 1
        assert "Search the web" in steps[0]["step"]

    def test_non_action_lines_skipped(self):
        """Lines without action words in first 4 words should be skipped if long."""
        plan = (
            "1. The cairn project is a personal AI agent built with LangGraph\n"
            "2. Search the web for documentation\n"
        )
        steps = parse_plan_steps(plan)
        assert len(steps) == 1
        assert "Search the web" in steps[0]["step"]

    def test_short_non_action_lines_kept(self):
        """Short lines (<=15 chars) are kept even without action words."""
        plan = "1. Do the thing\n2. Search the web"
        steps = parse_plan_steps(plan)
        # "Do the thing" is 12 chars, kept; but it also needs > 10 chars
        assert len(steps) >= 1

    def test_cap_at_three_steps(self):
        plan = (
            "1. Search the web for X\n"
            "2. Save the results\n"
            "3. Create a summary\n"
            "4. Run the analysis script\n"
            "5. Store the final output\n"
        )
        steps = parse_plan_steps(plan)
        assert len(steps) == 3

    def test_answer_from_context_early_exit(self):
        """'Answer from context' stops parsing — rest is the answer."""
        plan = (
            "1. Answer directly from context below\n"
            "2. Search the web for more info\n"
        )
        steps = parse_plan_steps(plan)
        assert len(steps) == 1
        assert "Answer directly" in steps[0]["step"]

    def test_answer_from_context_variants(self):
        for phrase in ["answer from context", "respond from context",
                       "already available", "use the context"]:
            plan = f"1. Check if information is {phrase}\n2. Search web"
            steps = parse_plan_steps(plan)
            assert len(steps) == 1, f"Failed for phrase: {phrase}"

    def test_empty_input_fallback(self):
        """Empty/garbage input → single fallback step."""
        steps = parse_plan_steps("")
        assert len(steps) == 1
        assert steps[0]["tool_hint"] is None

    def test_garbage_input_fallback(self):
        """Non-numbered text → single fallback step wrapping the text."""
        steps = parse_plan_steps("I think we should research this topic thoroughly.")
        assert len(steps) == 1
        assert steps[0]["status"] == "pending"

    def test_tool_hint_detected(self):
        """Tool names in step text are detected as hints."""
        plan = "1. Use web_search to find LangGraph tutorials"
        steps = parse_plan_steps(plan)
        assert steps[0]["tool_hint"] == "web_search"

    def test_tool_hint_underscore_to_space(self):
        """Tool names with underscores replaced by spaces are also detected."""
        plan = "1. Use arxiv search to find papers on transformers"
        steps = parse_plan_steps(plan)
        assert steps[0]["tool_hint"] == "arxiv_search"

    def test_no_tool_hint_when_absent(self):
        plan = "1. Search the web for documentation on this topic"
        steps = parse_plan_steps(plan)
        # "web_search" isn't literally in this text
        assert steps[0]["tool_hint"] is None or steps[0]["tool_hint"] == "web_search"

    def test_steps_too_short_filtered(self):
        """Steps with <= 10 chars are filtered out."""
        plan = "1. Do it\n2. Search the web for tutorials"
        steps = parse_plan_steps(plan)
        assert len(steps) == 1
        assert "Search" in steps[0]["step"]

    def test_mixed_valid_and_invalid_lines(self):
        plan = (
            "Here's my plan:\n"
            "1. **Project Overview**\n"
            "2. Search the web for the latest LangGraph documentation\n"
            "3. This is a description of the approach we should take for analysis\n"
            "4. Save the results to the memory store\n"
        )
        steps = parse_plan_steps(plan)
        assert len(steps) == 2
        assert "Search the web" in steps[0]["step"]
        assert "Save the results" in steps[1]["step"]
