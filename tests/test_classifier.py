"""Tests for the keyword-based task classifier and project detector."""

import pytest

from agent.classifier import classify_task, detect_project


# ------------------------------------------------------------------
# classify_task tests
# ------------------------------------------------------------------


class TestClassifyTask:
    """Tests for classify_task()."""

    def test_research_keyword_search(self):
        task_type, tools = classify_task("search for LangGraph tutorials")
        assert task_type == "research"
        assert "web_search" in tools

    def test_research_keyword_arxiv(self):
        task_type, _ = classify_task("find arxiv papers on transformers")
        assert task_type == "research"

    def test_research_keyword_what_is(self):
        task_type, _ = classify_task("what is retrieval augmented generation?")
        assert task_type == "research"

    def test_knowledge_management_keyword_save(self):
        task_type, tools = classify_task("save this for later")
        assert task_type == "knowledge_management"
        assert "scms_store" in tools

    def test_knowledge_management_keyword_recall(self):
        task_type, _ = classify_task("recall what I learned yesterday")
        assert task_type == "knowledge_management"

    def test_productivity_keyword_write_file(self):
        task_type, tools = classify_task("write file report.md")
        assert task_type == "productivity"
        assert "file_writer" in tools

    def test_productivity_keyword_note(self):
        task_type, _ = classify_task("take a note about the meeting")
        assert task_type == "productivity"

    def test_technical_keyword_run(self):
        task_type, tools = classify_task("run this Python script")
        assert task_type == "technical"
        assert "code_executor" in tools

    def test_technical_keyword_debug(self):
        task_type, _ = classify_task("debug this code snippet")
        assert task_type == "technical"

    def test_metatool_keyword_create_tool(self):
        task_type, tools = classify_task("create tool for CSV parsing")
        assert task_type == "metatool"
        assert "create_tool" in tools

    def test_metatool_keyword_pending(self):
        task_type, _ = classify_task("show pending tool approvals")
        assert task_type == "metatool"

    def test_multi_two_categories(self):
        """Keywords from 2+ categories → 'multi' with all tools."""
        task_type, tools = classify_task("search for Python code and run it")
        assert task_type == "multi"
        # 'multi' gets all tools
        assert "web_search" in tools
        assert "code_executor" in tools

    def test_no_match_defaults_to_research(self):
        """No keyword match → defaults to 'research'."""
        task_type, tools = classify_task("hello there")
        assert task_type == "research"
        assert "web_search" in tools

    def test_empty_string_defaults_to_research(self):
        task_type, _ = classify_task("")
        assert task_type == "research"

    def test_returns_tool_names_not_objects(self):
        """Tool names should be strings, not tool objects."""
        _, tools = classify_task("search the web")
        assert all(isinstance(t, str) for t in tools)


# ------------------------------------------------------------------
# detect_project tests
# ------------------------------------------------------------------


class TestDetectProject:
    """Tests for detect_project()."""

    SAMPLE_PROJECTS = [
        {"name": "cairn"},
        {"name": "KelemsClaw"},
        {"name": "my-portfolio"},
    ]

    def test_exact_match(self):
        result = detect_project("work on the cairn agent", self.SAMPLE_PROJECTS)
        assert result == "cairn"

    def test_exact_match_case_insensitive(self):
        result = detect_project("Updates for CAIRN", self.SAMPLE_PROJECTS)
        assert result == "cairn"

    def test_fuzzy_match(self):
        result = detect_project("the kelemscla project", self.SAMPLE_PROJECTS)
        # Fuzzy match should find KelemsClaw
        assert result == "KelemsClaw"

    def test_no_match(self):
        result = detect_project("unrelated task about weather", self.SAMPLE_PROJECTS)
        assert result == ""

    def test_empty_project_list(self):
        result = detect_project("work on cairn", [])
        assert result == ""

    def test_empty_task(self):
        result = detect_project("", self.SAMPLE_PROJECTS)
        assert result == ""

    def test_hyphenated_project_name(self):
        result = detect_project("update my-portfolio site", self.SAMPLE_PROJECTS)
        assert result == "my-portfolio"
