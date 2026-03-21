"""Tests for the MCP server tool functions."""

import pytest
from unittest.mock import patch, MagicMock


# We need to patch SCMSClient before importing the server module,
# because server.py imports SCMSClient at module level.
# We also need mcp_base_url empty to avoid OAuth init.

@pytest.fixture(autouse=True)
def mock_mcp_settings():
    """Ensure mcp_base_url is empty (no OAuth) for all tests."""
    with patch("mcp_server.config.mcp_settings") as mock_settings:
        mock_settings.mcp_base_url = ""
        mock_settings.mcp_host = "0.0.0.0"
        mock_settings.mcp_port = 8000
        mock_settings.daily_budget_usd = 5.0
        yield mock_settings


@pytest.fixture
def mock_client():
    """Create a mock SCMSClient."""
    client = MagicMock()
    return client


@pytest.fixture
def patch_get_client(mock_client):
    """Patch _get_client to return our mock."""
    with patch("mcp_server.server._get_client", return_value=mock_client):
        yield mock_client


# ------------------------------------------------------------------
# Test MCP tool functions directly (not via MCP protocol)
# ------------------------------------------------------------------


class TestMcpScmsSearch:

    def test_search_returns_formatted_results(self, patch_get_client):
        from mcp_server.server import scms_search

        patch_get_client.search_memories.return_value = [
            {"content": "LangGraph is a framework", "memory_type": "learning", "similarity": 0.85},
            {"content": "Docker sandbox for safety", "memory_type": "concept", "similarity": 0.72},
        ]

        result = scms_search("LangGraph")
        assert "LangGraph is a framework" in result
        assert "[0.85]" in result
        assert "(learning)" in result

    def test_search_no_results(self, patch_get_client):
        from mcp_server.server import scms_search

        patch_get_client.search_memories.return_value = []
        result = scms_search("nonexistent query")
        assert "No matching memories" in result

    def test_search_error_handled(self, patch_get_client):
        from mcp_server.server import scms_search

        patch_get_client.search_memories.side_effect = Exception("DB connection failed")
        result = scms_search("test")
        assert "Error" in result


class TestMcpListProjects:

    def test_list_projects_formatted(self, patch_get_client):
        from mcp_server.server import list_projects

        patch_get_client.list_projects.return_value = [
            {"name": "cairn", "status": "active"},
            {"name": "portfolio", "status": "planning"},
        ]

        result = list_projects()
        assert "cairn" in result
        assert "active" in result
        assert "portfolio" in result

    def test_list_projects_empty(self, patch_get_client):
        from mcp_server.server import list_projects

        patch_get_client.list_projects.return_value = []
        result = list_projects()
        assert "No projects" in result


class TestMcpAgentStatus:

    def test_agent_status_formatted(self, patch_get_client, mock_mcp_settings):
        from mcp_server.server import agent_status

        patch_get_client.get_queue_status.return_value = {
            "pending": 3, "running": 1, "completed": 15, "failed": 0
        }
        patch_get_client.get_daily_spend.return_value = 0.05

        result = agent_status()
        assert "pending: 3" in result
        assert "completed: 15" in result
        assert "$0.05" in result

    def test_agent_status_error(self, patch_get_client):
        from mcp_server.server import agent_status

        patch_get_client.get_queue_status.side_effect = Exception("timeout")
        result = agent_status()
        assert "Error" in result


class TestMcpQueueTask:

    def test_queue_task_success(self, patch_get_client):
        from mcp_server.server import queue_task

        patch_get_client.enqueue_task.return_value = {
            "id": "abc-123", "priority": 3, "status": "pending"
        }

        result = queue_task("Research MCP best practices", priority=3)
        assert "abc-123" in result
        assert "priority: 3" in result
