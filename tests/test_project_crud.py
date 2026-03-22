"""Tests for project CRUD — SCMSClient methods, MCP tools, and classifier routing."""

import pytest
from unittest.mock import patch, MagicMock


# ------------------------------------------------------------------
# SCMSClient CRUD tests (mocked Supabase)
# ------------------------------------------------------------------


class TestSCMSClientProjectCRUD:
    """Tests for create_project, update_project, archive_project on SCMSClient."""

    @pytest.fixture
    def mock_supabase(self):
        with patch("scms.client.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def client(self, mock_supabase):
        with patch("scms.embeddings.get_embeddings_batch", return_value=[[0.0] * 1536]):
            from scms.client import SCMSClient
            return SCMSClient()

    def test_create_project(self, client, mock_supabase):
        table = mock_supabase.table.return_value
        table.insert.return_value.execute.return_value.data = [
            {"id": "proj-1", "name": "ridgeline", "status": "active",
             "description": "audit tool", "metadata": {"stack": ["flowise"]}}
        ]

        result = client.create_project(
            name="ridgeline",
            description="audit tool",
            metadata={"stack": ["flowise"]},
        )
        assert result["name"] == "ridgeline"
        assert result["status"] == "active"
        mock_supabase.table.assert_called_with("projects")
        table.insert.assert_called_once()

    def test_update_project(self, client, mock_supabase):
        table = mock_supabase.table.return_value
        table.update.return_value.eq.return_value.execute.return_value.data = [
            {"id": "proj-1", "name": "ridgeline", "status": "paused",
             "description": "updated desc", "metadata": {}}
        ]

        result = client.update_project(name="ridgeline", description="updated desc", status="paused")
        assert result["status"] == "paused"
        assert result["description"] == "updated desc"

    def test_update_project_no_fields(self, client):
        result = client.update_project(name="ridgeline")
        assert result == {"error": "No fields to update"}

    def test_update_project_not_found(self, client, mock_supabase):
        table = mock_supabase.table.return_value
        table.update.return_value.eq.return_value.execute.return_value.data = []

        result = client.update_project(name="nonexistent", status="active")
        assert "error" in result
        assert "not found" in result["error"]

    def test_archive_project(self, client, mock_supabase):
        table = mock_supabase.table.return_value
        table.update.return_value.eq.return_value.execute.return_value.data = [
            {"id": "proj-1", "name": "ridgeline", "status": "archived",
             "description": "audit tool", "metadata": {}}
        ]

        result = client.archive_project(name="ridgeline")
        assert result["status"] == "archived"


# ------------------------------------------------------------------
# MCP tool tests
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_mcp_settings():
    with patch("mcp_server.config.mcp_settings") as mock_settings:
        mock_settings.mcp_base_url = ""
        mock_settings.mcp_host = "0.0.0.0"
        mock_settings.mcp_port = 8000
        mock_settings.daily_budget_usd = 5.0
        yield mock_settings


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def patch_get_client(mock_client):
    with patch("mcp_server.server._get_client", return_value=mock_client):
        yield mock_client


class TestMcpCreateProject:

    def test_create_success(self, patch_get_client):
        from mcp_server.server import create_project

        patch_get_client.create_project.return_value = {
            "id": "proj-1", "name": "ridgeline", "status": "active",
        }

        result = create_project("ridgeline", description="audit tool")
        assert "ridgeline" in result
        assert "proj-1" in result

    def test_create_duplicate(self, patch_get_client):
        from mcp_server.server import create_project

        patch_get_client.create_project.side_effect = Exception(
            'duplicate key value violates unique constraint "projects_name_key"'
        )

        result = create_project("ridgeline")
        assert "already exists" in result

    def test_create_error(self, patch_get_client):
        from mcp_server.server import create_project

        patch_get_client.create_project.side_effect = Exception("DB timeout")
        result = create_project("test")
        assert "Error" in result


class TestMcpUpdateProject:

    def test_update_success(self, patch_get_client):
        from mcp_server.server import update_project

        patch_get_client.update_project.return_value = {
            "name": "ridgeline", "status": "paused",
        }

        result = update_project("ridgeline", status="paused")
        assert "paused" in result

    def test_update_not_found(self, patch_get_client):
        from mcp_server.server import update_project

        patch_get_client.update_project.return_value = {
            "error": "Project 'nonexistent' not found"
        }

        result = update_project("nonexistent", status="active")
        assert "not found" in result


class TestMcpArchiveProject:

    def test_archive_success(self, patch_get_client):
        from mcp_server.server import archive_project

        patch_get_client.archive_project.return_value = {
            "name": "ridgeline", "status": "archived",
        }

        result = archive_project("ridgeline")
        assert "Archived" in result

    def test_archive_error(self, patch_get_client):
        from mcp_server.server import archive_project

        patch_get_client.archive_project.side_effect = Exception("DB error")
        result = archive_project("ridgeline")
        assert "Error" in result


# ------------------------------------------------------------------
# Classifier routing tests
# ------------------------------------------------------------------


class TestClassifierProjectRouting:

    def test_create_project_routes_to_knowledge_management(self):
        from agent.classifier import classify_task

        task_type, tools = classify_task("create new project ridgeline")
        assert task_type == "knowledge_management"
        assert "create_project" in tools

    def test_update_project_routes_correctly(self):
        from agent.classifier import classify_task

        task_type, tools = classify_task("update project ridgeline status")
        assert task_type == "knowledge_management"
        assert "update_project" in tools

    def test_archive_project_routes_correctly(self):
        from agent.classifier import classify_task

        task_type, tools = classify_task("archive project old-thing")
        assert task_type == "knowledge_management"
        assert "archive_project" in tools

    def test_new_project_has_crud_tools_available(self):
        from agent.classifier import classify_task

        _, tools = classify_task("add project for competitor analysis")
        assert "create_project" in tools
        assert "update_project" in tools
        assert "archive_project" in tools
