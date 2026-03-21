"""Integration test: metatool-generated tools load correctly as LangChain tools."""

import importlib.util
import textwrap
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def bare_tool_file(tmp_path):
    """Write a plain Python function (no @tool decorator) to a temp file."""
    code = textwrap.dedent("""\
        def greet(name: str) -> str:
            \"\"\"Return a greeting for the given name.\"\"\"
            return f"Hello, {name}!"
    """)
    tool_path = tmp_path / "greet.py"
    tool_path.write_text(code)
    return tool_path


@pytest.fixture
def mock_approved_tool(bare_tool_file):
    """Return SCMS tool record pointing at the bare function file."""
    return {
        "id": "test-id-123",
        "name": "greet",
        "function_name": "greet",
        "tool_type": "metatool_generated",
        "approval_status": "approved",
        "config": {
            "source_file": str(bare_tool_file),
            "categories": ["productivity"],
        },
    }


def test_bare_function_gets_tool_decorator(mock_approved_tool):
    """A bare function loaded by load_approved_custom_tools() should have
    .name, .description, and be invocable via .invoke()."""
    from agent.tools import TOOL_REGISTRY, ALL_TOOLS, load_approved_custom_tools

    # Clean up any prior state from module-level auto-load
    TOOL_REGISTRY.pop("greet", None)
    initial_count = len(ALL_TOOLS)

    with patch("scms.client.SCMSClient") as MockClient:
        MockClient.return_value.list_tools.return_value = [mock_approved_tool]
        load_approved_custom_tools()

    assert "greet" in TOOL_REGISTRY, "greet tool should be in registry"

    tool = TOOL_REGISTRY["greet"]["tool"]
    assert hasattr(tool, "name"), "tool must have .name attribute"
    assert hasattr(tool, "description"), "tool must have .description attribute"
    assert tool.name == "greet"
    assert "greeting" in tool.description.lower() or "name" in tool.description.lower()

    result = tool.invoke({"name": "World"})
    assert result == "Hello, World!"

    # Cleanup
    TOOL_REGISTRY.pop("greet", None)
    while len(ALL_TOOLS) > initial_count:
        ALL_TOOLS.pop()
