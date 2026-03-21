"""Metatool — the agent can create, test, and register new tools."""

import logging
import re
from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings

logger = logging.getLogger(__name__)

CUSTOM_TOOLS_DIR = Path("agent/tools/custom")


def _validate_tool_name(name: str) -> str | None:
    """Validate a tool name. Returns error message or None."""
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return f"Invalid tool name '{name}': must be snake_case (lowercase, underscores)"
    if len(name) < 3 or len(name) > 50:
        return f"Tool name must be 3-50 characters, got {len(name)}"

    # Check for conflicts with existing tools
    from agent.tools import TOOL_REGISTRY
    if name in TOOL_REGISTRY:
        return f"Tool '{name}' already exists in the registry"

    return None


@tool
def create_tool(
    name: str,
    description: str,
    code: str,
    test_code: str,
    categories: str = "technical",
) -> str:
    """Create a new tool by writing Python code, testing it in the Docker sandbox,
    and submitting it for human approval.

    Args:
        name: Tool name in snake_case (e.g., 'csv_analyzer').
        description: What the tool does (shown when selecting tools).
        code: Python source code implementing the tool function.
        test_code: Python code that tests the tool to verify it works.
        categories: Comma-separated categories (research, technical, productivity, etc.).
    """
    logger.info("Creating tool: %s", name)

    # Validate name
    name_error = _validate_tool_name(name)
    if name_error:
        return f"Error: {name_error}"

    # Test the code in the sandbox
    logger.info("Testing tool code in sandbox...")
    try:
        from sandbox.manager import SandboxManager
        manager = SandboxManager()

        # Combine tool code + test code for execution
        combined = f"{code}\n\n# === Test code ===\n{test_code}"
        result = manager.execute_code(combined)

        if result["timed_out"]:
            return f"Test timed out. Please optimize your code and try again."

        if result["exit_code"] != 0:
            error_output = result["stderr"] or result["stdout"] or "Unknown error"
            return (
                f"Tests failed (exit code {result['exit_code']}):\n"
                f"{error_output}\n\n"
                f"Fix the code and try again."
            )

        test_output = result["stdout"]
        logger.info("Tests passed: %s", test_output[:200])

    except ConnectionError:
        return "Error: Docker is not available. Cannot test tool in sandbox."
    except Exception as e:
        return f"Error testing tool: {e}"

    # Tests passed — save the tool
    CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tool_file = CUSTOM_TOOLS_DIR / f"{name}.py"
    tool_file.write_text(code, encoding="utf-8")
    logger.info("Tool code saved to: %s", tool_file)

    # Register in SCMS as pending approval
    try:
        from scms.client import SCMSClient
        client = SCMSClient()

        cat_list = [c.strip() for c in categories.split(",") if c.strip()]

        record = client.register_tool(
            name=name,
            description=description,
            tool_type="metatool_generated",
            function_name=name,
            config={
                "source_file": str(tool_file),
                "categories": cat_list,
                "test_code": test_code,
                "test_results": [{"passed": True, "output": test_output[:500]}],
            },
            approval_status="pending",
        )

        tool_id = record.get("id", "unknown")
        return (
            f"Tool '{name}' created and tests passed!\n"
            f"ID: {tool_id}\n"
            f"Status: pending human approval\n"
            f"Test output: {test_output[:200]}\n\n"
            f"To approve: python main.py --approve-tool {tool_id}"
        )

    except Exception as e:
        return f"Tool tested successfully but registration failed: {e}"


@tool
def test_tool(name: str, test_code: str) -> str:
    """Test an existing custom tool with new test code in the Docker sandbox.

    Args:
        name: Name of the tool to test.
        test_code: Python code that tests the tool.
    """
    logger.info("Testing tool: %s", name)

    # Load tool source
    tool_file = CUSTOM_TOOLS_DIR / f"{name}.py"
    if not tool_file.exists():
        return f"Tool '{name}' not found in {CUSTOM_TOOLS_DIR}"

    tool_code = tool_file.read_text(encoding="utf-8")

    try:
        from sandbox.manager import SandboxManager
        manager = SandboxManager()

        combined = f"{tool_code}\n\n# === Test code ===\n{test_code}"
        result = manager.execute_code(combined)

        output_parts = []
        if result["stdout"]:
            output_parts.append(f"Output:\n{result['stdout']}")
        if result["stderr"]:
            output_parts.append(f"Errors:\n{result['stderr']}")

        status = "PASSED" if result["exit_code"] == 0 else "FAILED"
        return f"Test {status} (exit code: {result['exit_code']})\n" + "\n".join(output_parts)

    except ConnectionError:
        return "Error: Docker is not available. Cannot test tool in sandbox."
    except Exception as e:
        return f"Error testing tool: {e}"


@tool
def list_pending_tools() -> str:
    """List all custom tools that are pending human approval.

    Shows tool name, description, creation date, and test results.
    """
    logger.info("Listing pending tools")
    try:
        from scms.client import SCMSClient
        client = SCMSClient()
        tools = client.list_pending_tools()

        if not tools:
            return "No tools pending approval."

        lines = [f"Found {len(tools)} pending tool(s):\n"]
        for t in tools:
            config = t.get("config", {})
            test_results = config.get("test_results", [])
            passed = sum(1 for r in test_results if r.get("passed"))

            lines.append(f"**{t['name']}** (ID: {t['id'][:8]}...)")
            lines.append(f"  Description: {t.get('description', 'N/A')}")
            lines.append(f"  Created: {t.get('created_at', 'N/A')}")
            lines.append(f"  Tests: {passed}/{len(test_results)} passed")
            lines.append(f"  Approve: python main.py --approve-tool {t['id']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing pending tools: {e}"
