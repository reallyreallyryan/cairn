"""Python code executor — Docker sandbox with subprocess fallback."""

import ast
import logging
import os
import subprocess
import textwrap
from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 10000

# Modules blocked in subprocess fallback mode
BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "sys", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "ctypes", "importlib", "signal", "multiprocessing",
}


def _check_code_safety(code: str) -> str | None:
    """Check code for dangerous operations (used in subprocess fallback)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    return f"Blocked import: '{module}' is not allowed"
        if isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    return f"Blocked import: '{module}' is not allowed"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("exec", "eval"):
                return f"Blocked: {func.id}() is not allowed"

    return None


def _subprocess_fallback(code: str) -> str:
    """Execute code via subprocess when Docker is unavailable."""
    safety_error = _check_code_safety(code)
    if safety_error:
        return f"Code rejected: {safety_error}"

    tmpdir = Path("/tmp/cairn")
    tmpdir.mkdir(parents=True, exist_ok=True)

    safe_env = {
        k: v for k, v in os.environ.items()
        if not any(secret in k.upper() for secret in
                   ["KEY", "TOKEN", "SECRET", "PASSWORD", "SUPABASE"])
    }
    safe_env["HOME"] = str(tmpdir)
    safe_env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        result = subprocess.run(
            ["python3", "-c", textwrap.dedent(code)],
            capture_output=True, text=True,
            timeout=settings.code_execution_timeout,
            cwd=str(tmpdir), env=safe_env,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(output_parts) if output_parts else "[No output]"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n[Truncated]"
        return f"[subprocess fallback — Docker unavailable]\n{output}"
    except subprocess.TimeoutExpired:
        return f"Timed out after {settings.code_execution_timeout}s"
    except Exception as e:
        return f"Error: {e}"


@tool
def code_executor(code: str) -> str:
    """Execute a Python code snippet in a Docker sandbox.

    The code runs in an isolated Docker container with no network access,
    limited memory (256MB), and a 60-second timeout. If Docker is unavailable,
    falls back to a restricted subprocess.

    Args:
        code: Python code to execute.
    """
    logger.info("Executing code (%d chars)", len(code))

    # Try Docker sandbox first
    try:
        from sandbox.manager import SandboxManager
        manager = SandboxManager()
        result = manager.execute_code(code)

        output_parts = []
        if result["stdout"]:
            output_parts.append(result["stdout"])
        if result["stderr"]:
            output_parts.append(f"[stderr]\n{result['stderr']}")
        if result["exit_code"] != 0 and not result["timed_out"]:
            output_parts.append(f"[exit code: {result['exit_code']}]")
        if result["timed_out"]:
            output_parts.append("[Timed out]")

        output = "\n".join(output_parts) if output_parts else "[No output]"

        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n[Truncated at {MAX_OUTPUT_CHARS} chars]"

        return output

    except ConnectionError as e:
        logger.warning("Docker unavailable, falling back to subprocess: %s", e)
        return _subprocess_fallback(code)
    except Exception as e:
        logger.warning("Sandbox error, falling back to subprocess: %s", e)
        return _subprocess_fallback(code)
