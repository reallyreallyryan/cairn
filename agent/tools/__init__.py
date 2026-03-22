"""Tool registry with category metadata for the cairn agent."""

import importlib.util
import logging
from pathlib import Path

from agent.tools.scms_tools import scms_search, scms_store
from agent.tools.web_search import web_search
from agent.tools.url_reader import url_reader
from agent.tools.file_reader import file_reader
from agent.tools.file_writer import file_writer
from agent.tools.note_taker import note_taker
from agent.tools.arxiv_search import arxiv_search
from agent.tools.github_search import github_search
from agent.tools.code_executor import code_executor
from agent.tools.metatool import create_tool, test_tool, list_pending_tools
from agent.tools.project_tools import (
    archive_project,
    create_project,
    update_project,
)

logger = logging.getLogger(__name__)

# Category-aware tool registry
TOOL_REGISTRY = {
    "scms_search": {
        "tool": scms_search,
        "categories": ["knowledge_management", "research"],
        "keywords": ["search memory", "recall", "what do I know", "what have I"],
    },
    "scms_store": {
        "tool": scms_store,
        "categories": ["knowledge_management"],
        "keywords": ["save", "store", "remember"],
    },
    "web_search": {
        "tool": web_search,
        "categories": ["research", "technical"],
        "keywords": ["search", "look up", "find out", "google", "web"],
    },
    "url_reader": {
        "tool": url_reader,
        "categories": ["research", "productivity"],
        "keywords": ["read url", "fetch", "open link", "read page", "http://", "https://"],
    },
    "file_reader": {
        "tool": file_reader,
        "categories": ["productivity", "technical", "knowledge_management"],
        "keywords": ["read file", "open file", "cat", "show file", "view file"],
    },
    "file_writer": {
        "tool": file_writer,
        "categories": ["productivity", "technical", "knowledge_management"],
        "keywords": ["write file", "create file", "save to file", "write to"],
    },
    "note_taker": {
        "tool": note_taker,
        "categories": ["knowledge_management", "productivity"],
        "keywords": ["take note", "note about", "jot down", "write note", "create note"],
    },
    "arxiv_search": {
        "tool": arxiv_search,
        "categories": ["research"],
        "keywords": ["arxiv", "paper", "academic", "research paper", "papers on"],
    },
    "github_search": {
        "tool": github_search,
        "categories": ["research", "technical"],
        "keywords": ["github", "repo", "repository", "open source"],
    },
    "code_executor": {
        "tool": code_executor,
        "categories": ["technical"],
        "keywords": ["run code", "execute", "python", "script", "calculate"],
    },
    "create_tool": {
        "tool": create_tool,
        "categories": ["metatool", "technical"],
        "keywords": ["create tool", "build tool", "new tool", "make tool", "write a tool"],
    },
    "test_tool": {
        "tool": test_tool,
        "categories": ["metatool", "technical"],
        "keywords": ["test tool", "try tool", "verify tool"],
    },
    "list_pending_tools": {
        "tool": list_pending_tools,
        "categories": ["metatool"],
        "keywords": ["pending tools", "unapproved tools", "tools awaiting"],
    },
    "create_project": {
        "tool": create_project,
        "categories": ["knowledge_management", "productivity"],
        "keywords": ["create project", "new project", "start project", "add project"],
    },
    "update_project": {
        "tool": update_project,
        "categories": ["knowledge_management", "productivity"],
        "keywords": ["update project", "edit project", "change project", "modify project"],
    },
    "archive_project": {
        "tool": archive_project,
        "categories": ["knowledge_management", "productivity"],
        "keywords": ["archive project", "delete project", "remove project", "close project"],
    },
}

# Flat list of all tools (backward compat)
ALL_TOOLS = [entry["tool"] for entry in TOOL_REGISTRY.values()]

# Category -> tool mapping
CATEGORY_TOOLS = {
    "research": ["web_search", "arxiv_search", "github_search", "url_reader", "scms_search", "scms_store"],
    "knowledge_management": ["scms_search", "scms_store", "note_taker", "file_reader", "file_writer", "create_project", "update_project", "archive_project"],
    "productivity": ["file_writer", "file_reader", "note_taker", "url_reader", "scms_store", "create_project", "update_project", "archive_project"],
    "technical": ["code_executor", "file_reader", "file_writer", "web_search", "scms_store", "github_search"],
    "metatool": ["create_tool", "test_tool", "list_pending_tools", "code_executor", "scms_store"],
}


def get_tools_for_category(category: str) -> list:
    """Get tool objects for a given category. Returns all tools for 'multi' or unknown."""
    if category == "multi" or category not in CATEGORY_TOOLS:
        return ALL_TOOLS
    tool_names = CATEGORY_TOOLS[category]
    return [TOOL_REGISTRY[name]["tool"] for name in tool_names if name in TOOL_REGISTRY]


def get_tool_names_for_category(category: str) -> list[str]:
    """Get tool names for a given category."""
    if category == "multi" or category not in CATEGORY_TOOLS:
        return list(TOOL_REGISTRY.keys())
    return CATEGORY_TOOLS.get(category, list(TOOL_REGISTRY.keys()))


def load_approved_custom_tools():
    """Load approved metatool-generated tools from tools/custom/ directory."""
    try:
        from scms.client import SCMSClient
        client = SCMSClient()
        custom_tools = client.list_tools(enabled_only=True)

        loaded = 0
        for t in custom_tools:
            if t.get("tool_type") != "metatool_generated":
                continue
            if t.get("approval_status") != "approved":
                continue

            config = t.get("config", {})
            source_file = config.get("source_file")
            if not source_file or not Path(source_file).exists():
                continue

            try:
                func_name = t.get("function_name") or t["name"]
                spec = importlib.util.spec_from_file_location(t["name"], source_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                tool_func = getattr(module, func_name)

                # Wrap bare functions with @tool decorator if needed
                if not hasattr(tool_func, "description"):
                    from langchain_core.tools import tool as tool_decorator
                    tool_func = tool_decorator(tool_func)

                categories = config.get("categories", ["technical"])
                TOOL_REGISTRY[t["name"]] = {
                    "tool": tool_func,
                    "categories": categories,
                    "keywords": [t["name"].replace("_", " ")],
                }
                ALL_TOOLS.append(tool_func)
                loaded += 1
                logger.info("Loaded custom tool: %s", t["name"])
            except Exception as e:
                logger.warning("Failed to load custom tool %s: %s", t["name"], e)

        if loaded:
            logger.info("Loaded %d approved custom tool(s)", loaded)

    except Exception as e:
        logger.debug("Could not load custom tools: %s", e)


# Auto-load approved custom tools at import time
load_approved_custom_tools()
