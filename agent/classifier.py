"""Task classifier and project detector for the cairn agent.

Uses deterministic keyword matching (no LLM calls) for reliability.
"""

import difflib
import logging
import re

from agent.tools import CATEGORY_TOOLS, get_tool_names_for_category

logger = logging.getLogger(__name__)

# Keywords that indicate each task category
CATEGORY_KEYWORDS = {
    "research": [
        "research", "paper", "arxiv", "find", "search", "learn about",
        "look up", "what is", "how does", "explore", "investigate",
        "discover", "github",
    ],
    "knowledge_management": [
        "save", "store", "remember", "recall", "what do i know",
        "what have i", "memory", "memories", "knowledge",
        "create project", "new project", "add project",
        "update project", "edit project", "modify project",
        "archive project", "delete project", "remove project",
    ],
    "productivity": [
        "write file", "create file", "note", "summarize", "organize",
        "draft", "document", "write a note", "take note",
    ],
    "technical": [
        "run", "execute", "code", "debug", "test", "script",
        "calculate", "python", "function", "program",
    ],
    "metatool": [
        "create tool", "make tool", "build tool", "new tool",
        "generate tool", "write a tool", "custom tool",
        "test tool", "pending tool", "approve tool",
    ],
}


def classify_task(task: str) -> tuple[str, list[str]]:
    """Classify a task and return (task_type, available_tool_names).

    Uses keyword matching across categories. If keywords from 2+ categories
    are detected, returns "multi" with all tools available.
    """
    task_lower = task.lower()
    matched_categories = set()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in task_lower:
                matched_categories.add(category)
                break

    if len(matched_categories) >= 2:
        task_type = "multi"
    elif len(matched_categories) == 1:
        task_type = matched_categories.pop()
    else:
        # Default to research if no keywords match
        task_type = "research"

    tool_names = get_tool_names_for_category(task_type)
    logger.info("Classified task as '%s' -> %d tools available", task_type, len(tool_names))
    return task_type, tool_names


def detect_project(task: str, projects: list[dict]) -> str:
    """Detect a project name from the task text.

    Uses exact substring matching first, then fuzzy matching as fallback.
    Returns empty string if no project detected.
    """
    if not projects:
        return ""

    task_lower = task.lower()
    project_names = [p["name"] for p in projects]

    # Exact substring match (case-insensitive)
    for name in project_names:
        if name.lower() in task_lower:
            logger.info("Detected project (exact match): %s", name)
            return name

    # Fuzzy match — extract words from task and compare
    task_words = re.findall(r"\b\w+\b", task)
    for word in task_words:
        matches = difflib.get_close_matches(
            word.lower(),
            [n.lower() for n in project_names],
            n=1,
            cutoff=0.7,
        )
        if matches:
            # Find the original-case name
            for name in project_names:
                if name.lower() == matches[0]:
                    logger.info("Detected project (fuzzy match): %s", name)
                    return name

    return ""
