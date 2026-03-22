"""Web search tool using DuckDuckGo."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo for current information.

    Args:
        query: The search query to look up on the web.
    """
    logger.info("Web search: query='%s'", query)
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No web results found for: {query}"

        lines = [f"Web search results for '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['body']}")
            lines.append(f"   URL: {r['href']}")
            lines.append("")
        return "\n".join(lines)

    except Exception as e:
        logger.error("Web search failed: %s", e)
        return f"Error performing web search: {e}"
