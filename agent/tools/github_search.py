"""GitHub repository search tool."""

import logging

import httpx
from langchain_core.tools import tool

from config.settings import settings

logger = logging.getLogger(__name__)


@tool
def github_search(query: str, language: str = "") -> str:
    """Search GitHub for repositories.

    Args:
        query: The search query for GitHub repositories.
        language: Optional language filter (e.g., 'python', 'typescript').
    """
    logger.info("GitHub search: query='%s', language='%s'", query, language)
    try:
        search_query = query
        if language:
            search_query += f" language:{language}"

        headers = {"Accept": "application/vnd.github.v3+json"}
        if settings.github_token:
            headers["Authorization"] = f"token {settings.github_token}"

        response = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": search_query, "sort": "stars", "per_page": 5},
            headers=headers,
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        if not items:
            return f"No GitHub repositories found for: {query}"

        lines = [f"Found {len(items)} repositories on GitHub:\n"]
        for i, repo in enumerate(items, 1):
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language", "Unknown")
            desc = repo.get("description", "No description")
            if desc and len(desc) > 150:
                desc = desc[:150] + "..."

            lines.append(f"{i}. **{repo['full_name']}** ({stars:,} stars)")
            lines.append(f"   Language: {lang}")
            lines.append(f"   {desc}")
            lines.append(f"   URL: {repo['html_url']}")
            lines.append("")

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return "GitHub API rate limit reached. Set GITHUB_TOKEN in .env for higher limits."
        return f"GitHub API error: {e.response.status_code}"
    except Exception as e:
        logger.error("GitHub search failed: %s", e)
        return f"Error searching GitHub: {e}"
