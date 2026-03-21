"""URL content reader tool."""

import logging

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def url_reader(url: str, max_chars: int = 5000) -> str:
    """Fetch and extract the main text content from a URL.

    Args:
        url: The URL to read content from.
        max_chars: Maximum characters to return (default 5000).
    """
    logger.info("Reading URL: %s", url)
    try:
        # Fetch the page
        response = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "cairn/1.0 (research agent)"},
        )
        response.raise_for_status()
        html = response.text

        # Try trafilatura first (better content extraction)
        try:
            import trafilatura
            content = trafilatura.extract(html, include_links=False, include_comments=False)
            if content:
                if len(content) > max_chars:
                    content = content[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"
                return content
        except ImportError:
            pass

        # Fallback: BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style tags
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            content = soup.get_text(separator="\n", strip=True)
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"
            return content
        except ImportError:
            pass

        # Last resort: raw text truncation
        content = html[:max_chars]
        return f"[Raw HTML - no parser available]\n{content}"

    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}"
    except Exception as e:
        logger.error("URL reader failed: %s", e)
        return f"Error reading URL: {e}"
