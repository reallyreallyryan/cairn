"""Digest Compiler — fetches full articles and produces summarized digest documents."""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import yaml

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompiledArticle:
    """A digest item with fetched content and generated summaries."""

    title: str
    source_name: str
    url: str
    relevance_score: float
    cross_encoder_score: float
    original_summary: str
    full_content: str | None
    deep_summary: str = ""
    brief_summary: str = ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load digest_sources.yaml."""
    path = Path(settings.digest_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Digest config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Content fetching
# ---------------------------------------------------------------------------

def fetch_article_content(url: str, max_chars: int = 15000) -> str | None:
    """Fetch full article text from a URL.

    Uses httpx + trafilatura (primary) + BeautifulSoup (fallback).
    Returns None on any failure (timeout, HTTP error, extraction failure).
    """
    if not url:
        return None

    logger.info("Fetching article: %s", url)
    try:
        response = httpx.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "cairn/1.0 (research agent)"},
        )
        response.raise_for_status()
        html = response.text
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    # Try trafilatura first
    try:
        import trafilatura
        content = trafilatura.extract(html, include_links=False, include_comments=False)
        if content:
            return content[:max_chars]
    except ImportError:
        pass

    # Fallback: BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)
        if content:
            return content[:max_chars]
    except ImportError:
        pass

    logger.warning("No parser could extract content from %s", url)
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_approved_items(client=None, since: str | None = None) -> list[dict]:
    """Load approved digest items from SCMS, parsed into structured dicts.

    Filters to status='completed' (approved) only. Parses each item's task
    text via parse_task_text() from the evaluation module.
    """
    from agent.evaluation import parse_task_text

    if client is None:
        from scms.client import SCMSClient
        client = SCMSClient()

    rows = client.get_reviewed_digest_items(since=since)

    items = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        parsed = parse_task_text(row.get("task", ""))
        if parsed is None:
            logger.debug("Skipping unparseable task: %s", row.get("id", "?"))
            continue
        items.append(parsed)

    logger.info("Loaded %d approved items", len(items))
    return items


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

DEEP_PROMPT = """\
Summarize this technical article for an AI engineer who builds autonomous agents, \
memory systems, and evaluation pipelines. Preserve all technical details, model names, \
architecture specifics, and benchmark results. Mention specific techniques and why they \
matter. 150-300 words.

Title: {title}

{content_section}\
"""

BRIEF_PROMPT = """\
Summarize this article for a smart reader who isn't deep in AI. Explain the same \
insights but translate jargon — when you use a technical term, add a brief parenthetical \
explanation. Focus on the "so what?" — why does this matter, what's the real-world \
implication. 100-200 words.

Title: {title}

{content_section}\
"""

SNIPPET_FALLBACK_NOTE = "\n\n_[Summary from snippet — full article not accessible]_"


def summarize_article(
    title: str,
    content: str | None,
    original_summary: str,
    style: str,
) -> str:
    """Summarize article content using the local LLM.

    Args:
        title: Article title.
        content: Full article text, or None if fetch failed.
        original_summary: Original 1-2 sentence snippet from digest extraction.
        style: "deep" or "brief".

    Returns:
        Summary text. Includes a snippet-fallback marker if content was unavailable.
    """
    from agent.utils import get_llm

    if content:
        content_section = f"Article content:\n{content[:12000]}"
    else:
        content_section = f"Brief description (full article not available):\n{original_summary}"

    template = DEEP_PROMPT if style == "deep" else BRIEF_PROMPT
    prompt = template.format(title=title, content_section=content_section)

    try:
        llm = get_llm("local")
        response = llm.invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
        # Strip <think> blocks from Qwen output
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if not content:
            text += SNIPPET_FALLBACK_NOTE
        return text
    except Exception as e:
        logger.warning("LLM summarization failed for '%s': %s", title, e)
        fallback = original_summary or title
        return f"{fallback}{SNIPPET_FALLBACK_NOTE}"


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_articles(items: list[dict]) -> list[CompiledArticle]:
    """Fetch content and generate both summaries for each approved item."""
    articles = []
    for item in items:
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        original_summary = item.get("summary", "")

        logger.info("Compiling: %s", title[:80])
        full_content = fetch_article_content(url)

        deep_summary = summarize_article(title, full_content, original_summary, "deep")
        brief_summary = summarize_article(title, full_content, original_summary, "brief")

        articles.append(CompiledArticle(
            title=title,
            source_name=item.get("source_name", ""),
            url=url,
            relevance_score=item.get("relevance_score", 0.0),
            cross_encoder_score=item.get("cross_encoder_score", 0.0),
            original_summary=original_summary,
            full_content=full_content,
            deep_summary=deep_summary,
            brief_summary=brief_summary,
        ))

    return articles


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------

def _count_sources(articles: list[CompiledArticle]) -> int:
    """Count unique source names."""
    return len({a.source_name for a in articles if a.source_name})


def build_digest(articles: list[CompiledArticle], style: str) -> str:
    """Build a markdown digest document.

    Args:
        articles: Compiled articles sorted by relevance (highest first).
        style: "deep" or "brief" — selects which summary and heading to use.
    """
    today = date.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    source_count = _count_sources(articles)

    style_label = "Deep dive" if style == "deep" else "Briefing"
    style_note = "" if style == "deep" else " Explained for a general audience."

    lines = [
        f"# cairn digest — {today}",
        f"## {style_label}",
        "",
        f"*{len(articles)} articles from {source_count} sources.{style_note} Generated {now}.*",
        "",
        "### Contents",
    ]

    # Table of contents
    for i, article in enumerate(articles, 1):
        source_tag = f" ({article.source_name})" if article.source_name else ""
        lines.append(f"{i}. {article.title}{source_tag}")

    lines.append("")
    lines.append("---")

    # Article sections
    for i, article in enumerate(articles, 1):
        lines.append("")
        if article.url:
            lines.append(f"### {i}. [{article.title}]({article.url})")
        else:
            lines.append(f"### {i}. {article.title}")

        meta_parts = []
        if article.source_name:
            meta_parts.append(f"**Source:** {article.source_name}")
        if article.relevance_score > 0:
            meta_parts.append(f"**Relevance:** {article.relevance_score:.2f}")
        if article.cross_encoder_score != 0.0:
            meta_parts.append(f"**Cross-encoder:** {article.cross_encoder_score:.2f}")
        if meta_parts:
            lines.append(" | ".join(meta_parts))

        summary = article.deep_summary if style == "deep" else article.brief_summary
        lines.append("")
        lines.append(summary)
        lines.append("")
        lines.append("---")

    # Footer
    full_count = sum(1 for a in articles if a.full_content is not None)
    snippet_count = len(articles) - full_count
    lines.append("")
    lines.append(
        f"*{len(articles)} articles compiled. "
        f"{full_count} with full content, {snippet_count} from snippets.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _get_digest_dir(config: dict) -> Path:
    """Get the digest output directory from config."""
    digest_dir = Path(
        config.get("settings", {}).get(
            "digest_notes_dir", "~/Documents/cairn/digests"
        )
    ).expanduser()
    digest_dir.mkdir(parents=True, exist_ok=True)
    return digest_dir


def save_compiled_digest(markdown: str, style: str, config: dict) -> Path:
    """Save a compiled digest document.

    Args:
        markdown: The markdown content.
        style: "deep" or "briefing" — used in the filename.
        config: Loaded YAML config (for digest_notes_dir).

    Returns:
        Path to the saved file.
    """
    digest_dir = _get_digest_dir(config)
    today = date.today().strftime("%Y-%m-%d")
    filename = f"{today}_digest_{style}.md"
    path = digest_dir / filename
    path.write_text(markdown, encoding="utf-8")
    logger.info("Saved %s digest to %s", style, path)
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_compile_digest(since: str | None = None) -> dict:
    """Compile approved digest items into deep-dive and briefing documents.

    Args:
        since: ISO date string (YYYY-MM-DD) to filter by. Default: last 24 hours.

    Returns:
        Dict with deep_path, briefing_path, articles_compiled,
        articles_with_full_content, and errors.
    """
    config = _load_config()

    # Default: 24 hours ago
    if since is None:
        since_dt = datetime.now(timezone.utc) - timedelta(hours=24)
        since = since_dt.isoformat()
    elif len(since) == 10:
        # Bare date like "2026-03-25" → start of that day UTC
        since = f"{since}T00:00:00+00:00"

    logger.info("Compiling digest (since %s)", since)

    items = load_approved_items(since=since)
    if not items:
        logger.info("No approved items found since %s", since)
        return {
            "deep_path": "",
            "briefing_path": "",
            "articles_compiled": 0,
            "articles_with_full_content": 0,
            "errors": [],
        }

    # Sort by relevance score descending
    items.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    articles = compile_articles(items)

    deep_md = build_digest(articles, "deep")
    briefing_md = build_digest(articles, "brief")

    deep_path = save_compiled_digest(deep_md, "deep", config)
    briefing_path = save_compiled_digest(briefing_md, "briefing", config)

    full_count = sum(1 for a in articles if a.full_content is not None)
    errors = [
        f"Could not fetch: {a.title}" for a in articles if a.full_content is None
    ]

    # Notify
    try:
        from agent.notifications import notify
        notify(
            "Digest Compiled",
            f"{len(articles)} articles compiled ({full_count} with full content)",
        )
    except Exception:
        pass

    return {
        "deep_path": str(deep_path),
        "briefing_path": str(briefing_path),
        "articles_compiled": len(articles),
        "articles_with_full_content": full_count,
        "errors": errors,
    }
