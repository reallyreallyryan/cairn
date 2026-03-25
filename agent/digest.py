"""Daily Research Digest Pipeline — fetches, summarizes, and queues research items."""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DigestItem:
    """A single item extracted from a source."""

    title: str
    summary: str
    url: str = ""
    source_name: str = ""
    relevance_score: float = 0.0
    embedding_score: float = 0.0
    cross_encoder_score: float = 0.0


@dataclass
class SourceResult:
    """Results from fetching and processing one source."""

    source_name: str
    items: list[DigestItem] = field(default_factory=list)
    filtered_items: list[DigestItem] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load digest_sources.yaml."""
    path = Path(settings.digest_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Digest config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_sources(frequency_filter: str | None = None) -> list[dict]:
    """Return source dicts, optionally filtered by frequency (daily/weekly)."""
    config = _load_config()
    sources = config.get("sources", [])
    if frequency_filter:
        sources = [s for s in sources if s.get("frequency") == frequency_filter]
    return sources


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_source(source: dict) -> SourceResult:
    """Fetch content from a single source via url_reader or arxiv_search.

    Returns a SourceResult with raw items or an error.
    """
    name = source.get("name", "Unknown")
    source_type = source.get("type", "blog")

    try:
        if source_type == "arxiv":
            return _fetch_arxiv(source)
        else:
            return _fetch_blog(source)
    except Exception as e:
        logger.warning("Failed to fetch source '%s': %s", name, e)
        return SourceResult(source_name=name, error=str(e))


def _fetch_blog(source: dict) -> SourceResult:
    """Fetch a blog/web source via url_reader."""
    from agent.tools.url_reader import url_reader

    name = source["name"]
    url = source["url"]
    max_items = source.get("max_items", 5)

    logger.info("Fetching blog source: %s (%s)", name, url)
    raw_text = url_reader.invoke({"url": url, "max_chars": 10000})

    if raw_text.startswith("Error") or raw_text.startswith("HTTP error"):
        return SourceResult(source_name=name, error=raw_text)

    # Use LLM to extract individual items from blog index page
    items = _extract_items_from_text(raw_text, source, max_items)

    # Apply keyword filter if configured
    filter_kw = source.get("filter_keywords")
    if filter_kw:
        filter_kw_lower = [kw.lower() for kw in filter_kw]
        items = [
            item for item in items
            if any(kw in item.title.lower() or kw in item.summary.lower() for kw in filter_kw_lower)
        ]

    return SourceResult(source_name=name, items=items)


def _fetch_arxiv(source: dict) -> SourceResult:
    """Fetch an arXiv source via arxiv_search."""
    from agent.tools.arxiv_search import arxiv_search

    name = source["name"]
    query = source.get("query", "")
    max_items = source.get("max_items", 5)

    logger.info("Fetching arXiv source: %s (query=%s)", name, query)
    raw_text = arxiv_search.invoke({"query": query, "max_results": max_items})

    if raw_text.startswith("Error") or raw_text.startswith("No arXiv"):
        return SourceResult(source_name=name, error=raw_text)

    items = _parse_arxiv_results(raw_text, name)
    return SourceResult(source_name=name, items=items)


def _parse_arxiv_results(raw_text: str, source_name: str) -> list[DigestItem]:
    """Parse the formatted arxiv_search output into DigestItem objects."""
    items: list[DigestItem] = []
    current_title = ""
    current_abstract = ""
    current_url = ""

    for line in raw_text.split("\n"):
        line = line.strip()
        if re.match(r"^\d+\.\s+\*\*", line):
            # Save previous item
            if current_title:
                items.append(DigestItem(
                    title=current_title,
                    summary=current_abstract,
                    url=current_url,
                    source_name=source_name,
                ))
            # Extract title (remove markdown bold and leading number)
            current_title = re.sub(r"^\d+\.\s+\*\*", "", line).rstrip("*").strip()
            current_abstract = ""
            current_url = ""
        elif line.startswith("Abstract:"):
            current_abstract = line.replace("Abstract:", "").strip()
        elif line.startswith("URL:"):
            current_url = line.replace("URL:", "").strip()

    # Don't forget last item
    if current_title:
        items.append(DigestItem(
            title=current_title,
            summary=current_abstract,
            url=current_url,
            source_name=source_name,
        ))

    return items


# ---------------------------------------------------------------------------
# LLM-based extraction and scoring
# ---------------------------------------------------------------------------

def _extract_items_from_text(
    raw_text: str, source: dict, max_items: int
) -> list[DigestItem]:
    """Use the local 32B model to extract article items from raw blog text.

    Sends the raw text to the LLM with a structured JSON prompt.
    Falls back to a single-item summary if JSON parsing fails.
    """
    from agent.utils import get_llm

    name = source.get("name", "Unknown")
    llm = get_llm("local")

    prompt = (
        "Extract the most recent blog posts or articles from the following webpage text. "
        f"Return at most {max_items} items.\n\n"
        "Return ONLY a JSON array — no markdown fences, no commentary. Each object must have:\n"
        '  {"title": "...", "summary": "1-2 sentence summary", "url": "full URL or empty string"}\n\n'
        f"Webpage text from {name}:\n\n{raw_text[:8000]}"
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences and /think blocks if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            parsed = [parsed]

        items = []
        for obj in parsed[:max_items]:
            if isinstance(obj, dict) and obj.get("title"):
                items.append(DigestItem(
                    title=obj["title"],
                    summary=obj.get("summary", ""),
                    url=obj.get("url", ""),
                    source_name=name,
                ))
        if items:
            return items
    except Exception as e:
        logger.warning("LLM item extraction failed for '%s': %s", name, e)

    # Fallback: treat entire text as one item
    snippet = raw_text[:500].replace("\n", " ")
    return [DigestItem(
        title=f"Latest from {name}",
        summary=snippet,
        source_name=name,
    )]


def _build_few_shot_context() -> str:
    """Build few-shot examples from digest approval/rejection history.

    Returns a prompt fragment with labeled examples of approved and rejected
    items, or an empty string if fewer than 3 approved items exist.
    """
    from scms.client import SCMSClient

    try:
        client = SCMSClient()
        reviewed = client.get_reviewed_digest_items(limit=30)
    except Exception as e:
        logger.debug("Could not fetch review history for few-shot: %s", e)
        return ""

    approved = [r for r in reviewed if r["status"] == "completed"][:8]
    rejected = [r for r in reviewed if r["status"] == "cancelled"][:5]

    if len(approved) < 3:
        return ""

    lines = ["Here are examples of items the user previously reviewed:\n"]
    lines.append("APPROVED (user found relevant):")
    for item in approved:
        title = _extract_title(item["task"])
        if title:
            lines.append(f'- "{title}"')

    if rejected:
        lines.append("\nREJECTED (user found irrelevant):")
        for item in rejected:
            title = _extract_title(item["task"])
            if title:
                lines.append(f'- "{title}"')

    lines.append("\nUse these preferences to calibrate your scoring.\n")
    return "\n".join(lines)


def summarize_and_score(
    items: list[DigestItem], source: dict
) -> list[DigestItem]:
    """Use local 32B model to assign relevance scores to digest items.

    Scores are 0.0–1.0 based on relevance to the user's active projects.
    Includes few-shot examples from approval/rejection history when available.
    """
    if not items:
        return items

    from agent.utils import get_llm

    projects = source.get("relevance_projects", [])
    llm = get_llm("local")

    items_text = "\n".join(
        f'{i+1}. "{item.title}" — {item.summary}'
        for i, item in enumerate(items)
    )

    few_shot = _build_few_shot_context()
    prompt = (
        "Score the relevance of these items for someone building AI agents, "
        "working on personal productivity tools, and tracking AI industry news. "
        f"Active projects: {', '.join(projects)}.\n\n"
        f"{few_shot}"
        f"Items:\n{items_text}\n\n"
        "Return ONLY a JSON array of numbers (0.0 to 1.0), one per item. "
        "No markdown fences, no commentary. Example: [0.8, 0.3, 0.9]"
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Strip think blocks and markdown fences
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        scores = json.loads(text)
        if isinstance(scores, list):
            for i, score in enumerate(scores):
                if i < len(items) and isinstance(score, (int, float)):
                    items[i].relevance_score = max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.warning("Relevance scoring failed for '%s': %s", source.get("name"), e)
        # Default to 0.5 on failure
        for item in items:
            item.relevance_score = 0.5

    return items


# ---------------------------------------------------------------------------
# Embedding-based pre-filtering
# ---------------------------------------------------------------------------


def embedding_prefilter(
    items: list[DigestItem],
    source: dict,
    config: dict,
) -> tuple[list[DigestItem], list[DigestItem]]:
    """Pre-filter items by embedding similarity against SCMS project memories.

    Embeds each item's title+summary, compares against memories in each
    relevance_project, and filters items below the similarity threshold.
    On cold start (no memories), all items pass through.

    Returns (passed_items, filtered_items).
    """
    if not items:
        return [], []

    relevance_projects = source.get("relevance_projects", [])
    if not relevance_projects:
        logger.info("No relevance_projects for '%s', skipping pre-filter", source.get("name"))
        return items, []

    # Per-source threshold overrides global default
    threshold = source.get(
        "similarity_threshold",
        config.get("settings", {}).get("similarity_threshold", 0.3),
    )

    try:
        from scms.client import SCMSClient
        from scms.embeddings import get_embeddings_batch

        client = SCMSClient()

        # Resolve project IDs once up front
        project_ids = {}
        for proj_name in relevance_projects:
            pid = client._resolve_project_id(proj_name)
            if pid:
                project_ids[proj_name] = pid
        if not project_ids:
            logger.info("No valid projects resolved, skipping pre-filter")
            return items, []

        # Batch-embed all items in one API call
        query_texts = [f"{item.title}. {item.summary}" for item in items]
        embeddings = get_embeddings_batch(query_texts)

        # Score each item against each project's memories
        passed: list[DigestItem] = []
        filtered: list[DigestItem] = []
        any_results_found = False

        for item, emb in zip(items, embeddings):
            max_similarity = 0.0

            for pid in project_ids.values():
                results = client.search_memories_by_embedding(
                    embedding=emb, project_id=pid, limit=1, threshold=0.0,
                )
                if results:
                    any_results_found = True
                    sim = results[0].get("similarity", 0.0)
                    max_similarity = max(max_similarity, sim)

            item.embedding_score = max_similarity
            if max_similarity >= threshold:
                passed.append(item)
            else:
                filtered.append(item)

        # Cold start: if SCMS returned nothing for ANY item, pass everything
        if not any_results_found:
            logger.info("SCMS has no memories for target projects — cold start bypass, all items pass")
            for item in filtered:
                item.embedding_score = 1.0
            return items, []

        logger.info(
            "Embedding pre-filter for '%s': %d/%d passed (threshold=%.2f)",
            source.get("name"), len(passed), len(items), threshold,
        )
        return passed, filtered

    except Exception as e:
        logger.warning("Embedding pre-filter failed, passing all items: %s", e)
        return items, []


def _rerank_items(
    items: list[DigestItem],
    source: dict,
) -> list[DigestItem]:
    """Score items with cairn-rank cross-encoder against project queries.

    Additive only — sets cross_encoder_score on each item but does not
    filter or reorder.  Falls back gracefully if cairn-rank is unavailable.
    """
    if not items:
        return items

    relevance_projects = source.get("relevance_projects", [])
    if not relevance_projects:
        return items

    try:
        from cairn_rank import CrossEncoderReranker, Document as RankDocument
    except ImportError:
        logger.warning("cairn-rank not installed, skipping cross-encoder scoring")
        return items

    try:
        reranker = CrossEncoderReranker()

        for item in items:
            doc = RankDocument(
                content=f"{item.title}. {item.summary}",
                title=item.title,
                url=item.url,
                source=item.source_name,
            )
            max_score = float("-inf")
            for project in relevance_projects:
                ranked = reranker.rank(project, [doc])
                if ranked:
                    max_score = max(max_score, ranked[0].score)

            item.cross_encoder_score = max_score if max_score > float("-inf") else 0.0

        logger.info(
            "Cross-encoder scoring for '%s': scored %d items",
            source.get("name"), len(items),
        )
    except Exception as e:
        logger.warning("Cross-encoder scoring failed, continuing without: %s", e)

    return items


# ---------------------------------------------------------------------------
# Digest assembly and output
# ---------------------------------------------------------------------------

def build_digest(all_results: list[SourceResult], config: dict) -> str:
    """Assemble all results into a markdown digest document."""
    threshold = config.get("settings", {}).get("relevance_threshold", 0.6)
    today = date.today().isoformat()

    lines = [
        f"# Daily Research Digest — {today}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Collect all items
    high_items: list[DigestItem] = []
    other_items: list[DigestItem] = []
    errors: list[tuple[str, str]] = []

    for result in all_results:
        if result.error:
            errors.append((result.source_name, result.error))
            continue
        for item in result.items:
            if item.relevance_score >= threshold:
                high_items.append(item)
            else:
                other_items.append(item)

    # High relevance section
    lines.append("## High Relevance")
    lines.append("")
    if high_items:
        high_items.sort(key=lambda x: x.relevance_score, reverse=True)
        for item in high_items:
            url_part = f" — [link]({item.url})" if item.url else ""
            emb_part = f", similarity: {item.embedding_score:.2f}" if item.embedding_score > 0 else ""
            ce_part = f", CE: {item.cross_encoder_score:.1f}" if item.cross_encoder_score != 0.0 else ""
            lines.append(f"- **{item.title}** (relevance: {item.relevance_score:.1f}{emb_part}{ce_part}, source: {item.source_name}){url_part}")
            lines.append(f"  {item.summary}")
            lines.append("")
    else:
        lines.append("*No high-relevance items today.*")
        lines.append("")

    # Other items section
    lines.append("## Other Items")
    lines.append("")
    if other_items:
        for item in other_items:
            url_part = f" — [link]({item.url})" if item.url else ""
            emb_part = f", similarity: {item.embedding_score:.2f}" if item.embedding_score > 0 else ""
            ce_part = f", CE: {item.cross_encoder_score:.1f}" if item.cross_encoder_score != 0.0 else ""
            lines.append(f"- **{item.title}** ({item.relevance_score:.1f}{emb_part}{ce_part}, {item.source_name}){url_part}")
            lines.append(f"  {item.summary}")
            lines.append("")
    else:
        lines.append("*No other items.*")
        lines.append("")

    # Filtered items section (below embedding pre-filter threshold)
    filtered_items: list[DigestItem] = []
    for result in all_results:
        filtered_items.extend(result.filtered_items)

    if filtered_items:
        lines.append("## Filtered (below embedding threshold)")
        lines.append("")
        filtered_items.sort(key=lambda x: x.embedding_score, reverse=True)
        for item in filtered_items:
            lines.append(f"- ~~{item.title}~~ (similarity: {item.embedding_score:.2f}, source: {item.source_name})")
        lines.append("")

    # Errors section
    if errors:
        lines.append("## Errors")
        lines.append("")
        for source_name, error in errors:
            lines.append(f"- **{source_name}**: {error}")
        lines.append("")

    # Stats
    total_items = len(high_items) + len(other_items)
    lines.append("---")
    lines.append(f"*{total_items} items from {len(all_results)} sources. "
                 f"{len(high_items)} high-relevance, {len(filtered_items)} pre-filtered, "
                 f"{len(errors)} errors.*")

    return "\n".join(lines)


def save_digest(markdown: str, config: dict) -> Path:
    """Save digest markdown to the digests directory."""
    digest_dir = Path(
        config.get("settings", {}).get(
            "digest_notes_dir", "~/Documents/cairn/digests"
        )
    ).expanduser()
    digest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{date.today().isoformat()}_digest.md"
    filepath = digest_dir / filename
    filepath.write_text(markdown)

    logger.info("Saved digest to %s", filepath)
    return filepath


def _extract_title(task_text: str) -> str:
    """Extract title from a [Digest Review] task text."""
    match = re.search(r"\[Digest Review\]\s*(.+)", task_text)
    return match.group(1).strip() if match else ""


def _extract_url(task_text: str) -> str:
    """Extract URL from a [Digest Review] task text."""
    match = re.search(r"URL:\s*(\S+)", task_text)
    return match.group(1).strip() if match else ""


def queue_for_review(
    all_results: list[SourceResult], config: dict
) -> list[str]:
    """Store high-relevance items in task_queue for human review.

    Deduplicates against existing queue items by URL (primary) then
    title (fallback) to prevent reviewed items from reappearing.
    """
    from scms.client import SCMSClient

    threshold = config.get("settings", {}).get("relevance_threshold", 0.6)
    client = SCMSClient()
    task_ids: list[str] = []

    # Fetch existing digest review items (any status) for dedup
    existing = client.get_digest_review_items()
    existing_urls = {
        _extract_url(t["task"]) for t in existing if _extract_url(t["task"])
    }
    existing_titles = {
        _extract_title(t["task"]) for t in existing if _extract_title(t["task"])
    }

    for result in all_results:
        for item in result.items:
            if item.relevance_score >= threshold:
                # Skip items already in the queue (by URL or title)
                if item.url and item.url in existing_urls:
                    logger.debug("Skipping duplicate (URL): %s", item.title)
                    continue
                if item.title and item.title in existing_titles:
                    logger.debug("Skipping duplicate (title): %s", item.title)
                    continue

                # Build a reviewable task description
                url_part = f"\nURL: {item.url}" if item.url else ""
                emb_part = f"\nEmbedding: {item.embedding_score:.2f}" if item.embedding_score > 0 else ""
                ce_part = f"\nCrossEncoder: {item.cross_encoder_score:.4f}" if item.cross_encoder_score != 0.0 else ""
                task_text = (
                    f"[Digest Review] {item.title}\n"
                    f"Source: {item.source_name}\n"
                    f"Relevance: {item.relevance_score:.2f}{emb_part}{ce_part}\n"
                    f"Summary: {item.summary}{url_part}"
                )
                record = client.enqueue_task(
                    task=task_text,
                    priority=3,
                    project="_digest_review",
                )
                task_ids.append(record["id"])

                # Track newly inserted items for intra-batch dedup
                if item.url:
                    existing_urls.add(item.url)
                if item.title:
                    existing_titles.add(item.title)

    logger.info("Queued %d items for review", len(task_ids))
    return task_ids


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_digest(frequency: str = "daily") -> dict:
    """Run the full digest pipeline.

    Returns:
        {"digest_path": str, "items_found": int, "items_queued": int,
         "errors": list[str], "sources_processed": int}
    """
    from agent.notifications import notify

    logger.info("=== Starting %s digest ===", frequency)

    # 1. Load config and sources
    config = _load_config()
    sources = load_sources(frequency_filter=frequency)

    if not sources:
        logger.info("No %s sources to process", frequency)
        return {
            "digest_path": "",
            "items_found": 0,
            "items_queued": 0,
            "errors": [],
            "sources_processed": 0,
        }

    # 2. Fetch each source, pre-filter, then score
    all_results: list[SourceResult] = []
    for source in sources:
        result = fetch_source(source)
        if result.items and not result.error:
            # 3a. Embedding pre-filter: remove items not relevant to SCMS memories
            passed, filtered = embedding_prefilter(result.items, source, config)
            result.filtered_items = filtered
            # 3b. Cross-encoder scoring (additive, does not filter)
            if passed:
                passed = _rerank_items(passed, source)
            # 3c. LLM scoring on items that passed pre-filter only
            if passed:
                result.items = summarize_and_score(passed, source)
            else:
                result.items = []
        all_results.append(result)

    # 4. Build and save digest markdown
    markdown = build_digest(all_results, config)
    digest_path = save_digest(markdown, config)

    # 5. Queue high-relevance items for review
    queued_ids = queue_for_review(all_results, config)

    # 6. Compute stats
    total_items = sum(len(r.items) for r in all_results)
    total_filtered = sum(len(r.filtered_items) for r in all_results)
    errors = [f"{r.source_name}: {r.error}" for r in all_results if r.error]

    # 7. Notify
    notify(
        "Morning digest ready",
        f"{total_items} items from {len(sources)} sources, "
        f"{total_filtered} pre-filtered, {len(queued_ids)} queued for review",
    )

    logger.info(
        "=== Digest complete: %d items, %d filtered, %d queued, %d errors ===",
        total_items, total_filtered, len(queued_ids), len(errors),
    )

    return {
        "digest_path": str(digest_path),
        "items_found": total_items,
        "items_filtered": total_filtered,
        "items_queued": len(queued_ids),
        "errors": errors,
        "sources_processed": len(sources),
    }
