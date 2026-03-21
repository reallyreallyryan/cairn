"""arXiv academic paper search tool."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers on a topic.

    Args:
        query: The search query for academic papers.
        max_results: Maximum number of results to return (default 5).
    """
    logger.info("arXiv search: query='%s', max=%d", query, max_results)
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = list(client.results(search))

        if not results:
            return f"No arXiv papers found for: {query}"

        lines = [f"Found {len(results)} papers on arXiv:\n"]
        for i, paper in enumerate(results, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += f" (+{len(paper.authors) - 3} more)"

            abstract = paper.summary.replace("\n", " ")[:200]

            lines.append(f"{i}. **{paper.title}**")
            lines.append(f"   Authors: {authors}")
            lines.append(f"   Abstract: {abstract}...")
            lines.append(f"   URL: {paper.entry_id}")
            lines.append(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error("arXiv search failed: %s", e)
        return f"Error searching arXiv: {e}"
