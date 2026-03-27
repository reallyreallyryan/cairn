"""Tests for the digest compiler (agent/compile_digest.py)."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.compile_digest import (
    CompiledArticle,
    build_digest,
    compile_articles,
    fetch_article_content,
    load_approved_items,
    run_compile_digest,
    summarize_article,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TASK_TEXT = (
    "[Digest Review] Designing AI Systems for Critical Thinking\n"
    "Source: arXiv cs.AI\n"
    "Relevance: 0.92\n"
    "Embedding: 0.45\n"
    "CrossEncoder: 1.34\n"
    "Summary: This paper proposes a framework for evaluating AI systems.\n"
    "URL: https://arxiv.org/abs/2025.12345"
)

SAMPLE_TASK_TEXT_NO_URL = (
    "[Digest Review] Some Local Research\n"
    "Source: Blog\n"
    "Relevance: 0.70\n"
    "Summary: A brief overview of local research.\n"
)

MALFORMED_TASK_TEXT = "This is not a valid digest review item."


def _make_row(task: str, status: str = "completed") -> dict:
    """Build a mock task_queue row."""
    return {
        "id": "test-id",
        "task": task,
        "status": status,
        "project": "_digest_review",
        "completed_at": "2026-03-25T12:00:00+00:00",
    }


def _make_article(
    title: str = "Test Article",
    source_name: str = "arXiv cs.AI",
    url: str = "https://example.com/article",
    relevance_score: float = 0.9,
    cross_encoder_score: float = 1.5,
    full_content: str | None = "Full article text here.",
    deep_summary: str = "Deep summary text.",
    brief_summary: str = "Brief summary text.",
) -> CompiledArticle:
    return CompiledArticle(
        title=title,
        source_name=source_name,
        url=url,
        relevance_score=relevance_score,
        cross_encoder_score=cross_encoder_score,
        original_summary="Original snippet.",
        full_content=full_content,
        deep_summary=deep_summary,
        brief_summary=brief_summary,
    )


# ---------------------------------------------------------------------------
# fetch_article_content
# ---------------------------------------------------------------------------

class TestFetchArticleContent:
    @patch("agent.compile_digest.httpx.get")
    def test_success_with_trafilatura(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Article content here</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("trafilatura.extract", return_value="Extracted article content"):
            result = fetch_article_content("https://example.com/article")

        assert result == "Extracted article content"

    @patch("agent.compile_digest.httpx.get")
    def test_fallback_when_trafilatura_returns_none(self, mock_get):
        """When trafilatura returns None and bs4 is unavailable, returns None."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Paragraph text</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("trafilatura.extract", return_value=None):
            result = fetch_article_content("https://example.com/article")

        # Without bs4 installed, falls through to None
        # (bs4 is an optional dependency — url_reader.py has the same fallback)
        assert result is None

    @patch("agent.compile_digest.httpx.get")
    def test_returns_none_on_timeout(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = fetch_article_content("https://example.com/article")
        assert result is None

    @patch("agent.compile_digest.httpx.get")
    def test_returns_none_on_http_error(self, mock_get):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp,
        )

        result = fetch_article_content("https://example.com/article")
        assert result is None

    def test_returns_none_for_empty_url(self):
        assert fetch_article_content("") is None
        assert fetch_article_content(None) is None

    @patch("agent.compile_digest.httpx.get")
    def test_truncates_at_max_chars(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>x</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        long_content = "A" * 20000
        with patch("trafilatura.extract", return_value=long_content):
            result = fetch_article_content("https://example.com", max_chars=100)

        assert len(result) == 100


# ---------------------------------------------------------------------------
# load_approved_items
# ---------------------------------------------------------------------------

class TestLoadApprovedItems:
    def test_loads_only_approved(self):
        mock_client = MagicMock()
        mock_client.get_reviewed_digest_items.return_value = [
            _make_row(SAMPLE_TASK_TEXT, status="completed"),
            _make_row(SAMPLE_TASK_TEXT, status="cancelled"),
        ]

        items = load_approved_items(client=mock_client, since="2026-03-24")
        assert len(items) == 1
        assert items[0]["title"] == "Designing AI Systems for Critical Thinking"

    def test_skips_unparseable(self):
        mock_client = MagicMock()
        mock_client.get_reviewed_digest_items.return_value = [
            _make_row(SAMPLE_TASK_TEXT, status="completed"),
            _make_row(MALFORMED_TASK_TEXT, status="completed"),
        ]

        items = load_approved_items(client=mock_client)
        assert len(items) == 1

    def test_returns_empty_on_no_items(self):
        mock_client = MagicMock()
        mock_client.get_reviewed_digest_items.return_value = []

        items = load_approved_items(client=mock_client)
        assert items == []

    def test_passes_since_to_client(self):
        mock_client = MagicMock()
        mock_client.get_reviewed_digest_items.return_value = []

        load_approved_items(client=mock_client, since="2026-03-20")
        mock_client.get_reviewed_digest_items.assert_called_once_with(since="2026-03-20")

    def test_extracts_all_fields(self):
        mock_client = MagicMock()
        mock_client.get_reviewed_digest_items.return_value = [
            _make_row(SAMPLE_TASK_TEXT, status="completed"),
        ]

        items = load_approved_items(client=mock_client)
        item = items[0]
        assert item["title"] == "Designing AI Systems for Critical Thinking"
        assert item["source_name"] == "arXiv cs.AI"
        assert item["relevance_score"] == 0.92
        assert item["embedding_score"] == 0.45
        assert item["cross_encoder_score"] == 1.34
        assert "framework" in item["summary"]
        assert item["url"] == "https://arxiv.org/abs/2025.12345"


# ---------------------------------------------------------------------------
# summarize_article
# ---------------------------------------------------------------------------

class TestSummarizeArticle:
    @patch("agent.utils.get_llm")
    def test_deep_style_uses_technical_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Technical summary output.")
        mock_get_llm.return_value = mock_llm

        result = summarize_article("Title", "Article content", "snippet", "deep")

        assert result == "Technical summary output."
        prompt_used = mock_llm.invoke.call_args[0][0]
        assert "technical" in prompt_used.lower()
        assert "150-300 words" in prompt_used

    @patch("agent.utils.get_llm")
    def test_brief_style_uses_accessible_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Accessible summary output.")
        mock_get_llm.return_value = mock_llm

        result = summarize_article("Title", "Article content", "snippet", "brief")

        assert result == "Accessible summary output."
        prompt_used = mock_llm.invoke.call_args[0][0]
        assert "so what" in prompt_used.lower()
        assert "100-200 words" in prompt_used

    @patch("agent.utils.get_llm")
    def test_strips_think_tags(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="<think>reasoning here</think>Clean output."
        )
        mock_get_llm.return_value = mock_llm

        result = summarize_article("Title", "Content", "snippet", "deep")
        assert "<think>" not in result
        assert result == "Clean output."

    @patch("agent.utils.get_llm")
    def test_snippet_fallback_when_no_content(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary from snippet.")
        mock_get_llm.return_value = mock_llm

        result = summarize_article("Title", None, "Original snippet text", "deep")

        assert "Summary from snippet" in result
        assert "full article not accessible" in result
        prompt_used = mock_llm.invoke.call_args[0][0]
        assert "Original snippet text" in prompt_used

    @patch("agent.utils.get_llm")
    def test_handles_llm_error(self, mock_get_llm):
        mock_get_llm.side_effect = ConnectionError("Ollama not available")

        result = summarize_article("Title", None, "Snippet text", "deep")

        assert "Snippet text" in result
        assert "full article not accessible" in result


# ---------------------------------------------------------------------------
# build_digest
# ---------------------------------------------------------------------------

class TestBuildDigest:
    def test_deep_digest_format(self):
        articles = [
            _make_article(title="First Article", relevance_score=0.95),
            _make_article(title="Second Article", relevance_score=0.80),
        ]

        md = build_digest(articles, "deep")

        assert "# cairn digest" in md
        assert "## Deep dive" in md
        assert "2 articles from 1 sources" in md
        assert "### Contents" in md
        assert "1. First Article" in md
        assert "2. Second Article" in md
        assert "### 1. [First Article]" in md
        assert "Deep summary text." in md
        assert "**Relevance:** 0.95" in md

    def test_briefing_format(self):
        articles = [_make_article(title="Test Article")]

        md = build_digest(articles, "brief")

        assert "## Briefing" in md
        assert "general audience" in md
        assert "Brief summary text." in md

    def test_empty_articles(self):
        md = build_digest([], "deep")

        assert "0 articles" in md
        assert "### Contents" in md

    def test_article_without_url_no_link(self):
        articles = [_make_article(url="")]

        md = build_digest(articles, "deep")

        assert "### 1. Test Article" in md
        assert "[Test Article](" not in md

    def test_cross_encoder_shown_when_nonzero(self):
        articles = [_make_article(cross_encoder_score=1.5)]
        md = build_digest(articles, "deep")
        assert "**Cross-encoder:** 1.50" in md

    def test_cross_encoder_hidden_when_zero(self):
        articles = [_make_article(cross_encoder_score=0.0)]
        md = build_digest(articles, "deep")
        assert "Cross-encoder" not in md


# ---------------------------------------------------------------------------
# compile_articles
# ---------------------------------------------------------------------------

class TestCompileArticles:
    @patch("agent.compile_digest.summarize_article")
    @patch("agent.compile_digest.fetch_article_content")
    def test_compiles_with_full_content(self, mock_fetch, mock_summarize):
        mock_fetch.return_value = "Full article text"
        mock_summarize.side_effect = ["Deep summary", "Brief summary"]

        items = [{"title": "Test", "url": "https://example.com", "source_name": "Blog",
                  "relevance_score": 0.9, "cross_encoder_score": 1.0, "summary": "Snippet"}]

        articles = compile_articles(items)

        assert len(articles) == 1
        assert articles[0].full_content == "Full article text"
        assert articles[0].deep_summary == "Deep summary"
        assert articles[0].brief_summary == "Brief summary"

    @patch("agent.compile_digest.summarize_article")
    @patch("agent.compile_digest.fetch_article_content")
    def test_graceful_degradation_on_fetch_failure(self, mock_fetch, mock_summarize):
        mock_fetch.return_value = None
        mock_summarize.side_effect = ["Deep from snippet", "Brief from snippet"]

        items = [{"title": "Test", "url": "https://broken.com", "source_name": "Blog",
                  "relevance_score": 0.8, "cross_encoder_score": 0.0, "summary": "Snippet"}]

        articles = compile_articles(items)

        assert articles[0].full_content is None
        assert articles[0].deep_summary == "Deep from snippet"
        # summarize_article was called with content=None
        calls = mock_summarize.call_args_list
        assert calls[0][0][1] is None  # content arg for deep call


# ---------------------------------------------------------------------------
# run_compile_digest (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestRunDailyDigest:
    @patch("agent.compile_digest.save_compiled_digest")
    @patch("agent.compile_digest.compile_articles")
    @patch("agent.compile_digest.load_approved_items")
    @patch("agent.compile_digest._load_config")
    def test_full_pipeline(self, mock_config, mock_load, mock_compile, mock_save):
        mock_config.return_value = {"settings": {"digest_notes_dir": "/tmp/test-digests"}}
        mock_load.return_value = [
            {"title": "Art1", "url": "https://example.com", "source_name": "Blog",
             "relevance_score": 0.9, "cross_encoder_score": 1.0, "summary": "S1"},
        ]
        mock_compile.return_value = [
            _make_article(title="Art1", full_content="Full text"),
        ]
        mock_save.side_effect = [
            Path("/tmp/test-digests/2026-03-25_digest_deep.md"),
            Path("/tmp/test-digests/2026-03-25_digest_briefing.md"),
        ]

        result = run_compile_digest(since="2026-03-24")

        assert result["articles_compiled"] == 1
        assert result["articles_with_full_content"] == 1
        assert "deep" in result["deep_path"]
        assert "briefing" in result["briefing_path"]
        assert result["errors"] == []
        assert mock_save.call_count == 2

    @patch("agent.compile_digest.load_approved_items")
    @patch("agent.compile_digest._load_config")
    def test_no_items_returns_empty(self, mock_config, mock_load):
        mock_config.return_value = {"settings": {}}
        mock_load.return_value = []

        result = run_compile_digest(since="2026-03-24")

        assert result["articles_compiled"] == 0
        assert result["deep_path"] == ""
        assert result["briefing_path"] == ""

    @patch("agent.compile_digest.save_compiled_digest")
    @patch("agent.compile_digest.compile_articles")
    @patch("agent.compile_digest.load_approved_items")
    @patch("agent.compile_digest._load_config")
    def test_reports_fetch_errors(self, mock_config, mock_load, mock_compile, mock_save):
        mock_config.return_value = {"settings": {}}
        mock_load.return_value = [{"title": "Broken", "url": "https://broken.com",
                                   "source_name": "X", "relevance_score": 0.5,
                                   "cross_encoder_score": 0.0, "summary": "S"}]
        mock_compile.return_value = [
            _make_article(title="Broken", full_content=None),
        ]
        mock_save.side_effect = [Path("/tmp/deep.md"), Path("/tmp/briefing.md")]

        result = run_compile_digest(since="2026-03-24")

        assert result["articles_with_full_content"] == 0
        assert len(result["errors"]) == 1
        assert "Broken" in result["errors"][0]
