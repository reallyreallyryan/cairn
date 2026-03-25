"""Tests for the digest evaluation pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from agent.evaluation import (
    ReviewedItem,
    EvalMetrics,
    parse_task_text,
    load_reviewed_items,
    compute_metrics,
    suggest_thresholds,
    build_eval_report,
    run_evaluation,
    MIN_REVIEWS,
    MIN_BUCKET_SIZE,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_review_row(title: str, source: str, relevance: float,
                     embedding: float, approved: bool,
                     created_at: str = "2026-03-20T10:00:00+00:00") -> dict:
    """Build a mock task_queue row as returned by Supabase."""
    status = "completed" if approved else "cancelled"
    task_text = (
        f"[Digest Review] {title}\n"
        f"Source: {source}\n"
        f"Relevance: {relevance:.2f}\n"
        f"Embedding: {embedding:.2f}\n"
        f"Summary: Summary of {title}\n"
        f"URL: https://example.com/{title.lower().replace(' ', '-')}"
    )
    return {
        "id": f"uuid-{title[:8]}",
        "task": task_text,
        "project": "_digest_review",
        "status": status,
        "result": f"quality={'1' if approved else '0'}",
        "created_at": created_at,
        "completed_at": created_at if approved else None,
        "priority": 3,
    }


def _make_reviewed_item(title: str = "Test Article", source: str = "Test Blog",
                        relevance: float = 0.8, embedding: float = 0.4,
                        approved: bool = True,
                        created_at: str = "2026-03-20T10:00:00+00:00") -> ReviewedItem:
    return ReviewedItem(
        title=title,
        source_name=source,
        relevance_score=relevance,
        embedding_score=embedding,
        approved=approved,
        created_at=created_at,
    )


def _make_items_for_metrics(n_approved: int, n_rejected: int,
                            week_offset: int = 0) -> list[ReviewedItem]:
    """Generate a mix of approved and rejected items with varied scores."""
    items = []
    for i in range(n_approved):
        items.append(_make_reviewed_item(
            title=f"Good {i}",
            source="Blog A" if i % 2 == 0 else "Blog B",
            relevance=0.6 + (i % 4) * 0.1,
            embedding=0.2 + (i % 4) * 0.1,
            approved=True,
            created_at=f"2026-03-{10 + week_offset + (i % 7):02d}T10:00:00+00:00",
        ))
    for i in range(n_rejected):
        items.append(_make_reviewed_item(
            title=f"Bad {i}",
            source="Blog A" if i % 2 == 0 else "Blog C",
            relevance=0.6 + (i % 3) * 0.1,
            embedding=0.1 + (i % 3) * 0.1,
            approved=False,
            created_at=f"2026-03-{10 + week_offset + (i % 7):02d}T10:00:00+00:00",
        ))
    return items


# ---------------------------------------------------------------------------
# parse_task_text tests
# ---------------------------------------------------------------------------

class TestParseTaskText:

    def test_parse_full_text(self):
        text = (
            "[Digest Review] AI Agent Patterns\n"
            "Source: Anthropic Blog\n"
            "Relevance: 0.85\n"
            "Embedding: 0.42\n"
            "Summary: Overview of agent design patterns\n"
            "URL: https://anthropic.com/blog/agents"
        )
        result = parse_task_text(text)

        assert result is not None
        assert result["title"] == "AI Agent Patterns"
        assert result["source_name"] == "Anthropic Blog"
        assert result["relevance_score"] == pytest.approx(0.85)
        assert result["embedding_score"] == pytest.approx(0.42)
        assert result["summary"] == "Overview of agent design patterns"
        assert result["url"] == "https://anthropic.com/blog/agents"

    def test_parse_without_embedding(self):
        """Older items may not have an Embedding line."""
        text = (
            "[Digest Review] Old Article\n"
            "Source: Blog\n"
            "Relevance: 0.70\n"
            "Summary: Something interesting\n"
            "URL: https://example.com"
        )
        result = parse_task_text(text)

        assert result is not None
        assert result["title"] == "Old Article"
        assert result["embedding_score"] == 0.0

    def test_parse_without_url(self):
        text = (
            "[Digest Review] No URL Article\n"
            "Source: Blog\n"
            "Relevance: 0.75\n"
            "Summary: No link available"
        )
        result = parse_task_text(text)

        assert result is not None
        assert result["url"] == ""

    def test_parse_malformed_returns_none(self):
        assert parse_task_text("Just some random text") is None
        assert parse_task_text("") is None

    def test_parse_missing_scores_default_to_zero(self):
        text = "[Digest Review] Title Only\nSource: Somewhere"
        result = parse_task_text(text)
        assert result is not None
        assert result["relevance_score"] == 0.0
        assert result["embedding_score"] == 0.0


# ---------------------------------------------------------------------------
# load_reviewed_items tests
# ---------------------------------------------------------------------------

class TestLoadReviewedItems:

    def test_loads_and_parses_items(self):
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = [
            _make_review_row("Article A", "Blog", 0.8, 0.4, True),
            _make_review_row("Article B", "Blog", 0.7, 0.3, False),
        ]

        items = load_reviewed_items(client=client)

        assert len(items) == 2
        assert items[0].title == "Article A"
        assert items[0].approved is True
        assert items[1].title == "Article B"
        assert items[1].approved is False

    def test_skips_unparseable_items(self):
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = [
            _make_review_row("Good Article", "Blog", 0.8, 0.4, True),
            {"id": "bad", "task": "Malformed task text", "status": "completed",
             "created_at": "2026-03-20T10:00:00+00:00"},
        ]

        items = load_reviewed_items(client=client)
        assert len(items) == 1
        assert items[0].title == "Good Article"


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_overall_approval_rate(self):
        items = [
            _make_reviewed_item(approved=True),
            _make_reviewed_item(approved=True),
            _make_reviewed_item(approved=False),
        ]
        metrics = compute_metrics(items)

        assert metrics.total_reviewed == 3
        assert metrics.total_approved == 2
        assert metrics.total_rejected == 1
        assert metrics.approval_rate == pytest.approx(2 / 3)

    def test_empty_items(self):
        metrics = compute_metrics([])
        assert metrics.total_reviewed == 0
        assert metrics.approval_rate == 0.0

    def test_relevance_buckets(self):
        items = [
            _make_reviewed_item(relevance=0.65, approved=True),
            _make_reviewed_item(relevance=0.62, approved=False),
            _make_reviewed_item(relevance=0.75, approved=True),
            _make_reviewed_item(relevance=0.85, approved=True),
            _make_reviewed_item(relevance=0.95, approved=False),
        ]
        metrics = compute_metrics(items)

        assert metrics.by_relevance_bucket["0.6-0.7"]["total"] == 2
        assert metrics.by_relevance_bucket["0.6-0.7"]["approved"] == 1
        assert metrics.by_relevance_bucket["0.7-0.8"]["total"] == 1
        assert metrics.by_relevance_bucket["0.8-0.9"]["total"] == 1
        assert metrics.by_relevance_bucket["0.9+"]["total"] == 1

    def test_embedding_buckets(self):
        items = [
            _make_reviewed_item(embedding=0.15, approved=True),
            _make_reviewed_item(embedding=0.25, approved=False),
            _make_reviewed_item(embedding=0.35, approved=True),
            _make_reviewed_item(embedding=0.55, approved=True),
            _make_reviewed_item(embedding=0.7, approved=True),
        ]
        metrics = compute_metrics(items)

        assert metrics.by_embedding_bucket["0.0-0.2"]["total"] == 1
        assert metrics.by_embedding_bucket["0.2-0.3"]["total"] == 1
        assert metrics.by_embedding_bucket["0.3-0.4"]["total"] == 1
        assert metrics.by_embedding_bucket["0.4-0.6"]["total"] == 1
        assert metrics.by_embedding_bucket["0.6+"]["total"] == 1

    def test_per_source_breakdown(self):
        items = [
            _make_reviewed_item(source="Blog A", approved=True),
            _make_reviewed_item(source="Blog A", approved=False),
            _make_reviewed_item(source="Blog B", approved=True),
        ]
        metrics = compute_metrics(items)

        assert metrics.by_source["Blog A"]["total"] == 2
        assert metrics.by_source["Blog A"]["approved"] == 1
        assert metrics.by_source["Blog A"]["rate"] == pytest.approx(0.5)
        assert metrics.by_source["Blog B"]["total"] == 1
        assert metrics.by_source["Blog B"]["rate"] == pytest.approx(1.0)

    def test_insufficient_data_flagged(self):
        items = [_make_reviewed_item() for _ in range(5)]
        metrics = compute_metrics(items)
        assert metrics.data_sufficient is False

    def test_sufficient_data_flagged(self):
        items = _make_items_for_metrics(15, 10)
        metrics = compute_metrics(items)
        assert metrics.data_sufficient is True

    def test_optimal_threshold_computed(self):
        """With enough data, an optimal threshold should be found."""
        items = _make_items_for_metrics(15, 10)
        metrics = compute_metrics(items)
        assert metrics.optimal_relevance_threshold is not None
        assert 0.5 <= metrics.optimal_relevance_threshold <= 1.0

    def test_optimal_threshold_none_insufficient_data(self):
        items = [_make_reviewed_item() for _ in range(5)]
        metrics = compute_metrics(items)
        assert metrics.optimal_relevance_threshold is None

    def test_weekly_trend(self):
        items = [
            _make_reviewed_item(created_at="2026-03-10T10:00:00+00:00", approved=True),
            _make_reviewed_item(created_at="2026-03-11T10:00:00+00:00", approved=False),
            _make_reviewed_item(created_at="2026-03-17T10:00:00+00:00", approved=True),
            _make_reviewed_item(created_at="2026-03-18T10:00:00+00:00", approved=True),
        ]
        metrics = compute_metrics(items)

        assert len(metrics.weekly_trend) == 2
        # Week 11 (Mar 10-11): 1 approved / 2 total
        assert metrics.weekly_trend[0]["total"] == 2
        assert metrics.weekly_trend[0]["approved"] == 1
        # Week 12 (Mar 17-18): 2 approved / 2 total
        assert metrics.weekly_trend[1]["total"] == 2
        assert metrics.weekly_trend[1]["approved"] == 2

    def test_date_range(self):
        items = [
            _make_reviewed_item(created_at="2026-02-15T10:00:00+00:00"),
            _make_reviewed_item(created_at="2026-03-20T10:00:00+00:00"),
        ]
        metrics = compute_metrics(items)
        assert "2026-02-15" in metrics.date_range
        assert "2026-03-20" in metrics.date_range


# ---------------------------------------------------------------------------
# suggest_thresholds tests
# ---------------------------------------------------------------------------

class TestSuggestThresholds:

    def test_insufficient_data_returns_warning(self):
        metrics = EvalMetrics(
            total_reviewed=5, total_approved=3, total_rejected=2,
            approval_rate=0.6, data_sufficient=False,
        )
        suggestions = suggest_thresholds(metrics, {})
        assert len(suggestions) == 1
        assert "need at least" in suggestions[0].lower()

    def test_flags_low_approval_source(self):
        metrics = EvalMetrics(
            total_reviewed=30, total_approved=15, total_rejected=15,
            approval_rate=0.5, data_sufficient=True,
            by_source={
                "Good Source": {"total": 10, "approved": 8, "rate": 0.8},
                "Bad Source": {"total": 10, "approved": 1, "rate": 0.1},
            },
        )
        suggestions = suggest_thresholds(metrics, {"settings": {}})
        flagged = [s for s in suggestions if "Bad Source" in s]
        assert len(flagged) == 1
        assert "10%" in flagged[0]


# ---------------------------------------------------------------------------
# build_eval_report tests
# ---------------------------------------------------------------------------

class TestBuildReport:

    def test_report_contains_key_sections(self):
        items = _make_items_for_metrics(15, 10)
        metrics = compute_metrics(items)
        suggestions = ["Test suggestion"]
        report = build_eval_report(metrics, suggestions)

        assert "## Summary" in report
        assert "## Approval by Relevance Score" in report
        assert "## Approval by Embedding Score" in report
        assert "## Per-Source Breakdown" in report
        assert "## Recommendations" in report
        assert "Test suggestion" in report

    def test_report_with_insufficient_data_caveat(self):
        items = [_make_reviewed_item() for _ in range(3)]
        metrics = compute_metrics(items)
        report = build_eval_report(metrics, [])

        assert "not be statistically reliable" in report

    def test_report_with_no_items(self):
        metrics = compute_metrics([])
        report = build_eval_report(metrics, [])
        assert "## Summary" in report
        assert "Total reviewed | 0" in report


# ---------------------------------------------------------------------------
# run_evaluation end-to-end test
# ---------------------------------------------------------------------------

class TestRunEvaluation:

    @patch("agent.evaluation.save_eval_report")
    @patch("agent.evaluation.load_reviewed_items")
    def test_full_pipeline(self, mock_load, mock_save):
        mock_load.return_value = _make_items_for_metrics(15, 10)
        mock_save.return_value = "/tmp/eval_report.md"

        result = run_evaluation()

        assert result["total_reviewed"] == 25
        assert result["approval_rate"] == pytest.approx(15 / 25)
        assert result["report_path"] == "/tmp/eval_report.md"
        assert isinstance(result["suggestions"], list)
        mock_save.assert_called_once()

    @patch("agent.evaluation.load_reviewed_items")
    def test_no_items_returns_empty(self, mock_load):
        mock_load.return_value = []

        result = run_evaluation()

        assert result["total_reviewed"] == 0
        assert result["report_path"] == ""
