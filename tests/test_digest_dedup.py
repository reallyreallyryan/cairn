"""Tests for digest queue deduplication and task text extraction helpers."""

import pytest
from unittest.mock import patch, MagicMock

from agent.digest import (
    DigestItem,
    SourceResult,
    _extract_title,
    _extract_url,
    queue_for_review,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_item(title: str, url: str = "", relevance: float = 0.8,
               embedding: float = 0.4, source: str = "Test Source") -> DigestItem:
    return DigestItem(
        title=title,
        summary=f"Summary of {title}",
        url=url,
        source_name=source,
        relevance_score=relevance,
        embedding_score=embedding,
    )


def _make_existing_task(title: str, url: str = "", status: str = "completed") -> dict:
    """Build a mock task_queue row as returned by get_digest_review_items()."""
    url_part = f"\nURL: {url}" if url else ""
    task_text = (
        f"[Digest Review] {title}\n"
        f"Source: Test Source\n"
        f"Relevance: 0.80\n"
        f"Summary: Summary of {title}{url_part}"
    )
    return {"id": f"uuid-{title[:8]}", "task": task_text, "status": status}


SAMPLE_CONFIG = {"settings": {"relevance_threshold": 0.6}}


# ---------------------------------------------------------------------------
# _extract_title tests
# ---------------------------------------------------------------------------

class TestExtractTitle:

    def test_extracts_title_from_standard_format(self):
        text = "[Digest Review] My Article Title\nSource: Blog\nRelevance: 0.80"
        assert _extract_title(text) == "My Article Title"

    def test_returns_empty_for_missing_prefix(self):
        assert _extract_title("No prefix here") == ""

    def test_handles_whitespace(self):
        text = "[Digest Review]   Spaced Title  \nSource: Blog"
        assert _extract_title(text) == "Spaced Title"


# ---------------------------------------------------------------------------
# _extract_url tests
# ---------------------------------------------------------------------------

class TestExtractUrl:

    def test_extracts_url(self):
        text = "Summary: blah\nURL: https://example.com/article"
        assert _extract_url(text) == "https://example.com/article"

    def test_returns_empty_when_no_url(self):
        text = "[Digest Review] Title\nSource: Blog\nRelevance: 0.80"
        assert _extract_url(text) == ""

    def test_handles_url_with_query_params(self):
        text = "URL: https://example.com/a?foo=bar&baz=1"
        assert _extract_url(text) == "https://example.com/a?foo=bar&baz=1"


# ---------------------------------------------------------------------------
# queue_for_review dedup tests
# ---------------------------------------------------------------------------

class TestQueueForReviewDedup:

    @patch("scms.client.SCMSClient")
    def test_skips_existing_url(self, mock_client_cls):
        """Items with a URL already in the queue should be skipped."""
        client = MagicMock()
        client.get_digest_review_items.return_value = [
            _make_existing_task("Old Article", url="https://example.com/old"),
        ]
        client.enqueue_task.return_value = {"id": "new-uuid"}
        mock_client_cls.return_value = client

        items = [
            _make_item("Old Article", url="https://example.com/old"),  # duplicate
            _make_item("New Article", url="https://example.com/new"),  # novel
        ]
        results = [SourceResult(source_name="Test", items=items)]

        task_ids = queue_for_review(results, SAMPLE_CONFIG)

        assert len(task_ids) == 1
        assert client.enqueue_task.call_count == 1
        # Verify the enqueued item is the new one
        call_args = client.enqueue_task.call_args
        assert "New Article" in call_args.kwargs.get("task", call_args[1].get("task", ""))

    @patch("scms.client.SCMSClient")
    def test_skips_existing_title(self, mock_client_cls):
        """Items with a title already in the queue (no URL match) should be skipped."""
        client = MagicMock()
        client.get_digest_review_items.return_value = [
            _make_existing_task("Duplicate Title"),  # no URL
        ]
        client.enqueue_task.return_value = {"id": "new-uuid"}
        mock_client_cls.return_value = client

        items = [
            _make_item("Duplicate Title"),  # same title, no URL
            _make_item("Fresh Article"),
        ]
        results = [SourceResult(source_name="Test", items=items)]

        task_ids = queue_for_review(results, SAMPLE_CONFIG)

        assert len(task_ids) == 1
        assert client.enqueue_task.call_count == 1

    @patch("scms.client.SCMSClient")
    def test_inserts_new_items(self, mock_client_cls):
        """Novel items should be queued normally."""
        client = MagicMock()
        client.get_digest_review_items.return_value = []  # empty queue
        client.enqueue_task.return_value = {"id": "new-uuid"}
        mock_client_cls.return_value = client

        items = [
            _make_item("Article A", url="https://a.com"),
            _make_item("Article B", url="https://b.com"),
        ]
        results = [SourceResult(source_name="Test", items=items)]

        task_ids = queue_for_review(results, SAMPLE_CONFIG)

        assert len(task_ids) == 2
        assert client.enqueue_task.call_count == 2

    @patch("scms.client.SCMSClient")
    def test_below_threshold_not_queued(self, mock_client_cls):
        """Items below relevance threshold should not be queued."""
        client = MagicMock()
        client.get_digest_review_items.return_value = []
        mock_client_cls.return_value = client

        items = [_make_item("Low Score", relevance=0.3)]
        results = [SourceResult(source_name="Test", items=items)]

        task_ids = queue_for_review(results, SAMPLE_CONFIG)

        assert len(task_ids) == 0
        assert client.enqueue_task.call_count == 0

    @patch("scms.client.SCMSClient")
    def test_intra_batch_dedup(self, mock_client_cls):
        """Duplicate items within the same batch should be deduped."""
        client = MagicMock()
        client.get_digest_review_items.return_value = []
        client.enqueue_task.return_value = {"id": "new-uuid"}
        mock_client_cls.return_value = client

        # Same title from two different sources in one batch
        items_a = [_make_item("Same Article", url="https://a.com/same")]
        items_b = [_make_item("Same Article", url="https://b.com/same")]
        results = [
            SourceResult(source_name="Source A", items=items_a),
            SourceResult(source_name="Source B", items=items_b),
        ]

        task_ids = queue_for_review(results, SAMPLE_CONFIG)

        # First one inserts; second is caught by intra-batch title dedup
        assert len(task_ids) == 1

    @patch("scms.client.SCMSClient")
    def test_skips_regardless_of_status(self, mock_client_cls):
        """Dedup should work for items in any status (pending, completed, cancelled)."""
        client = MagicMock()
        client.get_digest_review_items.return_value = [
            _make_existing_task("Approved One", url="https://a.com", status="completed"),
            _make_existing_task("Rejected One", url="https://b.com", status="cancelled"),
            _make_existing_task("Pending One", url="https://c.com", status="pending"),
        ]
        client.enqueue_task.return_value = {"id": "new-uuid"}
        mock_client_cls.return_value = client

        items = [
            _make_item("Approved One", url="https://a.com"),
            _make_item("Rejected One", url="https://b.com"),
            _make_item("Pending One", url="https://c.com"),
            _make_item("Brand New", url="https://d.com"),
        ]
        results = [SourceResult(source_name="Test", items=items)]

        task_ids = queue_for_review(results, SAMPLE_CONFIG)

        assert len(task_ids) == 1
        assert client.enqueue_task.call_count == 1


# ---------------------------------------------------------------------------
# completed_at for cancelled status (secondary bug fix in scms/client.py)
# ---------------------------------------------------------------------------

class TestCancelledCompletedAt:

    @patch("scms.client.create_client")
    def test_cancelled_status_sets_completed_at(self, mock_create):
        """Cancelling a task should set completed_at timestamp."""
        from scms.client import SCMSClient

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[{"id": "t1", "status": "cancelled"}])
        mock_create.return_value = mock_supabase

        client = SCMSClient()
        client.update_task_status("t1", "cancelled", result="quality=0")

        # Verify update was called with completed_at in the payload
        update_call = mock_table.update.call_args[0][0]
        assert "completed_at" in update_call
        assert update_call["status"] == "cancelled"
