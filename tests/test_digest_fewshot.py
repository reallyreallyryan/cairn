"""Tests for few-shot calibration in digest relevance scoring."""

import pytest
from unittest.mock import patch, MagicMock

from agent.digest import _build_few_shot_context, _extract_title


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_review_row(title: str, status: str) -> dict:
    """Build a mock task_queue row as returned by get_reviewed_digest_items()."""
    task_text = (
        f"[Digest Review] {title}\n"
        f"Source: Test Source\n"
        f"Relevance: 0.80\n"
        f"Summary: Summary of {title}"
    )
    return {"id": f"uuid-{title[:8]}", "task": task_text, "status": status}


# ---------------------------------------------------------------------------
# _build_few_shot_context tests
# ---------------------------------------------------------------------------

class TestBuildFewShotContext:

    @patch("scms.client.SCMSClient")
    def test_returns_examples_with_sufficient_history(self, mock_client_cls):
        """With 3+ approved items, should return a prompt fragment with examples."""
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = [
            _make_review_row("Agent Design Patterns", "completed"),
            _make_review_row("LLM Tool Use Survey", "completed"),
            _make_review_row("RAG Best Practices", "completed"),
            _make_review_row("Cooking Blog Post", "cancelled"),
            _make_review_row("Sports News Update", "cancelled"),
        ]
        mock_client_cls.return_value = client

        result = _build_few_shot_context()

        assert "APPROVED" in result
        assert "Agent Design Patterns" in result
        assert "LLM Tool Use Survey" in result
        assert "RAG Best Practices" in result
        assert "REJECTED" in result
        assert "Cooking Blog Post" in result
        assert "calibrate" in result.lower()

    @patch("scms.client.SCMSClient")
    def test_returns_empty_with_insufficient_history(self, mock_client_cls):
        """With fewer than 3 approved items, should return empty string."""
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = [
            _make_review_row("Only One Article", "completed"),
            _make_review_row("Rejected One", "cancelled"),
        ]
        mock_client_cls.return_value = client

        result = _build_few_shot_context()
        assert result == ""

    @patch("scms.client.SCMSClient")
    def test_returns_empty_with_no_history(self, mock_client_cls):
        """With zero items, should return empty string."""
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = []
        mock_client_cls.return_value = client

        result = _build_few_shot_context()
        assert result == ""

    @patch("scms.client.SCMSClient")
    def test_works_without_rejected_items(self, mock_client_cls):
        """Should work with only approved items (no rejected section)."""
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = [
            _make_review_row("Article A", "completed"),
            _make_review_row("Article B", "completed"),
            _make_review_row("Article C", "completed"),
        ]
        mock_client_cls.return_value = client

        result = _build_few_shot_context()

        assert "APPROVED" in result
        assert "Article A" in result
        assert "REJECTED" not in result

    @patch("scms.client.SCMSClient")
    def test_limits_approved_to_8(self, mock_client_cls):
        """Should include at most 8 approved examples."""
        client = MagicMock()
        client.get_reviewed_digest_items.return_value = [
            _make_review_row(f"Article {i}", "completed") for i in range(15)
        ]
        mock_client_cls.return_value = client

        result = _build_few_shot_context()

        # Count quoted titles in APPROVED section
        approved_count = result.count('- "Article')
        assert approved_count == 8

    @patch("scms.client.SCMSClient")
    def test_limits_rejected_to_5(self, mock_client_cls):
        """Should include at most 5 rejected examples."""
        client = MagicMock()
        reviewed = [_make_review_row(f"Good {i}", "completed") for i in range(5)]
        reviewed += [_make_review_row(f"Bad {i}", "cancelled") for i in range(10)]
        client.get_reviewed_digest_items.return_value = reviewed
        mock_client_cls.return_value = client

        result = _build_few_shot_context()

        # Count rejected items
        rejected_section = result.split("REJECTED")[1] if "REJECTED" in result else ""
        rejected_count = rejected_section.count('- "Bad')
        assert rejected_count == 5

    @patch("scms.client.SCMSClient")
    def test_graceful_on_scms_error(self, mock_client_cls):
        """Should return empty string if SCMS client fails."""
        mock_client_cls.side_effect = Exception("Connection refused")

        result = _build_few_shot_context()
        assert result == ""
