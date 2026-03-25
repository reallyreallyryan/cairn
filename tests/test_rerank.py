"""Tests for cross-encoder reranking in the digest pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from agent.digest import DigestItem, _rerank_items


def _make_items(n: int) -> list[DigestItem]:
    return [
        DigestItem(
            title=f"Article {i}",
            summary=f"Summary about topic {i}",
            source_name="Test Source",
        )
        for i in range(n)
    ]


SAMPLE_SOURCE = {
    "name": "Test Source",
    "relevance_projects": ["cairn", "Research"],
}


class TestRerankItems:

    @patch("cairn_rank.CrossEncoderReranker")
    def test_scores_assigned(self, mock_reranker_cls):
        """Items should get cross_encoder_score from reranker."""
        mock_scored = MagicMock()
        mock_scored.score = 3.5

        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [mock_scored]
        mock_reranker_cls.return_value = mock_reranker

        items = _make_items(2)
        result = _rerank_items(items, SAMPLE_SOURCE)

        assert len(result) == 2
        assert result[0].cross_encoder_score == 3.5
        assert result[1].cross_encoder_score == 3.5

    @patch("cairn_rank.CrossEncoderReranker")
    def test_max_across_projects(self, mock_reranker_cls):
        """Should take max score across project queries."""
        mock_reranker = MagicMock()

        def rank_side_effect(query, docs):
            scored = MagicMock()
            scored.score = 1.0 if query == "cairn" else 5.0
            return [scored]

        mock_reranker.rank.side_effect = rank_side_effect
        mock_reranker_cls.return_value = mock_reranker

        items = _make_items(1)
        result = _rerank_items(items, SAMPLE_SOURCE)

        assert result[0].cross_encoder_score == 5.0

    def test_empty_items_returns_empty(self):
        result = _rerank_items([], SAMPLE_SOURCE)
        assert result == []

    def test_no_projects_returns_unmodified(self):
        source = {"name": "Test", "relevance_projects": []}
        items = _make_items(2)
        result = _rerank_items(items, source)
        assert all(item.cross_encoder_score == 0.0 for item in result)

    def test_missing_projects_key_returns_unmodified(self):
        source = {"name": "Test"}
        items = _make_items(2)
        result = _rerank_items(items, source)
        assert all(item.cross_encoder_score == 0.0 for item in result)

    @patch("cairn_rank.CrossEncoderReranker", create=True, side_effect=RuntimeError("Model load failed"))
    def test_runtime_error_graceful(self, mock_reranker_cls):
        """Should return items unmodified on runtime errors."""
        items = _make_items(2)
        result = _rerank_items(items, SAMPLE_SOURCE)
        assert all(item.cross_encoder_score == 0.0 for item in result)

    @patch("cairn_rank.CrossEncoderReranker")
    def test_negative_scores_preserved(self, mock_reranker_cls):
        """Negative logits should be preserved (not clamped)."""
        mock_scored = MagicMock()
        mock_scored.score = -3.5

        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [mock_scored]
        mock_reranker_cls.return_value = mock_reranker

        items = _make_items(1)
        result = _rerank_items(items, SAMPLE_SOURCE)

        assert result[0].cross_encoder_score == -3.5
