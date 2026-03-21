"""Tests for the embedding-based digest pre-filter."""

import pytest
from unittest.mock import patch, MagicMock

from agent.digest import DigestItem, embedding_prefilter


def _make_items(n: int) -> list[DigestItem]:
    """Create n sample DigestItems."""
    return [
        DigestItem(
            title=f"Article {i}",
            summary=f"Summary of article {i}",
            source_name="Test Source",
        )
        for i in range(n)
    ]


SAMPLE_CONFIG = {
    "settings": {
        "similarity_threshold": 0.3,
    }
}

SAMPLE_SOURCE = {
    "name": "Test Source",
    "relevance_projects": ["cairn", "Research"],
}


@pytest.fixture
def mock_scms():
    """Mock SCMSClient for pre-filter tests."""
    client = MagicMock()
    # Default: resolve both projects
    client._resolve_project_id.side_effect = lambda name: f"uuid-{name}"
    return client


@pytest.fixture
def mock_embeddings():
    """Return fixed embeddings (one per item)."""
    def _batch(texts):
        return [[0.1] * 1536 for _ in texts]
    return _batch


class TestEmbeddingPrefilter:

    @patch("scms.embeddings.get_embeddings_batch")
    @patch("scms.client.SCMSClient")
    def test_items_above_threshold_pass(self, mock_client_cls, mock_embed_batch):
        """Items with high similarity should pass through."""
        client = MagicMock()
        client._resolve_project_id.side_effect = lambda name: f"uuid-{name}"
        client.search_memories_by_embedding.return_value = [{"similarity": 0.5}]
        mock_client_cls.return_value = client

        mock_embed_batch.return_value = [[0.1] * 1536 for _ in range(3)]

        items = _make_items(3)
        passed, filtered = embedding_prefilter(items, SAMPLE_SOURCE, SAMPLE_CONFIG)

        assert len(passed) == 3
        assert len(filtered) == 0
        assert all(item.embedding_score == 0.5 for item in passed)

    @patch("scms.embeddings.get_embeddings_batch")
    @patch("scms.client.SCMSClient")
    def test_items_below_threshold_filtered(self, mock_client_cls, mock_embed_batch):
        """Items with low similarity should be filtered out."""
        client = MagicMock()
        client._resolve_project_id.side_effect = lambda name: f"uuid-{name}"
        client.search_memories_by_embedding.return_value = [{"similarity": 0.1}]
        mock_client_cls.return_value = client

        mock_embed_batch.return_value = [[0.1] * 1536 for _ in range(3)]

        items = _make_items(3)
        passed, filtered = embedding_prefilter(items, SAMPLE_SOURCE, SAMPLE_CONFIG)

        assert len(passed) == 0
        assert len(filtered) == 3
        assert all(item.embedding_score == 0.1 for item in filtered)

    @patch("scms.embeddings.get_embeddings_batch")
    @patch("scms.client.SCMSClient")
    def test_cold_start_bypass(self, mock_client_cls, mock_embed_batch):
        """When SCMS has no memories, all items should pass (cold start)."""
        client = MagicMock()
        client._resolve_project_id.side_effect = lambda name: f"uuid-{name}"
        client.search_memories_by_embedding.return_value = []  # No memories
        mock_client_cls.return_value = client

        mock_embed_batch.return_value = [[0.1] * 1536 for _ in range(3)]

        items = _make_items(3)
        passed, filtered = embedding_prefilter(items, SAMPLE_SOURCE, SAMPLE_CONFIG)

        assert len(passed) == 3
        assert len(filtered) == 0
        # Cold start sets embedding_score to 1.0
        assert all(item.embedding_score == 1.0 for item in passed)

    @patch("scms.embeddings.get_embeddings_batch")
    @patch("scms.client.SCMSClient")
    def test_per_source_threshold_overrides_global(self, mock_client_cls, mock_embed_batch):
        """Source-level similarity_threshold should override the global default."""
        client = MagicMock()
        client._resolve_project_id.side_effect = lambda name: f"uuid-{name}"
        client.search_memories_by_embedding.return_value = [{"similarity": 0.32}]
        mock_client_cls.return_value = client

        mock_embed_batch.return_value = [[0.1] * 1536 for _ in range(2)]

        source_with_high_threshold = {
            **SAMPLE_SOURCE,
            "similarity_threshold": 0.4,  # Higher than global 0.3
        }

        items = _make_items(2)
        passed, filtered = embedding_prefilter(items, source_with_high_threshold, SAMPLE_CONFIG)

        # 0.32 < 0.4 → all filtered
        assert len(passed) == 0
        assert len(filtered) == 2

    @patch("scms.embeddings.get_embeddings_batch")
    @patch("scms.client.SCMSClient")
    def test_max_across_projects(self, mock_client_cls, mock_embed_batch):
        """Item should pass if relevant to ANY project (max across projects)."""
        client = MagicMock()
        client._resolve_project_id.side_effect = lambda name: f"uuid-{name}"

        # First project: low similarity. Second project: high similarity.
        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            pid = kwargs.get("project_id", "")
            if "cairn" in pid:
                return [{"similarity": 0.1}]
            else:
                return [{"similarity": 0.5}]
        client.search_memories_by_embedding.side_effect = side_effect
        mock_client_cls.return_value = client

        mock_embed_batch.return_value = [[0.1] * 1536]

        items = _make_items(1)
        passed, filtered = embedding_prefilter(items, SAMPLE_SOURCE, SAMPLE_CONFIG)

        assert len(passed) == 1
        assert passed[0].embedding_score == 0.5  # Max of 0.1 and 0.5

    def test_empty_items_no_api_calls(self):
        """Empty items list should return immediately without any API calls."""
        passed, filtered = embedding_prefilter([], SAMPLE_SOURCE, SAMPLE_CONFIG)
        assert passed == []
        assert filtered == []

    def test_no_relevance_projects_passes_all(self):
        """Source with no relevance_projects should pass all items through."""
        source_no_projects = {"name": "Test", "relevance_projects": []}
        items = _make_items(3)
        passed, filtered = embedding_prefilter(items, source_no_projects, SAMPLE_CONFIG)

        assert len(passed) == 3
        assert len(filtered) == 0
