"""Tests for the model router — complexity classification and tier selection."""

import pytest
from unittest.mock import patch, MagicMock

import agent.model_router as model_router
from agent.model_router import classify_complexity, route_and_get_llm


# Sample config matching the real model_routing.yaml structure
SAMPLE_CONFIG = {
    "tiers": {
        "local_light": {"model": "local_light", "cost_per_call": 0.0},
        "local": {"model": "local", "cost_per_call": 0.0},
        "cloud_standard": {"model": "cloud", "cost_per_call": 0.01},
        "cloud_advanced": {"model": "cloud", "cost_per_call": 0.03},
    },
    "rules": [
        {
            "name": "simple_recall",
            "match": {"task_types": ["knowledge_management"], "keywords": ["list projects", "recall"]},
            "tier": "local_light",
        },
        {
            "name": "simple_notes",
            "match": {"task_types": ["productivity"], "keywords": ["take note", "create note"]},
            "tier": "local_light",
        },
        {
            "name": "summarization",
            "match": {"task_types": ["research", "productivity"], "keywords": ["summarize", "digest"]},
            "tier": "local",
        },
        {
            "name": "research",
            "match": {"task_types": ["research"]},
            "tier": "cloud_standard",
        },
        {
            "name": "multi_step",
            "match": {"task_types": ["multi"]},
            "tier": "cloud_standard",
        },
        {
            "name": "metatool",
            "match": {"task_types": ["metatool"]},
            "tier": "cloud_standard",
        },
        {
            "name": "complex_technical",
            "match": {"task_types": ["technical"], "keywords": ["debug", "architect", "refactor"]},
            "tier": "cloud_advanced",
        },
    ],
    "default_tier": "local",
}


@pytest.fixture(autouse=True)
def mock_config():
    """Inject sample config into the router's cache for all tests."""
    model_router._config_cache = SAMPLE_CONFIG
    yield
    model_router._config_cache = None


# ------------------------------------------------------------------
# classify_complexity tests
# ------------------------------------------------------------------


class TestClassifyComplexity:

    def test_simple_recall_route(self):
        tier = classify_complexity("list projects", "knowledge_management")
        assert tier == "local_light"

    def test_simple_notes_route(self):
        tier = classify_complexity("take note about meeting", "productivity")
        assert tier == "local_light"

    def test_summarization_route(self):
        tier = classify_complexity("summarize this article", "research")
        assert tier == "local"

    def test_research_route(self):
        tier = classify_complexity("find papers on attention", "research")
        assert tier == "cloud_standard"

    def test_multi_step_route(self):
        tier = classify_complexity("search and save results", "multi")
        assert tier == "cloud_standard"

    def test_metatool_route(self):
        tier = classify_complexity("create a CSV tool", "metatool")
        assert tier == "cloud_standard"

    def test_complex_technical_route(self):
        tier = classify_complexity("debug this async function", "technical")
        assert tier == "cloud_advanced"

    def test_default_tier_no_rule_match(self):
        """When no rule matches, fall back to default_tier."""
        tier = classify_complexity("hello world", "unknown_type")
        assert tier == "local"

    def test_keyword_matching_is_case_insensitive(self):
        tier = classify_complexity("SUMMARIZE the report", "research")
        assert tier == "local"

    def test_rule_order_matters(self):
        """'summarize' keyword with 'research' type should hit summarization before research."""
        tier = classify_complexity("summarize these papers", "research")
        assert tier == "local"  # summarization rule comes before research


# ------------------------------------------------------------------
# route_and_get_llm tests
# ------------------------------------------------------------------


class TestRouteAndGetLlm:

    @patch("agent.utils.get_llm")
    def test_override_local(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()
        llm, tier, cost = route_and_get_llm("test task", "research", override="local")
        assert tier == "local"
        mock_get_llm.assert_called_with("local")

    @patch("agent.utils.get_llm")
    def test_override_cloud(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()
        llm, tier, cost = route_and_get_llm("test task", "research", override="cloud")
        assert tier == "cloud_standard"
        mock_get_llm.assert_called_with("cloud")

    @patch("agent.utils.get_llm")
    def test_override_local_light(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()
        llm, tier, cost = route_and_get_llm("test task", "research", override="local_light")
        assert tier == "local_light"
        mock_get_llm.assert_called_with("local_light")

    @patch("agent.utils.get_llm")
    def test_routed_cost_returned(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()
        _, _, cost = route_and_get_llm("find papers", "research")
        assert cost == 0.01  # cloud_standard

    @patch("agent.utils.get_llm")
    def test_local_tier_zero_cost(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()
        _, _, cost = route_and_get_llm("hello", "unknown_type")
        assert cost == 0.0

    @patch("agent.utils.get_llm")
    @patch("scms.client.SCMSClient")
    def test_budget_exhaustion_downgrades(self, mock_client_class, mock_get_llm):
        """When daily budget is exhausted, cloud tier should downgrade to local."""
        mock_get_llm.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.get_daily_spend.return_value = 10.0  # Over $5 budget
        mock_client_class.return_value = mock_client

        with patch("agent.model_router.settings") as mock_settings:
            mock_settings.daily_budget_usd = 5.0
            mock_settings.budget_warn_threshold = 0.8
            _, tier, cost = route_and_get_llm("find papers", "research")
            assert tier == "local"
            assert cost == 0.0
