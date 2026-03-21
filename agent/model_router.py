"""Model router — routes tasks to local or cloud LLMs based on complexity and budget."""

import logging
from pathlib import Path

import yaml

from config.settings import settings

logger = logging.getLogger(__name__)

_config_cache: dict | None = None


def _load_config() -> dict:
    """Load model routing config (cached)."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    path = Path(settings.model_routing_config)
    if not path.exists():
        logger.warning("Model routing config not found at %s, using defaults", path)
        _config_cache = {
            "tiers": {"local": {"model": "local", "cost_per_call": 0.0}},
            "rules": [],
            "default_tier": "local",
            "fallback": {"local": "cloud_standard"},
        }
        return _config_cache

    with open(path) as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache


def classify_complexity(task: str, task_type: str) -> str:
    """Determine which model tier to use based on task type and keywords."""
    config = _load_config()
    task_lower = task.lower()

    for rule in config.get("rules", []):
        match = rule.get("match", {})
        type_match = task_type in match.get("task_types", [])
        keyword_match = not match.get("keywords") or any(
            kw in task_lower for kw in match.get("keywords", [])
        )
        if type_match and keyword_match:
            tier = rule["tier"]
            logger.info("Route rule '%s' matched → tier '%s'", rule["name"], tier)
            return tier

    default = config.get("default_tier", "local")
    logger.info("No routing rule matched → default tier '%s'", default)
    return default


def route_and_get_llm(
    task: str,
    task_type: str,
    override: str | None = None,
) -> tuple:
    """Route task to appropriate LLM and return (llm, tier_name, cost_per_call).

    Args:
        task: The task text.
        task_type: Classification from classifier (research, technical, etc.)
        override: If set ("local" or "cloud"), bypasses routing rules.

    Returns:
        Tuple of (llm_instance, tier_name, cost_per_call)
    """
    from agent.utils import get_llm

    config = _load_config()
    tiers = config.get("tiers", {})

    # CLI override bypasses routing
    if override:
        override_map = {"local": "local", "local_light": "local_light", "cloud": "cloud_standard"}
        tier_name = override_map.get(override, "cloud_standard")
        tier_config = tiers.get(tier_name, {"model": override, "cost_per_call": 0.0})
        llm = get_llm(override)
        return llm, tier_name, tier_config.get("cost_per_call", 0.0)

    # Route based on task complexity
    tier_name = classify_complexity(task, task_type)
    tier_config = tiers.get(tier_name, {"model": "local", "cost_per_call": 0.0})

    # Budget check for cloud tiers
    if tier_config.get("cost_per_call", 0) > 0:
        try:
            from scms.client import SCMSClient
            client = SCMSClient()
            daily_spend = client.get_daily_spend()
            budget = settings.daily_budget_usd
            warn_at = budget * settings.budget_warn_threshold

            if daily_spend >= budget:
                logger.warning(
                    "Daily budget exhausted ($%.2f/$%.2f), downgrading to local",
                    daily_spend, budget,
                )
                tier_name = "local"
                tier_config = tiers.get("local", {"model": "local", "cost_per_call": 0.0})
            elif daily_spend >= warn_at:
                logger.warning(
                    "Daily budget at %.0f%% ($%.2f/$%.2f)",
                    (daily_spend / budget) * 100, daily_spend, budget,
                )
        except Exception as e:
            logger.debug("Budget check failed (non-critical): %s", e)

    model_key = tier_config.get("model", "local")
    llm = get_llm(model_key)
    cost = tier_config.get("cost_per_call", 0.0)

    logger.info("Routed to tier '%s' (model=%s, cost=$%.4f)", tier_name, model_key, cost)
    return llm, tier_name, cost
