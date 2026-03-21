"""Shared utilities for agent nodes."""

import logging
import re

from langchain_ollama import ChatOllama

from config.settings import settings

logger = logging.getLogger(__name__)


def clean_output(text: str) -> str:
    """Strip narrated tool call XML and other artifacts from LLM output."""
    # Remove <function_calls>...</function_calls> blocks
    text = re.sub(r"<function_calls>.*?</function_calls>", "", text, flags=re.DOTALL)
    # Remove <function_result>...</function_result> blocks
    text = re.sub(r"<function_result>.*?</function_result>", "", text, flags=re.DOTALL)
    # Remove any remaining XML-ish tags (but keep markdown)
    text = re.sub(r"</?(?:tool_call|tool_result|search_results?|results?)>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_llm(model: str | None = None):
    """Create the appropriate LLM based on configuration."""
    use_cloud = (model == "cloud") or (
        model is None and settings.agent_model == "cloud"
    )

    if use_cloud:
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            logger.warning("No ANTHROPIC_API_KEY set, falling back to local Ollama model")
        else:
            logger.info("Using cloud model: claude-sonnet-4-20250514")
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                api_key=settings.anthropic_api_key,
            )

    if model == "local_light":
        logger.info("Using local-light model: %s", settings.ollama_model_light)
        return ChatOllama(
            model=settings.ollama_model_light,
            base_url=settings.ollama_base_url,
        )

    logger.info("Using local model: %s", settings.ollama_model)
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )
