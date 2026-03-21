"""Embedding generation for SCMS.

Supports Ollama (local, default) and OpenAI as providers.
"""

import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    if settings.embedding_provider == "ollama":
        return _ollama_embed(text)
    elif settings.embedding_provider == "openai":
        return _openai_embed(text)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    if settings.embedding_provider == "ollama":
        return _ollama_embed_batch(texts)
    elif settings.embedding_provider == "openai":
        return _openai_embed_batch(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def _ollama_embed(text: str) -> list[float]:
    """Generate embedding via Ollama API."""
    result = _ollama_embed_batch([text])
    return result[0]


def _ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via Ollama API (batch)."""
    url = f"{settings.ollama_base_url}/api/embed"
    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Is Ollama running? Try: ollama serve"
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Embedding model '{settings.embedding_model}' not found. "
                f"Pull it with: ollama pull {settings.embedding_model}"
            )
        raise

    data = response.json()
    embeddings = data.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(f"No embeddings returned from Ollama for model {settings.embedding_model}")

    # Validate dimensions
    for emb in embeddings:
        if len(emb) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(emb)}, "
                f"expected {settings.embedding_dim}. "
                f"Check EMBEDDING_DIM setting matches your model."
            )

    return embeddings


def _openai_embed(text: str) -> list[float]:
    """Generate embedding via OpenAI API."""
    return _openai_embed_batch([text])[0]


def _openai_embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via OpenAI API (batch — sends all texts in one call)."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    response.raise_for_status()

    data = response.json()
    # Sort by index to preserve input order
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    embeddings = [item["embedding"] for item in sorted_data]

    for emb in embeddings:
        if len(emb) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(emb)}, "
                f"expected {settings.embedding_dim}"
            )

    return embeddings
