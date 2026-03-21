"""Re-embed all memories using the current embedding provider.

Usage: uv run python scripts/migrate_embeddings.py

Prerequisites:
  1. Update .env with OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_PROVIDER
  2. Run 004a_embedding_pre_migrate.sql in Supabase SQL Editor
"""

import sys
import time
import logging
from pathlib import Path

# Ensure project root is on sys.path (project is not installed as a package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supabase import create_client

from config.settings import settings
from scms.embeddings import get_embeddings_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_RETRIES = 3


def main() -> None:
    client = create_client(settings.supabase_url, settings.supabase_key)

    # Fetch all memories
    logger.info("Fetching all memories...")
    result = client.table("memories").select("id, content").order("created_at").execute()
    rows = result.data

    if not rows:
        logger.info("No memories to migrate.")
        return

    logger.info("Found %d memories to re-embed with %s", len(rows), settings.embedding_model)

    success = 0
    failed = []

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [r["content"] for r in batch]
        ids = [r["id"] for r in batch]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                embeddings = get_embeddings_batch(texts)
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    logger.error("Failed batch starting at %d after %d attempts: %s", i, MAX_RETRIES, e)
                    failed.extend(ids)
                    embeddings = None
                    break
                wait = 2 ** attempt
                logger.warning("Attempt %d failed, retrying in %ds: %s", attempt, wait, e)
                time.sleep(wait)

        if embeddings is None:
            continue

        # Update each row
        for row_id, embedding in zip(ids, embeddings):
            client.table("memories").update({"embedding": embedding}).eq("id", row_id).execute()

        success += len(batch)
        logger.info("Progress: %d/%d migrated", success, len(rows))

        # Brief pause between batches to respect rate limits
        if i + BATCH_SIZE < len(rows):
            time.sleep(1)

    logger.info("Migration complete: %d succeeded, %d failed", success, len(failed))
    if failed:
        logger.error("Failed IDs: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
