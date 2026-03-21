-- 004b: Finalize embedding dimension migration (768 → 1536)
-- Run AFTER the re-embedding script has updated all rows.

-- Constrain column to 1536 dimensions
ALTER TABLE memories
  ALTER COLUMN embedding TYPE vector(1536);

-- Recreate HNSW index for cosine similarity search
CREATE INDEX idx_memories_embedding
  ON memories USING hnsw (embedding vector_cosine_ops);
