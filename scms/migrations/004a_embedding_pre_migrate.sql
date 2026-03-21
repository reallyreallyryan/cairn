-- 004a: Prepare for embedding dimension migration (768 → 1536)
-- Run BEFORE the re-embedding script.

-- Drop the HNSW index (can't alter dimension with index present)
DROP INDEX IF EXISTS idx_memories_embedding;

-- Remove dimension constraint so we can write 1536-dim vectors
-- while old 768-dim data still exists
ALTER TABLE memories
  ALTER COLUMN embedding TYPE vector;

-- Recreate match_memories() with 1536-dim parameter signature
DROP FUNCTION IF EXISTS match_memories;

CREATE OR REPLACE FUNCTION match_memories(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 5,
  filter_project_id uuid DEFAULT NULL,
  filter_memory_type text DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  content text,
  memory_type text,
  tags text[],
  source text,
  metadata jsonb,
  project_id uuid,
  created_at timestamptz,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id,
    m.content,
    m.memory_type,
    m.tags,
    m.source,
    m.metadata,
    m.project_id,
    m.created_at,
    1 - (m.embedding <=> query_embedding) AS similarity
  FROM memories m
  WHERE
    (filter_project_id IS NULL OR m.project_id = filter_project_id)
    AND (filter_memory_type IS NULL OR m.memory_type = filter_memory_type)
    AND 1 - (m.embedding <=> query_embedding) > match_threshold
  ORDER BY m.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
