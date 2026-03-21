-- cairn SCMS: Initial Schema Migration
-- Run this in your Supabase SQL Editor (Dashboard > SQL Editor > New query)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Projects table
-- ============================================================
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'idea')),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_projects_name ON projects (name);
CREATE INDEX idx_projects_status ON projects (status);

-- ============================================================
-- Memories table (core knowledge store)
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL
        CHECK (memory_type IN (
            'concept', 'pattern', 'decision', 'tool_eval',
            'problem_solution', 'reference', 'context', 'learning'
        )),
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    embedding vector(768) NOT NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    source TEXT DEFAULT 'manual',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast cosine similarity search (better than IVFFlat for small datasets)
CREATE INDEX idx_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_memories_type ON memories (memory_type);
CREATE INDEX idx_memories_project ON memories (project_id);
CREATE INDEX idx_memories_tags ON memories USING gin (tags);

-- ============================================================
-- Tool registry
-- ============================================================
CREATE TABLE IF NOT EXISTS tool_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    tool_type TEXT NOT NULL DEFAULT 'builtin'
        CHECK (tool_type IN ('builtin', 'custom', 'metatool_generated')),
    function_name TEXT,
    config JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tool_registry_name ON tool_registry (name);

-- ============================================================
-- Decision log
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision TEXT NOT NULL,
    reasoning TEXT,
    alternatives TEXT[] NOT NULL DEFAULT '{}',
    outcome TEXT,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    context JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_decision_log_project ON decision_log (project_id);

-- ============================================================
-- Semantic search function
-- ============================================================
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5,
    filter_project_id UUID DEFAULT NULL,
    filter_memory_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    memory_type TEXT,
    project_id UUID,
    tags TEXT[],
    source TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.memory_type,
        m.project_id,
        m.tags,
        m.source,
        m.metadata,
        (1 - (m.embedding <=> query_embedding))::FLOAT AS similarity
    FROM memories m
    WHERE (1 - (m.embedding <=> query_embedding)) > match_threshold
      AND (filter_project_id IS NULL OR m.project_id = filter_project_id)
      AND (filter_memory_type IS NULL OR m.memory_type = filter_memory_type)
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- Auto-update updated_at trigger
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Seed data: Projects
-- ============================================================
INSERT INTO projects (name, description, status, metadata) VALUES
(
    'cairn',
    'The cairn agent project itself — self-referential development',
    'active',
    '{"stack": ["LangGraph", "Supabase", "pgvector", "Docker"], "goals": ["Expand tool library", "Improve routing accuracy"]}'::jsonb
),
(
    'SideProject',
    'Example side project for testing multi-project memory isolation',
    'active',
    '{"stack": ["Python", "FastAPI"], "goals": ["Build MVP", "Launch beta"]}'::jsonb
),
(
    'Research',
    'Ongoing learning and research notes',
    'active',
    '{"stack": ["Papers", "Tutorials", "Experiments"], "goals": ["Stay current on agentic AI", "Evaluate new models"]}'::jsonb
);

-- ============================================================
-- Row Level Security (basic — allow all for service role)
-- ============================================================
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE decision_log ENABLE ROW LEVEL SECURITY;

-- Allow full access via anon key (personal project, single user)
CREATE POLICY "Allow all for anon" ON projects FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for anon" ON memories FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for anon" ON tool_registry FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for anon" ON decision_log FOR ALL USING (true) WITH CHECK (true);
