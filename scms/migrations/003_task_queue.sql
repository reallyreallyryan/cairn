-- cairn Sprint 4: Task Queue Migration
-- Run this in your Supabase SQL Editor
-- NOTE: Also run 002_sandbox_metatool.sql first if you haven't already!

CREATE TABLE IF NOT EXISTS task_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task TEXT NOT NULL,
    project TEXT,
    priority INT NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    recurring TEXT,             -- cron expression, NULL = one-shot
    model_used TEXT,
    cost_usd NUMERIC(10,6) DEFAULT 0,
    result TEXT,
    error TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_task_queue_status ON task_queue (status);
CREATE INDEX idx_task_queue_priority ON task_queue (priority DESC);
CREATE INDEX idx_task_queue_created ON task_queue (created_at);

-- RLS: same open policy as other tables (personal project)
ALTER TABLE task_queue ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all for anon" ON task_queue FOR ALL USING (true) WITH CHECK (true);
