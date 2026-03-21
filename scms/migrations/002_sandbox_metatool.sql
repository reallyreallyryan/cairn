-- cairn Sprint 3: Sandbox & Metatooling Migration
-- Run this in your Supabase SQL Editor

-- Add approval workflow columns to tool_registry
ALTER TABLE tool_registry
    ADD COLUMN IF NOT EXISTS approval_status TEXT NOT NULL DEFAULT 'approved'
        CHECK (approval_status IN ('approved', 'pending', 'rejected')),
    ADD COLUMN IF NOT EXISTS source_code TEXT,
    ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS approved_by TEXT;

-- Ensure existing builtin tools are marked approved
UPDATE tool_registry SET approval_status = 'approved' WHERE tool_type = 'builtin';

-- Index for quick pending-tool queries
CREATE INDEX IF NOT EXISTS idx_tool_registry_approval ON tool_registry (approval_status);
