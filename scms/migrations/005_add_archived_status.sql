-- 005: Add 'archived' as a valid project status
-- Required by archive_project() in scms/client.py

ALTER TABLE projects DROP CONSTRAINT IF EXISTS projects_status_check;
ALTER TABLE projects ADD CONSTRAINT projects_status_check
    CHECK (status IN ('active', 'paused', 'completed', 'idea', 'archived'));
