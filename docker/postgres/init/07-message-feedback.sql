-- ============================================================================
-- SUJBOT2 Message Feedback Schema
-- ============================================================================
-- This script creates the message_feedback table for user ratings:
-- 1. Thumbs up/down ratings on assistant messages
-- 2. Optional comments for feedback
-- 3. LangSmith trace ID correlation for debugging
--
-- Features:
-- - Unique constraint prevents duplicate feedback per user/message
-- - Cascade delete when message or user is deleted
-- - LangSmith sync tracking for retry logic
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- ============================================================================
-- MESSAGE FEEDBACK TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS auth.message_feedback (
    id SERIAL PRIMARY KEY,

    -- Foreign keys
    message_id INTEGER NOT NULL REFERENCES auth.messages(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- LangSmith correlation
    run_id TEXT,  -- LangSmith trace ID (nullable for old messages)

    -- Feedback data
    score INTEGER NOT NULL CHECK (score IN (-1, 1)),  -- -1=thumbs down, 1=thumbs up
    comment TEXT,  -- Optional user comment

    -- Sync status
    langsmith_synced BOOLEAN DEFAULT FALSE,  -- Track if feedback was sent to LangSmith

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Prevent duplicate feedback per message per user
    UNIQUE(message_id, user_id)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Fast lookup by message ID (for checking existing feedback)
CREATE INDEX IF NOT EXISTS idx_message_feedback_message_id
ON auth.message_feedback(message_id);

-- Analytics queries by score
CREATE INDEX IF NOT EXISTS idx_message_feedback_score
ON auth.message_feedback(score);

-- LangSmith reconciliation queries
CREATE INDEX IF NOT EXISTS idx_message_feedback_run_id
ON auth.message_feedback(run_id) WHERE run_id IS NOT NULL;

-- Find unsynced feedback for retry
CREATE INDEX IF NOT EXISTS idx_message_feedback_unsynced
ON auth.message_feedback(langsmith_synced) WHERE langsmith_synced = FALSE;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE auth.message_feedback IS
    'User feedback (thumbs up/down) on assistant messages for LangSmith integration';
COMMENT ON COLUMN auth.message_feedback.message_id IS
    'Reference to auth.messages(id) - the rated message';
COMMENT ON COLUMN auth.message_feedback.user_id IS
    'Reference to auth.users(id) - who submitted the feedback';
COMMENT ON COLUMN auth.message_feedback.run_id IS
    'LangSmith trace/run ID for correlation. NULL for messages before feature was added';
COMMENT ON COLUMN auth.message_feedback.score IS
    '1 = thumbs up (helpful), -1 = thumbs down (not helpful)';
COMMENT ON COLUMN auth.message_feedback.comment IS
    'Optional user comment explaining the rating';
COMMENT ON COLUMN auth.message_feedback.langsmith_synced IS
    'TRUE if feedback was successfully sent to LangSmith';

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================

DO $$
DECLARE
    table_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'auth' AND table_name = 'message_feedback'
    ) INTO table_exists;

    IF table_exists THEN
        RAISE NOTICE 'SUCCESS: auth.message_feedback table created successfully';
    ELSE
        RAISE WARNING 'WARNING: auth.message_feedback table was not created';
    END IF;
END $$;
