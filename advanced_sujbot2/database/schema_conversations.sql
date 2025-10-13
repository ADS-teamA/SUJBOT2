-- Conversation Persistence Schema for SUJBOT2
-- Stores chat conversations with messages (similar to ChatGPT)

-- ============================================================================
-- Conversations Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) DEFAULT 'New Conversation',

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Session context (optional - for analytics)
    session_id VARCHAR(255),

    -- Document context - which documents were used in this conversation
    document_ids TEXT[], -- Array of document IDs

    -- Flags
    is_archived BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,

    -- Statistics
    message_count INTEGER DEFAULT 0,

    CONSTRAINT title_not_empty CHECK (char_length(title) > 0)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_is_archived ON conversations(is_archived) WHERE is_archived = FALSE;

-- ============================================================================
-- Messages Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    -- Message content
    type VARCHAR(20) NOT NULL CHECK (type IN ('user', 'bot')),
    content TEXT NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Ordering (to maintain message sequence)
    sequence INTEGER NOT NULL,

    -- Optional: Intent routing information
    intent VARCHAR(100), -- e.g., 'compliance_check', 'simple_query'
    pipeline VARCHAR(100), -- e.g., 'compliance_analysis', 'simple_chat'

    -- Optional: Sources/citations (JSON)
    sources JSONB,

    -- Optional: Metadata (JSON) - for storing additional info
    metadata JSONB DEFAULT '{}'::jsonb,

    CONSTRAINT content_not_empty CHECK (char_length(content) > 0),
    CONSTRAINT sequence_non_negative CHECK (sequence >= 0)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_sequence ON messages(conversation_id, sequence);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);

-- ============================================================================
-- Triggers
-- ============================================================================

-- Update conversation.updated_at when a message is added
CREATE OR REPLACE FUNCTION update_conversation_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversations
    SET updated_at = CURRENT_TIMESTAMP,
        message_count = (
            SELECT COUNT(*)
            FROM messages
            WHERE conversation_id = NEW.conversation_id
        )
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_conversation_timestamp
AFTER INSERT ON messages
FOR EACH ROW
EXECUTE FUNCTION update_conversation_timestamp();

-- Auto-generate title from first user message
CREATE OR REPLACE FUNCTION auto_generate_conversation_title()
RETURNS TRIGGER AS $$
BEGIN
    -- Only generate title if it's still "New Conversation" and this is first user message
    IF NEW.type = 'user' AND NEW.sequence = 0 THEN
        UPDATE conversations
        SET title = CASE
            -- Truncate to 100 chars and add ellipsis if longer
            WHEN char_length(NEW.content) > 100 THEN
                substring(NEW.content, 1, 97) || '...'
            ELSE
                NEW.content
        END
        WHERE id = NEW.conversation_id
          AND title = 'New Conversation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_auto_generate_title
AFTER INSERT ON messages
FOR EACH ROW
EXECUTE FUNCTION auto_generate_conversation_title();

-- ============================================================================
-- Useful Views
-- ============================================================================

-- View for conversations with latest message preview
CREATE OR REPLACE VIEW conversation_list AS
SELECT
    c.id,
    c.title,
    c.created_at,
    c.updated_at,
    c.message_count,
    c.is_archived,
    c.is_favorite,
    c.document_ids,
    -- Latest message preview
    (
        SELECT content
        FROM messages
        WHERE conversation_id = c.id
        ORDER BY sequence DESC
        LIMIT 1
    ) AS latest_message_preview,
    (
        SELECT type
        FROM messages
        WHERE conversation_id = c.id
        ORDER BY sequence DESC
        LIMIT 1
    ) AS latest_message_type
FROM conversations c
ORDER BY c.updated_at DESC;

-- ============================================================================
-- Sample Queries
-- ============================================================================

-- Get all conversations (most recent first)
-- SELECT * FROM conversation_list WHERE is_archived = FALSE LIMIT 50;

-- Get full conversation with messages
-- SELECT c.*,
--        json_agg(
--            json_build_object(
--                'id', m.id,
--                'type', m.type,
--                'content', m.content,
--                'created_at', m.created_at,
--                'sequence', m.sequence,
--                'sources', m.sources
--            ) ORDER BY m.sequence
--        ) AS messages
-- FROM conversations c
-- LEFT JOIN messages m ON m.conversation_id = c.id
-- WHERE c.id = '...'
-- GROUP BY c.id;

-- Delete old archived conversations (cleanup)
-- DELETE FROM conversations WHERE is_archived = TRUE AND updated_at < NOW() - INTERVAL '90 days';
