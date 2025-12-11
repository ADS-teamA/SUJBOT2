-- ============================================================================
-- SUJBOT2 User Spending Tracking Schema
-- ============================================================================
-- This script adds spending tracking columns to auth.users:
-- 1. spending_limit_czk - Admin-configurable spending limit per user
-- 2. total_spent_czk - Running total of user's spending
-- 3. spending_reset_at - Timestamp for future monthly reset feature
--
-- Security features:
-- - Atomic updates to prevent race conditions
-- - Index for quick spending limit checks
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- ============================================================================
-- ADD SPENDING COLUMNS TO AUTH.USERS
-- ============================================================================

-- Add spending_limit_czk column (default 500 CZK)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'auth'
        AND table_name = 'users'
        AND column_name = 'spending_limit_czk'
    ) THEN
        ALTER TABLE auth.users
        ADD COLUMN spending_limit_czk DECIMAL(10,2) DEFAULT 500.00;
        RAISE NOTICE 'Added spending_limit_czk column to auth.users';
    ELSE
        RAISE NOTICE 'spending_limit_czk column already exists';
    END IF;
END $$;

-- Add total_spent_czk column (default 0)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'auth'
        AND table_name = 'users'
        AND column_name = 'total_spent_czk'
    ) THEN
        ALTER TABLE auth.users
        ADD COLUMN total_spent_czk DECIMAL(10,2) DEFAULT 0.00;
        RAISE NOTICE 'Added total_spent_czk column to auth.users';
    ELSE
        RAISE NOTICE 'total_spent_czk column already exists';
    END IF;
END $$;

-- Add spending_reset_at column (for future monthly reset feature)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'auth'
        AND table_name = 'users'
        AND column_name = 'spending_reset_at'
    ) THEN
        ALTER TABLE auth.users
        ADD COLUMN spending_reset_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        RAISE NOTICE 'Added spending_reset_at column to auth.users';
    ELSE
        RAISE NOTICE 'spending_reset_at column already exists';
    END IF;
END $$;

-- ============================================================================
-- INDEX FOR SPENDING CHECKS
-- ============================================================================

-- Index for quick spending limit queries
CREATE INDEX IF NOT EXISTS idx_users_spending
ON auth.users(id, total_spent_czk, spending_limit_czk);

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON COLUMN auth.users.spending_limit_czk IS
    'Maximum spending allowed for user in CZK (default 500). Set by admin.';
COMMENT ON COLUMN auth.users.total_spent_czk IS
    'Running total of user spending in CZK. Updated after each chat message.';
COMMENT ON COLUMN auth.users.spending_reset_at IS
    'Timestamp when spending was last reset. For future monthly reset feature.';

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================

-- Verify columns were added successfully
DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_schema = 'auth'
    AND table_name = 'users'
    AND column_name IN ('spending_limit_czk', 'total_spent_czk', 'spending_reset_at');

    IF col_count = 3 THEN
        RAISE NOTICE 'SUCCESS: All spending tracking columns present in auth.users';
    ELSE
        RAISE WARNING 'WARNING: Expected 3 spending columns, found %', col_count;
    END IF;
END $$;
