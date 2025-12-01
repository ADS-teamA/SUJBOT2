-- Migration: Add 'cheap' to agent_variant allowed values
-- Run this against an existing database to add support for the new 'cheap' variant.
--
-- This migration:
-- 1. Drops the existing CHECK constraint on agent_variant
-- 2. Adds a new CHECK constraint that allows 'premium', 'cheap', and 'local'
-- 3. Updates the default value to 'cheap' for new users
--
-- Usage:
--   psql -h localhost -U postgres -d sujbot -f scripts/migrate_add_cheap_variant.sql

BEGIN;

-- Drop existing constraint
ALTER TABLE auth.users DROP CONSTRAINT IF EXISTS users_agent_variant_check;

-- Add new constraint with 'cheap' included
ALTER TABLE auth.users ADD CONSTRAINT users_agent_variant_check
    CHECK (agent_variant IN ('premium', 'cheap', 'local'));

-- Update default to 'cheap' for new users
ALTER TABLE auth.users ALTER COLUMN agent_variant SET DEFAULT 'cheap';

-- Optionally migrate existing 'premium' users to 'cheap' if desired
-- Uncomment the following line to do so:
-- UPDATE auth.users SET agent_variant = 'cheap' WHERE agent_variant = 'premium';

COMMIT;

-- Verify the change
SELECT
    column_name,
    column_default,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'auth'
  AND table_name = 'users'
  AND column_name = 'agent_variant';
