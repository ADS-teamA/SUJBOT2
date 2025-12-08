-- ============================================================================
-- Add Admin Role System
-- ============================================================================
-- Adds is_admin column to users table for role-based access control
-- Used to protect /auth/register endpoint (admin-only user creation)
-- ============================================================================

-- Ensure we're in the correct database
\c sujbot

-- Add is_admin column (defaults to false for security)
ALTER TABLE auth.users
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;

-- Create index for admin lookups (used in authorization checks)
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON auth.users(is_admin) WHERE is_admin = true;

-- Update default admin user to have admin privileges
UPDATE auth.users
SET is_admin = true
WHERE email = 'admin@sujbot.local';

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON COLUMN auth.users.is_admin IS 'Admin role flag for user management and registration permissions';
