-- ============================================================================
-- SUJBOT2 Database Initialization Script
-- ============================================================================
-- Run this script to initialize the PostgreSQL database for SUJBOT2
-- Prerequisites:
-- 1. PostgreSQL 15+ installed
-- 2. pgvector extension available
-- ============================================================================

-- Create database (run as postgres superuser)
-- Uncomment if creating database for the first time:
-- CREATE DATABASE sujbot2 WITH ENCODING 'UTF8' LC_COLLATE='cs_CZ.UTF-8' LC_CTYPE='cs_CZ.UTF-8';

-- Connect to database
\c sujbot2

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create application role
CREATE ROLE sujbot_app WITH LOGIN PASSWORD 'change_this_password';

-- Grant permissions
GRANT CONNECT ON DATABASE sujbot2 TO sujbot_app;
GRANT USAGE ON SCHEMA public TO sujbot_app;

-- Load main schema
\i schema.sql

-- Grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sujbot_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sujbot_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO sujbot_app;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO sujbot_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO sujbot_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT EXECUTE ON FUNCTIONS TO sujbot_app;

-- Verify installation
SELECT
    extname AS extension,
    extversion AS version
FROM pg_extension
WHERE extname IN ('vector', 'pg_trgm', 'btree_gin');

-- Show table summary
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

\echo '============================================'
\echo 'Database initialization complete!'
\echo 'Application user: sujbot_app'
\echo 'IMPORTANT: Change the default password!'
\echo '============================================'
