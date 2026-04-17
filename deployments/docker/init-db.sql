-- =============================================================================
-- AI Financial Brain - Database Initialization
-- Run on first PostgreSQL container start
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create additional databases needed by services
-- Note: POSTGRES_DB creates aifb_db by default
SELECT 'CREATE DATABASE aifb_mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aifb_mlflow')\gexec
SELECT 'CREATE DATABASE aifb_prefect'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aifb_prefect')\gexec

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE aifb_db TO aifb_user;
GRANT ALL PRIVILEGES ON DATABASE aifb_mlflow TO aifb_user;
GRANT ALL PRIVILEGES ON DATABASE aifb_prefect TO aifb_user;

-- Print initialization complete
DO $$
BEGIN
    RAISE NOTICE 'AI Financial Brain database initialization complete';
END
$$;
