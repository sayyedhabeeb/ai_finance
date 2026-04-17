"""
PostgreSQL Schema Initialization.

Creates all tables, indexes, and TimescaleDB hypertables for the
AI Financial Brain platform. Designed to be idempotent (safe to re-run).
"""

import logging

logger = logging.getLogger(__name__)


async def init_database(connection):
    """
    Initialize the full database schema. Accepts an asyncpg connection.

    This function is idempotent: all statements use CREATE TABLE IF NOT EXISTS
    and CREATE INDEX IF NOT EXISTS so they can safely be re-run.
    """
    logger.info("Initializing database schema …")

    # ------------------------------------------------------------------
    # 1. Users
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              TEXT PRIMARY KEY,
            email           TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            phone           TEXT,
            risk_profile    TEXT NOT NULL DEFAULT 'moderate'
                            CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive', 'very_aggressive')),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
    """)

    # ------------------------------------------------------------------
    # 2. Portfolios
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            name            TEXT NOT NULL DEFAULT 'Default Portfolio',
            currency        TEXT NOT NULL DEFAULT 'USD' CHECK (length(currency) = 3),
            total_value     DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios (user_id);
    """)

    # ------------------------------------------------------------------
    # 3. Holdings
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            portfolio_id    UUID NOT NULL REFERENCES portfolios (id) ON DELETE CASCADE,
            symbol          TEXT NOT NULL,
            quantity        DECIMAL(18, 6) NOT NULL,
            avg_cost        DECIMAL(18, 4) NOT NULL,
            current_value   DECIMAL(18, 2) DEFAULT NULL,
            added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_holdings_portfolio_symbol UNIQUE (portfolio_id, symbol)
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_holdings_portfolio_id ON holdings (portfolio_id);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_holdings_symbol ON holdings (symbol);
    """)

    # ------------------------------------------------------------------
    # 4. Transactions
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            portfolio_id    UUID NOT NULL REFERENCES portfolios (id) ON DELETE CASCADE,
            symbol          TEXT NOT NULL,
            type            TEXT NOT NULL CHECK (type IN ('buy', 'sell', 'dividend', 'split', 'transfer_in', 'transfer_out')),
            quantity        DECIMAL(18, 6) NOT NULL,
            price           DECIMAL(18, 4) NOT NULL,
            fees            DECIMAL(10, 4) NOT NULL DEFAULT 0.00,
            date            DATE NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_id ON transactions (portfolio_id);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions (symbol);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions (date DESC);
    """)

    # ------------------------------------------------------------------
    # 5. Financial Goals
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS financial_goals (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            goal_type       TEXT NOT NULL CHECK (goal_type IN (
                'retirement', 'house', 'education', 'emergency_fund',
                'wealth_building', 'income', 'custom'
            )),
            target_amount   DECIMAL(18, 2) NOT NULL,
            current_amount  DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
            target_date     DATE,
            priority        INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 1 AND 10),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_financial_goals_user_id ON financial_goals (user_id);
    """)

    # ------------------------------------------------------------------
    # 6. Market Data (TimescaleDB hypertable)
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp       TIMESTAMPTZ NOT NULL,
            symbol          TEXT NOT NULL,
            open            DECIMAL(18, 4) NOT NULL,
            high            DECIMAL(18, 4) NOT NULL,
            low             DECIMAL(18, 4) NOT NULL,
            close           DECIMAL(18, 4) NOT NULL,
            volume          BIGINT NOT NULL DEFAULT 0
        );
    """)

    # Attempt to create TimescaleDB hypertable — graceful fallback
    try:
        await connection.execute("""
            SELECT create_hypertable('market_data', 'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE
            );
        """)
        logger.info("TimescaleDB hypertable created for market_data")
    except Exception as exc:
        logger.warning(
            "TimescaleDB hypertable creation failed (non-critical, table still usable): %s",
            exc,
        )

    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, timestamp DESC);
    """)
    await connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_ts ON market_data (symbol, timestamp);
    """)

    # ------------------------------------------------------------------
    # 7. News Articles
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title           TEXT NOT NULL,
            content         TEXT,
            source          TEXT NOT NULL,
            published_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            tickers         TEXT[] NOT NULL DEFAULT '{}',
            sentiment_score DECIMAL(5, 4),
            embedding_id    TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_articles_published ON news_articles (published_at DESC);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_articles_tickers ON news_articles USING GIN (tickers);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment ON news_articles (sentiment_score);
    """)

    # ------------------------------------------------------------------
    # 8. Agent Executions (query audit log)
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS agent_executions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         TEXT NOT NULL,
            query_id        TEXT NOT NULL,
            agent_type      TEXT NOT NULL,
            input_data      JSONB NOT NULL DEFAULT '{}',
            output          JSONB,
            latency_ms      DECIMAL(10, 2),
            status          TEXT NOT NULL DEFAULT 'completed'
                            CHECK (status IN ('pending', 'running', 'completed', 'failed')),
            error_message   TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_executions_user_id ON agent_executions (user_id, created_at DESC);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_executions_query_id ON agent_executions (query_id);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_executions_agent_type ON agent_executions (agent_type);
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_executions_created ON agent_executions (created_at DESC);
    """)

    # ------------------------------------------------------------------
    # 9. User Preferences
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            preference_key  TEXT NOT NULL,
            preference_value JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_user_preferences UNIQUE (user_id, preference_key)
        );
    """)
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences (user_id);
    """)

    # ------------------------------------------------------------------
    # 10. Updated-at trigger function (applies to any table with updated_at)
    # ------------------------------------------------------------------
    await connection.execute("""
        CREATE OR REPLACE FUNCTION trigger_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Apply triggers
    for table in ("users", "portfolios", "financial_goals", "user_preferences"):
        trigger_name = f" trg_{table}_updated_at"
        await connection.execute(f"""
            DO $$ BEGIN
                CREATE TRIGGER {trigger_name}
                    BEFORE UPDATE ON {table}
                    FOR EACH ROW
                    EXECUTE FUNCTION trigger_set_updated_at();
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$;
        """)

    # ------------------------------------------------------------------
    # 11. Continuous aggregates for market_data (TimescaleDB feature)
    # ------------------------------------------------------------------
    try:
        await connection.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', timestamp) AS bucket,
                symbol,
                FIRST(open, timestamp)   AS open,
                MAX(high)                AS high,
                MIN(low)                 AS low,
                LAST(close, timestamp)   AS close,
                SUM(volume)              AS volume
            FROM market_data
            GROUP BY bucket, symbol
            WITH NO DATA;
        """)
        await connection.execute("""
            SELECT add_continuous_aggregate_policy('market_data_daily',
                start_offset => INTERVAL '3 days',
                end_offset   => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
        logger.info("TimescaleDB continuous aggregate created for market_data_daily")
    except Exception as exc:
        logger.warning("Continuous aggregate creation failed (non-critical): %s", exc)

    logger.info("Database schema initialization complete")


async def seed_demo_data(connection):
    """
    Insert seed data for development / demo purposes.
    Safe to call multiple times — uses ON CONFLICT where applicable.
    """
    logger.info("Seeding demo data …")

    # Demo users
    await connection.execute("""
        INSERT INTO users (id, email, name, phone, risk_profile)
        VALUES
            ('usr_demo_001', 'alice@example.com', 'Alice Chen', '+1-555-0101', 'moderate'),
            ('usr_demo_002', 'bob@example.com',   'Bob Martinez', '+1-555-0102', 'aggressive'),
            ('usr_demo_003', 'carol@example.com',  'Carol Johnson', '+1-555-0103', 'conservative')
        ON CONFLICT (email) DO NOTHING;
    """)

    # Demo portfolio
    await connection.execute("""
        INSERT INTO portfolios (id, user_id, name, currency, total_value)
        VALUES
            ('a0000000-0000-0000-0000-000000000001', 'usr_demo_001', 'Alice Growth Portfolio', 'USD', 150000.00),
            ('a0000000-0000-0000-0000-000000000002', 'usr_demo_002', 'Bob Aggressive Portfolio', 'USD', 285000.00)
        ON CONFLICT DO NOTHING;
    """)

    # Demo holdings
    await connection.execute("""
        INSERT INTO holdings (portfolio_id, symbol, quantity, avg_cost, current_value)
        VALUES
            ('a0000000-0000-0000-0000-000000000001', 'AAPL', 50,  150.00, 9250.00),
            ('a0000000-0000-0000-0000-000000000001', 'MSFT', 30,  280.00, 9750.00),
            ('a0000000-0000-0000-0000-000000000001', 'GOOGL', 20, 135.00, 5520.00),
            ('a0000000-0000-0000-0000-000000000001', 'NVDA', 15,  450.00, 18825.00),
            ('a0000000-0000-0000-0000-000000000001', 'VTI',  100, 220.00, 24600.00),
            ('a0000000-0000-0000-0000-000000000001', 'BND',  200,  72.00, 14400.00),
            ('a0000000-0000-0000-0000-000000000002', 'TSLA', 40,  200.00, 36400.00),
            ('a0000000-0000-0000-0000-000000000002', 'NVDA', 50,  400.00, 62750.00),
            ('a0000000-0000-0000-0000-000000000002', 'AMD',  100, 120.00, 33000.00),
            ('a0000000-0000-0000-0000-000000000002', 'COIN', 25,  250.00, 12500.00),
            ('a0000000-0000-0000-0000-000000000002', 'QQQ',  150, 370.00, 83250.00)
        ON CONFLICT (portfolio_id, symbol) DO NOTHING;
    """)

    # Demo financial goals
    await connection.execute("""
        INSERT INTO financial_goals (user_id, goal_type, target_amount, current_amount, target_date, priority)
        VALUES
            ('usr_demo_001', 'retirement',       2000000.00, 350000.00, '2050-01-01', 10),
            ('usr_demo_001', 'house',            100000.00,  45000.00,  '2027-06-01', 8),
            ('usr_demo_001', 'emergency_fund',    25000.00,  18000.00,  NULL,          9),
            ('usr_demo_002', 'wealth_building', 5000000.00, 285000.00, '2045-01-01', 10),
            ('usr_demo_002', 'education',        200000.00,  15000.00,  '2030-09-01', 6)
        ON CONFLICT DO NOTHING;
    """)

    # Demo preferences
    await connection.execute("""
        INSERT INTO user_preferences (user_id, preference_key, preference_value)
        VALUES
            ('usr_demo_001', 'risk_tolerance', '"moderate"'),
            ('usr_demo_001', 'preferred_sectors', '["technology", "healthcare", "consumer_staples"]'),
            ('usr_demo_001', 'currency', '"USD"'),
            ('usr_demo_001', 'communication_style', '"detailed"'),
            ('usr_demo_002', 'risk_tolerance', '"aggressive"'),
            ('usr_demo_002', 'preferred_sectors', '["technology", "consumer_discretionary", "crypto"]'),
            ('usr_demo_002', 'currency', '"USD"'),
            ('usr_demo_002', 'communication_style', '"concise"')
        ON CONFLICT (user_id, preference_key) DO NOTHING;
    """)

    logger.info("Demo data seeded successfully")
