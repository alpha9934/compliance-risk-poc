"""
Creates all required tables in Neon Postgres.
Run once after setting DATABASE_URL_SYNC in .env

Usage:
    python -m scripts.init_db
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
import psycopg2

DDL = """
CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    transaction_id  TEXT UNIQUE NOT NULL,
    customer_id     TEXT NOT NULL,
    amount          NUMERIC(18,2),
    currency        CHAR(3),
    channel         TEXT,
    origin_account  TEXT,
    dest_account    TEXT,
    jurisdiction_origin TEXT,
    jurisdiction_dest   TEXT,
    product_type    TEXT,
    timestamp       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id              SERIAL PRIMARY KEY,
    transaction_id  TEXT REFERENCES transactions(transaction_id),
    risk_score      NUMERIC(5,4),
    risk_class      TEXT,
    model_version   TEXT,
    shap_values     JSONB,
    llm_explanation TEXT,
    llm_reason_codes TEXT[],
    policy_citations JSONB,
    judge_recommendation TEXT,
    judge_narrative TEXT,
    scored_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cases (
    id              SERIAL PRIMARY KEY,
    case_id         TEXT UNIQUE NOT NULL,
    transaction_id  TEXT REFERENCES transactions(transaction_id),
    status          TEXT DEFAULT 'UNDER_REVIEW',
    risk_class      TEXT,
    assigned_to     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reviewer_actions (
    id              SERIAL PRIMARY KEY,
    case_id         TEXT REFERENCES cases(case_id),
    reviewer_id     TEXT NOT NULL,
    action          TEXT NOT NULL,
    reason_code     TEXT NOT NULL,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_events (
    id              SERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,
    payload         JSONB,
    event_hash      TEXT UNIQUE NOT NULL,
    prev_hash       TEXT,
    session_id      TEXT,
    caller_id       TEXT,
    timestamp       TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_scores_txn ON risk_scores(transaction_id);
"""


def init():
    if not settings.database_url_sync:
        print("ERROR: DATABASE_URL_SYNC not set in .env")
        sys.exit(1)
    print("Connecting to Neon Postgres...")
    conn = psycopg2.connect(settings.database_url_sync)
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()
    cur.close()
    conn.close()
    print("All tables created successfully.")


if __name__ == "__main__":
    init()
