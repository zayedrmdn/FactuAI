-- Add users table for authentication
-- Idempotent migration

CREATE TABLE IF NOT EXISTS users (
    id              BIGSERIAL PRIMARY KEY,
    email           VARCHAR(120) UNIQUE NOT NULL,
    username        VARCHAR(80) UNIQUE,
    password_hash   VARCHAR(256) NOT NULL,
    profile_picture TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert a test user (password: test123)
-- Password hash generated with bcrypt for 'test123'
INSERT INTO users (email, username, password_hash)
VALUES ('test@example.com', 'testuser', '$2b$12$cYQ3HDDPG9r1BwfaxVg1Q.Ioi/u8dzQQi5gWpCUcHbq5yXlDQ.SLm')
ON CONFLICT (email) DO NOTHING;