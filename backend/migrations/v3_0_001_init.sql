-- Full Path: backend\migrations\v3_0_001_init.sql
-- FactuAI v3.0 (Phase 1) - Core tables + pgvector
-- Idempotent migration: safe to run on startup.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS verifications (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    user_id BIGINT NULL,
    input_text TEXT NOT NULL,
    model_used TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    verdict VARCHAR(20) NOT NULL,
    confidence NUMERIC(3, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_verifications_user_created ON verifications (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_verifications_verdict ON verifications (verdict);
CREATE TABLE IF NOT EXISTS claims (
    id BIGSERIAL PRIMARY KEY,
    verification_id BIGINT NOT NULL REFERENCES verifications(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    verdict VARCHAR(20) NOT NULL,
    confidence NUMERIC(3, 2) NOT NULL,
    reasoning TEXT NOT NULL,
    claim_embedding VECTOR(384) NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_claims_verification ON claims (verification_id);
CREATE INDEX IF NOT EXISTS idx_claims_embedding ON claims USING ivfflat (claim_embedding vector_cosine_ops) WITH (lists = 100);
CREATE TABLE IF NOT EXISTS sources (
    id BIGSERIAL PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    domain TEXT NOT NULL,
    credibility_score NUMERIC(3, 2) DEFAULT 0.50,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources (domain);
CREATE TABLE IF NOT EXISTS evidence (
    id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    source_id BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    snippet TEXT NOT NULL,
    relevance_score NUMERIC(4, 3) NOT NULL,
    snippet_embedding VECTOR(384) NULL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_claim_source_snippet UNIQUE (claim_id, source_id, snippet)
);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence (claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence (source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_embedding ON evidence USING ivfflat (snippet_embedding vector_cosine_ops) WITH (lists = 100);