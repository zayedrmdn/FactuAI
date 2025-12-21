# Database Schema

PostgreSQL 16 with pgvector extension for continuous learning.

---

## Overview

**Database:** PostgreSQL 16+  
**Extension:** pgvector (for vector similarity search)  
**Driver:** asyncpg (Python async)  
**ORM:** SQLAlchemy (async mode)

**Connection String:**
```
postgresql+asyncpg://postgres:postgres@localhost:5433/factuai
```

---

## Core Tables

### `verifications`

Top-level request record for each fact-check analysis.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | BIGSERIAL | PK | Auto-incrementing ID |
| `request_id` | UUID | UNIQUE, NOT NULL | External identifier |
| `input_text` | TEXT | NOT NULL | User's original input |
| `model_used` | TEXT | | LLM model for verification |
| `latency_ms` | INTEGER | | Total processing time |
| `verdict` | VARCHAR(50) | | Overall verdict (if single claim) |
| `confidence` | NUMERIC(3,2) | | Overall confidence (0.00-1.00) |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Timestamp |

**Indexes:**
- `request_id` (unique)
- `created_at` (for time-based queries)

---

### `claims`

One row per extracted claim.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | BIGSERIAL | PK | Auto-incrementing ID |
| `verification_id` | BIGINT | FK → verifications(id) | Parent verification |
| `claim_text` | TEXT | NOT NULL | Extracted claim text |
| `verdict` | VARCHAR(50) | | TRUE, FALSE, MIXED, etc. |
| `confidence` | NUMERIC(3,2) | | Confidence score (0.00-1.00) |
| `reasoning` | TEXT | | LLM's reasoning for verdict |
| `claim_embedding` | VECTOR(384) | | pgvector embedding for RAG |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Timestamp |

**Indexes:**
- `verification_id` (foreign key)
- `claim_embedding` (HNSW index for vector similarity)

**pgvector Index:**
```sql
CREATE INDEX idx_claims_embedding 
ON claims USING ivfflat (claim_embedding vector_cosine_ops) WITH (lists = 100);
```

---

### `sources`

Normalized source metadata (deduplicated by URL).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | BIGSERIAL | PK | Auto-incrementing ID |
| `url` | TEXT | UNIQUE, NOT NULL | Source URL |
| `title` | TEXT | | Page title |
| `domain` | VARCHAR(255) | | Extracted domain |
| `credibility_score` | NUMERIC(3,2) | | Credibility (0.00-1.00) |
| `first_seen_at` | TIMESTAMPTZ | DEFAULT NOW() | First appearance |
| `last_seen_at` | TIMESTAMPTZ | DEFAULT NOW() | Last appearance |

**Indexes:**
- `url` (unique)
- `domain` (for filtering)

---

### `evidence`

Snippets/quotes from sources tied to claims.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | BIGSERIAL | PK | Auto-incrementing ID |
| `claim_id` | BIGINT | FK → claims(id) | Associated claim |
| `source_id` | BIGINT | FK → sources(id) | Source reference |
| `snippet` | TEXT | NOT NULL | Evidence text |
| `relevance_score` | NUMERIC(4,3) | | Relevance (0.000-1.000) |
| `snippet_embedding` | VECTOR(384) | | pgvector embedding |
| `captured_at` | TIMESTAMPTZ | DEFAULT NOW() | Timestamp |

**Composite Unique Constraint:**
```sql
UNIQUE (claim_id, source_id, snippet)
```
This prevents duplicate evidence for the same claim from the same source.

**Indexes:**
- `claim_id` (foreign key)
- `source_id` (foreign key)
- `snippet_embedding` (HNSW index for vector similarity)

**pgvector Index:**
```sql
CREATE INDEX idx_evidence_embedding 
ON evidence USING ivfflat (snippet_embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Entity-Relationship Diagram

```mermaid
erDiagram
    VERIFICATIONS ||--o{ CLAIMS : contains
    CLAIMS ||--o{ EVIDENCE : supported_by
    SOURCES ||--o{ EVIDENCE : provides

    VERIFICATIONS {
        bigint id PK
        uuid request_id UK
        text input_text
        text model_used
        int latency_ms
        varchar verdict
        numeric confidence
        timestamptz created_at
    }

    CLAIMS {
        bigint id PK
        bigint verification_id FK
        text claim_text
        varchar verdict
        numeric confidence
        text reasoning
        vector claim_embedding
        timestamptz created_at
    }

    SOURCES {
        bigint id PK
        text url UK
        text title
        text domain
        numeric credibility_score
        timestamptz first_seen_at
        timestamptz last_seen_at
    }

    EVIDENCE {
        bigint id PK
        bigint claim_id FK
        bigint source_id FK
        text snippet
        numeric relevance_score
        vector snippet_embedding
        timestamptz captured_at
    }
```

---

## pgvector Integration

### Vector Embeddings

- **Dimension:** 384 (BAAI/bge-small-en-v1.5 model)
- **Distance Metric:** Cosine similarity
- **Index Type:** IVFFLAT (Inverted File with Flat Compression)

### Creating Vector Columns

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE claims (
    id BIGSERIAL PRIMARY KEY,
    claim_text TEXT NOT NULL,
    claim_embedding VECTOR(384)
);

-- Create IVFFLAT index for fast similarity search
CREATE INDEX idx_claims_embedding 
ON claims USING ivfflat (claim_embedding vector_cosine_ops) WITH (lists = 100);
```

### Querying by Similarity

```sql
-- Find similar claims (cosine distance < 0.20)
SELECT claim_text, 
       1 - (claim_embedding <=> :query_embedding) AS similarity
FROM claims
WHERE 1 - (claim_embedding <=> :query_embedding) > 0.80
ORDER BY claim_embedding <=> :query_embedding
LIMIT 5;
```

**Note:** `<=>` is the cosine distance operator. To convert to similarity: `1 - distance`.

---

## Migrations

**Location:** `backend/migrations/*.sql`

### Migration Strategy

- **Idempotent:** All migrations use `IF NOT EXISTS` checks
- **Auto-applied:** On startup if `DB_RUN_MIGRATIONS=true`
- **Manual application:** Via `psql` if needed

**Example:**
```bash
psql -h localhost -p 5433 -U postgres -d factuai -f backend/migrations/v4_0_001_core.sql
```

### Migration Files

- `v3_0_001_init.sql` - Core schema (verifications, claims, sources, evidence) + pgvector
- `v3_0_002_users.sql` - User authentication tables
- Additional migrations as features are added

---

## Performance Considerations

### Indexing Strategy

1. **Foreign keys** always indexed
2. **Timestamp columns** indexed for time-based queries
3. **Vector columns** use IVFFLAT indexes (faster than brute force)
4. **Unique constraints** automatically create indexes

### Connection Pooling

SQLAlchemy async engine uses connection pooling:

```python
# backend/app/core/db.py
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=False
)
```

---

## Data Retention

### No Automatic Cleanup

Currently, data is retained indefinitely for:
- Continuous learning (RAG retrieval)
- Analytics/insights
- Audit trail

### Future: Manual Cleanup

```sql
-- Delete old verifications (older than 90 days)
DELETE FROM verifications 
WHERE created_at < NOW() - INTERVAL '90 days';

-- Cascade deletes will remove related claims and evidence
```

---

## Backup & Restore

### Backup

```bash
pg_dump -h localhost -p 5433 -U postgres factuai > backup.sql
```

### Restore

```bash
psql -h localhost -p 5433 -U postgres -d factuai < backup.sql
```

---

See also:
- [Overview](overview.md) - High-level architecture
- [Continuous Learning](../05-features/continuous-learning.md) - RAG feedback loop
- [Backend Architecture](backend.md) - SQLAlchemy usage
