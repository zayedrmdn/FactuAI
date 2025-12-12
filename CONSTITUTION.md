---
title: FactuAI Constitution
version: 4.0.0
last_updated: 2025-12-12
audience: AI Agents, Developers, Code Contributors
status: Active Governance Document
format: Enforceable Markdown
---

# FactuAI Constitution

This file is the single source of truth for **engineering rules**.

## Non-Negotiables

### 1) Backend = Native Async FastAPI

- All backend I/O is truly async:
  - HTTP: `httpx.AsyncClient`
  - DB: SQLAlchemy async + `asyncpg`
  - Redis: `redis.asyncio`
- Do not add sync wrappers or thread bridges for core logic.

### 2) Vertical Slice Architecture (VSA)

- Code is organized by **feature** under `backend/app/features/*`.
- Each feature owns its own API boundary (`router.py`) and orchestration (`service.py`).
- Route handlers must be thin: validate → call service → return.

### 3) OCP + Dependency Injection

- The system must be open to extension without modifying orchestrators.
- Dependencies are wired via `backend/app/core/container.py` and env-configured dotted paths in `backend/app/core/settings.py`.

### 4) Cross-Feature Boundaries

- A feature must not import another feature.
- Shared types live in `backend/app/contracts/`.
- Shared infrastructure lives in `backend/app/core/`.

### 5) No Legacy / No Pipeline

- Do not reintroduce removed legacy stacks (old pipeline, old Flask layers, old adapter wrappers).

## Continuous Learning (RAG Feedback Loop)

- After **high-confidence** verifications, the backend attempts to store embeddings into pgvector columns.
- Learning failures must fail safely and be logged.

## Data & Migrations

- Migrations are SQL files in `backend/migrations/*.sql`.
- They are applied on startup (see `backend/app/core/db.py`).

## Quality Gates

- Backend tests must pass: run `pytest` in `backend/`.
- Keep normal runs free of new warnings (avoid deprecations).
- Never commit secrets.

## Frontend (Lightweight Rules)

- Keep pages thin; move logic into `frontend/src/lib/` and hooks.
- Prefer composition over prop drilling.
- Keep config in `frontend/src/config/`.
