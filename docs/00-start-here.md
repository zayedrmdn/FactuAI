# FactuAI Documentation

**Welcome!** This is your central navigation hub for all FactuAI documentation.

---

## 🚀 Quick Start Paths

### For AI Agents
1. Read [Engineering Rules](01-rules/constitution.md) (MUST READ FIRST)
2. Read [AI Agent Onboarding](01-rules/agents.md)
3. Explore [Architecture Overview](03-architecture/overview.md)
4. Understand the [4-Phase Pipeline](04-pipeline/00-overview.md)

### For New Developers
1. [Quick Setup Guide](02-setup/quickstart.md) (get up and running in 5 minutes)
2. [Architecture Overview](03-architecture/overview.md)
3. [Backend Architecture](03-architecture/backend.md)
4. [Frontend Architecture](03-architecture/frontend.md)

### For Product/Business
1. [Product Specifications](product-specs.md)
2. [Feature Catalog](05-features/)
3. [API Reference](07-api/endpoints.md)

---

## 📚 Documentation Structure

### 01 - Rules & Governance
- [constitution.md](01-rules/constitution.md) - Engineering rules (non-negotiable)
- [agents.md](01-rules/agents.md) - AI agent onboarding & quick reference
- [theme-standards.md](01-rules/theme-standards.md) - UI technical specifications
- [design-philosophy.md](01-rules/design-philosophy.md) - Anti-AI-slop creative direction

### 02 - Setup Guides
- [quickstart.md](02-setup/quickstart.md) - Docker + local development
- [windows.md](02-setup/windows.md) - Windows-specific setup
- [environment-vars.md](02-setup/environment-vars.md) - `.env` configuration reference
- [troubleshooting.md](02-setup/troubleshooting.md) - Common issues & solutions

### 03 - Architecture
- [overview.md](03-architecture/overview.md) - High-level system design
- [backend.md](03-architecture/backend.md) - FastAPI, VSA, DI patterns
- [frontend.md](03-architecture/frontend.md) - Next.js, feature modules
- [database.md](03-architecture/database.md) - Postgres + pgvector schema

### 04 - The Analysis Pipeline (Core Feature)
- [00-overview.md](04-pipeline/00-overview.md) - Pipeline flow diagram
- [01-intent.md](04-pipeline/01-intent.md) - Phase 0: Intent Extraction (LLM)
- [02-strategist.md](04-pipeline/02-strategist.md) - Phase 1: Multi-Angle Query Generation
- [03-search.md](04-pipeline/03-search.md) - Phase 2: Parallel Search (Tavily + RAG)
- [04-pivot.md](04-pipeline/04-pivot.md) - Phase 3: Pivot Loop (Iterative Research)
- [05-verification.md](04-pipeline/05-verification.md) - Phase 4: LLM Synthesis

### 05 - Features
- [continuous-learning.md](05-features/continuous-learning.md) - RAG feedback loop
- [source-filtering.md](05-features/source-filtering.md) - Social media blocklist
- [model-override.md](05-features/model-override.md) - Frontend model selection
- [search-providers.md](05-features/search-providers.md) - Adding new providers (OCP)

### 06 - Testing
- [test-claims.md](06-testing/test-claims.md) - Benchmark claims for QA
- [backend-tests.md](06-testing/backend-tests.md) - pytest guide
- [frontend-tests.md](06-testing/frontend-tests.md) - vitest guide

### 07 - API Reference
- [endpoints.md](07-api/endpoints.md) - API endpoints
- [schemas.md](07-api/schemas.md) - Request/response types

### 08 - Logs & History
- [changelog.md](08-logs/changelog.md) - Version history
- [bugs/](08-logs/bugs/) - Bug reports

---

## 🔍 Find What You Need

| I want to... | Go to... |
|-------------|----------|
| Set up FactuAI locally | [02-setup/quickstart.md](02-setup/quickstart.md) |
| Understand the 4-phase pipeline | [04-pipeline/00-overview.md](04-pipeline/00-overview.md) |
| Add a new search provider | [05-features/search-providers.md](05-features/search-providers.md) |
| Configure environment variables | [02-setup/environment-vars.md](02-setup/environment-vars.md) |
| Run tests | [06-testing/](06-testing/) |
| Check API endpoints | [07-api/endpoints.md](07-api/endpoints.md) |
| See recent changes | [08-logs/changelog.md](08-logs/changelog.md) |
| Report a bug | [08-logs/bugs/template.md](08-logs/bugs/template.md) |

---

## 📖 Reading Order (Suggested)

**For comprehensive understanding:**

1. **Rules** → [constitution.md](01-rules/constitution.md)
2. **Setup** → [quickstart.md](02-setup/quickstart.md)
3. **Architecture** → [overview.md](03-architecture/overview.md)
4. **Pipeline** → [00-overview.md](04-pipeline/00-overview.md)
5. **Features** → Browse [05-features/](05-features/)
6. **Testing** → [test-claims.md](06-testing/test-claims.md)

---

**Last Updated:** 2025-12-24  
**Version:** 4.0.5
