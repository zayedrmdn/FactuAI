---
title: FactuAI System Documentation
version: 3.1.0
last_updated: 2025-12-09
authors: [Zayed Ramadan Rahmat]
audience: AI Agents, Developers
status: Production Ready
repository: https://github.com/zayedrmdn/FactuAI
format: Structured Markdown for AI Parsing
---

# FactuAI - AI-Powered Fact-Checking System

**Document Type**: System Architecture & Operations Manual  
**Target Audience**: AI Agents, Automated Systems, Developers

---

## Executive Summary

FactuAI is a production-grade full-stack monorepo for AI-powered fact-checking and news verification. Built with Next.js 15, React 19, Flask 3, and PostgreSQL.

**Core Capabilities**:
- Text/Image/Video claim extraction and verification
- Multi-provider LLM orchestration (OpenRouter free tier, NVIDIA NIM paid tier)
- **Modular Search Architecture**: Easily extensible search providers (Google, NewsAPI, Tavily, etc.)
- Evidence-based fact-checking pipeline with semantic ranking
- Real-time progressive response streaming (SSE)
- Per-stage model selection for optimal performance

**Deployment Modes**:
- **Cloud Mode** (Default): API-based, ~50MB dependencies
- **Local Mode**: GPU-accelerated, ~4GB+ dependencies (requires NVIDIA CUDA 11.8+)

---

## Monorepo Structure

```
FactuAI/
├── frontend/                    # Next.js 15 + React 19 + TypeScript
│   ├── src/app/                 # App Router pages (layout only)
│   ├── src/components/          # Reusable components
│   │   ├── ui/                  # shadcn/ui primitives
│   │   ├── ai/                  # PipelineModelConfig
│   │   ├── dashboard/           # Header, Sidebar
│   │   └── landing/             # Landing page
│   ├── src/lib/                 # Utilities, hooks, validation
│   ├── src/config/              # AI models registry
│   ├── src/stores/              # Zustand state stores
│   └── src/types/               # TypeScript definitions
│
├── backend/                     # Flask 3 + Python 3.10+
│   ├── api/                     # REST API blueprints
│   ├── core/                    # Config, logging, exceptions
│   ├── database/                # SQLAlchemy models
│   ├── pipeline/                # Fact-checking orchestration
│   ├── services/                # LLM, classifiers, search, OCR
│   ├── schemas/                 # Pydantic validation
│   └── tests/                   # Pytest suite (61 test cases)
│
├── scripts/launch.bat           # Unified launcher (cloud/local modes)
├── .env                         # Environment config (gitignored)
├── README.md                    # This file
├── MODELS.md                    # AI model integration guide
└── CONSTITUTION.md              # Coding standards & governance
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 20+
- PostgreSQL 14+
- (Optional) NVIDIA GPU with CUDA 11.8+ for Local Mode

**Verify**:
```bash
python --version  # 3.10+
node --version    # 20+
psql --version    # 14+
```

### Installation

#### Option A: Automated (Recommended)

```bash
# Backend (Cloud Mode - uses OpenRouter API, ~50MB deps)
cd scripts
launch.bat cloud

# OR Local Mode (full ML stack, ~4GB+ deps, requires GPU)
launch.bat local

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

**Services**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

#### Option B: Manual

**Backend**:
```bash
# Cloud Mode
python -m venv .venv-cloud
.venv-cloud\Scripts\activate  # Windows: .venv-cloud\Scripts\activate
pip install -r backend/requirements-core.txt

# OR Local Mode
python -m venv .venv-local
.venv-local\Scripts\activate
pip install -r backend/requirements-local.txt

# Start
cd backend
python app.py
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

### Environment Configuration

Create `.env` in **project root**:
```env
# Run Mode: 'cloud' (default) or 'local'
APP_RUN_MODE=cloud

# LLM Provider: 'openrouter' (default), 'nvidia', or 'auto'
LLM_PROVIDER=openrouter

# Database
DB_URI=postgresql://user:pass@localhost:5432/factuai

# OpenRouter API (free tier)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b:free

# NVIDIA NIM API (paid, default provider)
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL=qwen/qwen2.5-7b-instruct

# Search & News
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_cse_id
NEWS_API_KEY=your_key_here

# Security
SECRET_KEY=your_secret_key_generated_here
```

---

## System Architecture

### Backend: Layered Architecture

```
API Layer (Flask Blueprints) ← REST endpoints, input validation
    ↓
Service Layer ← Business logic, external API calls
    ↓
Pipeline Orchestrator ← Fact-checking workflow
    ↓
Data Layer (SQLAlchemy ORM) ← Database models & queries
```

**Key Components**:
- **LLM Factory**: Dynamic provider switching (OpenRouter/NVIDIA/Local)
- **Pipeline Orchestrator**: Multi-stage fact-checking workflow (singleton pattern)
- **Evidence Ranker**: Semantic similarity scoring with SentenceTransformer (GPU-accelerated)
- **OCR Service**: Tesseract text extraction from images
- **Search Integration**: Google Custom Search + NewsAPI

**Dual-Mode Operation**:
- **Cloud Mode**: External APIs, minimal deps (~50MB)
- **Local Mode**: GPU inference, full ML stack (~4GB+)

### Frontend: Component Composition

```
App Router (Pages) ← Next.js routing, layout only
    ↓
Feature Modules ← Dashboard components, hooks, services
    ↓
UI Components (shadcn/ui) ← Reusable primitives
    ↓
State Stores (Zustand) ← Global state with localStorage persistence
```

**Architecture Patterns**:
- **App Shell**: Persistent sidebar + header layout
- **Mobile-First**: Responsive breakpoints (375px → 1280px+)
- **Config-Driven Models**: Registry in `config/ai-models.ts`
- **Type-Safe**: Full TypeScript coverage, strict mode
- **Progressive Enhancement**: SSE streaming for real-time updates

---

## Pipeline Flow

### Overview

```
┌─────────────────────────────────────────────────────────┐
│              Client Request (Text/Image/Video)           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│         Service Manager (Singleton Pattern)              │
│  • LLM Clients (cached by provider:model)               │
│  • Pipeline Orchestrator                                 │
│  • OCR Service                                           │
│  • SentenceTransformer (GPU/CPU)                        │
│  • Search Client                                         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│           Pipeline Orchestrator (7 Stages)               │
│  1. Intent Detection (Tier 1: Lightweight)              │
│  2. Claim Extraction (Tier 2: Medium)                   │
│  3. Evidence Collection (Google + NewsAPI)              │
│  4. Evidence Ranking (SentenceTransformer)              │
│  5. Evidence Selection (Tier 3: Heavyweight)            │
│  6. Source Quotes (Top 3, deduplicated)                 │
│  7. Summarization (Tier 3: Heavyweight)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          Fact-Check Results (JSON/SSE)                   │
│  • Claims with Evidence                                  │
│  • Source Quotes (URL, score)                           │
│  • Summary (input + evidence)                           │
└─────────────────────────────────────────────────────────┘
```

### Model Tier System

**Tier 1: Intent Detection (Lightweight)**
- **Purpose**: Quick classification (fact_claim, opinion, etc.)
- **Models**: `google/gemini-flash-1.5` (OpenRouter), `qwen/qwen2.5-7b-instruct` (NVIDIA)
- **Parameters**: `max_tokens=32`, `temperature=0.1`
- **Latency**: <500ms

**Tier 2: Claim Extraction (Medium)**
- **Purpose**: Extract verifiable claims
- **Models**: `anthropic/claude-3.5-haiku` (OpenRouter), `qwen/qwen2.5-7b-instruct` (NVIDIA)
- **Parameters**: `max_tokens=512`, `temperature=0.3`
- **Latency**: 1-2s

**Tier 3: Reasoning & Verification (Heavyweight)**
- **Purpose**: Complex reasoning, evidence selection, summarization
- **Models**: `anthropic/claude-3.5-sonnet` (OpenRouter), `meta/llama-3.1-70b-instruct` (NVIDIA)
- **Parameters**: `max_tokens=2048`, `temperature=0.5`
- **Latency**: 2-5s

**Configuration**: `backend/core/model_tiers.py`

### Example: Real Request Flow

**Input**:
```json
{
  "text": "The current president of Indonesia is Jokowi Widodo in 2025.",
  "provider": "openrouter"
}
```

**Processing Steps** (7.3s total):

1. **Intent Detection** (0.3s): Lightweight model → `fact_claim`
2. **Input Summary** (1.2s): Heavyweight model → "User claims Jokowi is president..."
3. **Search Collection** (0.8s): Google (3) + NewsAPI (2) → 5 URLs
4. **Article Fetching** (2.1s): 4/5 successful (1 blocked with 403)
5. **Sentence Extraction** (0.4s): 18 candidate sentences
6. **Semantic Ranking** (0.6s): SentenceTransformer → top score 0.89
7. **LLM Selection** (1.8s): Heavyweight picks sentences 1, 2
8. **Build Results** (0.1s): 47 words evidence, 3 quotes, 4 URLs

**Output**:
```json
{
  "results": [{
    "claim": "The current president of Indonesia is Jokowi Widodo in 2025.",
    "evidence": "Prabowo Subianto was inaugurated as Indonesia's 8th president on October 20, 2024. He succeeds Jokowi Widodo, who served two terms from 2014 to 2024.",
    "source_quotes": [
      {"quote": "Prabowo Subianto was inaugurated...", "source": "BBC", "url": "...", "score": 0.89},
      {"quote": "He succeeds Jokowi Widodo...", "source": "Reuters", "url": "...", "score": 0.85},
      {"quote": "The transition marks the end...", "source": "CNN", "url": "...", "score": 0.78}
    ],
    "urls": ["...", "...", "...", "..."]
  }],
  "summary": "According to multiple sources, Prabowo Subianto became president on Oct 20, 2024, succeeding Jokowi. The claim is incorrect."
}
```

---

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - Authentication
- `POST /api/auth/forgot-password` - Password reset request
- `POST /api/auth/reset-password` - Password reset confirmation

### Fact-Checking
- `POST /api/process` - Progressive fact-check (SSE streaming)
- `POST /api/factcheck` - Simple fact-check (JSON response)
- `POST /api/validate` - Content validation

### User Profile
- `GET /api/profile/:id` - Get profile
- `PUT /api/profile/:id` - Update profile
- `POST /api/profile/:id/upload` - Upload profile picture

### Media Processing
- `POST /api/image/upload` - Image OCR
- `POST /api/video/analyze` - Video analysis (YouTube transcripts)

---

## AI Model Registry

See **MODELS.md** for comprehensive integration guide.

### Supported Providers

**OpenRouter (Free Tier)**:
- `alibaba/tongyi-deepresearch-30b-a3b:free` (Research, 128K context)
- `allenai/olmo-3-32b-think:free` (Reasoning, 32K context)
- `openai/gpt-oss-120b:free` (General, 128K context)
- `google/gemma-3-27b-it:free` (Multimodal, 128K context)
- `nvidia/nemotron-nano-9b-v2:free` (Fast, 8K context)

**NVIDIA NIM (Paid)**:
- `meta/llama-3.1-405b-instruct` (Premium, 128K context)
- `meta/llama-3.1-70b-instruct` (High performance, 128K context)
- `meta/llama-3.1-8b-instruct` (Lightweight, 128K context)
- `mistralai/mistral-nemotron` (Balanced, 32K context)
- `qwen/qwen2.5-7b-instruct` (Default, 32K context)

**Defaults**:
- Provider: `nvidia`
- Model: `qwen/qwen2.5-7b-instruct`
- Fallback: `alibaba/tongyi-deepresearch-30b-a3b:free` (OpenRouter)

### Adding New Models

1. Open `frontend/src/config/ai-models.ts`
2. Add `ModelConfig` to provider's `models` array (see **MODELS.md** for schema)
3. Ensure `modelId` matches API spec EXACTLY (case-sensitive)
4. No code changes required (config-driven registry)

---

## Performance Optimizations

### 1. Singleton Pattern
- Pipeline Orchestrator: Single instance for all requests
- OCR Service: Shared Tesseract instance
- SentenceTransformer: Loaded once, GPU-accelerated
- Search Client: Connection pooling

### 2. LLM Caching
- Clients cached by `provider:model_id`
- Reused across requests with same config
- Prevents repeated initialization

### 3. Model Tiering
- Lightweight: Simple tasks (intent, <500ms)
- Medium: Extraction (1-2s)
- Heavyweight: Complex reasoning (2-5s)
- Reduces API costs and latency

### 4. Article Caching
- Fetched articles saved to `article_cache.json`
- Prevents re-fetching same URLs
- Cache persists across server restarts

### 5. Lazy Loading
- Heavy dependencies loaded on-demand
- NLTK downloads on first use
- Embedding models loaded dynamically

---

## Testing

### Backend (Pytest)

**Test Suite**: 61 test cases across multiple modules

**Coverage**:
- `tests/test_unit/test_evidence_pipeline.py` (10 tests - Evidence collection pipeline)
- `tests/test_services/test_intent_reasoning_models.py` (11 tests - Reasoning model JSON parsing with live API tests)
- `tests/test_services/test_intent_classifier.py` (24 tests - Intent detection classifier)
- `tests/test_services/test_extractors*.py` (11 tests - Article extraction and caching)
- `tests/test_routes/` (API endpoint tests)
- `tests/test_models/` (Database model tests)

**Run Tests**:
```bash
cd backend
pytest tests/                                                    # All tests
pytest tests/test_services/test_intent_reasoning_models.py      # Reasoning model parsing (includes 2 live API tests)
pytest tests/test_unit/test_evidence_pipeline.py                # Evidence pipeline unit tests
```

**Key Test Suites**:
- **Reasoning Model Tests**: Validates JSON extraction from reasoning models (Alibaba Tongyi, GLM 4.5 Air) including step-by-step analysis, markdown formatting, and malformed responses
- **Live API Tests**: 2 tests using OpenRouter GLM 4.5 Air to ensure production readiness
- **Intent Classification**: 24 tests covering fact claims, questions, opinions, multi-claims, and edge cases

### Frontend (Manual)

- Component rendering tests
- Hook behavior tests
- E2E user flows with Playwright (future)

---

## Troubleshooting

### Issue: "OpenRouter generation failed"
**Cause**: API key invalid or model ID incorrect  
**Solution**: 
- Verify `OPENROUTER_API_KEY` in `.env`
- Check model ID includes `:free` suffix if applicable
- Robust null checks now prevent crashes

### Issue: Unicode errors in console (Windows)
**Status**: ✅ Fixed  
**Solution**: `SafeFormatter` sanitizes console output, full UTF-8 in log files

### Issue: Slow performance
**Solution**:
- Verify SentenceTransformer uses GPU (`torch.cuda.is_available()`)
- Check model tier configuration (`backend/core/model_tiers.py`)
- Ensure LLM caching works (check logs for "Using cached LLM client")

### Issue: Scraping 403 errors
**Status**: ✅ Handled gracefully  
**Solution**: System skips blocked URLs and continues with others

### Issue: Database connection failed
**Solution**:
- Verify PostgreSQL is running: `psql -U postgres`
- Check `DB_URI` in `.env`
- Run `python backend/scripts/test_db_connection.py`

---

## Project Status

### ✅ Production Ready
- Full-stack authentication (registration, login, password reset)
- Progressive fact-checking pipeline with SSE streaming
- Multi-provider LLM orchestration (OpenRouter + NVIDIA NIM)
- Responsive dashboard UI (mobile, tablet, desktop)
- Image OCR (Tesseract) and video processing (YouTube transcripts)
- Evidence retrieval (Google + NewsAPI) and semantic ranking
- Database migrations and user management
- Model tier system for performance optimization
- Singleton pattern for resource efficiency
- Robust error handling and logging

### ⏳ In Progress
- Advanced summarization models (T5 fine-tuning on MultiNews)
- BERT/NeoBERT classification deployment (fine-tuned on LIAR2)
- Real-time collaboration features

---

## Recent Improvements (December 2024)

### Critical Fixes
1. **OpenRouter Error Handling**: Robust null checks, validation, graceful errors
2. **Unicode Logging**: `SafeFormatter` for Windows console, UTF-8 log files
3. **Singleton Pattern**: No duplicate service initialization
4. **Model Caching**: SentenceTransformer, KeyBERT, LLM clients cached
5. **Scraper Improvements**: 403 handling, graceful fallback

### Performance
- **Model Tiering**: Lightweight → Medium → Heavyweight
- **LLM Caching**: Clients reused across requests
- **Article Caching**: Disk persistence
- **Lazy Loading**: On-demand dependency loading

---

## Technology Stack

### Frontend
- Next.js 15 + React 19 + TypeScript 5
- Tailwind CSS 4 + shadcn/ui
- Zustand 5 (state management)
- Zod 3 (validation)

### Backend
- Flask 3 + Python 3.10+
- PostgreSQL 14+ + SQLAlchemy 2
- Pydantic 2 (validation)
- Pytest 8 (testing)
- PyTorch 2 + Hugging Face Transformers 4

### AI/ML
- OpenRouter (free tier models)
- NVIDIA NIM (paid tier models)
- Google Custom Search API
- NewsAPI
- Tesseract OCR
- Sentence Transformers (semantic search)

---

## Documentation Structure

- **README.md** (this file): System architecture, setup, operations, pipeline overview
- **MODELS.md**: AI model integration guide (OpenRouter + NVIDIA NIM specs, code examples)
- **CONSTITUTION.md**: Coding standards, design system, governance rules

**Governance**: All code contributions and AI agent operations must follow **CONSTITUTION.md** standards.

---

**Document Version**: 3.0.0  
**Last Updated**: December 2025  
**Maintained By**: Zayed Ramadan Rahmat
- Local state - UI interactions (modals, drawers)

---

## TROUBLESHOOTING

### Backend

**Unicode Encoding Errors:**
```
Solution: Logs now use UTF-8 encoding. All emoji removed from log messages.
Status: Fixed in v2.1.0 (see UNICODE_FIX_REPORT.md)
```

**Model ID Errors (400 Bad Request):**
```
Problem: openrouter-tongyi-deepresearch-30b is not a valid model ID
Solution: OpenRouter free models require :free suffix
Example: alibaba/tongyi-deepresearch-30b-a3b:free
Status: Fixed - all model IDs updated
```

**Module Import Errors:**
```bash
# Verify correct venv activated
which python  # Should show .venv-cloud or .venv-local
pip install -r backend/requirements-core.txt
```

**Database Connection Failed:**
```bash
# Test connection
cd backend
python scripts/test_db_connection.py

# Verify PostgreSQL running
psql -U postgres -c "SELECT version();"
```

### Frontend

**Build Errors:**
```bash
# Clean and rebuild
rm -rf .next node_modules
npm install
npm run build
```

**API Proxy Not Working:**
```
Check: next.config.ts rewrites configuration
Ensure: Backend running on localhost:5000
Verify: No CORS errors in browser console
```

**Pipeline Model Configuration:**
```
Status: Active in v2.1.0+
Feature: Task-specific model selection (Intent, Extraction, Reasoning)
Location: Dashboard page, above input area
Note: Removed global ModelSelector from Header for better UX
```

---

## MAINTENANCE

### Adding New Models
1. Update `frontend/src/config/ai-models.ts`
2. Ensure exact model ID match with provider API
3. Update `.env` if changing defaults
4. No code changes required (config-driven)

### Database Migrations
```bash
# Create migration (if using Alembic)
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

### Dependency Updates
```bash
# Frontend
cd frontend && npm update

# Backend (Cloud Mode)
pip install --upgrade -r backend/requirements-core.txt

# Backend (Local Mode)
pip install --upgrade -r backend/requirements-local.txt
```

---

## LICENSE

MIT License - Free for academic, research, and learning purposes.

---

## METADATA

**Version:** 3.0.1  
**Last Updated:** 2025-12-09T12:00:00Z  
**Author:** Zayed Ramadan Rahmat  
**Institution:** Asia Pacific University, Malaysia  
**Project Type:** Final Year Project (Computer Science - AI)  
**Repository:** https://github.com/zayedrmdn/FactuAI  
**Documentation Format:** AI Agent-Optimized Structured Markdown

---

## RECENT UPDATES

### Search Provider Toggle Feature (v3.1.0 - 2025-12-09)
1. **User-Configurable Search Providers**:
   - Toggle Google Custom Search and NewsAPI independently
   - Intuitive UI with clear visual feedback
   - Automatic validation (at least one must be enabled)
   - State persisted to localStorage
2. **Backend Implementation**:
   - Scalable provider filtering in `evidence.py`
   - Validation ensures at least one provider is active
   - Full error handling and logging
   - Easy to extend with new providers
3. **Frontend Implementation**:
   - New Zustand store for provider state management
   - SearchProvidersConfig component with toggle switches
   - Integration with useFactCheck hook
   - Console logging for debugging
4. **Benefits**:
   - Cost savings (disable expensive APIs)
   - Flexibility for different use cases
   - Better quota management
   - Extensible architecture

### Critical LLM Response Handling (v3.0.2 - 2025-12-09)
1. **Enhanced Response Validation**: 
   - Added strict validation for empty/minimal responses (< 10 chars)
   - Implemented detailed logging with `repr()` to detect special characters
   - Raises `LLMClientError` for truly empty responses
   - Warning for suspiciously short responses that may indicate prompt issues
2. **Retry Logic with Adaptive Parameters**:
   - Verification: 2 retries with increasing `max_tokens` (800 → 2500)
   - Summarization: 2 retries with quality validation (>30 chars)
   - Temperature optimization: 0.3 for fact-checking, 0.5 for summaries
3. **Response Quality Checks**:
   - Minimum 50 chars for verification responses
   - Minimum 30 chars for summary responses
   - Graceful fallback to UNVERIFIABLE when LLM fails
4. **Model-Specific Fixes**:
   - Documented Mistral-7b-instruct issues in MODELS.md
   - Recommended parameters: `max_tokens=800-2000`, `temperature=0.3-0.5`
   - Better model alternatives suggested for fact-checking tasks
   
### Backend Fixes (v3.0.1)
1. **Verdict Normalization**: Added mapping function to convert backend verdicts (TRUE, FALSE, UNVERIFIABLE, etc.) to frontend-compatible labels (true, false, unknown, etc.)
2. **Executive Summary**: Improved summarization prompt to generate proper executive summaries with key findings, context, and stakes
3. **Dynamic Token Limits**: Token allocation now scales based on input/evidence size for efficiency
4. **HF_HOME Warning**: Fixed TRANSFORMERS_CACHE deprecation by setting HF_HOME in config.py
5. **Reasoning Model Support**: Enhanced empty response handling to extract from `reasoning` and `reasoning_details` fields
6. **Finish Reason Logging**: Added verbose logging for `finish_reason=length` warnings

### Frontend Fixes (v3.0.1)
1. **Confidence Display**: Fixed circular progress bar display (was multiplying by 100 twice)
2. **Copy Function**: Implemented full structured plain text export with:
   - Executive summary
   - Overall scores (trust score, AI detection)
   - Detailed findings with evidence and sources
   - Proper formatting for clipboard
3. **Label Mapping**: Frontend now correctly displays verdict labels (Unknown → proper verdict)

### Code Quality
- Followed KISS and DRY principles throughout
- Maintained backward compatibility
- All fixes are dynamic and model-agnostic
- Centralized logging for debugging
- Production-ready error handling with no temporary workarounds

---

## REFERENCES

- [Next.js 15 Documentation](https://nextjs.org/docs)
- [Flask 3 Documentation](https://flask.palletsprojects.com/)
- [OpenRouter API](https://openrouter.ai/docs)
- [NVIDIA NIM API](https://docs.api.nvidia.com/)
- [Tailwind CSS 4](https://tailwindcss.com/docs)
- [shadcn/ui](https://ui.shadcn.com/)
- [CONSTITUTION.md](./CONSTITUTION.md) - Coding Standards

---

**END OF DOCUMENT**
