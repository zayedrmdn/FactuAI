---
title: FactuAI System Documentation
version: 2.1.0
last_updated: 2025-12-07T21:30:00Z
authors: [Zayed Ramadan Rahmat]
audience: AI Agents, Developers
status: Production Ready
repository: https://github.com/zayedrmdn/FactuAI
---

# FactuAI - AI-Powered Fact-Checking System

## META

**Document Type:** System Architecture & Operations Manual  
**Target Audience:** AI Agents, Automated Systems, Developers  
**Parsing Format:** Structured Markdown with YAML frontmatter  
**Update Frequency:** On every significant system change  
**Schema Version:** 2.1

---

## EXECUTIVE SUMMARY

FactuAI is a production-grade full-stack monorepo for AI-powered news summarization and fact-checking. Built with Next.js 15, React 19, Flask 3, and PostgreSQL.

**Primary Functions:**
- Text/Image/Video claim extraction and verification
- Multi-provider LLM orchestration (OpenRouter, NVIDIA NIM)
- Evidence-based fact-checking pipeline
- Real-time progressive response streaming

**Deployment Modes:**
- **Cloud Mode** (Default): API-based, ~50MB dependencies
- **Local Mode**: GPU-accelerated, ~4GB+ dependencies

---

## MONOREPO STRUCTURE

```
FactuAI/
├── frontend/                     # Next.js 15 + React 19 + TypeScript
│   ├── src/app/                  # Next.js App Router pages
│   ├── src/components/           # Reusable React components
│   ├── src/config/               # Configuration (AI models registry)
│   ├── src/stores/               # Zustand state management
│   └── src/types/                # TypeScript definitions
├── backend/                      # Flask 3 + Python 3.10+
│   ├── api/                      # REST API blueprints
│   ├── core/                     # Config, logging, exceptions
│   ├── database/                 # SQLAlchemy models
│   ├── pipeline/                 # Fact-checking orchestration
│   ├── services/                 # LLM, classifiers, search
│   ├── schemas/                  # Pydantic validation
│   └── tests/                    # Pytest suite
├── scripts/                      # Orchestration tools
│   └── launch.bat                # Unified launcher (cloud/local)
├── .env                          # Environment configuration (gitignored)
├── .env.example                  # Environment template
├── CONSTITUTIONS.md              # Coding standards (AI agent rules)
├── UNICODE_FIX_REPORT.md         # Recent fixes documentation
└── README.md                     # This file

**REMOVED:** frontend/README.md, backend/README.md (consolidated here)
```

---

## SYSTEM CAPABILITIES

### Primary Functions
- **Claim Extraction:** NLP-based extraction from text, images (OCR), and videos
- **Evidence Retrieval:** Google Search API + NewsAPI integration
- **Fact Verification:** Multi-stage pipeline with LLM reasoning
- **Progressive Streaming:** Server-Sent Events for real-time updates

### Supported Input Types
- Text (up to 10,000 characters)
- Images (PNG, JPG, WebP) with Tesseract OCR
- Videos (YouTube URLs) with transcript extraction

### Output Formats
- JSON API responses
- Server-Sent Events (SSE) streams
- Structured fact-check reports with confidence scores

---

## TECHNOLOGY STACK

### Frontend (Next.js 15)
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js | 15.3+ |
| Language | TypeScript | 5.x |
| UI Library | React | 19.x |
| Styling | Tailwind CSS | 4.x |
| Components | shadcn/ui | Latest |
| State | Zustand | 5.x |
| Validation | Zod | 3.x |
| HTTP Client | Fetch API | Native |

### Backend (Flask 3)
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Flask | 3.0+ |
| Language | Python | 3.10+ |
| Database | PostgreSQL | 14+ |
| ORM | SQLAlchemy | 2.x |
| Validation | Pydantic | 2.x |
| Testing | Pytest | 8.x |
| ML Library | PyTorch | 2.x |
| Transformers | Hugging Face | 4.x |

### AI/ML Infrastructure
| Service | Provider | Purpose |
|---------|----------|---------|
| LLM API | OpenRouter | Free/paid model access |
| LLM API | NVIDIA NIM | High-performance inference |
| Search | Google Custom Search | Evidence retrieval |
| News | NewsAPI | Article fetching |
| OCR | Tesseract | Image text extraction |

---

## INSTALLATION

### Prerequisites

**System Requirements:**
- Python 3.10 or higher
- Node.js 20 or higher with npm
- PostgreSQL 14 or higher
- Windows 10/11, Linux, or macOS
- (Optional) NVIDIA GPU with CUDA 11.8+ for Local Mode

**Verify Prerequisites:**
```bash
python --version    # Should show 3.10+
node --version      # Should show 20+
psql --version      # Should show 14+
```

### Option A: Automated Setup (Recommended)

Use the unified launcher script:

```bash
cd scripts
launch.bat cloud   # Lightweight mode - uses OpenRouter API (~50MB deps)
# OR
launch.bat local   # Full ML stack - uses local models (~4GB+ deps, requires GPU)
```

This script will:
1. Create the appropriate virtual environment (`.venv-cloud/` or `.venv-local/`)
2. Install backend dependencies
3. Start the Flask backend on `http://localhost:5000`

Then in a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:3000`

### Manual Setup

**Backend:**
```bash
# Cloud Mode (lightweight - uses OpenRouter API)
python -m venv .venv-cloud
.venv-cloud\Scripts\activate     # Windows
pip install -r backend/requirements-core.txt

# OR Local Mode (full ML stack - requires CUDA)
python -m venv .venv-local
.venv-local\Scripts\activate     # Windows
pip install -r backend/requirements-local.txt

# Start backend
cd backend
python app.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Environment Configuration

Create a `.env` file in the **project root**:
```env
# Run Mode: 'cloud' (default) or 'local'
APP_RUN_MODE=cloud

# LLM Provider (cloud mode only): 'openrouter' (default), 'nvidia', or 'auto'
LLM_PROVIDER=openrouter

# Database
DB_URI=postgresql://user:pass@localhost:5432/factuai

# OpenRouter API (free tier models)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# NVIDIA NIM API (default provider)
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL=qwen/qwen2.5-7b-instruct
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Search & News APIs
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_cse_id
NEWS_API_KEY=your_key_here

# Security
SECRET_KEY=your_secret_key
```

### Running the Application

**Using Launch Script (Recommended):**
```bash
cd scripts
launch.bat cloud   # Start in Cloud Mode
launch.bat local   # Start in Local Mode
launch.bat         # Defaults to Cloud Mode
```

**Manual Start:**
```bash
# Terminal 1: Backend (activate appropriate venv first)
.venv-cloud\Scripts\activate
cd backend && python app.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

---

## AI MODEL REGISTRY

### Supported Providers

#### OpenRouter (Free Tier Models)
| Model ID | Display Name | Context | Max Tokens | Temperature | Tier |
|----------|--------------|---------|------------|-------------|------|
| `alibaba/tongyi-deepresearch-30b-a3b:free` | Alibaba: Tongyi DeepResearch 30B A3B | 128K | 8000 | 0.3 | free |
| `allenai/olmo-3-32b-think:free` | AllenAI: Olmo 3 32B Think | 32K | 6000 | 0.2 | free |
| `openai/gpt-oss-120b:free` | OpenAI: GPT-OSS 120B | 8K | 4096 | 0.7 | free |
| `nvidia/nemotron-nano-9b-v2:free` | NVIDIA: Nemotron Nano 9B V2 | 4K | 2048 | 0.5 | free |
| `meituan/longcat-flash-chat:free` | Meituan: LongCat Flash Chat | 32K | 4096 | 0.8 | free |

**Default:** `alibaba/tongyi-deepresearch-30b-a3b:free` (Recommended for research)

#### NVIDIA NIM (Premium Models)
| Model ID | Display Name | Context | Max Tokens | Temperature | Tier |
|----------|--------------|---------|------------|-------------|------|
| `meta/llama-3.1-405b-instruct` | Meta Llama 3.1 405B Instruct | 128K | 1024 | 0.2 | premium |
| `meta/llama-3.1-70b-instruct` | Meta Llama 3.1 70B Instruct | 128K | 1024 | 0.2 | high |
| `meta/llama-3.1-8b-instruct` | Meta Llama 3.1 8B Instruct | 128K | 1024 | 0.2 | low |
| `mistralai/mistral-nemotron` | Mistral Nemotron | 32K | 4096 | 0.6 | medium |
| `qwen/qwen2.5-7b-instruct` | Qwen 2.5 7B Instruct | 32K | 1024 | 0.2 | low |

**Default:** `qwen/qwen2.5-7b-instruct` (Fast and efficient)

### Model Selection Guidelines

**For Agents:**
- Read `frontend/src/config/ai-models.ts` for full registry
- Model IDs must match exactly (case-sensitive, include `:free` suffix where applicable)
- Default provider: `nvidia` (set in `.env`)
- Default model: `qwen/qwen2.5-7b-instruct`

**Adding Models:**
1. Update `frontend/src/config/ai-models.ts` with new `ModelConfig` entry
2. Ensure `modelId` matches provider API specification exactly
3. Set appropriate `defaultTemperature`, `defaultMaxTokens`, `defaultTopP`
4. No code changes required - registry is configuration-driven

### ML Training Models
- **Summarization:** T5 fine-tuned on MultiNews dataset
- **Classification:** BERT/NeoBERT fine-tuned on LIAR2 dataset
- **Embeddings:** Sentence Transformers for semantic search
- **Optimization:** Optuna for hyperparameter tuning

---

---

## API ENDPOINTS

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User authentication
- `POST /api/auth/forgot-password` - Password reset request
- `POST /api/auth/reset-password` - Password reset confirmation

### Fact-Checking
- `POST /api/process` - Progressive fact-check (SSE streaming)
- `POST /api/factcheck` - Simple fact-check (JSON response)
- `POST /api/validate` - Content validation

### User Profile
- `GET /api/profile/:id` - Get user profile
- `PUT /api/profile/:id` - Update profile
- `POST /api/profile/:id/upload` - Upload profile picture

### Media Processing
- `POST /api/image/upload` - Image upload with OCR
- `POST /api/video/analyze` - Video analysis

---

## DEPLOYMENT STATUS

**Production Ready Components:**
- ✅ Full-stack authentication system
- ✅ Progressive fact-checking pipeline with SSE
- ✅ Multi-provider LLM orchestration
- ✅ Responsive dashboard UI (mobile, tablet, desktop)
- ✅ Image OCR and video processing
- ✅ Evidence retrieval and ranking
- ✅ Database migrations and user management

**In Progress:**
- ⏳ Advanced summarization models
- ⏳ Fine-tuned BERT/NeoBERT deployment
- ⏳ Real-time collaboration features

---

## 📁 Detailed Structure

```bash
FactuAI/                       # Monorepo root
├── backend/                   # Flask Python API
│   ├── api/                   # API route handlers (blueprints)
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── factcheck.py       # Fact-checking endpoints
│   │   ├── image.py           # Image processing endpoints
│   │   ├── profile.py         # User profile endpoints
│   │   └── ...
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration management
│   │   ├── exceptions.py      # Custom exceptions
│   │   ├── helpers.py         # Helper functions
│   │   └── logging.py         # Logging setup
│   ├── database/              # Database layer
│   │   ├── connection.py      # DB connection management
│   │   └── models/            # SQLAlchemy models
│   ├── pipeline/              # ML pipeline & orchestration
│   │   ├── orchestrator.py    # Main fact-checking flow
│   │   ├── evidence/          # Evidence retrieval
│   │   ├── extraction/        # Feature extraction
│   │   ├── fetchers/          # Data fetchers (News API, scraping)
│   │   └── summarization/     # Text summarization
│   ├── services/              # Business logic
│   │   ├── llm/               # LLM providers (OpenRouter, NVIDIA, Local)
│   │   ├── classifier/        # Claim classification (BERT)
│   │   ├── search/            # Google Search integration
│   │   ├── factcheck_service.py
│   │   ├── ocr.py             # Tesseract OCR
│   │   └── ...
│   ├── schemas/               # Request/response validation (Pydantic)
│   ├── scripts/               # Backend utilities
│   │   ├── test_db_connection.py
│   │   └── verify_structure.py
│   ├── tests/                 # Pytest suite
│   ├── uploads/               # User uploads (profile pictures)
│   ├── app.py                 # Flask app entry point
│   ├── requirements-core.txt  # Cloud mode dependencies (~50MB)
│   └── requirements-local.txt # Local mode dependencies (~4GB+)
│
├── frontend/                  # Next.js 15 + React 19 UI
│   ├── src/
│   │   ├── app/               # Next.js App Router
│   │   │   ├── dashboard/     # Dashboard pages
│   │   │   │   ├── features/  # Feature modules
│   │   │   │   │   ├── inputs/    # Text, Image, Video tabs
│   │   │   │   │   ├── results/   # Results display
│   │   │   │   │   └── history/   # History panel
│   │   │   │   ├── hooks/         # Custom React hooks
│   │   │   │   ├── profile/       # Profile page
│   │   │   │   ├── services/      # API service layer
│   │   │   │   ├── layout.tsx     # Dashboard layout (Sidebar + Header)
│   │   │   │   └── page.tsx       # Dashboard home
│   │   │   ├── login/, register/, forgot-password/ # Auth pages
│   │   │   ├── globals.css    # Global styles + Tailwind
│   │   │   └── layout.tsx     # Root layout
│   │   ├── components/
│   │   │   ├── ai/            # AI model selector
│   │   │   ├── dashboard/     # Header, Sidebar
│   │   │   ├── landing/       # Landing page components
│   │   │   └── ui/            # shadcn/ui primitives
│   │   ├── config/            # AI models registry
│   │   ├── stores/            # Zustand state stores
│   │   └── lib/               # Utility functions
│   ├── public/                # Static assets
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts         # API proxy configuration
│   └── .env.local             # Frontend environment variables
│
├── scripts/                   # Cross-project orchestration
│   └── launch.bat             # Unified launcher (cloud/local modes)
│
├── .env                       # Root environment variables (gitignored)
├── .env.example               # Environment template
├── CONSTITUTIONS.md           # Coding standards (AI agent rules)
├── UNICODE_FIX_REPORT.md      # Recent Unicode & model fixes
├── LICENSE                    # MIT License
└── README.md                  # This file (consolidated documentation)
```

---

## ARCHITECTURE OVERVIEW

### Backend (Flask 3 + Python)

**Layer Structure:**
```
API Layer (Flask Blueprints)
    ↓
Service Layer (Business Logic)
    ↓
Pipeline Layer (Orchestration)
    ↓
Data Layer (SQLAlchemy ORM)
```

**Key Components:**
- **LLM Factory:** Dynamic provider switching (OpenRouter/NVIDIA/Local)
- **Pipeline Orchestrator:** Fact-checking workflow coordination
- **Evidence Ranker:** Sentence embedding similarity scoring
- **Classifier Service:** BERT-based claim classification
- **OCR Service:** Tesseract text extraction from images

**Dual-Mode Operation:**
- **Cloud Mode** (Default): External API calls, minimal dependencies
- **Local Mode**: GPU-accelerated local inference, full ML stack

### Frontend (Next.js 15 + React 19)

**Architecture Pattern:**
```
App Router (Pages)
    ↓
Feature Modules (Dashboard)
    ↓
UI Components (shadcn/ui)
    ↓
State Stores (Zustand)
```

**Key Features:**
- **App Shell Pattern:** Persistent sidebar + header layout
- **Mobile-First Design:** Responsive breakpoints (375px → 1280px+)
- **Config-Driven Models:** Registry pattern for AI provider management
- **Type-Safe API:** Full TypeScript coverage with strict mode
- **Progressive Enhancement:** SSE streaming for real-time updates

**State Management:**
- `ai-store.ts` - Model selection (persisted to localStorage)
- `useUser` hook - Authentication context
- `useFactCheck` hook - Fact-checking workflow
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

**Model Selector Text Overflow:**
```
Status: Fixed in v2.1.0
Solution: Removed third line (modelId display) from selector UI
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

**Version:** 2.1.0  
**Last Updated:** 2025-12-07T21:35:00Z  
**Author:** Zayed Ramadan Rahmat  
**Institution:** Asia Pacific University, Malaysia  
**Project Type:** Final Year Project (Computer Science - AI)  
**Repository:** https://github.com/zayedrmdn/FactuAI  
**Documentation Format:** AI Agent-Optimized Structured Markdown

---

## REFERENCES

- [Next.js 15 Documentation](https://nextjs.org/docs)
- [Flask 3 Documentation](https://flask.palletsprojects.com/)
- [OpenRouter API](https://openrouter.ai/docs)
- [NVIDIA NIM API](https://docs.api.nvidia.com/)
- [Tailwind CSS 4](https://tailwindcss.com/docs)
- [shadcn/ui](https://ui.shadcn.com/)
- [CONSTITUTIONS.md](./CONSTITUTIONS.md) - Coding Standards
- [UNICODE_FIX_REPORT.md](./UNICODE_FIX_REPORT.md) - Recent Fixes

---

**END OF DOCUMENT**
