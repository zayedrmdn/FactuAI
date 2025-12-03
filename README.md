# FactuAI

**FactuAI** is a full-stack AI-powered application for:
- 📰 **News summarization**
- 🔍 **Fact-checking political claims**

Built as part of my **Final Year Project** in Computer Science (AI) at APU, Malaysia.

---

## 🎯 Goals

- Extract concise summaries from long-form news articles
- Automatically classify factual accuracy of statements (e.g. `true`, `false`, `pants_on_fire`)
- Empower responsible media consumption using NLP and deep learning

---

## 🛠️ Tech Stack

| Layer       | Tech                                      |
|-------------|-------------------------------------------|
| Frontend    | React (Next.js), Tailwind CSS, Shadcn UI  |
| Backend     | Flask (REST API), SQLAlchemy              |
| Database    | PostgreSQL                                |
| ML Models   | Hugging Face Transformers (BERT, T5)      |
| Tools       | PyTorch, Optuna, xFormers, Jupyter        |

---

## 🚀 Quick Start

### Installation

FactuAI uses **separate virtual environments** for each run mode:

| Mode | venv | Requirements | Size |
|------|------|--------------|------|
| Cloud | `.venv-cloud/` | `requirements-core.txt` | ~50MB |
| Local | `.venv-local/` | `requirements-local.txt` | ~4GB+ |

**Option 1: Automatic Setup (Recommended)**
```bash
cd scripts
launch.bat cloud   # Creates .venv-cloud and starts in Cloud Mode
# OR
launch.bat local   # Creates .venv-local and starts in Local Mode
```

**Option 2: Manual Setup**
```bash
# Cloud Mode (lightweight - uses OpenRouter API)
python -m venv .venv-cloud
.venv-cloud\Scripts\activate     # Windows
pip install -r requirements-core.txt

# Local Mode (full ML stack - requires CUDA)
python -m venv .venv-local
.venv-local\Scripts\activate     # Windows
pip install -r requirements-local.txt
```

### Configuration

Create a `.env` file in the project root:
```env
# Run Mode: 'cloud' (default) or 'local'
APP_RUN_MODE=cloud

# LLM Provider (cloud mode only): 'openrouter' (default), 'nvidia', or 'auto'
LLM_PROVIDER=openrouter

# Database
DB_URI=postgresql://user:pass@localhost:5432/factuai

# OpenRouter API (default cloud provider)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=meta-llama/llama-4-maverick:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Nvidia NIM API (alternative cloud provider)
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
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

## 🧠 ML Models

- **Summarization**: fine-tuning T5 on [MultiNews](https://huggingface.co/datasets/alexfabbri/multi_news)
- **Fact-checking**: fine-tuning BERT and NeoBERT on [LIAR2](https://github.com/chengxuphd/liar2)

All models are trained with optimized preprocessing and hyperparameter tuning (Optuna).

---

## 📌 Current Status

- ✅ Dataset preprocessing (`LIAR2`, `MultiNews`)
- ✅ Tokenization, padding, vocabulary
- ✅ Model fine-tuning with Hugging Face
- ✅ Production-ready Dashboard with App Shell and modern UI
- ✅ Flask backend with PostgreSQL integration
- ✅ Dual run mode support (Cloud/Local)
- ⏳ In progress: dashboard summarization + fact-check endpoint
- ⏳ In progress: model evaluation, export, deployment

---

## 📁 Folder Structure

```bash
FactuAI/
├── backend/                   # Flask app
│   ├── core/                  # Centralized config, logging, exceptions, helpers
│   ├── api/                   # API endpoints (blueprints)
│   ├── services/              # Business logic
│   │   ├── llm/               # LLM providers (OpenRouter, Nvidia, Local)
│   │   ├── classifier/        # Claim classification (BERT-based)
│   │   ├── classifier_intent/ # Intent detection
│   │   └── search/            # Google Search integration
│   ├── pipeline/              # Fact-checking pipeline & orchestration
│   ├── database/              # Database models and connection
│   ├── schemas/               # Request/response schemas
│   └── tests/                 # Unit and integration tests
├── frontend/                  # Next.js app
│   └── src/
│       ├── app/dashboard/
│       │   ├── features/      # Feature-based components
│       │   │   ├── inputs/    # Text, image, video input
│       │   │   ├── results/   # Fact-check results display
│       │   │   ├── history/   # History panel
│       │   │   └── settings/  # Settings dialog
│       │   ├── hooks/         # React hooks
│       │   ├── services/      # API service layer
│       │   └── types/         # TypeScript types
│       └── components/ui/     # Shared UI components
├── scripts/                   # Launch scripts, verification tools
├── requirements-core.txt      # Lightweight deps (Cloud Mode ~50MB)
├── requirements-local.txt     # Full ML deps (Local Mode ~4GB+)
└── README.md
```

---

## 🛡️ License

Licensed under the [MIT License](LICENSE).  
Free for academic, research, and learning purposes.

---

## 👨‍💻 Author

**Zayed Ramadan Rahmat**  
Final Year BSc (Hons) Computer Science (AI), APU  
📍 Kuala Lumpur, Malaysia  
🔗 [LinkedIn](https://linkedin.com/in/zayedrmdn) · 📧 [Email](mailto:zayedrmdn@email.com)
