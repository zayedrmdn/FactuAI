---
title: FactuAI Constitution
version: 3.0.0
last_updated: 2025-12-08
authors: [Zayed Ramadan Rahmat]
audience: AI Agents, Developers, Code Contributors
status: Active Governance Document
format: Structured Markdown for AI Parsing
---

# FactuAI Constitution

**Document Type**: Coding Standards & Architectural Governance  
**Enforcement**: Mandatory for all code contributions and AI agent operations  
**Scope**: Full-stack monorepo (Next.js frontend, Flask backend)

---

## 🎯 Core Principles

### 1. Single Responsibility
- Each module, class, function has ONE clear purpose
- Avoid bloated multi-concern components
- Extract reusable logic into separate utilities/hooks

### 2. Defensive Programming
- Validate ALL inputs at API boundaries and function entry
- Handle edge cases explicitly (empty strings, null, network failures)
- Provide fallbacks for critical operations
- Never assume data is well-formed

### 3. Lazy Loading
- Load expensive resources only when needed
- Use dynamic imports for code splitting
- Defer non-critical initialization until after first render

### 4. Reuse Before Create
- Check existing components/hooks/utilities before writing new code
- Extend or compose existing logic vs duplicating
- Follow established patterns in the codebase

### 5. Production-Ready Code Only
- All code must be **final and complete**
- No placeholders, TODOs, or half-implemented features in commits
- Test edge cases before merging

---

## 📂 Project Structure

### Backend (`backend/`)

```
backend/
├── core/              # Config, logging, exceptions, helpers
├── api/               # REST API endpoints (Flask blueprints)
├── services/          # Business logic (LLM, classifiers, search)
├── pipeline/          # Fact-checking orchestration
├── database/          # Models and connection
├── schemas/           # Request/response validation (Pydantic)
└── tests/             # Unit and integration tests
```

**Rules**:
- API routes use blueprints (`api/`)
- Business logic in `services/`, NOT route handlers
- Database models in `database/models/`
- Shared utilities in `core/`

### Frontend (`frontend/src/`)

```
frontend/src/
├── app/               # Next.js App Router pages (layout only)
├── components/        # Reusable components
│   ├── ui/            # shadcn/ui components
│   ├── ai/            # AI components (PipelineModelConfig)
│   ├── dashboard/     # Dashboard-specific components
│   └── landing/       # Landing page components
├── lib/               # Utilities (API clients, hooks, validation)
├── config/            # Configuration (ai-models.ts)
├── stores/            # Zustand state stores (ai-store.ts)
└── types/             # Global TypeScript types
```

**Rules**:
- Pages are **layout only** - logic in hooks/services
- Shared UI in `components/ui/`
- Configuration in `config/`, NOT components
- Global state uses Zustand stores in `stores/`
- TypeScript interfaces for ALL props/responses

---

## 🎨 Code Style

### General

- **File Size**: < 500 lines (split large files)
- **Naming**:
  - PascalCase: Components, classes
  - camelCase: Functions, variables, hooks
  - UPPER_SNAKE_CASE: Constants
- **Imports**: Group and sort (stdlib → third-party → local)

### TypeScript/React

- Use `Readonly<>` for all props (prevent mutations)
- Prefer functional components with hooks
- Use `useCallback`/`useMemo` for expensive operations
- Destructure props at function signature
- Use Tailwind classes (avoid inline styles)

### Python

- Type hints **mandatory** for all functions
- Docstrings for public functions/classes (Google style)
- Pydantic models for ALL API schemas
- Custom exceptions from `core/exceptions.py`
- Centralized logger from `core/logging.py`

---

## 🏗️ Architecture Patterns

### Backend: Layered Architecture

1. **API Layer** (`api/`): Validates input → calls service → formats response
2. **Service Layer** (`services/`): Business logic, external API/DB interactions
3. **Data Layer** (`database/`): SQLAlchemy models and queries

**Dependency Injection**:
- Services initialized in `services/service_manager.py`
- Pass dependencies explicitly (no global state)

**Error Handling**:
- `try/except` around external API calls
- Structured responses: `{"error": "message", "details": {...}}`

### Frontend: Component Composition

1. **Component Composition**: Build complex UIs from small, focused components
2. **State Management**:
   - `useState`: UI-only state (modal open/close)
   - Custom hooks: Shared logic (`useAppState`, `useSettings`)
   - localStorage: Persistence (preferences, history)
3. **Data Fetching**:
   - Centralize API calls in `lib/api.ts`
   - Explicit loading/error states

---

## 🎨 Design System

### Colors (Tailwind CSS Variables)

| Token            | Light          | Dark           | Usage                  |
|------------------|----------------|----------------|------------------------|
| `--primary`      | Indigo 600     | Indigo 500     | Buttons, links, accents|
| `--secondary`    | Slate 100      | Slate 800      | Secondary buttons      |
| `--destructive`  | Red 600        | Red 900        | Delete actions, errors |
| `--muted`        | Slate 100      | Slate 800      | Disabled states        |
| `--sidebar`      | Slate 50       | Slate 950      | Sidebar background     |

#### Score Colors (Semantic)

| Token              | Color     | Usage (Confidence)    |
|--------------------|-----------|-----------------------|
| `--score-very-high`| Green 600 | 80-100% (excellent)   |
| `--score-high`     | Lime 600  | 60-79% (good)         |
| `--score-medium`   | Amber 600 | 40-59% (moderate)     |
| `--score-low`      | Orange 600| 20-39% (poor)         |
| `--score-very-low` | Red 600   | 0-19% (failing)       |

**Example**:
```tsx
// ✅ CORRECT: Semantic CSS variables
const color = score >= 80 ? "oklch(var(--score-very-high))" : "oklch(var(--score-low))";
```

#### Model Tier Colors

| Class                 | Color           | Tier Level            |
|-----------------------|-----------------|-----------------------|
| `.badge-tier-free`    | Green 50/700    | Free / Open Source    |
| `.badge-tier-low`     | Blue 50/700     | Low cost / Standard   |
| `.badge-tier-medium`  | Yellow 50/700   | Medium cost / Balanced|
| `.badge-tier-high`    | Orange 50/700   | High cost / Advanced  |
| `.badge-tier-premium` | Purple 50/700   | Premium / SOTA        |

### Typography

- **Font**: Geist Sans (variable font, next/font)
- **Sizes**: `text-2xs` (10px), `text-xs` (12px), `text-sm` (14px), `text-base` (16px), `text-lg` (18px), `text-xl` (20px)

### Semantic Layout Utilities

| Class         | Value    | Usage                      |
|---------------|----------|----------------------------|
| `w-modal-sm`  | 20rem    | Small modals (320px)       |
| `w-modal-md`  | 25rem    | Medium modals (400px)      |
| `w-modal-lg`  | 32rem    | Large modals (512px)       |
| `h-card`      | 18.75rem | Card height (300px)        |
| `h-panel`     | 37.5rem  | Panel height (600px)       |
| `min-h-card`  | 18.75rem | Minimum card height        |
| `min-h-panel` | 37.5rem  | Minimum panel height       |

**Example**:
```tsx
// ✅ CORRECT: Semantic utilities
<DropdownMenuContent className="w-modal-sm" />
<div className="min-h-panel" />
```

### Spacing

- Use Tailwind scale: `p-2`, `m-4`, `gap-6`
- Consistent: `p-4` for card padding, `gap-6` for grid gaps

---

## 📱 Mobile Responsiveness

### Breakpoints

- Mobile: `< 768px` (base, no prefix)
- Tablet: `md:` ≥ 768px
- Desktop: `lg:` ≥ 1024px
- Wide: `xl:` ≥ 1280px

### Guidelines

1. **Mobile-first**: Base styles for smallest screen
2. **Progressive enhancement**: Use `md:`/`lg:` for larger screens
3. **Test on 375px** (iPhone SE/8 minimum)

### Dashboard Layout Pattern

**Desktop (≥ 768px)**:
- Sidebar visible (collapsible: 64px ↔ 256px)
- Header shows branding + user profile
- Content fills remaining space

**Mobile (< 768px)**:
- Sidebar hidden by default
- Hamburger menu in Header
- Sidebar as overlay drawer
- Backdrop closes drawer
- Body scroll prevented when open

**Touch Targets**:
- Minimum: 44×44px for buttons/links
- Spacing: 8px between interactive elements
- Icons: 20-24px (`h-5 w-5` or `h-6 w-6`)

**Example**:
```tsx
// ✅ CORRECT: Mobile stacks, desktop side-by-side
<div className="flex flex-col md:flex-row gap-4">
  <aside className="w-full md:w-64">Sidebar</aside>
  <main className="flex-1">Content</main>
</div>
```

---

## 🤖 AI Model Registry Pattern

### Architecture

Config-driven registry for scalability and maintainability.

**Key Files**:
- `frontend/src/types/ai.ts` - TypeScript interfaces
- `frontend/src/config/ai-models.ts` - Model registry (single source of truth)
- `frontend/src/stores/ai-store.ts` - Zustand store with localStorage
- `frontend/src/components/ai/PipelineModelConfig.tsx` - UI component

### Current Model Registry

**OpenRouter (Free Tier)**:
```
alibaba/tongyi-deepresearch-30b-a3b:free       (Research)
allenai/olmo-3-32b-think:free                  (Reasoning)
openai/gpt-oss-120b:free                       (General)
nvidia/nemotron-nano-9b-v2:free                (Fast)
meituan/longcat-flash-chat:free                (Conversational)
```

**NVIDIA NIM**:
```
meta/llama-3.1-405b-instruct                   (Premium SOTA)
meta/llama-3.1-70b-instruct                    (High Performance)
meta/llama-3.1-8b-instruct                     (Lightweight)
mistralai/mistral-nemotron                     (Balanced)
qwen/qwen2.5-7b-instruct                       (Default: Fast & Efficient)
```

**Defaults**:
- Provider: `nvidia`
- Model: `qwen/qwen2.5-7b-instruct`
- Fallback: `alibaba/tongyi-deepresearch-30b-a3b:free` (OpenRouter)

### Adding a New Model

**CRITICAL**: Model IDs must match provider API specs EXACTLY (case-sensitive, include suffixes).

**For Existing Provider**:

1. Open `frontend/src/config/ai-models.ts`
2. Add new `ModelConfig` to provider's `models` array:

```typescript
{
  id: 'nvidia-qwen2.5-7b',
  displayName: 'Qwen 2.5 7B Instruct',
  provider: 'nvidia',
  modelId: 'qwen/qwen2.5-7b-instruct',    // EXACT API match
  description: 'Fast and efficient model',
  defaultTemperature: 0.2,
  defaultMaxTokens: 1024,
  defaultTopP: 0.7,
  defaultSystemPrompt: GENERAL_SYSTEM_PROMPT,
  capabilities: {
    contextWindow: 32768,
    supportsStreaming: true,
    supportsFunctionCalling: true,
    supportsVision: false,
  },
  tier: 'low',                              // free, low, medium, high, premium
  isRecommended: true,
}
```

3. Save file (no code changes required - configuration-driven)
4. Update `.env` if changing defaults:
```env
NVIDIA_MODEL=qwen/qwen2.5-7b-instruct
OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b:free
```

### State Management

- User selection persisted to localStorage automatically
- Session overrides (temperature, tokens) are temporary
- Invalid selections fall back to defaults

### UI Integration

- **PipelineModelConfig**: Task-specific model selection (Intent/Extraction/Reasoning)
- **Visual feedback**: Active pipeline stage indicators
- **Persistence**: Selections stored in localStorage via Zustand
- **No prop drilling**: Components access stores directly

---

## 🚀 Performance

### Frontend

- **Code Splitting**: `next/dynamic` for heavy components
- **Image Optimization**: `next/image` for all images
- **Lazy Hydration**: Defer non-critical JS until after first paint

### Backend

- **Caching**: Cache API responses (e.g., news articles)
- **Connection Pooling**: SQLAlchemy's connection pool
- **Async Operations**: Threading/async for parallel API calls

---

## 🧪 Testing

### Backend

- **Unit Tests**: Test services/utilities in isolation
- **Integration Tests**: Test API endpoints end-to-end
- **Fixtures**: pytest fixtures for DB/service mocks

### Frontend

- **Component Tests**: Rendering and user interactions
- **Hook Tests**: `@testing-library/react-hooks`
- **E2E Tests**: Playwright for critical flows

---

## 📝 Git Workflow

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `refactor/description`: Code improvements
- `docs/description`: Documentation changes

### Commit Messages (Conventional Commits)

- `feat: add video upload support`
- `fix: resolve sidebar overlap issue`
- `refactor: simplify Sidebar component`
- `docs: update README with troubleshooting`

### Pull Requests

- **Title**: Clear and descriptive
- **Description**: Explain "why" and "what"
- **Testing**: Describe how to test changes
- **Screenshots**: Include for UI changes

---

## 🔒 Security

- **Never commit secrets** (API keys, passwords)
- Use `.env` files for environment variables (add to `.gitignore`)
- **Validate and sanitize** all user input on backend
- **Use HTTPS** for production
- **Rate limiting** on public API endpoints

---

## 🤝 Contributing

All contributors must:
1. Read and follow this document
2. Run linters/formatters before committing
3. Write tests for new features
4. Update documentation for API changes
5. Submit PRs for review (no direct commits to `main`)

---

**Document Version**: 3.0.0  
**Last Updated**: December 2025
