---
title: FactuAI Constitutions
version: 2.1.0
last_updated: 2025-12-07T21:35:00Z
authors: [Zayed Ramadan Rahmat]
audience: AI Agents, Automated Systems, Developers
status: Active Governance Document
repository: https://github.com/zayedrmdn/FactuAI
format: Structured Markdown for AI Parsing
---

# FactuAI Constitutions

**Document Type:** Coding Standards & Architectural Governance  
**Enforcement:** Mandatory for all code contributions and AI agent operations  
**Scope:** Full-stack monorepo (frontend, backend, configuration)

This document defines the **coding conventions**, **architectural principles**, and **governance rules** for the FactuAI project. All contributors and AI agents working on this codebase must follow these standards.

---

## 🎯 Core Principles

### 1. Single Responsibility
- Each module, class, and function should have **one clear purpose**.
- Avoid bloated components with multiple concerns.
- Extract reusable logic into separate utilities or hooks.

### 2. Defensive Programming
- **Validate all inputs** at API boundaries and function entry points.
- **Handle edge cases** explicitly (e.g., empty strings, null values, network failures).
- **Provide fallbacks** for critical operations (e.g., default values, error states).
- Never assume data is well-formed or complete.

### 3. Lazy Loading
- Load expensive resources only when needed.
- Use dynamic imports for code splitting.
- Defer non-critical initialization until after first render.

### 4. Reuse Before Create
- Check existing components, hooks, and utilities before writing new code.
- Extend or compose existing logic rather than duplicating it.
- Follow established patterns in the codebase.

### 5. Production-Ready Code Only
- All code must be **final and complete**.
- No placeholders, TODOs, or half-implemented features in commits.
- Test edge cases before merging.

---

## 📂 Project Structure

### Backend (`backend/`)

```
backend/
├── core/              # Config, logging, exceptions, helpers (shared utilities)
├── api/               # REST API endpoints (Flask blueprints)
├── services/          # Business logic (LLM, classifiers, search)
├── pipeline/          # Fact-checking orchestration
├── database/          # Models and connection logic
├── schemas/           # Request/response validation
└── tests/             # Unit and integration tests
```

**Rules:**
- All API routes must use blueprints (`api/`).
- Business logic lives in `services/`, not in route handlers.
- Database models go in `database/models/`.
- Shared utilities (config, logging) live in `core/`.

### Frontend (`frontend/src/`)

```
frontend/src/
├── app/               # Next.js App Router pages
│   ├── dashboard/     # Dashboard feature module
│   │   ├── features/  # Feature-specific components
│   │   ├── hooks/     # Custom React hooks
│   │   ├── services/  # API client logic
│   │   └── types/     # TypeScript interfaces
├── components/        # Reusable components
│   ├── ui/            # shadcn/ui components
│   ├── ai/            # AI-related components (ModelSelector)
│   ├── dashboard/     # Dashboard-specific components
│   └── landing/       # Landing page components
├── config/            # Configuration files (ai-models.ts)
├── stores/            # Zustand state stores (ai-store.ts)
├── types/             # Global TypeScript types (ai.ts)
└── lib/               # Utilities (cn, API clients, etc.)
```

**Rules:**
- Pages are **layout only**. Logic lives in hooks or services.
- Feature modules are self-contained (`features/inputs/`, `features/results/`).
- Shared UI components go in `components/ui/`.
- Configuration belongs in `config/`, not components.
- Global state uses Zustand stores in `stores/`.
- Use TypeScript interfaces for all props and API responses.

---

## 🎨 Code Style

### General

- **File Size**: Keep files under **500 lines** where possible. Split large files into modules.
- **Naming**:
  - Use **PascalCase** for components and classes.
  - Use **camelCase** for functions, variables, and hooks.
  - Use **UPPER_SNAKE_CASE** for constants.
- **Imports**: Group and sort imports (standard library → third-party → local).

### TypeScript/React

- **Use `Readonly<>` for all props** to prevent accidental mutations.
- **Prefer functional components** with hooks over class components.
- **Use `useCallback` and `useMemo`** for expensive computations or frequently rendered components.
- **Destructure props** at function signature: `({ prop1, prop2 }: Props)`.
- **Avoid inline styles**. Use Tailwind classes or CSS modules.

### Python

- **Type hints are mandatory** for all function signatures.
- **Docstrings** for public functions and classes (Google style).
- **Use `Pydantic` models** for all API request/response schemas.
- **Error handling**: Use custom exceptions from `core/exceptions.py`.
- **Logging**: Use the centralized logger from `core/logging.py`.

---

## 🏗️ Architecture Patterns

### Backend

1. **Layered Architecture**:
   - **API Layer** (`api/`): Validates input, calls service layer, formats response.
   - **Service Layer** (`services/`): Contains business logic, interacts with external APIs/DBs.
   - **Data Layer** (`database/`): SQLAlchemy models and queries.

2. **Dependency Injection**:
   - Services are initialized in `services/service_manager.py`.
   - Pass dependencies explicitly (no global state).

3. **Error Handling**:
   - Use `try/except` blocks around external API calls.
   - Return structured error responses: `{"error": "message", "details": {...}}`.

### Frontend

1. **Component Composition**:
   - Build complex UIs by composing small, focused components.
   - Use `children` prop for flexible layouts.

2. **State Management**:
   - **Local state** (`useState`) for UI-only state (e.g., modal open/close).
   - **Custom hooks** for shared logic (e.g., `useAppState`, `useSettings`).
   - **localStorage** for persistence (e.g., user preferences, history).

3. **Data Fetching**:
   - Centralize API calls in `services/` (e.g., `services/api.ts`).
   - Handle loading and error states explicitly.

---

## 🔧 Design System

### Colors (Tailwind CSS Variables)

| Token                | Light Mode       | Dark Mode        | Usage                     |
|----------------------|------------------|------------------|---------------------------|
| `--primary`          | Indigo 600       | Indigo 500       | Buttons, links, accents   |
| `--secondary`        | Slate 100        | Slate 800        | Secondary buttons         |
| `--destructive`      | Red 600          | Red 900          | Delete actions, errors    |
| `--muted`            | Slate 100        | Slate 800        | Disabled states           |
| `--sidebar`          | Slate 50         | Slate 950        | Sidebar background        |
| `--sidebar-accent`   | Slate 100        | Slate 800        | Active sidebar items      |

#### Score Colors (Semantic)

Use these CSS variables for confidence/detection scores:

| Token                | Color            | Usage                          |
|----------------------|------------------|--------------------------------|
| `--score-very-high`  | Green 600        | 80-100% confidence (excellent) |
| `--score-high`       | Lime 600         | 60-79% confidence (good)       |
| `--score-medium`     | Amber 600        | 40-59% confidence (moderate)   |
| `--score-low`        | Orange 600       | 20-39% confidence (poor)       |
| `--score-very-low`   | Red 600          | 0-19% confidence (failing)     |
| `--score-trail`      | Gray 200/Slate 800| Progress bar background       |

**Usage Example:**
```tsx
// ❌ WRONG: Hard-coded hex colors
const color = score >= 80 ? "#16a34a" : "#dc2626";

// ✅ CORRECT: Semantic CSS variables
const color = score >= 80 ? "oklch(var(--score-very-high))" : "oklch(var(--score-very-low))";
```

#### Model Tier Colors

Use these utility classes for AI model tiers:

| Class                 | Color            | Tier Level                     |
|-----------------------|------------------|--------------------------------|
| `.badge-tier-free`    | Green 50/700     | Free / Open Source models      |
| `.badge-tier-low`     | Blue 50/700      | Low cost / Standard models     |
| `.badge-tier-medium`  | Yellow 50/700    | Medium cost / Balanced models  |
| `.badge-tier-high`    | Orange 50/700    | High cost / Advanced models    |
| `.badge-tier-premium` | Purple 50/700    | Premium / SOTA models          |

**Usage Example:**
```tsx
<Badge className="badge-tier-premium">Premium</Badge>
```

### Typography

- **Font Family**: Geist Sans (variable font via `next/font`)
- **Sizes**:
  - `text-2xs`: 0.625rem (10px) - **Use for tiny labels, badges**
  - `text-xs`: 0.75rem (12px)
  - `text-sm`: 0.875rem (14px)
  - `text-base`: 1rem (16px)
  - `text-lg`: 1.125rem (18px)
  - `text-xl`: 1.25rem (20px)

### Semantic Layout Utilities

Use these custom classes instead of arbitrary values:

| Class           | Value      | Usage                          |
|-----------------|------------|--------------------------------|
| `w-modal-sm`    | 20rem      | Small modals/dropdowns (320px) |
| `w-modal-md`    | 25rem      | Medium modals (400px)          |
| `w-modal-lg`    | 32rem      | Large modals (512px)           |
| `h-card`        | 18.75rem   | Card height (300px)            |
| `h-panel`       | 37.5rem    | Panel height (600px)           |
| `min-h-card`    | 18.75rem   | Minimum card height            |
| `min-h-panel`   | 37.5rem    | Minimum panel height           |

**Usage Example:**
```tsx
// ❌ WRONG: Arbitrary Tailwind values
<DropdownMenuContent className="w-[320px]" />
<div className="min-h-[600px]" />

// ✅ CORRECT: Semantic utilities
<DropdownMenuContent className="w-modal-sm" />
<div className="min-h-panel" />
```

### Spacing

- Use Tailwind's spacing scale: `p-2`, `m-4`, `gap-6`, etc.
- Consistent spacing: `p-4` for card padding, `gap-6` for grid gaps.

---

## 📱 Mobile Responsiveness

### Breakpoints (Tailwind)

- **Mobile**: `< 768px` (base styles, no prefix)
- **Tablet**: `md:` ≥ 768px
- **Desktop**: `lg:` ≥ 1024px
- **Wide**: `xl:` ≥ 1280px

### Mobile-First Guidelines

1. **Write base styles for mobile** (smallest screen first)
2. **Use `md:` and `lg:` prefixes** to enhance for larger screens
3. **Test on 375px viewport** (iPhone SE/8 minimum)

### Dashboard Layout Pattern

The dashboard uses a responsive sidebar-drawer pattern:

**Desktop (≥ 768px):**
- Sidebar visible on left (collapsible: 64px ↔ 256px)
- Content area fills remaining space
- Both render simultaneously

**Mobile (< 768px):**
- Sidebar hidden by default
- Hamburger menu button in Header
- Sidebar appears as overlay drawer when opened
- Backdrop overlay closes drawer on click
- Body scroll prevented when drawer open

**Implementation:**
```tsx
// ✅ CORRECT: Mobile-responsive sidebar
<div className="hidden md:flex">
  <Sidebar collapsed={collapsed} onToggle={toggle} isMobile={false} />
</div>

<div className={`mobile-drawer md:hidden ${open ? 'translate-x-0' : '-translate-x-full'}`}>
  <Sidebar collapsed={false} onToggle={close} isMobile={true} />
</div>

// Hamburger button (mobile only)
<Button className="md:hidden" onClick={toggleMobile}>
  <Menu className="h-5 w-5" />
</Button>
```

### Touch Target Guidelines

- **Minimum size**: 44×44px for tap targets (buttons, links)
- **Spacing**: 8px minimum between interactive elements
- **Icons**: 20-24px for mobile (use `h-5 w-5` or `h-6 w-6`)

### Responsive Component Patterns

```tsx
// ❌ WRONG: Fixed layout, not responsive
<div className="flex gap-4">
  <aside className="w-64">Sidebar</aside>
  <main className="flex-1">Content</main>
</div>

// ✅ CORRECT: Mobile stacks, desktop side-by-side
<div className="flex flex-col md:flex-row gap-4">
  <aside className="w-full md:w-64">Sidebar</aside>
  <main className="flex-1">Content</main>
</div>

// ❌ WRONG: Text hidden on mobile
<span className="text-sm">Welcome back, {name}</span>

// ✅ CORRECT: Hide secondary text on small screens
<span className="hidden sm:inline text-sm">Welcome back, {name}</span>
```

---

## 🚀 Performance

### Frontend

- **Code Splitting**: Use `next/dynamic` for heavy components.
- **Image Optimization**: Use `next/image` for all images.
- **Lazy Hydration**: Defer non-critical JS until after first paint.

### Backend

- **Caching**: Cache API responses where appropriate (e.g., news articles).
- **Connection Pooling**: Use SQLAlchemy's connection pool.
- **Async Operations**: Use threading/async for parallel API calls.

---

## 🧪 Testing

### Backend

- **Unit Tests**: Test services and utilities in isolation.
- **Integration Tests**: Test API endpoints end-to-end.
- **Fixtures**: Use `pytest` fixtures for database and service mocks.

### Frontend

- **Component Tests**: Test rendering and user interactions.
- **Hook Tests**: Test custom hooks with `@testing-library/react-hooks`.
- **E2E Tests**: Use Playwright for critical user flows.

---

## 📝 Git Workflow

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `refactor/description`: Code improvements
- `docs/description`: Documentation changes

### Commit Messages

Follow **Conventional Commits**:
- `feat: add video upload support`
- `fix: resolve sidebar overlap issue`
- `refactor: simplify Sidebar component`
- `docs: update README with troubleshooting`

### Pull Requests

- **Title**: Clear and descriptive
- **Description**: Explain the "why" and "what"
- **Testing**: Describe how to test the changes
- **Screenshots**: Include for UI changes

---

## 🤖 AI Model Registry Pattern

### Architecture

The application uses a **config-driven registry pattern** for managing AI providers and models. This ensures scalability and maintainability.

**Key Files:**
- `frontend/src/types/ai.ts` - TypeScript interfaces and types
- `frontend/src/config/ai-models.ts` - Model registry (single source of truth)
- `frontend/src/stores/ai-store.ts` - Zustand store with localStorage persistence
- `frontend/src/components/ai/ModelSelector.tsx` - UI component

### Current Model Registry (Production)

**OpenRouter Models (Free Tier):**
```
alibaba/tongyi-deepresearch-30b-a3b:free       (Recommended - Research)
allenai/olmo-3-32b-think:free                  (Reasoning)
openai/gpt-oss-120b:free                       (General Purpose)
nvidia/nemotron-nano-9b-v2:free                (Fast)
meituan/longcat-flash-chat:free                (Conversational)
```

**NVIDIA NIM Models:**
```
meta/llama-3.1-405b-instruct                   (Premium - SOTA)
meta/llama-3.1-70b-instruct                    (High Performance)
meta/llama-3.1-8b-instruct                     (Lightweight)
mistralai/mistral-nemotron                     (Balanced)
qwen/qwen2.5-7b-instruct                       (Default - Fast & Efficient)
```

**Default Configuration:**
- Provider: `nvidia`
- Model: `qwen/qwen2.5-7b-instruct`
- Fallback: `alibaba/tongyi-deepresearch-30b-a3b:free` (OpenRouter)

### Adding a New Model

**CRITICAL:** Model IDs must match provider API specifications EXACTLY (case-sensitive, include suffixes).

**For Existing Provider:**

1. Open `frontend/src/config/ai-models.ts`
2. Add new `ModelConfig` to provider's `models` array:

```typescript
{
  id: 'nvidia-qwen2.5-7b',               // Internal ID
  displayName: 'Qwen 2.5 7B Instruct',   // UI display
  provider: 'nvidia',                     // Provider ID
  modelId: 'qwen/qwen2.5-7b-instruct',   // API model ID (EXACT match)
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
  tier: 'low',                            // free, low, medium, high, premium
  isRecommended: true,
}
```

3. Save file. **No code changes required** - registry is configuration-driven.
4. Update `.env` if changing defaults:
   ```env
   NVIDIA_MODEL=qwen/qwen2.5-7b-instruct
   OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b:free
   ```

### Adding a New Provider

1. Add the provider ID to `AIProvider` type in `frontend/src/types/ai.ts`
2. Create model configs in `frontend/src/config/ai-models.ts`
3. Add a `ProviderConfig` entry to the `providers` array
4. Update `defaultProvider` in `modelRegistry` if needed

### State Management

- User selection is persisted to `localStorage` automatically
- Session overrides (temperature, tokens, etc.) are temporary
- Invalid selections fall back to defaults on load

### UI Integration

The `ModelSelector` component:
- Groups models by provider
- Shows model metadata (context window, tier, recommendations)
- Provides settings popover for parameter adjustment
- Handles all state updates through the Zustand store

**No prop drilling required** - all components can access the store directly via `useAIStore()`.

---

## 🔒 Security

- **Never commit secrets** (API keys, passwords) to Git.
- Use `.env` files for environment variables (add to `.gitignore`).
- **Validate and sanitize all user input** on the backend.
- **Use HTTPS** for production deployments.
- **Rate limiting**: Implement on public API endpoints.

---

## 📚 References

- [Next.js Documentation](https://nextjs.org/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [shadcn/ui](https://ui.shadcn.com/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

---

## 🤝 Contributing

All contributors must:
1. Read and follow this document.
2. Run linters and formatters before committing.
3. Write tests for new features.
4. Update documentation for API changes.
5. Submit PRs for review (no direct commits to `main`).

---

**Last Updated**: December 2024
