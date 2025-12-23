# Frontend Architecture

Next.js 16 frontend with feature-based colocation and Zustand state management.

---

## Tech Stack

- **Framework:** Next.js 16 (App Router, Turbopack)
- **Language:** TypeScript
- **Styling:** Tailwind CSS v4
- **State Management:** Zustand (feature stores)
- **HTTP Client:** Fetch API
- **Testing:** Vitest
- **Package Manager:** pnpm

---

## Directory Structure

```
frontend/src/
├── app/                    # Next.js App Router pages
│   ├── page.tsx            # Landing page
│   ├── dashboard/          # Main fact-check interface
│   ├── login/             
│   └── register/
├── features/               # Feature-based colocation (MANDATORY)
│   ├── ai-providers/       # Model selection, pipeline config
│   │   ├── index.ts        # Barrel exports
│   │   ├── types.ts
│   │   ├── constants.ts
│   │   ├── registry.ts     # Model/provider registry
│   │   ├── components/     # PipelineModelConfig, ActiveModelDisplay
│   │   └── stores/         # Zustand stores
│   ├── search/             # Search input, provider config
│   ├── analyze/            # Results display, claim cards
│   └── history/            # History panel, session management
├── components/             # Generic UI primitives ONLY
│   ├── ui/                 # shadcn/ui components
│   └── layout/             # Layout components
├── lib/                    # Shared utilities
│   └── hooks/              # Shared hooks (useFactCheck, useSystemConfig)
├── config/                 # Static configuration
└── types/                  # Cross-cutting TypeScript types
```

---

## Feature-Based Colocation (MANDATORY)

> [!IMPORTANT]
> **Rule:** Domain-specific components, hooks, and state *must* live in `frontend/src/features/*/`, NOT in `components/`.
>
> The `components/` directory is reserved for **generic, reusable UI primitives only** (buttons, modals, cards, etc.).

### Feature Module Anatomy

```
frontend/src/features/analyze/
├── index.ts          # Barrel exports (single entry point)
├── components/       # Feature-specific components
│   ├── AnalyzeCard.tsx
│   ├── ResultsDisplay.tsx
│   └── ClaimCard.tsx
├── hooks/            # Feature-specific hooks
│   └── useAnalyze.ts
└── types.ts          # Feature-specific types
```

### Barrel Exports Pattern

```typescript
// frontend/src/features/analyze/index.ts
export { AnalyzeCard } from './components/AnalyzeCard';
export { ResultsDisplay } from './components/ResultsDisplay';
export { useAnalyze } from './hooks/useAnalyze';
export type { AnalyzeResult } from './types';
```

**Usage:**
```typescript
// ✅ Good: Import from feature barrel
import { AnalyzeCard, useAnalyze } from '@/features/analyze';

// ❌ Bad: Import from deep paths
import { AnalyzeCard } from '@/features/analyze/components/AnalyzeCard';
```

---

## State Management (Zustand)

### Feature Stores

Each feature owns its state via Zustand stores:

```typescript
// frontend/src/features/ai-providers/stores/selection.ts
import { create } from 'zustand';

interface AIStoreState {
  selection: {
    provider: string;
    modelId: string;
    sessionOverrides?: {
      temperature?: number;
      max_tokens?: number;
    };
  };
  setProvider: (provider: string) => void;
  setModelId: (modelId: string) => void;
}

export const useAIStore = create<AIStoreState>((set) => ({
  selection: {
    provider: 'openrouter',
    modelId: 'openrouter-llama-3.3-70b',
  },
  setProvider: (provider) =>
    set((state) => ({ selection: {... state.selection, provider } })),
  setModelId: (modelId) =>
    set((state) => ({ selection: { ...state.selection, modelId } })),
}));
```

### Store Usage

```typescript
import { useAIStore } from '@/features/ai-providers';

function ModelSelector() {
  const { selection, setModelId } = useAIStore();
  
  return (
    <select value={selection.modelId} onChange={(e) => setModelId(e.target.value)}>
      {/* options */}
    </select>
  );
}
```

---

## Optimistic UI Patterns

### Pipeline Progress Visualization

The `PipelineStepLoader` component provides immersive visual feedback for the 4-phase process with orbital animations and progress tracking:

```typescript
// frontend/src/features/analyze/components/PipelineStepLoader.tsx
export function PipelineStepLoader() {
  // Features:
  // - Orbital particle animation around central icon
  // - SVG progress ring with gradient stroke
  // - Animated icon transitions between phases (Brain → Search → Globe → Shield)
  // - Step progress dots with animated connectors
  // - Overall progress bar with percentage
  
  // Steps: 'Extracting Claims' → 'Strategizing' → 'Gathering Evidence' → 'Synthesizing Verdict'
}
```

This ensures perceived performance and engagement while the backend executes complex chains.

---

## API Communication

### Centralized API Client

**File:** `frontend/src/lib/api.ts` (or hooks like `useFactCheck.ts`)

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export async function analyzeText(text: string, modelId?: string) {
  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model_id: modelId }),
  });
  
  if (!response.ok) {
    throw new Error(await response.text());
  }
  
  return response.json();
}
```

### Error Handling

```typescript
// frontend/src/lib/hooks/useFactCheck.ts
try {
  const response = await fetch(API_ANALYZE_URL, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  
  if (!response.ok) {
    const data = await response.json();
    const errorMessage = data.detail || data.error || `Server responded ${response.status}`;
    
    // Handle specific errors with user-friendly messages
    if (errorMessage.toLowerCase().includes('no claims extracted')) {
      setFactCheckError('No verifiable claims found...');
      return;
    }
    
    throw new Error(errorMessage);
  }
} catch (e) {
  // Handle network errors
}
```

---

## Routing & Pages

### App Router Structure

```
frontend/src/app/
├── page.tsx                # Landing page
├── layout.tsx              # Root layout
├── dashboard/
│   ├── page.tsx            # Main dashboard
│   └── layout.tsx          # Dashboard layout
├── login/page.tsx
└── register/page.tsx
```

### Page Components (Thin)

**Rule:** Pages should be thin - delegate logic to hooks and feature components.

```typescript
// app/dashboard/page.tsx
import { AnalyzeCard } from '@/features/analyze';
import { useFactCheck } from '@/lib/hooks/useFactCheck';

export default function DashboardPage() {
  const { input, setInput, handleFactCheck, factResults, loading } = useFactCheck();
  
  return (
    <div>
      <AnalyzeCard
        input={input}
        onInputChange={setInput}
        onSubmit={handleFactCheck}
        results={factResults}
        loading={loading}
      />
    </div>
  );
}
```

---

## Styling Conventions

### Tailwind CSS v4

- Use utility classes for styling
- Create reusable components in `components/ui/` for common patterns
- Use CSS variables for theming (defined in `globals.css`)

```css
/* app/globals.css */
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 221.2 83.2% 53.3%;
}
```

---

## Testing

### Vitest Setup

```bash
cd frontend
pnpm test
```

### Test Structure

```
frontend/src/features/analyze/__tests__/
├── AnalyzeCard.test.tsx
└── useAnalyze.test.ts
```

See [../06-testing/frontend-tests.md](../06-testing/frontend-tests.md).

---

## Best Practices

1. **Feature-based colocation**: Domain logic in `features/`, primitives in `components/`
2. **Barrel exports**: Use `index.ts` in feature modules
3. **Thin pages**: Delegate to hooks and feature components
4. **Zustand for feature state**: One store per feature
5. **Optimistic UI**: Show progress immediately, update on response
6. **Error handling**: User-friendly messages for common errors

---

See also:
- [Overview](overview.md) - High-level architecture
- [Backend Architecture](backend.md) - FastAPI patterns
- [Constitution](../01-rules/constitution.md) - Frontend rules
