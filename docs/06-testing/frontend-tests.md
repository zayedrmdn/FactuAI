# Frontend Testing Guide

Using Vitest for testing the FactuAI frontend.

---

## Overview

**Framework:** Vitest  
**Test Location:** `frontend/__tests__/` (minimal coverage)  
**Coverage:** Component tests, hook tests, integration tests

> [!NOTE]
> **Current State:** The project has minimal test coverage. The test structure and examples shown below represent the **planned/recommended** approach for future test development, not the current implementation.

---

## Setup

```bash
cd frontend
pnpm install
```

**Dependencies:**
- `vitest` - Test framework (Vite-native)
- `@testing-library/react` - React component testing
- `@testing-library/dom` - DOM utilities
- `jsdom` - DOM environment simulation

---

## Running Tests

### All Tests

```bash
pnpm test
```

### Watch Mode

```bash
pnpm test --watch
```

### Coverage

```bash
pnpm test --coverage
```

### Specific Test File

```bash
pnpm test src/features/analyze/__tests__/AnalyzeCard.test.tsx
```

---

## Test Structure

```
frontend/src/features/
├── analyze/
│   ├── components/
│   │   └── AnalyzeCard.tsx
│   └── __tests__/
│       ├── AnalyzeCard.test.tsx
│       └── useAnalyze.test.ts
├── ai-providers/
│   └── __tests__/
│       └── selection.test.ts
└── search/
    └── __tests__/
        └── useSearch.test.ts
```

**Pattern:** Tests should live in `__tests__/` folder next to the code they test (planned structure).

> [!WARNING]
> **Actual Current State:** Only `frontend/__tests__/sanity.test.tsx` exists. The feature-colocated test structure shown above is aspirational and should be implemented as testing coverage expands.

---

## Writing Tests

### Component Tests

```tsx
// src/features/analyze/__tests__/AnalyzeCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { AnalyzeCard } from '../components/AnalyzeCard';

describe('AnalyzeCard', () => {
  it('renders input field', () => {
    render(<AnalyzeCard input="" onInputChange={() => {}} onSubmit={() => {}} />);
    
    const input = screen.getByPlaceholderText(/enter text/i);
    expect(input).toBeInTheDocument();
  });
  
  it('calls onSubmit when button clicked', () => {
    const handleSubmit = vi.fn();
    render(<AnalyzeCard input="test" onInputChange={() => {}} onSubmit={handleSubmit} />);
    
    const button = screen.getByRole('button', { name: /analyze/i });
    fireEvent.click(button);
    
    expect(handleSubmit).toHaveBeenCalled();
  });
});
```

### Hook Tests

```typescript
// src/lib/hooks/__tests__/useFactCheck.test.ts
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { useFactCheck } from '../useFactCheck';

describe('useFactCheck', () => {
  it('initializes with empty input', () => {
    const { result } = renderHook(() => useFactCheck());
    
    expect(result.current.input).toBe('');
    expect(result.current.factResults).toEqual([]);
  });
  
  it('updates input when setInput called', () => {
    const { result } = renderHook(() => useFactCheck());
    
    act(() => {
      result.current.setInput('test claim');
    });
    
    expect(result.current.input).toBe('test claim');
  });
});
```

### Zustand Store Tests

```typescript
// src/features/ai-providers/stores/__tests__/selection.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useAIStore } from '../selection';

describe('AI Selection Store', () => {
  beforeEach(() => {
    // Reset store before each test
    useAIStore.setState({
      selection: { modelId: 'default', provider: 'openrouter' }
    });
  });
  
  it('updates model ID', () => {
    const { setModelId } = useAIStore.getState();
    
    setModelId('new-model-id');
    
    const { selection } = useAIStore.getState();
    expect(selection.modelId).toBe('new-model-id');
  });
});
```

---

## Mocking

### Mocking fetch

```typescript
import { vi } from 'vitest';

global.fetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({
      request_id: '123',
      claims: [{ claim_text: 'test', verdict: 'TRUE', confidence: 0.95 }]
    })
  })
) as any;

it('calls API and displays results', async () => {
  const { result } = renderHook(() => useFactCheck());
  
  await act(async () => {
    await result.current.handleFactCheck();
  });
  
  expect(result.current.factResults).toHaveLength(1);
});
```

### Mocking Zustand Stores

```typescript
import { vi } from 'vitest';

vi.mock('@/features/ai-providers', () => ({
  useAIStore: () => ({
    selection: { modelId: 'test-model', provider: 'test' },
    setModelId: vi.fn()
  })
}));
```

---

## Example Tests

### Testing Error Handling

```tsx
it('displays error message on API failure', async () => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: false,
      json: () => Promise.resolve({ detail: 'No claims extracted' })
    })
  ) as any;
  
  const { result } = renderHook(() => useFactCheck());
  
  await act(async () => {
    await result.current.handleFactCheck();
  });
  
  expect(result.current.factCheckError).toContain('No verifiable claims');
});
```

### Testing Loading States

```tsx
it('shows loading state during fact-check', async () => {
  const { result } = renderHook(() => useFactCheck());
  
  act(() => {
    result.current.handleFactCheck();
  });
  
  expect(result.current.loading).toBe('factcheck');
  
  // Wait for completion
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 100));
  });
  
  expect(result.current.loading).toBe(null);
});
```

---

## Vitest Configuration

**File:** `frontend/vitest.config.ts`

```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts']
  }
});
```

**Setup File:** `frontend/src/test/setup.ts`

```typescript
import '@testing-library/jest-dom';
```

---

## Coverage Goals

- **Components:** 80%+ coverage
- **Hooks:** 90%+ coverage
- **Stores:** 90%+ coverage
- **Utilities:** 95%+ coverage

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Frontend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Setup Node
      uses: actions/setup-node@v2
      with:
        node-version: '20'
    
    - name: Install pnpm
      run: npm install -g pnpm
    
    - name: Install dependencies
      run: |
        cd frontend
        pnpm install
    
    - name: Run tests
      run: |
        cd frontend
        pnpm test --coverage
```

---

## Best Practices

1. **Test user behavior**, not implementation details
2. **Use `data-testid`** sparingly (prefer accessible queries)
3. **Mock external dependencies** (APIs, stores)
4. **Test error states** and edge cases
5. **Keep tests focused** (one assertion per test ideally)
6. **Use descriptive test names** (`it('displays error when API fails')`)

---

See also:
- [Frontend Architecture](../03-architecture/frontend.md)
- [Backend Testing](backend-tests.md)
- [Test Claims Benchmark](test-claims.md)
