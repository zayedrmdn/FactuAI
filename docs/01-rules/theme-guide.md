---
title: FactuAI Theme Guide
version: 1.0.0
last_updated: 2025-12-21
audience: AI Agents, Developers, UI/UX Contributors
status: Active UI Standards
---

# FactuAI Theme Guide

Centralized UI/UX standards for professional, enterprise-ready interfaces.

---

## Typography

### Font Families

| Usage | Font | CSS Variable |
|-------|------|--------------|
| Body | Geist Sans | `var(--font-sans)` |
| Code | Geist Mono | `var(--font-mono)` |

### Font Sizes

| Class | Size | Line Height | Usage |
|-------|------|-------------|-------|
| `text-2xs` | 0.625rem (10px) | 0.875rem | Micro labels, badges |
| `text-xs` | 0.75rem (12px) | 1rem | Secondary info, timestamps |
| `text-sm` | 0.875rem (14px) | 1.25rem | Body text, descriptions |
| `text-base` | 1rem (16px) | 1.5rem | Primary body content |
| `text-lg` | 1.125rem (18px) | 1.75rem | Subheadings |
| `text-xl` | 1.25rem (20px) | 1.75rem | Section headers |
| `text-2xl` | 1.5rem (24px) | 2rem | Page titles |

### Font Weights

- `font-normal` (400): Body text
- `font-medium` (500): Labels, UI elements
- `font-semibold` (600): Headings, emphasis
- `font-bold` (700): Strong emphasis (use sparingly)

---

## Semantic Colors

### Base Palette

| Variable | Light | Dark | Usage |
|----------|-------|------|-------|
| `--background` | Slate 50 | Slate 950 | Page background |
| `--foreground` | Slate 900 | White | Primary text |
| `--card` | White | Slate 900 | Card surfaces |
| `--muted` | Slate 100 | Slate 800 | Subtle backgrounds |
| `--muted-foreground` | Slate 500 | Slate 400 | Secondary text |

### Accent Colors

| Variable | Color (Light) | Usage |
|----------|---------------|-------|
| `--primary` | Indigo 600 | CTAs, links, focus states |
| `--secondary` | Slate 100 | Secondary buttons |
| `--destructive` | Red 600 | Delete, errors |
| `--accent` | Slate 100 | Hover states |

### Verdict Colors

| Verdict | Color | Tailwind Class |
|---------|-------|----------------|
| TRUE | Green 600 | `text-green-600` |
| FALSE | Red 600 | `text-red-600` |
| MOSTLY_TRUE | Lime 600 | `text-lime-600` |
| MOSTLY_FALSE | Orange 600 | `text-orange-600` |
| MIXED | Amber 600 | `text-amber-600` |
| UNVERIFIABLE | Slate 500 | `text-slate-500` |

### Confidence Score Colors

| Level | Variable | Usage |
|-------|----------|-------|
| Very High (90-100%) | `--score-very-high` | Green 600 |
| High (70-89%) | `--score-high` | Lime 600 |
| Medium (50-69%) | `--score-medium` | Amber 600 |
| Low (30-49%) | `--score-low` | Orange 600 |
| Very Low (0-29%) | `--score-very-low` | Red 600 |

---

## Icons

### Library: Lucide React

**Mandatory:** Use [Lucide React](https://lucide.dev/) for all icons.

```tsx
import { Zap, FlaskConical, Check, X, AlertTriangle } from 'lucide-react';
```

### Standard Icon Sizes

| Size | Class | Usage |
|------|-------|-------|
| Small | `h-4 w-4` | Inline with text, buttons |
| Medium | `h-5 w-5` | Standalone icons, cards |
| Large | `h-6 w-6` | Headers, emphasis |

### Common Icons

| Purpose | Icon | Import |
|---------|------|--------|
| Success | Check | `Check` |
| Error | X | `X` |
| Warning | AlertTriangle | `AlertTriangle` |
| Info | Info | `Info` |
| Loading | Loader2 | `Loader2` (with `animate-spin`) |
| Quick Mode | Zap | `Zap` |
| Deep Mode | FlaskConical | `FlaskConical` |
| Search | Search | `Search` |
| Settings | Settings | `Settings` |

### Do Not Use

- Emoji icons (no Unicode emoji)
- Font Awesome (not installed)
- Custom SVGs (unless absolutely necessary)
- Heroicons (deprecated in favor of Lucide)

---

## Component Patterns

### Buttons

```tsx
// Primary action
<Button variant="default">Analyze</Button>

// Secondary action
<Button variant="secondary">Cancel</Button>

// Destructive action
<Button variant="destructive">Delete</Button>

// Ghost (minimal)
<Button variant="ghost">Learn More</Button>
```

### Cards

```tsx
<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
    <CardDescription>Description</CardDescription>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

### Badges / Tier Indicators

| Tier | Class |
|------|-------|
| Free | `badge-tier-free` |
| Low | `badge-tier-low` |
| Medium | `badge-tier-medium` |
| High | `badge-tier-high` |
| Premium | `badge-tier-premium` |

---

## Spacing

### Standard Scale

| Size | Value | Usage |
|------|-------|-------|
| `gap-1` | 0.25rem (4px) | Tight groupings |
| `gap-2` | 0.5rem (8px) | Related elements |
| `gap-4` | 1rem (16px) | Standard spacing |
| `gap-6` | 1.5rem (24px) | Section gaps |
| `gap-8` | 2rem (32px) | Major sections |

### Modal Widths

| Class | Width | Usage |
|-------|-------|-------|
| `w-modal-sm` | 320px | Confirmations |
| `w-modal-md` | 400px | Standard dialogs |
| `w-modal-lg` | 512px | Complex forms |

---

## Accessibility

### Focus States

All interactive elements must have visible focus indicators:
```css
@apply outline-ring/50;
```

### Color Contrast

- All text must meet WCAG 2.1 AA minimum (4.5:1 for normal text).
- Use `text-foreground` for primary text, `text-muted-foreground` for secondary.

### Motion

- Respect `prefers-reduced-motion`.
- Default transition: `transition-colors duration-200`.
- Avoid animations longer than 300ms.

---

## Animation

### Standard Transitions

```css
transition-colors duration-200   /* Color changes */
transition-all duration-300      /* Complex animations */
```

### Loading States

- Use `Loader2` icon with `animate-spin`.
- Pair with descriptive text (e.g., "Analyzing...").

### Page Transitions

```css
.page-enter { opacity: 0; transform: translateY(4px); }
.page-enter-active { opacity: 1; transform: translateY(0); transition: all 300ms; }
```

---

## Best Practices

### Do

1. Use semantic color variables (`--primary`, `--destructive`).
2. Use Lucide icons exclusively.
3. Follow feature-based colocation (components in `features/`).
4. Use `cn()` utility for conditional classes.
5. Use Tailwind utility classes over custom CSS.
6. Test with both light and dark themes.

### Do Not

1. Use emojis as icons.
2. Use hard-coded color values (e.g., `#ff0000`).
3. Place domain components in `src/components/`.
4. Create inline styles.
5. Use px values for spacing (use Tailwind scale).
6. Ignore dark mode compatibility.

---

## File Structure

```
frontend/src/
├── app/globals.css          # CSS variables, theme tokens
├── components/ui/           # Generic primitives (shadcn/ui)
├── features/*/components/   # Domain components
└── lib/utils.ts             # cn() utility
```

---

## Related Documents

- [Constitution](constitution.md) - Engineering rules
- [Frontend Architecture](../03-architecture/frontend.md) - Component patterns
