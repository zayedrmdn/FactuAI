---
title: FactuAI Theme Standards
version: 1.2.0
last_updated: 2025-12-22
audience: AI Agents, Developers, UI/UX Contributors
status: Active Technical Standards
---

# FactuAI Theme Standards

Technical UI specifications for consistent, accessible interfaces.

> **For creative direction and anti-template design philosophy, see [design-philosophy.md](design-philosophy.md).**

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

| Variable | Color (Light) | Tailwind Class | Usage |
|----------|---------------|----------------|-------|
| `--primary` | Indigo 600 | `text-primary`, `bg-primary` | CTAs, links, focus states |
| `--secondary` | Slate 100 | `text-secondary`, `bg-secondary` | Secondary buttons |
| `--destructive` | Red 600 | `text-destructive`, `bg-destructive` | Delete, errors |
| `--accent` | Slate 100 | `text-accent`, `bg-accent` | Hover states |

### Status Colors

| Status | Variable | Tailwind Class | Usage |
|--------|----------|----------------|-------|
| Success | `--success` | `text-success`, `bg-success` | Positive outcomes, TRUE verdicts |
| Warning | `--warning` | `text-warning`, `bg-warning` | Caution, MIXED/MOSTLY_FALSE verdicts |
| Info | `--info` | `text-info`, `bg-info` | Informational states |
| Error | `--destructive` | `text-destructive`, `bg-destructive` | Errors, FALSE verdicts |

### Verdict Color Mapping

| Verdict | Recommended Class |
|---------|-------------------|
| TRUE | `text-success` |
| FALSE | `text-destructive` |
| MOSTLY_TRUE | `text-success` |
| MOSTLY_FALSE | `text-warning` |
| MIXED | `text-warning` |
| UNVERIFIABLE | `text-muted-foreground` |

### Confidence Score Colors

| Level | Variable | Tailwind Class |
|-------|----------|----------------|
| Very High (90-100%) | `--score-very-high` | `text-score-very-high` |
| High (70-89%) | `--score-high` | `text-score-high` |
| Medium (50-69%) | `--score-medium` | `text-score-medium` |
| Low (30-49%) | `--score-low` | `text-score-low` |
| Very Low (0-29%) | `--score-very-low` | `text-score-very-low` |

### Palette Migration Guide

Replace raw Tailwind palette classes with semantic tokens:

| Avoid | Use Instead |
|-------|-------------|
| `text-blue-*`, `bg-blue-*` | `text-primary`, `bg-primary` |
| `text-gray-*` | `text-muted-foreground` |
| `bg-gray-*` | `bg-muted` |
| `border-gray-*` | `border-border` |
| `text-green-*` | `text-success` |
| `text-red-*` | `text-destructive` |
| `text-yellow-*`, `text-amber-*` | `text-warning` |

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

### Prohibited

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

### Form Elements

```tsx
// Text input
<Input placeholder="Enter text..." />

// With label
<div className="space-y-2">
  <Label htmlFor="email">Email</Label>
  <Input id="email" type="email" />
</div>

// Error state
<Input className="border-destructive" />
<p className="text-sm text-destructive">Error message</p>

// Textarea
<Textarea placeholder="Describe your claim..." rows={4} />

// Select
<Select>
  <SelectTrigger>
    <SelectValue placeholder="Choose..." />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="a">Option A</SelectItem>
    <SelectItem value="b">Option B</SelectItem>
  </SelectContent>
</Select>
```

---

## Responsive Design

### Approach: Mobile-First

Write base styles for mobile, then add responsive modifiers for larger screens.

### Breakpoints

| Prefix | Min Width | Usage |
|--------|-----------|-------|
| (none) | 0px | Mobile base |
| `sm:` | 640px | Large phones, small tablets |
| `md:` | 768px | Tablets |
| `lg:` | 1024px | Laptops, small desktops |
| `xl:` | 1280px | Desktops |
| `2xl:` | 1536px | Large desktops |

### Common Patterns

```tsx
// Stack on mobile, row on desktop
<div className="flex flex-col gap-4 md:flex-row">

// Hide on mobile, show on desktop
<div className="hidden lg:block">

// Full width on mobile, constrained on desktop
<div className="w-full max-w-md mx-auto lg:max-w-2xl">

// Responsive text sizing
<h1 className="text-xl md:text-2xl lg:text-3xl">
```

### Container Widths

| Class | Max Width | Usage |
|-------|-----------|-------|
| `max-w-sm` | 24rem (384px) | Narrow cards, modals |
| `max-w-md` | 28rem (448px) | Standard cards |
| `max-w-lg` | 32rem (512px) | Wide cards, forms |
| `max-w-xl` | 36rem (576px) | Content areas |
| `max-w-2xl` | 42rem (672px) | Article content |
| `max-w-4xl` | 56rem (896px) | Wide content |
| `max-w-6xl` | 72rem (1152px) | Full layouts |

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

### Keyboard Navigation

- All interactive elements must be keyboard-accessible.
- Use `tabIndex={0}` for custom interactive elements.
- Follow logical tab order (no `tabIndex > 0`).

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

1. Use semantic color variables (`text-primary`, `bg-destructive`).
2. Use Lucide icons exclusively.
3. Follow feature-based colocation (components in `features/`).
4. Use `cn()` utility for conditional classes.
5. Use Tailwind utility classes over custom CSS.
6. Define raw color values only in `frontend/src/app/globals.css`.
7. Test with both light and dark themes.
8. Write mobile-first responsive styles.

### Do Not

1. Use emojis as icons.
2. Use hard-coded color values (e.g., `#ff0000`) in components.
3. Use Tailwind palette utilities (`text-blue-600`, `bg-gray-200`) in components.
4. Place domain components in `src/components/`.
5. Create inline styles (exception: numeric CSS variables for dynamic values like progress widths).
6. Use px values for spacing (use Tailwind scale).
7. Ignore dark mode compatibility.
8. Use `tabIndex > 0` for navigation order.

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

- [Design Philosophy](design-philosophy.md) - Anti-AI-slop creative direction
- [Constitution](constitution.md) - Engineering rules
- [Frontend Architecture](../03-architecture/frontend.md) - Component patterns

---

## Quick Checklist

Use this for code review:

- [ ] Icons: Lucide only
- [ ] Colors: Semantic tokens only (no `*-blue-600`, `*-gray-200`)
- [ ] Raw values: Only in `globals.css` tokens
- [ ] Domain UI: Lives under `src/features/*`
- [ ] Responsive: Mobile-first with breakpoint modifiers
- [ ] Dark mode: Works correctly
- [ ] Accessibility: Focus states visible, keyboard navigable
