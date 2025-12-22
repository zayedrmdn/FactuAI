---
title: FactuAI Design Philosophy
version: 1.0.0
last_updated: 2025-12-22
audience: AI Agents, UI/UX Contributors
status: Active Creative Direction
---

# FactuAI Design Philosophy

> **Your role:** You are an award-winning UI/UX designer known for creating innovative, 
> non-generic web interfaces that would be featured on Awwwards—not cookie-cutter templates.

This document guides AI agents and designers to create **world-class, distinctive UI** while still respecting the technical standards in [theme-standards.md](theme-standards.md).

---

## The Anti-Slop Manifesto

Every LLM has been trained on millions of identical SaaS websites. We explicitly **reject** these lazy defaults:

### Banned Patterns

| ❌ Pattern to Avoid | Why It's Bad | ✅ Do This Instead |
|---------------------|--------------|-------------------|
| Centered hero with purple/blue gradients | Every AI-generated landing page looks identical | Asymmetric compositions, unexpected focal points, editorial layouts |
| shadcn/ui or Tailwind UI aesthetic clones | Instantly recognizable as template-based | Custom component styling with unique personality |
| Inter, Roboto, Poppins as body fonts | The "default AI font stack" | We use **Geist** exclusively—embrace its character |
| Card grids with subtle shadows | Predictable, forgettable | Dynamic layouts, overlapping elements, intentional negative space |
| Generic SaaS dashboard patterns | Every AI tool looks the same | Data storytelling, contextual hierarchy, meaningful visualization |
| Stock illustration style (undraw.co, blush) | Screams "placeholder" | Purpose-built assets, abstract geometric shapes, or photography |
| Bland loading spinners | Zero personality | Branded micro-animations, skeleton states with character |
| "Get Started Free" with blue rounded button | The most forgettable CTA possible | Contextual messaging, unexpected placements, compelling microcopy |
| Hero stats in a 3-column grid (📊 10K+, 👥 500+, ⭐ 4.9) | AI's favorite empty filler | Real, specific data or remove entirely—no vanity metrics |
| Testimonial cards with circular avatars | Template energy | Quote treatments, pull-quotes, editorial layouts |
| Gradient text on dark backgrounds | Every AI landing page in 2024 | Typography as design element, not decoration |

---

## Design Principles

### 1. Editorial, Not Template

Think **magazine layout**, not SaaS boilerplate. Every screen should feel intentionally designed:

- **Hierarchy through scale contrast**—not just bold vs. regular
- **Purposeful asymmetry** over rigid, predictable grids
- **Generous whitespace** that lets content breathe
- **Typography as a design element**, not just a content carrier
- **Unexpected visual anchors** that guide the eye

### 2. Interaction Over Static

Interfaces must feel **alive and responsive**:

- Micro-interactions on every interactive element
- State transitions that communicate meaning and feedback
- Hover states that reward exploration and curiosity
- Scroll-driven animations that enhance, not distract
- Responsive feedback that acknowledges user actions instantly

### 3. World-Class Details

The gap between good and exceptional lives in the details:

| Element | Generic Approach | World-Class Approach |
|---------|------------------|----------------------|
| **Focus states** | Blue outline | Custom, beautiful, on-brand |
| **Error states** | Red text below input | Helpful AND visually considered |
| **Empty states** | "No data" placeholder | Opportunity for delight and guidance |
| **Loading states** | Spinner animation | Branded skeleton, progress storytelling |
| **Tooltips** | Default browser or tiny popover | Designed with care, optional delight |
| **Form validation** | Inline red text | Progressive disclosure, encouraging copy |

### 4. Bold Creative Choices

Playing it safe produces forgettable interfaces. Make decisions that spark reaction:

- **Unexpected color combinations** within our semantic palette
- **Dramatic scale differences** that create visual drama
- **Unconventional navigation** when usability permits
- **Memorable micro-moments** users will actually remember
- **Confident negative space** that refuses to fill every pixel

---

## FactuAI-Specific Direction

FactuAI is a **fact-checking system**. The UI should communicate:

### Trust & Authority
- Clean, confident layouts that instill credibility
- Clear information hierarchy for complex verification results
- Professional without being sterile or corporate

### Trust & Authority
- Clean, confident layouts that instill credibility
- Clear information hierarchy for complex verification results
- Professional without being sterile or corporate

### Tactility & Depth (Anti-Flat)
- **Avoid "Floating Text":** Interactive elements must look clickable. Use subtle borders, backgrounds, or shadows. Pure text buttons are for secondary navigation only, not primary tools.
- **Micro-Shadows:** Use `shadow-sm` or `shadow-md` to lift distinct tools off the canvas.
- **Defined Boundaries:** Inputs and toolbars should have clear (but elegant) containment, not just whitespace.

### Intelligence & Precision
- The interface should feel "smart"—every element purposeful
- Verdicts and confidence scores deserve visual presence
- Evidence presentation that aids comprehension

### Transparency
- Complex AI processes made understandable
- Progress states that explain what's happening
- Source attribution that's clear and traceable

---

## Creative Constraints

> Creativity flourishes within constraints. These are yours.

All designs **must** comply with [theme-standards.md](theme-standards.md):

1. **Font family:** Geist Sans (body), Geist Mono (code)
2. **Colors:** Semantic tokens only (`--primary`, `--destructive`, etc.)
3. **Icons:** Lucide React exclusively—no emoji, no Font Awesome
4. **Accessibility:** WCAG 2.1 AA minimum (4.5:1 contrast)
5. **Dark mode:** Full parity required
6. **Responsive:** Mobile-first breakpoints

These aren't limitations—they're the frame that makes distinctive design possible.

---

## Inspiration Sources

Look here for direction. **NOT** at Dribbble SaaS templates:

| Source | Why |
|--------|-----|
| [awwwards.com](https://www.awwwards.com) | Site of the Day winners—the actual bar |
| [httpster.net](https://httpster.net) | Curated, edgy web design |
| [siteinspire.com](https://www.siteinspire.com) | Minimal, editorial excellence |
| **Bloomberg, NYT, FT** | Data visualization, information design |
| **Stripe, Linear, Vercel** | Premium SaaS that broke the template mold |
| **Experimental studios** | Active Theory, Locomotive, Resn |

---

## The Pre-Design Checklist

Before finalizing **any** interface, ask:

1. **Would this stand out on Awwwards?**
2. **Can someone tell this apart from a template in 3 seconds?**
3. **Is there at least one memorable detail or interaction?**
4. **Does this feel premium, or generic and forgettable?**
5. **Have I avoided every pattern in the "Banned" table?**

If the answer isn't "yes" to all five, **iterate**.

---

## Example: Verdict Card

❌ **Generic AI approach:**
```
┌─────────────────────────────────┐
│ ✅ TRUE                         │
│ Confidence: 87%                 │
│                                 │
│ This claim has been verified... │
│                                 │
│ [Sources] [Share]               │
└─────────────────────────────────┘
```

✅ **FactuAI approach:**
- Verdict as bold typographic statement, not a badge
- Confidence visualized with intentional representation
- Evidence hierarchy that aids scanning
- Micro-interactions on source expansion
- Asymmetric layout that creates visual interest

---

## Related Documents

- [theme-standards.md](theme-standards.md) — Technical specifications (colors, typography, components)
- [constitution.md](constitution.md) — Engineering rules
- [Frontend Architecture](../03-architecture/frontend.md) — Component patterns
