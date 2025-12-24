// Path: frontend/src/features/dashboard-shell/components/DashboardFooter.tsx
'use client';

/**
 * Minimal dashboard footer.
 * Subtle, unobtrusive version info.
 */
export function DashboardFooter() {
  return (
    <footer className="shrink-0 border-t border-border/40 bg-card/30 backdrop-blur-sm px-4 py-3 text-xs">
      <div className="max-w-4xl mx-auto flex items-center justify-center">
        <span className="text-muted-foreground/60">
          <span className="font-semibold text-muted-foreground">FactuAI</span> v4.0.4
        </span>
      </div>
    </footer>
  );
}
