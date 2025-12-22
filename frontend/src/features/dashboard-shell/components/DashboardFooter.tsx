// Path: frontend/src/features/dashboard-shell/components/DashboardFooter.tsx
'use client';

import { Keyboard, Info } from 'lucide-react';

/**
 * Minimal, contextual dashboard footer.
 * Provides version info and keyboard hints without being a generic copyright bar.
 */
export function DashboardFooter() {
  return (
    <footer className="shrink-0 border-t border-border/40 bg-background/50 backdrop-blur-sm px-4 py-2.5 text-xs text-muted-foreground">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-2 max-w-[1600px] mx-auto">
        {/* Left: Version */}
        <div className="flex items-center gap-2 order-2 sm:order-1">
          <Info className="h-3.5 w-3.5" aria-hidden="true" />
          <span className="font-medium">FactuAI</span>
          <span className="text-muted-foreground/60">v4.0.4</span>
        </div>

        {/* Center: Keyboard Hint (hidden on mobile) */}
        <div className="hidden md:flex items-center gap-1.5 order-1 sm:order-2">
          <Keyboard className="h-3.5 w-3.5" aria-hidden="true" />
          <span>
            <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border/50 font-mono text-2xs">
              Ctrl
            </kbd>
            {' + '}
            <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border/50 font-mono text-2xs">
              Enter
            </kbd>
            {' to analyze'}
          </span>
        </div>

        {/* Right: Help link */}
        <div className="order-3">
          <a
            href="#"
            className="text-muted-foreground hover:text-foreground transition-colors duration-200 underline-offset-4 hover:underline"
          >
            Help & Feedback
          </a>
        </div>
      </div>
    </footer>
  );
}
