// Full Path: src/features/analyze/components/InvestigationTrace.tsx
import { useState } from 'react';
import { ChevronDown, Search, Brain, Zap } from 'lucide-react';
import { InvestigationTrace as TraceType } from '@/types/dashboard/factcheck';
import { Card, CardContent } from '@/components/ui/card';

interface Props {
  trace: TraceType;
}

export function InvestigationTrace({ trace }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  if (!trace) return null;

  return (
    <Card className="border-l-4 border-l-primary bg-muted/50 shadow-sm mb-6">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between p-4 text-left hover:bg-muted transition-colors"
      >
        <div className="flex items-center gap-3">
          <Brain className="h-5 w-5 text-primary" />
          <div>
            <h3 className="text-sm font-semibold text-foreground">Investigation Logic</h3>
            <p className="text-xs text-muted-foreground">
              {trace.queries.length} queries • {trace.pivot?.triggered ? 'Pivoted' : 'Direct'}
            </p>
          </div>
        </div>
        <ChevronDown
          className={`h-5 w-5 text-muted-foreground transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <CardContent className="pt-0 pb-4 px-4 space-y-4">
          {/* Phase 1: Queries */}
          <div className="space-y-2">
            <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Search className="h-3 w-3" /> Initial Strategy
            </h4>
            <div className="grid gap-2">
              {trace.queries.map((q, i) => (
                <div
                  key={i}
                  className="text-xs bg-card border border-border px-2 py-1.5 rounded-md text-foreground font-mono"
                >
                  {q}
                </div>
              ))}
            </div>
          </div>

          {/* Phase 2: Pivot */}
          {trace.pivot && (
            <div className="space-y-2">
              <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Zap className="h-3 w-3" /> Pivot Decision
              </h4>
              <div
                className={`text-xs px-3 py-2 rounded-md border ${
                  trace.pivot.triggered
                    ? 'bg-primary/5 border-primary/10 text-foreground'
                    : 'bg-muted border-border text-muted-foreground'
                }`}
              >
                {trace.pivot.triggered ? (
                  <>
                    <span className="font-semibold block mb-1">Triggered Follow-up Search:</span>
                    <span className="font-mono block mb-1">&ldquo;{trace.pivot.query}&rdquo;</span>
                    <span className="opacity-90 italic">Reason: {trace.pivot.reason}</span>
                  </>
                ) : (
                  <span className="italic">No pivot required: Evidence was sufficient.</span>
                )}
              </div>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}
