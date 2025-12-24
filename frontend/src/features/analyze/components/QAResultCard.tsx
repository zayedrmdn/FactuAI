'use client';

import { useState } from 'react';
import { QAResult } from '@/types/dashboard/factcheck';
import { TextSize } from '@/types/dashboard/ui';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { HelpCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

export function QAResultCard({
  result,
  index,
  textSize,
}: Readonly<{
  result: QAResult;
  index: number;
  textSize: TextSize;
}>) {
  const { question, answer, sources, confidence } = result;
  const [expanded, setExpanded] = useState(false);

  // Ensure confidence is a valid number
  const safeConfidence =
    typeof confidence === 'number' && !Number.isNaN(confidence) ? confidence : 0.8;

  const textSizeClass = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  }[textSize];

  return (
    <Card
      className={cn(
        'group relative overflow-hidden bg-card border border-border shadow-sm hover:shadow-md transition-all duration-300',
        expanded ? 'ring-1 ring-primary/5' : ''
      )}
    >
      {/* Side Accent (Primary for QA) */}
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary" />

      <div className="p-5 pl-7 flex flex-col h-full">
        {/* Header: Label & Confidence */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest bg-primary/10 text-primary">
              <HelpCircle className="h-3 w-3" />
              Question
            </span>
            <span className="text-[10px] font-medium text-muted-foreground/70">#{index + 1}</span>
          </div>
          <span className="text-[10px] font-mono text-muted-foreground">
            {(safeConfidence * 100).toFixed(0)}% CONFIDENCE
          </span>
        </div>

        {/* Question Text (Title) */}
        <h3 className={cn('font-semibold text-foreground leading-snug mb-3', textSizeClass)}>
          {question}
        </h3>

        {/* Answer Content */}
        <div className={cn('text-muted-foreground leading-relaxed mb-4', textSizeClass)}>
          {answer}
        </div>

        {/* Sources (Compact Footer) */}
        <div className="mt-auto pt-2 border-t border-border/50">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setExpanded(!expanded)}
                className="h-6 px-0 text-xs text-muted-foreground hover:text-foreground hover:bg-transparent p-0 flex items-center gap-1"
                disabled={sources.length === 0}
              >
                {sources.length > 0 ? (
                  expanded ? (
                    <>
                      Hide Sources <ChevronUp className="h-3 w-3" />
                    </>
                  ) : (
                    <>
                      View Sources{' '}
                      <span className="px-1.5 py-0.5 rounded-full bg-muted text-[9px] font-medium ml-1">
                        {sources.length}
                      </span>
                      <ChevronDown className="h-3 w-3" />
                    </>
                  )
                ) : (
                  <span className="opacity-50 cursor-not-allowed">No sources available</span>
                )}
              </Button>
            </div>

            {/* Expandable Sources List */}
            {expanded && sources.length > 0 && (
              <div className="mt-1 animate-in fade-in slide-in-from-top-1 duration-200">
                <ul className="space-y-1.5">
                  {sources.map((url, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-[10px] text-muted-foreground mt-0.5">{i + 1}.</span>
                      <a
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline truncate"
                      >
                        {url}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
