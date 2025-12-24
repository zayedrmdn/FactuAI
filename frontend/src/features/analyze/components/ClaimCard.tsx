'use client';

import { useState } from 'react';
import { TextSize } from '@/types/dashboard/ui';
import { FactCheckResult } from '@/types/dashboard/factcheck';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ClaimCardProps {
  result: FactCheckResult;
  index: number;
  textSize: TextSize;
}

const VERDICT_CONFIG: Record<
  string,
  {
    textClass: string;
    bgClass: string;
    borderClass: string;
    label: string;
    icon: React.ElementType;
  }
> = {
  true: {
    textClass: 'text-success',
    bgClass: 'bg-success/10',
    borderClass: 'border-l-success',
    label: 'True',
    icon: CheckCircle2,
  },
  mostly_true: {
    textClass: 'text-success',
    bgClass: 'bg-success/10',
    borderClass: 'border-l-success',
    label: 'Mostly True',
    icon: CheckCircle2,
  },
  half_true: {
    textClass: 'text-warning',
    bgClass: 'bg-warning/10',
    borderClass: 'border-l-warning',
    label: 'Half True',
    icon: AlertTriangle,
  },
  barely_true: {
    textClass: 'text-warning',
    bgClass: 'bg-warning/10',
    borderClass: 'border-l-warning',
    label: 'Barely True',
    icon: AlertTriangle,
  },
  false: {
    textClass: 'text-destructive',
    bgClass: 'bg-destructive/10',
    borderClass: 'border-l-destructive',
    label: 'False',
    icon: XCircle,
  },
  mostly_false: {
    textClass: 'text-destructive',
    bgClass: 'bg-destructive/10',
    borderClass: 'border-l-destructive',
    label: 'Mostly False',
    icon: XCircle,
  },
  unknown: {
    textClass: 'text-muted-foreground',
    bgClass: 'bg-muted',
    borderClass: 'border-l-muted-foreground',
    label: 'Unknown',
    icon: HelpCircle,
  },
};

export default function ClaimCard({ result, index, textSize }: Readonly<ClaimCardProps>) {
  const [expanded, setExpanded] = useState(false);

  const sourceUrls = Array.isArray(result.sources)
    ? result.sources
        .map((s) => {
          if (typeof s === 'string') return s;
          if (s && typeof s === 'object') {
            const obj = s as Record<string, unknown>;
            return String(obj.url || obj.text || obj.title || '');
          }
          return '';
        })
        .filter((s) => !!s)
    : [];

  const label = result.label?.toLowerCase() || 'unknown';
  const config = (VERDICT_CONFIG[label] || VERDICT_CONFIG.unknown)!;
  const VerdictIcon = config.icon;

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
      {/* Verdict Strip - Left Border Accent */}
      <div
        className={cn('absolute left-0 top-0 bottom-0 w-1', config.bgClass.replace('/10', ''))}
      />

      <div className="p-5 pl-7 flex flex-col h-full">
        {/* Header: Verdict & Confidence */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest',
                config.bgClass,
                config.textClass
              )}
            >
              <VerdictIcon className="h-3 w-3" />
              {config.label}
            </span>
            <span className="text-[10px] font-medium text-muted-foreground/70">
              CLAIM #{index + 1}
            </span>
          </div>
          {result.confidence !== undefined && (
            <span className="text-[10px] font-mono text-muted-foreground">
              {(result.confidence * 100).toFixed(0)}% CONFIDENCE
            </span>
          )}
        </div>

        {/* Claim Text */}
        <h3 className={cn('font-medium text-foreground leading-snug mb-4', textSizeClass)}>
          {result.claim}
        </h3>

        {/* Evidence / Reasoning Preview */}
        <div className="mt-auto space-y-3">
          {/* Analysis (Collapsed by default unless very short) */}
          {expanded && result.reasoning && (
            <div className="text-sm text-muted-foreground animate-in fade-in slide-in-from-top-1 duration-200">
              <span className="font-semibold text-foreground text-xs uppercase tracking-wide block mb-1">
                Analysis
              </span>
              {result.reasoning}
            </div>
          )}

          {/* Sources List (Collapsed) */}
          {expanded && sourceUrls.length > 0 && (
            <div className="pt-2 border-t border-border/50 animate-in fade-in slide-in-from-top-1 duration-300">
              <span className="font-semibold text-foreground text-xs uppercase tracking-wide block mb-2">
                Sources
              </span>
              <ul className="space-y-1.5">
                {sourceUrls.map((url, i) => (
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

          {/* Expand/Collapse Action */}
          <div className="flex items-center justify-between pt-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(!expanded)}
              className="h-6 px-0 text-xs text-muted-foreground hover:text-foreground hover:bg-transparent p-0 flex items-center gap-1"
            >
              {expanded ? (
                <>
                  Read Less <ChevronUp className="h-3 w-3" />
                </>
              ) : (
                <>
                  Read Analysis & Sources{' '}
                  <span className="px-1.5 py-0.5 rounded-full bg-muted text-[9px] font-medium">
                    {sourceUrls.length + (result.reasoning ? 1 : 0)}
                  </span>
                  <ChevronDown className="h-3 w-3" />
                </>
              )}
            </Button>

            {/* Small visual indicator if not expanded but has content */}
            {!expanded && (
              <div className="text-[10px] text-muted-foreground/50 font-mono">+ Details</div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
