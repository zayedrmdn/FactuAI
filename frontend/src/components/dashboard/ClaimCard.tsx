'use client';

import { TextSize } from '@/types/dashboard/ui';
import { FactCheckResult } from '@/types/dashboard/factcheck';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge, Progress } from '@/components/ui/primitives';
import { Accordion, AccordionItem } from '@/components/ui/accordion';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  ExternalLink,
  Quote,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ClaimCardProps {
  result: FactCheckResult;
  index: number;
  textSize: TextSize;
  animationDelay: number;
}

const VERDICT_CONFIG: Record<
  string,
  {
    variant: 'success' | 'warning' | 'destructive' | 'secondary';
    label: string;
    icon: React.ElementType;
    borderColor: string;
    headerBg: string;
  }
> = {
  true: {
    variant: 'success',
    label: 'True',
    icon: CheckCircle2,
    borderColor: 'border-l-emerald-500',
    headerBg: 'bg-emerald-50/30',
  },
  mostly_true: {
    variant: 'success',
    label: 'Mostly True',
    icon: CheckCircle2,
    borderColor: 'border-l-emerald-500',
    headerBg: 'bg-emerald-50/30',
  },
  half_true: {
    variant: 'warning',
    label: 'Half True',
    icon: AlertTriangle,
    borderColor: 'border-l-amber-500',
    headerBg: 'bg-amber-50/30',
  },
  barely_true: {
    variant: 'warning',
    label: 'Barely True',
    icon: AlertTriangle,
    borderColor: 'border-l-amber-500',
    headerBg: 'bg-amber-50/30',
  },
  false: {
    variant: 'destructive',
    label: 'False',
    icon: XCircle,
    borderColor: 'border-l-rose-500',
    headerBg: 'bg-rose-50/30',
  },
  mostly_false: {
    variant: 'destructive',
    label: 'Mostly False',
    icon: XCircle,
    borderColor: 'border-l-rose-500',
    headerBg: 'bg-rose-50/30',
  },
  unknown: {
    variant: 'secondary',
    label: 'Unknown',
    icon: HelpCircle,
    borderColor: 'border-l-slate-300',
    headerBg: 'bg-slate-50/50',
  },
};

export default function ClaimCard({
  result,
  index,
  textSize,
  animationDelay,
}: Readonly<ClaimCardProps>) {
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

  const config = VERDICT_CONFIG[result.label] ?? VERDICT_CONFIG.unknown;
  const VerdictIcon = config!.icon;

  const textSizeClass = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  }[textSize];

  // Calculate progress color based on confidence
  const getProgressColor = (score: number) => {
    if (score >= 0.8) return 'oklch(0.623 0.214 163.525)'; // emerald-500
    if (score >= 0.5) return 'oklch(0.769 0.188 70.08)'; // amber-500
    return 'oklch(0.627 0.265 303.9)'; // purple/indigo or red
  };

  return (
    <Card
      className={cn(
        'w-full overflow-hidden border border-slate-200 shadow-sm rounded-xl bg-white transition-all duration-300 hover:shadow-md animate-in fade-in slide-in-from-bottom-4 border-l-4',
        config!.borderColor
      )}
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {/* Card Header - Vertical Stack for Mobile, Horizontal for Desktop */}
      <CardHeader className={cn('p-6 pb-4 space-y-4', config!.headerBg)}>
        {/* Top Row: Badge & Claim Number */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Claim {index + 1}
            </span>
            <Badge variant={config!.variant} className="gap-1.5 px-2.5 py-0.5">
              <VerdictIcon className="h-3.5 w-3.5" />
              {config!.label}
            </Badge>
          </div>

          {/* Confidence Score - Top Right */}
          {result.confidence !== undefined && (
            <div className="hidden sm:flex items-center gap-3">
              <div className="text-xs font-medium text-slate-500">Confidence</div>
              <div className="flex items-center gap-2 min-w-[100px]">
                <Progress
                  value={result.confidence * 100}
                  className="h-2 w-16"
                  indicatorColor={getProgressColor(result.confidence)}
                />
                <span className="text-sm font-bold text-slate-700">
                  {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Claim Text */}
        <div className="space-y-2">
          <h3
            className={cn(
              'font-medium text-slate-900 leading-relaxed break-words whitespace-normal',
              textSizeClass
            )}
          >
            {result.claim}
          </h3>
        </div>

        {/* Mobile Confidence (if needed) */}
        {result.confidence !== undefined && (
          <div className="sm:hidden pt-2 border-t border-slate-200/60 mt-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-500">Confidence Score</span>
              <div className="flex items-center gap-2">
                <Progress
                  value={result.confidence * 100}
                  className="h-2 w-20"
                  indicatorColor={getProgressColor(result.confidence)}
                />
                <span className="text-sm font-bold text-slate-700">
                  {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="p-6 pt-0 space-y-4">
        {/* Explanation */}
        {result.explanation && (
          <div className="rounded-lg bg-slate-50 border border-slate-200 p-4 text-sm text-slate-700 leading-relaxed">
            {result.explanation}
          </div>
        )}

        {/* Evidence & Sources Accordion */}
        <div className="border border-slate-200 rounded-lg">
          <Accordion>
            {/* Evidence Section */}
            <AccordionItem
              title={
                <div className="flex items-center gap-2">
                  <Quote className="h-4 w-4 text-slate-500" />
                  <span className="text-slate-700 font-medium">Evidence & Analysis</span>
                </div>
              }
            >
              <div className="px-4 space-y-3">
                {result.source_quotes && result.source_quotes.length > 0 ? (
                  <div className="space-y-3">
                    {result.source_quotes.map((quote, idx) => (
                      <div
                        key={`quote-${idx}`}
                        className="pl-3 border-l-2 border-slate-200 hover:border-primary/50 transition-colors"
                      >
                        <blockquote className="text-sm text-slate-600 italic mb-1.5">
                          &quot;{quote.quote}&quot;
                        </blockquote>
                        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
                          <span className="font-medium">{quote.source}</span>
                          <a
                            href={quote.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-blue-600 hover:underline"
                          >
                            View Source <ExternalLink className="h-3 w-3" />
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">
                    {Array.isArray(result.evidence) 
                      ? result.evidence.map(item => item.text || item).join('. ') 
                      : (result.evidence || 'No detailed evidence available.')}
                  </p>
                )}
              </div>
            </AccordionItem>

            {/* Sources Section */}
            {result.sources?.length > 0 && (
              <AccordionItem
                title={
                  <div className="flex items-center gap-2">
                    <ExternalLink className="h-4 w-4 text-slate-500" />
                    <span className="text-slate-700 font-medium">
                      All Sources ({sourceUrls.length})
                    </span>
                  </div>
                }
              >
                <div className="px-4">
                  <ul className="space-y-2">
                    {sourceUrls.map((url, idx) => (
                      <li key={`source-${idx}`} className="flex items-start gap-2 text-sm">
                        <span className="text-xs text-slate-400 w-5 shrink-0">{idx + 1}.</span>
                        <a
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline break-all"
                        >
                          {url}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              </AccordionItem>
            )}
          </Accordion>
        </div>
      </CardContent>
    </Card>
  );
}
