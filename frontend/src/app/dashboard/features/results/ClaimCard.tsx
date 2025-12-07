'use client';

import { TextSize } from '../../types/ui';
import { FactCheckResult } from '../../types/factcheck';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
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

      <CardContent className="p-6 pt-0 space-y-4 mt-4">
        {/* Explanation Box */}
        {result.explanation && (
          <div className="rounded-lg bg-slate-50 border border-slate-200 p-4 text-sm text-slate-700 leading-relaxed">
            {result.explanation}
          </div>
        )}

        {/* Evidence Accordion */}
        <div className="border border-slate-200 rounded-lg divide-y divide-slate-200">
          <Accordion>
            <AccordionItem
              title={
                <div className="flex items-center gap-2 px-4 pr-4">
                  <Quote className="h-4 w-4 text-slate-500" />
                  <span className="text-slate-700">Evidence & Analysis</span>
                </div>
              }
              className="border-b-0"
            >
              <div className="px-4 pb-4 pt-2 space-y-4">
                {result.source_quotes && result.source_quotes.length > 0 ? (
                  <div className="grid gap-4">
                    {result.source_quotes.map((quote) => (
                      <div
                        key={`quote-${quote.source}-${quote.url.slice(0, 30)}`}
                        className="relative pl-4 border-l-2 border-slate-200 hover:border-primary/50 transition-colors"
                      >
                        <blockquote className="text-sm text-slate-600 italic mb-2">
                          &quot;{quote.quote}&quot;
                        </blockquote>
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                          <span className="font-medium text-slate-900">{quote.source}</span>
                          <a
                            href={quote.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-blue-600 hover:underline"
                          >
                            Source <ExternalLink className="h-3 w-3" />
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">
                    {result.evidence || 'No specific evidence quotes available.'}
                  </p>
                )}
              </div>
            </AccordionItem>

            {result.sources?.length > 0 && (
              <AccordionItem
                title={
                  <div className="flex items-center gap-2 px-4 pr-4">
                    <ExternalLink className="h-4 w-4 text-slate-500" />
                    <span className="text-slate-700">Sources ({result.sources.length})</span>
                  </div>
                }
                className="border-t"
              >
                <div className="px-4 pb-4 pt-2">
                  <ul className="space-y-2">
                    {result.sources.map((url, sourceIndex) => (
                      <li
                        key={`source-${url.slice(0, 50)}`}
                        className="flex items-start gap-2 text-sm"
                      >
                        <span className="mt-0.5 text-xs text-slate-400 w-4">
                          {sourceIndex + 1}.
                        </span>
                        <a
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline break-all flex-1"
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
