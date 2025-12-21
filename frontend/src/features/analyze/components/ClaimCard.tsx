'use client';

import { TextSize } from '@/types/dashboard/ui';
import { FactCheckResult } from '@/types/dashboard/factcheck';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/primitives';
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
    borderColor: 'border-l-success',
    headerBg: 'bg-success/5',
  },
  mostly_true: {
    variant: 'success',
    label: 'Mostly True',
    icon: CheckCircle2,
    borderColor: 'border-l-success',
    headerBg: 'bg-success/5',
  },
  half_true: {
    variant: 'warning',
    label: 'Half True',
    icon: AlertTriangle,
    borderColor: 'border-l-warning',
    headerBg: 'bg-warning/5',
  },
  barely_true: {
    variant: 'warning',
    label: 'Barely True',
    icon: AlertTriangle,
    borderColor: 'border-l-warning',
    headerBg: 'bg-warning/5',
  },
  false: {
    variant: 'destructive',
    label: 'False',
    icon: XCircle,
    borderColor: 'border-l-destructive',
    headerBg: 'bg-destructive/5',
  },
  mostly_false: {
    variant: 'destructive',
    label: 'Mostly False',
    icon: XCircle,
    borderColor: 'border-l-destructive',
    headerBg: 'bg-destructive/5',
  },
  unknown: {
    variant: 'secondary',
    label: 'Unknown',
    icon: HelpCircle,
    borderColor: 'border-l-border',
    headerBg: 'bg-muted/30',
  },
};

export default function ClaimCard({ result, index, textSize }: Readonly<ClaimCardProps>) {
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

  return (
    <Card
      className={cn(
        'w-full overflow-hidden border border-border shadow-sm rounded-xl bg-card transition-all duration-300 hover:shadow-md animate-in fade-in slide-in-from-bottom-4 border-l-4',
        config!.borderColor
      )}
    >
      {/* Card Header - Vertical Stack for Mobile, Horizontal for Desktop */}
      <CardHeader className={cn('p-6 pb-4 space-y-4', config!.headerBg)}>
        {/* Top Row: Claim Number & Confidence (Demoted) */}
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Claim {index + 1}
          </span>

          {result.confidence !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Confidence
              </span>
              <span className="text-xs font-bold text-foreground bg-background/50 px-1.5 py-0.5 rounded border border-border">
                {(result.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>

        {/* Hero Verdict Badge (Promoted) */}
        <div>
          <Badge
            variant={config!.variant}
            className="text-sm sm:text-base px-4 py-1.5 gap-2 shadow-sm"
          >
            <VerdictIcon className="h-4 w-4 sm:h-5 sm:w-5" />
            {config!.label}
          </Badge>
        </div>

        {/* Claim Text */}
        <div className="space-y-2 pt-1">
          <h3
            className={cn(
              'font-medium text-foreground leading-relaxed break-words whitespace-normal',
              textSizeClass
            )}
          >
            {result.claim}
          </h3>
        </div>
      </CardHeader>

      <CardContent className="p-6 pt-0 space-y-4">
        {/* Explanation */}
        {result.explanation && (
          <div
            className={cn(
              'rounded-lg bg-muted border border-border p-4 text-foreground leading-relaxed',
              textSizeClass
            )}
          >
            {result.explanation}
          </div>
        )}

        {/* Analysis / Reasoning */}
        {result.reasoning && (
          <div
            className={cn(
              'rounded-lg bg-primary/5 border border-primary/10 p-4 text-foreground leading-relaxed',
              textSizeClass
            )}
          >
            <h4 className="text-xs font-semibold text-primary uppercase tracking-wider mb-2 flex items-center gap-2">
              <HelpCircle className="h-4 w-4" />
              Analysis
            </h4>
            {result.reasoning}
          </div>
        )}

        {/* Evidence & Sources Accordion */}
        <div className="border border-border rounded-lg">
          <Accordion>
            {/* Evidence Section */}
            <AccordionItem
              title={
                <div className="flex items-center gap-2">
                  <Quote className="h-4 w-4 text-muted-foreground" />
                  <span className="text-foreground font-medium">Evidence & Analysis</span>
                </div>
              }
            >
              <div className="px-4 space-y-3">
                {result.source_quotes && result.source_quotes.length > 0 ? (
                  <div className="space-y-3">
                    {result.source_quotes.map((quote, idx) => (
                      <div
                        key={`quote-${idx}`}
                        className="pl-3 border-l-2 border-border hover:border-primary/50 transition-colors"
                      >
                        <blockquote className="text-sm text-muted-foreground italic mb-1.5">
                          &quot;{quote.quote}&quot;
                        </blockquote>
                        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                          <span className="font-medium">{quote.source}</span>
                          <a
                            href={quote.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-primary hover:underline"
                          >
                            View Source <ExternalLink className="h-3 w-3" />
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {Array.isArray(result.evidence)
                      ? result.evidence.map((item) => item.text || item).join('. ')
                      : result.evidence || 'No detailed evidence available.'}
                  </p>
                )}
              </div>
            </AccordionItem>

            {/* Sources Section */}
            {result.sources?.length > 0 && (
              <AccordionItem
                title={
                  <div className="flex items-center gap-2">
                    <ExternalLink className="h-4 w-4 text-muted-foreground" />
                    <span className="text-foreground font-medium">
                      All Sources ({sourceUrls.length})
                    </span>
                  </div>
                }
              >
                <div className="px-4">
                  <ul className="space-y-2">
                    {sourceUrls.map((url, idx) => (
                      <li key={`source-${idx}`} className="flex items-start gap-2 text-sm">
                        <span className="text-xs text-muted-foreground w-5 shrink-0">
                          {idx + 1}.
                        </span>
                        <a
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline break-all"
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
