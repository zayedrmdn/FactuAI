'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  FileText,
  Copy,
  Share2,
  Settings,
  X,
  RefreshCw,
  Download,
  Activity,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { PipelineStepLoader } from './PipelineStepLoader';
import { InvestigationTrace } from './InvestigationTrace';
import { toast } from 'sonner';

import { ErrorState } from '@/components/ui/feedback-states';
import ClaimCard from './ClaimCard';
import { FactCheckResult, QAResult } from '@/types/dashboard/factcheck';
import { TextSize } from '@/types/dashboard/ui';
import { QAResultCard } from './QAResultCard';
import { usePdfExport } from '@/lib/hooks/usePdfExport';
import { cn } from '@/lib/utils';
import { Separator } from '@/components/ui/separator';

type CombinedResult = FactCheckResult | QAResult;

const toSourceStrings = (sources: unknown): string[] => {
  if (!Array.isArray(sources)) return [];
  return sources
    .map((s) => {
      if (typeof s === 'string') return s;
      if (s && typeof s === 'object') {
        const obj = s as { url?: string; text?: string; title?: string };
        return obj.url || obj.text || obj.title || '';
      }
      return '';
    })
    .filter((s) => !!s);
};

/** True if this is a QAResult, false otherwise */
function isQAResult(r: CombinedResult): r is QAResult {
  return (r as QAResult).answer !== undefined;
}

interface ResultsViewProps {
  results: CombinedResult[];
  summary: string;
  updated: string;
  loading: null | 'summary' | 'factcheck';
  loadingPhase?: string;
  progress?: number;
  currentClaim?: number;
  prefs: {
    textSize: TextSize;
  };
  averageConfidence: number;
  aiScore?: number | null;
  aiError?: string;
  onRetry: () => void;
  onClear: () => void;
  onCancel?: () => void;
  openSettings: () => void;
  error?: string;
  className?: string;
}

export default function ResultsView({
  results,
  summary,
  updated,
  loading,
  prefs,
  averageConfidence,
  aiScore,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  aiError,
  onRetry,
  onClear,
  onCancel,
  openSettings,
  error,
  className = '',
}: Readonly<ResultsViewProps>): React.JSX.Element | null {
  // detect pure QA mode
  const isQAOnly = results.length > 0 && results.every((r) => isQAResult(r));

  // PDF export hook
  const { exportPdf } = usePdfExport({
    results,
    summary,
    averageConfidence,
    aiScore,
    isQAOnly,
  });

  // Calculate stats
  const stats = results.reduce(
    (acc, r) => {
      if (!isQAResult(r)) {
        const label = r.label || 'unknown';
        if (['true', 'mostly_true'].includes(label)) acc.trueCount++;
        else if (['false', 'mostly_false'].includes(label)) acc.falseCount++;
        else acc.mixedCount++;
        acc.total++;
      }
      return acc;
    },
    { total: 0, trueCount: 0, falseCount: 0, mixedCount: 0 }
  );

  const copyFullResults = async () => {
    if (!results.length && !summary) {
      toast.error('No results to copy');
      return;
    }

    // Build comprehensive text report
    const sections: string[] = [];

    // Header
    sections.push('FACTUAI ANALYSIS REPORT');
    sections.push('=' + '='.repeat(50));
    sections.push(
      `Generated: ${updated ? new Date(updated).toLocaleString() : new Date().toLocaleString()} `
    );
    sections.push('');

    // Executive Summary
    if (summary) {
      sections.push('EXECUTIVE SUMMARY');
      sections.push('-'.repeat(50));
      sections.push(summary);
      sections.push('');
    }

    // Overall Scores
    sections.push('OVERALL ASSESSMENT');
    sections.push('-'.repeat(50));
    sections.push(`Trust Score: ${averageConfidence.toFixed(0)}% `);
    if (aiScore !== null && aiScore !== undefined) {
      sections.push(
        `AI Detection: ${aiScore.toFixed(1)}% (${aiScore >= 60 ? 'Likely AI' : 'Likely Human'})`
      );
    }
    sections.push(`Total Claims Analyzed: ${stats.total} `);
    sections.push(
      `Verified: ${stats.trueCount} | False: ${stats.falseCount} | Unclear: ${stats.mixedCount} `
    );
    sections.push('');

    // Detailed Findings
    sections.push('DETAILED FINDINGS');
    sections.push('='.repeat(50));
    sections.push('');

    results.forEach((r, idx) => {
      if (isQAResult(r)) {
        sections.push(`Q${idx + 1}: ${r.question} `);
        sections.push(`Answer: ${r.answer} `);
        sections.push(`Confidence: ${(r.confidence * 100).toFixed(1)}% `);
        sections.push(`Sources: ${toSourceStrings(r.sources).join(', ')} `);
      } else {
        sections.push(`Claim ${idx + 1} `);
        sections.push('-'.repeat(50));
        sections.push(`Statement: ${r.claim} `);
        sections.push(`Verdict: ${r.label?.toUpperCase() || 'UNKNOWN'} `);
        sections.push(`Confidence: ${(r.confidence * 100).toFixed(0)}% `);
        sections.push('');

        if (r.reasoning) {
          sections.push(`Analysis:`);
          // Normalize text to match PDF export quality
          const cleanReasoning = r.reasoning
            .replace(/[\u2018\u2019]/g, "'")
            .replace(/[\u201C\u201D]/g, '"')
            .replace(/\u00AD/g, '')
            .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015]/g, '-')
            .replace(/\s+/g, ' ')
            .trim();
          sections.push(cleanReasoning);
          sections.push('');
        }

        if (r.source_quotes && r.source_quotes.length > 0) {
          sections.push('Key Evidence:');
          r.source_quotes.forEach((sq, i) => {
            sections.push(`  ${i + 1}. "${sq.quote}"`);
            sections.push(`     — ${sq.source}`);
            if (sq.url) sections.push(`     URL: ${sq.url}`);
          });
          sections.push('');
        }

        const sourceUrls = toSourceStrings(r.sources);
        if (sourceUrls.length > 0) {
          sections.push('Sources:');
          sourceUrls.forEach((url, i) => {
            sections.push(`  ${i + 1}. ${url}`);
          });
        }
      }
      sections.push('');
      sections.push('');
    });

    // Footer
    sections.push('='.repeat(50));
    sections.push('Powered by FactuAI');

    const fullText = sections.join('\n');

    try {
      await navigator.clipboard.writeText(fullText);
      toast.success('Full analysis copied to clipboard');
    } catch (err) {
      console.error('Copy failed:', err);
      toast.error('Failed to copy to clipboard');
    }
  };

  const shareResults = async () => {
    if (!navigator.share) {
      toast.error('Sharing not supported on this device');
      return;
    }

    const text = `FactuAI Results: \n\n${summary} \n\nDetailed results: ${results.length} claims analyzed`;

    try {
      await navigator.share({
        title: 'FactuAI Results',
        text: text,
      });
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        toast.error('Failed to share');
      }
    }
  };

  // Show loading state with smart loader
  if (loading) {
    return (
      <Card className={cn('shadow-sm min-h-[400px] flex items-center justify-center', className)}>
        <CardContent className="p-4 md:p-8 w-full max-w-xl">
          <PipelineStepLoader />
          {onCancel && (
            <div className="mt-8 text-center">
              <Button
                variant="ghost"
                onClick={onCancel}
                className="text-muted-foreground hover:text-destructive"
              >
                Cancel Analysis
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  // Show error state
  if (error) {
    return (
      <Card className={cn('shadow-sm', className)}>
        <CardContent className="p-4 md:p-6">
          <ErrorState title="Analysis Failed" error={error} onRetry={onRetry} />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn('flex flex-col w-full space-y-8 max-w-6xl mx-auto pb-10', className)}>
      {/* 1. Header Section: Title, Stats, Actions */}
      <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
        {/* Top Bar: Title & Primary Actions */}
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Activity className="h-4 w-4" />
              <span className="text-xs uppercase tracking-wider font-semibold">
                Analysis Report
              </span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-black text-foreground tracking-tight">
              Validation Results
            </h1>
            <p className="text-sm text-muted-foreground">
              Generated {updated ? new Date(updated).toLocaleString() : 'Just now'}
            </p>
          </div>

          <div className="flex items-center gap-2 self-start">
            <Button variant="ghost" size="sm" onClick={openSettings} title="Settings">
              <Settings className="h-4 w-4" />
            </Button>
            <Separator orientation="vertical" className="h-4" />
            <Button variant="ghost" size="sm" onClick={onRetry} title="New Analysis">
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              title="Close"
              className="text-destructive hover:bg-destructive/10"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-6 sm:gap-12 p-1">
          {/* Trust Score */}
          <div className="flex flex-col">
            <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
              Trust Score
            </span>
            <div className="flex items-baseline gap-1">
              <span className="text-4xl font-black text-foreground">
                {averageConfidence !== undefined && !isNaN(averageConfidence)
                  ? averageConfidence.toFixed(0)
                  : 'N/A'}
                <span className="text-xl text-muted-foreground font-normal">%</span>
              </span>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                Verified
              </span>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-success" />
                <span className="text-xl font-bold text-foreground">{stats.trueCount}</span>
              </div>
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                False
              </span>
              <div className="flex items-center gap-2">
                <XCircle className="h-5 w-5 text-destructive" />
                <span className="text-xl font-bold text-foreground">{stats.falseCount}</span>
              </div>
            </div>

            {/* AI Score (Conditional) */}
            {aiScore !== null && aiScore !== undefined && (
              <>
                <Separator orientation="vertical" className="h-8 hidden sm:block" />
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                    AI Probability
                  </span>
                  <span className="text-xl font-bold text-foreground">{aiScore.toFixed(0)}%</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Action Toolbar */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={copyFullResults}
            className="h-8 text-xs gap-2"
          >
            <Copy className="h-3.5 w-3.5" /> Copy Report
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => exportPdf()}
            className="h-8 text-xs gap-2"
          >
            <Download className="h-3.5 w-3.5" /> Export PDF
          </Button>
          <Button variant="ghost" size="sm" onClick={shareResults} className="h-8 w-8 p-0">
            <Share2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <Separator />

      {/* 2. Executive Summary - Editorial Style */}
      {summary && (
        <div className="animate-in fade-in slide-in-from-bottom-3 duration-500 delay-100">
          <div className="flex items-center gap-2 mb-3">
            <FileText className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-bold text-foreground uppercase tracking-wider">
              Executive Summary
            </h3>
          </div>
          <p className="text-lg md:text-xl leading-relaxed text-foreground/90 max-w-4xl font-medium">
            {summary}
          </p>
        </div>
      )}

      {/* 3. Detailed Results Grid */}
      <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-200">
        <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
          Analysis Breakdown{' '}
          <span className="px-1.5 py-0.5 rounded-full bg-muted text-foreground text-[10px]">
            {results.length}
          </span>
        </h3>

        {/* Investigation Trace (New) */}
        {(() => {
          const first = results[0];
          if (first && !isQAResult(first) && first.trace) {
            return (
              <div className="mb-4">
                <InvestigationTrace trace={first.trace} />
              </div>
            );
          }
          return null;
        })()}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 auto-rows-fr">
          {results.map((result, idx) =>
            isQAResult(result) ? (
              <QAResultCard
                key={`qa - ${result.question.slice(0, 50)} -${idx} `}
                result={result}
                index={idx}
                textSize={prefs.textSize}
              />
            ) : (
              <ClaimCard
                key={`claim - ${result.claim.slice(0, 50)} -${idx} `}
                result={result}
                index={idx}
                textSize={prefs.textSize}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
