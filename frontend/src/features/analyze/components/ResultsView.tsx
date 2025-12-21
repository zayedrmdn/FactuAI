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
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
} from 'lucide-react';
import { PipelineStepLoader } from './PipelineStepLoader';
import { InvestigationTrace } from './InvestigationTrace';
import { toast } from 'sonner';

import { ErrorState } from '@/components/ui/feedback-states';
// Score displays are embedded directly in the component
import ClaimCard from './ClaimCard';
import { FactCheckResult, QAResult } from '@/types/dashboard/factcheck';
import { TextSize } from '@/types/dashboard/ui';
import { QAResultCard } from './QAResultCard';
import { usePdfExport } from '@/lib/hooks/usePdfExport';
import { Progress } from '@/components/ui/primitives';
import { cn } from '@/lib/utils';

type CombinedResult = FactCheckResult | QAResult;

const VERDICT_CONFIG: Record<
  string,
  {
    variant: 'success' | 'warning' | 'destructive' | 'secondary';
    label: string;
    icon: React.ElementType;
  }
> = {
  true: { variant: 'success', label: 'Verified', icon: CheckCircle2 },
  mostly_true: { variant: 'success', label: 'Mostly True', icon: CheckCircle2 },
  false: { variant: 'destructive', label: 'False', icon: XCircle },
  mostly_false: { variant: 'destructive', label: 'Mostly False', icon: XCircle },
  unclear: { variant: 'warning', label: 'Unclear', icon: AlertTriangle },
  mixture: { variant: 'warning', label: 'Mixed', icon: AlertTriangle },
  unknown: { variant: 'secondary', label: 'Unknown', icon: HelpCircle },
};

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
  aiError, // Reserved for future AI error display
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
          sections.push(`Analysis: `);
          sections.push(r.reasoning);
          sections.push('');
        }

        if (r.source_quotes && r.source_quotes.length > 0) {
          sections.push('Evidence & Analysis');
          r.source_quotes.forEach((sq, i) => {
            sections.push(`${i + 1}."${sq.quote}"`);
            sections.push(`   Source: ${sq.source} `);
            if (sq.url) sections.push(`   URL: ${sq.url} `);
          });
        }

        const sourceUrls = toSourceStrings(r.sources);
        if (sourceUrls.length > 0) {
          sections.push('');
          sections.push('All Sources');
          sourceUrls.forEach((url, i) => {
            sections.push(`${i + 1}. ${url} `);
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

  // Full Path: src/features/analyze/components/ResultsView.tsx

  // ... (inside ResultsView component)

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
    <div className={cn('flex flex-col w-full space-y-6 max-w-6xl mx-auto pb-10', className)}>
      {/* Unified Summary Card - Mobile First Design */}
      <Card className="border shadow-sm rounded-xl overflow-hidden">
        <CardContent className="p-4 sm:p-6">
          {/* Header Row with Title and Actions */}
          <div className="flex items-start justify-between mb-6 pb-4 border-b border-border">
            <div className="space-y-1">
              <h2 className="text-lg sm:text-xl font-semibold text-foreground">Analysis Results</h2>
              <p className="text-xs text-muted-foreground">
                Last updated: {updated ? new Date(updated).toLocaleTimeString() : 'Just now'}
              </p>
            </div>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={openSettings}
                className="h-8 w-8 text-muted-foreground hover:text-foreground"
              >
                <Settings className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClear}
                className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* HERO SECTION: Claim Verdicts (Promoted) */}
          <div className="mb-8">
            <h3 className="text-xs font-bold text-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
              Verdict Summary
            </h3>
            <div className="flex flex-wrap gap-3">
              {results.map((r, idx) => {
                if (isQAResult(r)) return null;

                const config = (VERDICT_CONFIG[r.label?.toLowerCase() || 'unknown'] ??
                  VERDICT_CONFIG.unknown)!;
                const VerdictIcon = config.icon;
                return (
                  <div
                    key={idx}
                    className="flex items-center gap-3 bg-card border border-border rounded-lg p-2 pr-4 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className="flex flex-col items-center justify-center w-8 h-8 rounded-md bg-muted text-xs font-bold text-muted-foreground border border-border">
                      #{idx + 1}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                        Verdict
                      </span>
                      <div
                        className={cn(
                          'flex items-center gap-1.5 font-bold text-sm',
                          config.variant === 'success'
                            ? 'text-success'
                            : config.variant === 'destructive'
                              ? 'text-destructive'
                              : config.variant === 'warning'
                                ? 'text-warning'
                                : 'text-foreground'
                        )}
                      >
                        <VerdictIcon className="h-4 w-4" />
                        {config.label}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* SECONDARY GRID: Trust Score, Stats, Actions (Demoted/Compacted) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6 border-t border-border">
            {/* 1. Trust Score (Demoted to simple stat) */}
            <div className="flex flex-col justify-center space-y-1">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Confidence Level
              </h4>
              <div className="flex items-baseline gap-3">
                <span className="text-3xl font-bold text-foreground">
                  {averageConfidence !== undefined && !isNaN(averageConfidence)
                    ? averageConfidence.toFixed(0)
                    : 'N/A'}
                  %
                </span>
                <Progress value={averageConfidence || 0} className="h-2 w-24" />
              </div>
              {aiScore !== null && aiScore !== undefined && (
                <p className="text-xs text-muted-foreground mt-1">
                  AI Probability: {aiScore.toFixed(0)}%
                </p>
              )}
            </div>

            {/* 2. Analysis Stats (Tightened) */}
            <div className="flex flex-col justify-center space-y-2 md:border-l md:border-border md:pl-6">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Breakdown
              </h4>
              <div className="flex gap-4 text-sm">
                <div className="flex items-center gap-1.5 text-foreground">
                  <CheckCircle2 className="h-3.5 w-3.5 text-success" />
                  <span className="font-medium">{stats.trueCount}</span> Verified
                </div>
                <div className="flex items-center gap-1.5 text-foreground">
                  <XCircle className="h-3.5 w-3.5 text-destructive" />
                  <span className="font-medium">{stats.falseCount}</span> False
                </div>
                <div className="flex items-center gap-1.5 text-foreground">
                  <AlertTriangle className="h-3.5 w-3.5 text-warning" />
                  <span className="font-medium">{stats.mixedCount}</span> Other
                </div>
              </div>
            </div>

            {/* 3. Quick Actions (Minimalist Toolbar) */}
            <div className="flex flex-col justify-center md:items-end space-y-2 md:border-l md:border-border md:pl-6">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider md:text-right">
                Tools
              </h4>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyFullResults}
                  className="h-8 px-3 text-xs gap-2"
                  title="Copy Report"
                >
                  <Copy className="h-3.5 w-3.5" /> Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportPdf()}
                  className="h-8 px-3 text-xs gap-2"
                  title="Download PDF"
                >
                  <Download className="h-3.5 w-3.5" /> PDF
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={shareResults}
                  className="h-8 w-8 p-0"
                  title="Share"
                >
                  <Share2 className="h-3.5 w-3.5" />
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  onClick={onRetry}
                  className="h-8 w-8 p-0 ml-2"
                  title="New Analysis"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Executive Summary Text Block */}
      {summary && (
        <Card className="border border-border shadow-sm rounded-xl bg-card">
          <CardContent className="p-6">
            <h4 className="text-sm font-semibold text-foreground mb-3 uppercase tracking-wider flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              Executive Summary
            </h4>
            <p className="text-base leading-relaxed text-foreground whitespace-pre-wrap break-words">
              {summary}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Detailed Results */}
      <div className="space-y-6">
        <div className="flex items-center justify-between px-1">
          <h3 className="text-lg font-semibold text-foreground">Detailed Findings</h3>
          <span className="text-sm text-muted-foreground">{results.length} Claims Analyzed</span>
        </div>

        {/* Investigation Trace (New) */}
        {(() => {
          const first = results[0];
          if (first && !isQAResult(first) && first.trace) {
            return <InvestigationTrace trace={first.trace} />;
          }
          return null;
        })()}

        <div className="flex flex-col space-y-6">
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
