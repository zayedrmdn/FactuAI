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
} from 'lucide-react';
import { toast } from 'sonner';

import { LoadingAnimation, ErrorState } from '@/components/ui/feedback-states';
import { AIDetectionScore, OverallScore } from '@/components/ui/score-display';
import ClaimCard from './ClaimCard';
import { FactCheckResult, QAResult } from '@/types/dashboard/factcheck';
import { TextSize } from '@/types/dashboard/ui';
import { QAResultCard } from './QAResultCard';
import { usePdfExport } from '@/lib/hooks/usePdfExport';
import { cn } from '@/lib/utils';

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
    labelStyle: 'badge' | 'text';
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
  loadingPhase = '',
  progress = 0,
  currentClaim = 0,
  prefs,
  averageConfidence,
  aiScore,
  aiError,
  onRetry,
  onClear,
  onCancel,
  openSettings,
  error,
  className = '',
}: Readonly<ResultsViewProps>) {
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
    sections.push(`Generated: ${updated ? new Date(updated).toLocaleString() : new Date().toLocaleString()}`);
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
    sections.push(`Trust Score: ${averageConfidence.toFixed(0)}%`);
    if (aiScore !== null && aiScore !== undefined) {
      sections.push(`AI Detection: ${aiScore.toFixed(1)}% (${aiScore >= 60 ? 'Likely AI' : 'Likely Human'})`);
    }
    sections.push(`Total Claims Analyzed: ${stats.total}`);
    sections.push(`Verified: ${stats.trueCount} | False: ${stats.falseCount} | Unclear: ${stats.mixedCount}`);
    sections.push('');

    // Detailed Findings
    sections.push('DETAILED FINDINGS');
    sections.push('='.repeat(50));
    sections.push('');

    results.forEach((r, idx) => {
      if (isQAResult(r)) {
        sections.push(`Q${idx + 1}: ${r.question}`);
        sections.push(`Answer: ${r.answer}`);
        sections.push(`Confidence: ${(r.confidence * 100).toFixed(1)}%`);
        sections.push(`Sources: ${toSourceStrings(r.sources).join(', ')}`);
      } else {
        sections.push(`Claim ${idx + 1}`);
        sections.push('-'.repeat(50));
        sections.push(`Statement: ${r.claim}`);
        sections.push(`Verdict: ${r.label?.toUpperCase() || 'UNKNOWN'}`);
        sections.push(`Confidence: ${(r.confidence * 100).toFixed(0)}%`);
        sections.push('');
        
        if (r.reasoning) {
          sections.push(`Analysis:`);
          sections.push(r.reasoning);
          sections.push('');
        }

        if (r.source_quotes && r.source_quotes.length > 0) {
          sections.push('Evidence & Analysis');
          r.source_quotes.forEach((sq, i) => {
            sections.push(`${i + 1}. "${sq.quote}"`);
            sections.push(`   Source: ${sq.source}`);
            if (sq.url) sections.push(`   URL: ${sq.url}`);
          });
        }

        const sourceUrls = toSourceStrings(r.sources);
        if (sourceUrls.length > 0) {
          sections.push('');
          sections.push('All Sources');
          sourceUrls.forEach((url, i) => {
            sections.push(`${i + 1}. ${url}`);
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

  /* Copy all results to clipboard - currently unused but kept for future use */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _copyResults = async () => {
    if (!results.length) {
      toast.error('No results to copy');
      return;
    }

    const text = results
      .map((r, idx) => {
        if (isQAResult(r)) {
          // QAResult branch
          return [
            `Q${idx + 1}: ${r.question}`,
            `Answer: ${r.answer}`,
            `Confidence: ${(r.confidence * 100).toFixed(1)}%`,
            `Sources: ${toSourceStrings(r.sources).join(', ')}`,
          ].join('\n');
        } else {
          // FactCheckResult branch
          let evidenceText = '';
          if (r.source_quotes && r.source_quotes.length > 0) {
            evidenceText = r.source_quotes.map((sq) => `"${sq.quote}" - ${sq.source}`).join('\n');
          } else {
            evidenceText = Array.isArray(r.evidence) ? r.evidence.join('. ') : r.evidence;
          }
          return (
            `Claim ${idx + 1}: ${r.claim}\n` +
            `Verdict: ${r.label}\n` +
            `Confidence: ${(r.confidence * 100).toFixed(1)}%\n` +
            `Evidence:\n${evidenceText}\n` +
            `Sources: ${toSourceStrings(r.sources).join(', ')}\n`
          );
        }
      })
      .join('\n\n');

    try {
      await navigator.clipboard.writeText(text);
      toast.success('Results copied to clipboard');
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

    const text = `FactuAI Results:\n\n${summary}\n\nDetailed results: ${results.length} claims analyzed`;

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

  // Show loading state
  if (loading) {
    return (
      <Card className={cn('shadow-sm', className)}>
        <CardContent className="p-4 md:p-8">
          <LoadingAnimation phase={loadingPhase} progress={progress} currentClaim={currentClaim} />
          {onCancel && (
            <div className="mt-6 text-center">
              <Button variant="link" onClick={onCancel} className="text-muted-foreground">
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
          <div className="flex items-start justify-between mb-6 pb-4 border-b">
            <div className="space-y-1">
              <h2 className="text-lg sm:text-xl font-semibold text-slate-900">Analysis Results</h2>
              <p className="text-xs text-slate-500">
                Last updated: {updated ? new Date(updated).toLocaleTimeString() : 'Just now'}
              </p>
            </div>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={openSettings}
                className="h-8 w-8 text-slate-400 hover:text-slate-700"
              >
                <Settings className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClear}
                className="h-8 w-8 text-slate-400 hover:text-rose-600 hover:bg-rose-50"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Main Content Grid - Mobile: stack, Tablet: 2-col, Desktop: 3-col */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
            {/* Trust Score Section */}
            <div className="flex flex-col items-center justify-center p-4 rounded-lg bg-slate-50/50 border border-slate-100">
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
                Trust Score
              </h3>
              {averageConfidence !== undefined && averageConfidence !== null && !isNaN(averageConfidence) ? (
                <OverallScore score={averageConfidence} />
              ) : (
                <div className="w-24 h-24 flex items-center justify-center text-slate-400">
                  <span className="text-sm">N/A</span>
                </div>
              )}
              {aiScore !== null && aiScore !== undefined && (
                <div className="pt-4 w-full border-t border-slate-200 mt-4">
                  <AIDetectionScore score={aiScore} error={aiError} />
                </div>
              )}
            </div>

            {/* Analysis Breakdown Section */}
            <div className="flex flex-col justify-center p-4 rounded-lg bg-slate-50/50 border border-slate-100">
              <div className="space-y-1 mb-4">
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Analysis Breakdown
                </h3>
                <p className="text-2xl font-bold text-slate-900">
                  {stats.total}{' '}
                  <span className="text-base font-normal text-slate-500">
                    Claim{stats.total !== 1 ? 's' : ''}
                  </span>
                </p>
              </div>

              <div className="space-y-2.5">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-slate-700">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 flex-shrink-0" />
                    <span>Verified</span>
                  </div>
                  <span className="font-semibold text-slate-900">{stats.trueCount}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-slate-700">
                    <XCircle className="h-4 w-4 text-rose-500 flex-shrink-0" />
                    <span>False</span>
                  </div>
                  <span className="font-semibold text-slate-900">{stats.falseCount}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-slate-700">
                    <AlertTriangle className="h-4 w-4 text-amber-500 flex-shrink-0" />
                    <span>Unclear</span>
                  </div>
                  <span className="font-semibold text-slate-900">{stats.mixedCount}</span>
                </div>
              </div>
            </div>

            {/* Actions Section */}
            <div className="flex flex-col justify-center p-4 rounded-lg bg-slate-50/50 border border-slate-100 md:col-span-2 lg:col-span-1">
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
                Quick Actions
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-1 gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyFullResults}
                  className="w-full justify-start"
                  title="Copy complete analysis to clipboard"
                >
                  <Copy className="mr-2 h-3.5 w-3.5" /> Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportPdf()}
                  className="w-full justify-start"
                >
                  <Download className="mr-2 h-3.5 w-3.5" /> PDF
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={shareResults}
                  className="w-full justify-start"
                >
                  <Share2 className="mr-2 h-3.5 w-3.5" /> Share
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  onClick={onRetry}
                  className="w-full justify-start sm:col-span-4 lg:col-span-1"
                >
                  <RefreshCw className="mr-2 h-3.5 w-3.5" /> New Analysis
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Executive Summary Text Block */}
      {summary && (
        <Card className="border border-slate-200 shadow-sm rounded-xl bg-white">
          <CardContent className="p-6">
            <h4 className="text-sm font-semibold text-slate-900 mb-3 uppercase tracking-wider flex items-center gap-2">
              <FileText className="h-4 w-4 text-slate-400" />
              Executive Summary
            </h4>
            <p className="text-base leading-relaxed text-slate-700 whitespace-pre-wrap break-words">
              {summary}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Detailed Results */}
      <div className="space-y-6">
        <div className="flex items-center justify-between px-1">
          <h3 className="text-lg font-semibold text-slate-900">Detailed Findings</h3>
          <span className="text-sm text-slate-500">{results.length} Claims Analyzed</span>
        </div>

        <div className="flex flex-col space-y-6">
          {results.map((result, idx) =>
            isQAResult(result) ? (
              <QAResultCard
                key={`qa-${result.question.slice(0, 50)}-${idx}`}
                result={result}
                index={idx}
                textSize={prefs.textSize}
                animationDelay={idx * 100}
              />
            ) : (
              <ClaimCard
                key={`claim-${result.claim.slice(0, 50)}-${idx}`}
                result={result}
                index={idx}
                textSize={prefs.textSize}
                animationDelay={idx * 100}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
