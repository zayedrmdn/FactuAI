"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
} from "lucide-react";
import { toast } from "sonner";

import LoadingAnimation from "@/components/ui/LoadingAnimation";
import ErrorState from "@/components/ui/ErrorState";
import AIDetectionScore from "@/components/ui/AIDetectionScore";
import OverallScore from "@/components/ui/OverallScore";
import ClaimCard from "./ClaimCard";
import { FactCheckResult, QAResult } from "../../types/factcheck";
import { TextSize } from "../../types/ui";
import { QAResultCard } from "./QAResultCard";
import { usePdfExport } from "../export";
import { cn } from "@/lib/utils";

type CombinedResult = FactCheckResult | QAResult;

/** True if this is a QAResult, false otherwise */
function isQAResult(r: CombinedResult): r is QAResult {
  return (r as QAResult).answer !== undefined;
}

interface ResultsViewProps {
  results: CombinedResult[];
  summary: string;
  updated: string;
  loading: null | "summary" | "factcheck";
  loadingPhase?: string;
  progress?: number;
  currentClaim?: number;
  prefs: { 
    labelStyle: "badge" | "text"; 
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
  loadingPhase = "",
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
  className = ""
}: Readonly<ResultsViewProps>) {

  // detect pure QA mode
  const isQAOnly = results.length > 0 
    && results.every(r => isQAResult(r));

  // PDF export hook
  const { exportPdf } = usePdfExport({
    results,
    summary,
    averageConfidence,
    aiScore,
    isQAOnly
  });

  const copySummary = async () => {
    if (!summary) {
      toast.error("No summary to copy");
      return;
    }
    
    try {
      await navigator.clipboard.writeText(summary);
      toast.success("Summary copied to clipboard");
    } catch (err) {
      console.error("Copy failed:", err);
      toast.error("Failed to copy to clipboard");
    }
  };

  /* Copy all results to clipboard - currently unused but kept for future use */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _copyResults = async () => {
    if (!results.length) {
      toast.error("No results to copy");
      return;
    }

    const text = results.map((r, idx) => {
      if (isQAResult(r)) {
        // QAResult branch
        return [
          `Q${idx+1}: ${r.question}`,
          `Answer: ${r.answer}`,
          `Confidence: ${(r.confidence * 100).toFixed(1)}%`,
          `Sources: ${r.sources.join(", ")}`,
        ].join("\n");
      } else {
        // FactCheckResult branch
        let evidenceText = "";
        if (r.source_quotes && r.source_quotes.length > 0) {
          evidenceText = r.source_quotes
            .map(sq => `"${sq.quote}" - ${sq.source}`)
            .join("\n");
        } else {
          evidenceText = Array.isArray(r.evidence)
            ? r.evidence.join(". ")
            : r.evidence;
        }
        return (
          `Claim ${idx+1}: ${r.claim}\n` +
          `Verdict: ${r.label}\n` +
          `Confidence: ${(r.confidence * 100).toFixed(1)}%\n` +
          `Evidence:\n${evidenceText}\n` +
          `Sources: ${r.sources.join(", ")}\n`
        );
      }
    }).join("\n\n");

    try {
      await navigator.clipboard.writeText(text);
      toast.success("Results copied to clipboard");
    } catch (err) {
      console.error("Copy failed:", err);
      toast.error("Failed to copy to clipboard");
    }
  };

  const shareResults = async () => {
    if (!navigator.share) {
      toast.error("Sharing not supported on this device");
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
        toast.error("Failed to share");
      }
    }
  };

  // Calculate stats
  const stats = results.reduce((acc, r) => {
    if (!isQAResult(r)) {
      const label = r.label || 'unknown';
      if (['true', 'mostly_true'].includes(label)) acc.trueCount++;
      else if (['false', 'mostly_false'].includes(label)) acc.falseCount++;
      else acc.mixedCount++;
      acc.total++;
    }
    return acc;
  }, { total: 0, trueCount: 0, falseCount: 0, mixedCount: 0 });


  // Show loading state
  if (loading) {
    return (
      <Card className={cn("shadow-sm", className)}>
        <CardContent className="p-4 md:p-8">
          <LoadingAnimation
            phase={loadingPhase}
            progress={progress}
            currentClaim={currentClaim}
          />
          {onCancel && (
            <div className="mt-6 text-center">
              <Button
                variant="link"
                onClick={onCancel}
                className="text-muted-foreground"
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
      <Card className={cn("shadow-sm", className)}>
        <CardContent className="p-4 md:p-6">
          <ErrorState
            title="Analysis Failed"
            error={error}
            onRetry={onRetry}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("flex flex-col w-full space-y-8 max-w-6xl mx-auto pb-10", className)}>
      {/* Summary Card Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Column 1: Overall Score */}
        <Card className="border border-slate-200 shadow-sm rounded-xl bg-white overflow-hidden flex flex-col">
           <CardContent className="p-6 flex-1 flex flex-col items-center justify-center space-y-4">
              <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Trust Score</h3>
              <div className="scale-110">
                <OverallScore score={averageConfidence} />
              </div>
              {aiScore !== null && aiScore !== undefined && (
                 <div className="pt-2 w-full border-t border-slate-100 mt-2">
                    <AIDetectionScore score={aiScore} error={aiError} />
                 </div>
              )}
           </CardContent>
        </Card>

        {/* Column 2: Stats & Breakdown */}
        <Card className="border border-slate-200 shadow-sm rounded-xl bg-white overflow-hidden flex flex-col">
           <CardContent className="p-6 flex-1 flex flex-col justify-center space-y-6">
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Analysis Breakdown</h3>
                <p className="text-2xl font-bold text-slate-900">{stats.total} <span className="text-base font-normal text-slate-500">Claims Analyzed</span></p>
              </div>
              
              <div className="space-y-3">
                 <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2 text-slate-700">
                       <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                       <span>Verified / True</span>
                    </div>
                    <span className="font-medium text-slate-900">{stats.trueCount}</span>
                 </div>
                 <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2 text-slate-700">
                       <XCircle className="h-4 w-4 text-rose-500" />
                       <span>False / Misleading</span>
                    </div>
                    <span className="font-medium text-slate-900">{stats.falseCount}</span>
                 </div>
                 <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2 text-slate-700">
                       <AlertTriangle className="h-4 w-4 text-amber-500" />
                       <span>Mixed / Unclear</span>
                    </div>
                    <span className="font-medium text-slate-900">{stats.mixedCount}</span>
                 </div>
              </div>
           </CardContent>
        </Card>

        {/* Column 3: Actions & Metadata */}
        <Card className="border border-slate-200 shadow-sm rounded-xl bg-slate-50/50 overflow-hidden flex flex-col">
           <CardContent className="p-6 flex-1 flex flex-col justify-between space-y-4">
              <div className="space-y-2">
                 <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Actions</h3>
                    <div className="flex gap-1">
                       <Button variant="ghost" size="icon" onClick={openSettings} className="h-8 w-8 text-slate-400 hover:text-slate-700">
                         <Settings className="h-4 w-4" />
                       </Button>
                       <Button variant="ghost" size="icon" onClick={onClear} className="h-8 w-8 text-slate-400 hover:text-rose-600 hover:bg-rose-50">
                         <X className="h-4 w-4" />
                       </Button>
                    </div>
                 </div>
                 <p className="text-xs text-slate-400">
                   Last updated: {updated ? new Date(updated).toLocaleTimeString() : "Just now"}
                 </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                 <Button variant="outline" size="sm" onClick={copySummary} className="w-full bg-white">
                    <Copy className="mr-2 h-3.5 w-3.5" /> Copy
                 </Button>
                 <Button variant="outline" size="sm" onClick={() => exportPdf()} className="w-full bg-white">
                    <Download className="mr-2 h-3.5 w-3.5" /> PDF
                 </Button>
                 <Button variant="outline" size="sm" onClick={shareResults} className="w-full bg-white col-span-2">
                    <Share2 className="mr-2 h-3.5 w-3.5" /> Share Report
                 </Button>
              </div>
              
              <Button variant="default" size="sm" onClick={onRetry} className="w-full mt-auto">
                <RefreshCw className="mr-2 h-3.5 w-3.5" /> New Analysis
              </Button>
           </CardContent>
        </Card>
      </div>

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
          {results.map((result, idx) => (
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
          ))}
        </div>
      </div>
    </div>
  );
}
