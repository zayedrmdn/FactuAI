"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  FileText,
  Copy,
  Share2,
  Settings,
  X,
  RefreshCw,
  Download,
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
}: ResultsViewProps) {

  // detect pure QA vs pure claim vs mixed
  const isQAOnly = results.length > 0 
    && results.every(r => isQAResult(r));
  const isClaimOnly = results.length > 0 
    && results.every(r => !isQAResult(r));

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
      toast.error("Failed to copy to clipboard");
    }
  };

  const copyResults = async () => {
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
            message={error}
            onRetry={onRetry}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-6 w-full max-w-7xl mx-auto", className)}>
      {/* Summary Card */}
      <Card className="border-border/60 shadow-sm">
        <CardHeader className="flex flex-row items-start justify-between pb-4 p-4 md:p-6">
          <div className="space-y-1">
            <CardTitle className="text-xl font-semibold tracking-tight">Analysis Summary</CardTitle>
            <CardDescription>
              {updated ? `Last updated: ${new Date(updated).toLocaleString()}` : "Analysis complete"}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="icon" onClick={openSettings} title="Settings">
              <Settings className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" onClick={onClear} title="Clear Results">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-6 p-4 md:p-6">
          {/* Scores */}
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2">
            <div className="flex items-center justify-center rounded-lg border bg-muted/30 p-4 w-full min-w-0">
              <OverallScore score={averageConfidence} />
            </div>
            {aiScore !== null && aiScore !== undefined && (
              <div className="flex items-center justify-center rounded-lg border bg-muted/30 p-4 w-full min-w-0">
                <AIDetectionScore score={aiScore} error={aiError} />
              </div>
            )}
          </div>

          {/* Summary Text */}
          <div className="rounded-lg bg-muted/50 p-4 min-w-0">
            <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap break-words">
              {summary || "No summary available."}
            </p>
          </div>

          {/* Actions */}
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" size="sm" onClick={copySummary}>
              <Copy className="mr-2 h-4 w-4" />
              Copy Summary
            </Button>
            <Button variant="secondary" size="sm" onClick={copyResults}>
              <FileText className="mr-2 h-4 w-4" />
              Copy Details
            </Button>
            <Button variant="secondary" size="sm" onClick={() => exportPdf()}>
              <Download className="mr-2 h-4 w-4" />
              Export PDF
            </Button>
            <Button variant="secondary" size="sm" onClick={shareResults}>
              <Share2 className="mr-2 h-4 w-4" />
              Share
            </Button>
            <Button variant="outline" size="sm" onClick={onRetry} className="ml-auto">
              <RefreshCw className="mr-2 h-4 w-4" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Results */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold tracking-tight">Detailed Findings</h3>
        {results.map((result, idx) => (
          isQAResult(result) ? (
            <QAResultCard
              key={idx}
              result={result}
              index={idx}
              textSize={prefs.textSize}
              animationDelay={idx * 100}
            />
          ) : (
            <ClaimCard
              key={idx}
              result={result}
              index={idx}
              textSize={prefs.textSize}
              animationDelay={idx * 100}
            />
          )
        ))}
      </div>
    </div>
  );
}
