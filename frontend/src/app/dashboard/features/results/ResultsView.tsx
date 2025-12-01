"use client";

import { motion } from "framer-motion";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  DocumentIcon,
  ClipboardIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  Cog6ToothIcon,
  XMarkIcon,
  ArrowPathIcon,
} from "@heroicons/react/24/outline";
import { toast } from "sonner";

import LoadingAnimation from "@/components/ui/LoadingAnimation";
import ErrorState from "@/components/ui/ErrorState";
import AIDetectionScore from "@/components/ui/AIDetectionScore";
import OverallScore from "@/components/ui/OverallScore";
import ClaimCard from "./ClaimCard";
import { FactCheckResult, QAResult, SourceQuote } from "../../types/factcheck";
import { TextSize } from "../../types/ui";
import { QAResultCard } from "./QAResultCard";
import { usePdfExport } from "../export";

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
      <Card className={className}>
        <CardContent className="p-6">
          <LoadingAnimation
            phase={loadingPhase}
            progress={progress}
            currentClaim={currentClaim}
          />
          {onCancel && (
            <div className="text-center mt-4">
              <button
                onClick={onCancel}
                className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 underline"
              >
                Cancel
              </button>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  // Show error state
  if (error) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <ErrorState
            error={error}
            onRetry={onRetry}
            onClear={onClear}
            title="Fact-checking failed"
            retryText="Try Again"
            clearText="Clear & Start Over"
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-5xl mx-auto"
    >
      <Card className={`${className} shadow-lg border-0 bg-white dark:bg-gray-800`}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-6 border-b border-gray-100 dark:border-gray-700">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <CardTitle className="text-2xl font-bold text-gray-900 dark:text-white">Fact-Check Results</CardTitle>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Analysis completed • {results.length} {results.length === 1 ? 'item' : 'items'} verified
            </p>
          </motion.div>
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={openSettings}
            className="p-3 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
            title="Settings"
          >
            <Cog6ToothIcon className="w-6 h-6" />
          </motion.button>
        </CardHeader>

        <CardContent className="p-8 space-y-8">
          {/* Scores Section - Enhanced */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-8 items-center justify-center bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-2xl p-6 border border-blue-100 dark:border-blue-800/30"
          >
            {averageConfidence > 0 && (
              <OverallScore 
                score={averageConfidence} 
                title="Average Confidence"
              />
            )}
            
            {(aiScore !== undefined && aiScore !== null) && (
              <AIDetectionScore 
                score={aiScore} 
                error={aiError}
              />
            )}
          </motion.div>

          {/* Summary Section - Enhanced with Visual Appeal */}
          {!isQAOnly && summary && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-3">
                  <span className="text-2xl">🧾</span>
                  Summary
                </h3>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={copySummary}
                  className="flex items-center gap-2 text-sm bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/50 px-4 py-2 rounded-lg transition-all duration-200 font-medium"
                  title="Copy summary"
                >
                  <ClipboardIcon className="w-4 h-4" />
                  Copy
                </motion.button>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700/30 shadow-sm">
                <p className={`text-gray-800 dark:text-gray-200 leading-relaxed font-medium ${
                  prefs.textSize === "sm" ? "text-sm" :
                  prefs.textSize === "lg" ? "text-lg" : "text-base"
                }`}>
                  {summary}
                </p>
                {updated && (
                  <div className="text-xs text-blue-600 dark:text-blue-400 mt-4 font-medium">
                    Last updated: {new Date(updated).toLocaleString()}
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Claims Section - Enhanced with Better Spacing */}
          {results.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center gap-3">
                  <span className="text-2xl">
                    {isQAOnly ? "❓" : isClaimOnly ? "🔍" : "📊"}
                  </span>
                  {isQAOnly
                    ? `Questions Answered (${results.length})`
                    : isClaimOnly
                      ? `Claims Analyzed (${results.length})`
                      : `Results (${results.length})`
                  }
                </h3>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={copyResults}
                  className="flex items-center gap-2 text-sm bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/50 px-4 py-2 rounded-lg transition-all duration-200 font-medium"
                  title="Copy all results"
                >
                  <ClipboardIcon className="w-4 h-4" />
                  Copy All
                </motion.button>
              </div>
              
              <div className="grid gap-6">
                {results.map((r, i) => {
                  const delay = 0.6 + (i * 0.1);
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay }}
                      className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow duration-200"
                    >
                      {isQAResult(r) ? (
                        <QAResultCard
                          result={r}
                          index={i}
                          textSize={prefs.textSize}
                          animationDelay={0}
                        />
                      ) : (
                        <ClaimCard
                          result={r}
                          index={i}
                          textSize={prefs.textSize}
                          animationDelay={0}
                        />
                      )}
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {/* Action Buttons - Enhanced with Better Spacing and Hover Effects */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="flex flex-wrap gap-4 pt-8 border-t border-gray-200 dark:border-gray-700"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onRetry}
              className="flex items-center justify-center gap-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-6 py-3 rounded-xl transition-all duration-200 font-semibold shadow-lg hover:shadow-xl min-w-[160px]"
            >
              <ArrowPathIcon className="w-5 h-5" />
              Analyze Another
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={exportPdf}
              className="flex items-center justify-center gap-3 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white px-6 py-3 rounded-xl transition-all duration-200 font-semibold shadow-lg hover:shadow-xl min-w-[140px]"
            >
              <ArrowDownTrayIcon className="w-5 h-5" />
              Export PDF
            </motion.button>

            {'share' in navigator && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={shareResults}
                className="flex items-center justify-center gap-3 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white px-6 py-3 rounded-xl transition-all duration-200 font-semibold shadow-lg hover:shadow-xl min-w-[120px]"
              >
                <ShareIcon className="w-5 h-5" />
                Share
              </motion.button>
            )}

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onClear}
              className="flex items-center justify-center gap-3 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600 hover:border-gray-400 dark:hover:border-gray-500 px-6 py-3 rounded-xl transition-all duration-200 font-semibold min-w-[100px]"
            >
              <XMarkIcon className="w-5 h-5" />
              Clear
            </motion.button>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
