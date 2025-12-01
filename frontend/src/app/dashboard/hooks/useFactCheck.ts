"use client";

import { useState, useCallback } from "react";
import { toast } from "sonner";
import { validateForFactCheck } from "./useInputValidation";

const API_URL = "http://localhost:5000/api/process";

export function useFactCheck() {
  const [input, setInput] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState<null | "summary" | "factcheck">(null);
  const [loadingPhase, setLoadingPhase] = useState("");
  const [progress, setProgress] = useState(0);
  const [currentClaim, setCurrentClaim] = useState(0);
  const [factResults, setFactResults] = useState<any[]>([]);
  const [summary, setSummary] = useState("");
  const [updated, setUpdated] = useState("");
  const [factCheckError, setFactCheckError] = useState("");
  const [aiScore, setAIScore] = useState<number | null>(null);
  const [aiError, setAIError] = useState<string | undefined>(undefined);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // Compute average confidence
  const avgConfidence = factResults.length > 0
    ? (factResults.reduce((sum, r) => sum + (r.confidence ?? 0), 0) / factResults.length) * 100
    : 0;

  const handleFactCheck = useCallback(async () => {
    if (!input.trim()) return;

    console.log("=== FACT CHECK STARTED ===");

    // Clear previous state
    setFactResults([]);
    setSummary("");
    setUpdated("");
    setProgress(0);
    setCurrentClaim(0);
    setLoadingPhase("");
    setFactCheckError("");

    // Show results view with delay
    setTimeout(() => setShowResults(true), 100);

    // Create abort controller
    const controller = new AbortController();
    setAbortController(controller);
    setLoading("factcheck");

    try {
      // Validate input
      const validation = await validateForFactCheck(input);
      if (!validation.isValid) {
        setFactCheckError(validation.error);
        setLoading(null);
        setAbortController(null);
        return;
      }
    } catch (validationError) {
      console.error("Validation error:", validationError);
      setFactCheckError("Validation service temporarily unavailable. Please try again.");
      setLoading(null);
      setAbortController(null);
      return;
    }

    try {
      setLoadingPhase("Extracting claims...");
      setProgress(10);

      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, include_summary: true, progressive: true }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `Server responded ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";
      let allResults: any[] = [];
      let textSummary = "";

      while (true) {
        if (controller.signal.aborted) {
          console.log("Request was cancelled by user");
          reader.cancel();
          return;
        }

        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'phase') {
              setLoadingPhase(data.message);
              setProgress(data.progress || 0);
              if (data.claim_index !== undefined) {
                setCurrentClaim(data.claim_index + 1);
              }
            } else if (data.type === 'result') {
              allResults.push(data.result);
              setFactResults([...allResults]);
            } else if (data.type === 'summary') {
              textSummary = data.summary;
              setSummary(textSummary);
            } else if (data.type === 'complete') {
              setProgress(100);
              setLoadingPhase("Complete!");
            } else if (data.type === 'error') {
              throw new Error(data.message || "Server error");
            }
          } catch (e) {
            console.warn("Failed to parse SSE data:", line);
          }
        }
      }

      setUpdated(new Date().toISOString());

      if (allResults.length > 0) {
        // TODO: Push to history here
        toast.success("Fact‑check complete");
      } else {
        setFactCheckError("No verifiable claims found in the provided text. Please try with content that contains specific factual statements.");
      }

    } catch (e: any) {
      if (e.name === 'AbortError') {
        console.log("Request was cancelled");
        toast.info("Fact-check cancelled");
        // Reset progress only on cancel
        setProgress(0);
        setCurrentClaim(0);
        return;
      }
      console.error("Fact-check error:", e);
      setFactCheckError(e.message || "Server error occurred. Please try again.");
      // Reset progress only on error
      setProgress(0);
      setCurrentClaim(0);
    } finally {
      setLoading(null);
      setLoadingPhase("");
      setAbortController(null);
      // Don't reset progress here - let it stay at final value (usually 100%)
    }
  }, [input]);

  const handleCancel = useCallback(() => {
    if (abortController) {
      console.log("Cancelling fact-check request...");
      abortController.abort();
      setAbortController(null);
    }
    setLoading(null);
    setLoadingPhase("");
    setProgress(0);
    setCurrentClaim(0);
    setShowResults(false);
    toast.info("Fact-check cancelled");
  }, [abortController]);

  const handleRetryInput = useCallback(() => {
    const resultsElement = document.querySelector('[data-results-view]');
    if (resultsElement) {
      resultsElement.classList.add('animate-out', 'slide-out-to-top', 'duration-300');
      setTimeout(() => {
        setShowResults(false);
        setFactCheckError("");
      }, 300);
    } else {
      setShowResults(false);
      setFactCheckError("");
    }
  }, []);

  const handleClear = useCallback(() => {
    const element = document.querySelector('[data-results-view]') || document.querySelector('[data-input-card]');
    if (element) {
      element.classList.add('animate-out', 'fade-out', 'duration-200');
      setTimeout(() => {
        setInput("");
        setFactResults([]);
        setSummary("");
        setUpdated("");
        setAIScore(null);
        setAIError(undefined);
        setShowResults(false);
        setFactCheckError("");
        setProgress(0);
        setCurrentClaim(0);
      }, 200);
    } else {
      setInput("");
      setFactResults([]);
      setSummary("");
      setUpdated("");
      setAIScore(null);
      setAIError(undefined);
      setShowResults(false);
      setFactCheckError("");
      setProgress(0);
      setCurrentClaim(0);
    }
  }, []);

  const handleAIDetection = useCallback((score: number | null, error?: string) => {
    setAIScore(score);
    setAIError(error);
  }, []);

  // Add setters for loading historical data
  const loadResults = useCallback((results: any[], summaryText: string, updatedTime: string) => {
    setFactResults(results);
    setSummary(summaryText);
    setUpdated(updatedTime);
    setShowResults(true);
    setFactCheckError("");
  }, []);

  return {
    input,
    setInput,
    showResults,
    loading,
    loadingPhase,
    progress,
    currentClaim,
    factResults,
    summary,
    updated,
    factCheckError,
    aiScore,
    aiError,
    avgConfidence,
    handleFactCheck,
    handleCancel,
    handleRetryInput,
    handleClear,
    handleAIDetection,
    loadResults, // Add this new function
  };
}
