'use client';

import { useState, useCallback } from 'react';
import { toast } from 'sonner';
import { validateForFactCheck } from './useInputValidation';
import { FactCheckResult } from '../types/factcheck';

const API_URL = 'http://localhost:5000/api/process';

// Type for SSE data messages
interface SSEMessage {
  type: 'phase' | 'result' | 'summary' | 'complete' | 'error';
  message?: string;
  progress?: number;
  claim_index?: number;
  result?: FactCheckResult;
  summary?: string;
}

// Type for progress state setters
interface ProgressSetters {
  setLoadingPhase: (phase: string) => void;
  setProgress: (progress: number) => void;
  setCurrentClaim: (claim: number) => void;
  setFactResults: React.Dispatch<React.SetStateAction<FactCheckResult[]>>;
  setSummary: (summary: string) => void;
}

/** Process a single SSE message and update state accordingly */
function processSSEMessage(
  data: SSEMessage,
  allResults: FactCheckResult[],
  setters: ProgressSetters
): void {
  const { setLoadingPhase, setProgress, setCurrentClaim, setFactResults, setSummary } = setters;

  switch (data.type) {
    case 'phase':
      setLoadingPhase(data.message ?? '');
      setProgress(data.progress ?? 0);
      if (data.claim_index !== undefined) {
        setCurrentClaim(data.claim_index + 1);
      }
      break;
    case 'result':
      if (data.result) {
        allResults.push(data.result);
        setFactResults([...allResults]);
      }
      break;
    case 'summary':
      if (data.summary) {
        setSummary(data.summary);
      }
      break;
    case 'complete':
      setProgress(100);
      setLoadingPhase('Complete!');
      break;
    case 'error':
      throw new Error(data.message ?? 'Server error');
  }
}

/** Parse SSE line and extract JSON data */
function parseSSELine(line: string): SSEMessage | null {
  if (!line.trim() || !line.startsWith('data: ')) {
    return null;
  }
  return JSON.parse(line.slice(6)) as SSEMessage;
}

export function useFactCheck() {
  const [input, setInput] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState<null | 'summary' | 'factcheck'>(null);
  const [loadingPhase, setLoadingPhase] = useState('');
  const [progress, setProgress] = useState(0);
  const [currentClaim, setCurrentClaim] = useState(0);
  const [factResults, setFactResults] = useState<FactCheckResult[]>([]);
  const [summary, setSummary] = useState('');
  const [updated, setUpdated] = useState('');
  const [factCheckError, setFactCheckError] = useState('');
  const [aiScore, setAIScore] = useState<number | null>(null);
  const [aiError, setAIError] = useState<string | undefined>(undefined);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // Compute average confidence
  const avgConfidence =
    factResults.length > 0
      ? (factResults.reduce((sum, r) => sum + (r.confidence ?? 0), 0) / factResults.length) * 100
      : 0;

  /** Reset all state to initial values */
  const resetState = useCallback(() => {
    setFactResults([]);
    setSummary('');
    setUpdated('');
    setProgress(0);
    setCurrentClaim(0);
    setLoadingPhase('');
    setFactCheckError('');
  }, []);

  /** Reset progress state on error or cancel */
  const resetProgressState = useCallback(() => {
    setProgress(0);
    setCurrentClaim(0);
  }, []);

  /** Cleanup after fact-check completes */
  const cleanup = useCallback(() => {
    setLoading(null);
    setLoadingPhase('');
    setAbortController(null);
  }, []);

  /** Handle validation errors */
  const handleValidationError = useCallback((error: string) => {
    setFactCheckError(error);
    setLoading(null);
    setAbortController(null);
  }, []);

  /** Process the SSE stream from the server */
  const processStream = useCallback(
    async (
      reader: ReadableStreamDefaultReader<Uint8Array>,
      controller: AbortController,
      setters: ProgressSetters
    ): Promise<FactCheckResult[]> => {
      const decoder = new TextDecoder();
      let buffer = '';
      const allResults: FactCheckResult[] = [];

      while (true) {
        if (controller.signal.aborted) {
          console.log('Request was cancelled by user');
          reader.cancel();
          return allResults;
        }

        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const data = parseSSELine(line);
          if (data) {
            processSSEMessage(data, allResults, setters);
          }
        }
      }

      return allResults;
    },
    []
  );

  const handleFactCheck = useCallback(async () => {
    if (!input.trim()) return;

    console.log('=== FACT CHECK STARTED ===');
    resetState();

    // Show results view with delay
    setTimeout(() => setShowResults(true), 100);

    // Create abort controller
    const controller = new AbortController();
    setAbortController(controller);
    setLoading('factcheck');

    // Validate input
    try {
      const validation = await validateForFactCheck(input);
      if (!validation.isValid) {
        handleValidationError(validation.error);
        return;
      }
    } catch {
      handleValidationError('Validation service temporarily unavailable. Please try again.');
      return;
    }

    // Get AI model selection from store
    const { useAIStore } = await import('@/stores/ai-store');
    const { getModelById } = await import('@/config/ai-models');
    const { selection } = useAIStore.getState();
    const baseModel = getModelById(selection.modelId);

    // Compute effective model config with session overrides
    const temperature =
      selection.sessionOverrides?.temperature ?? baseModel?.defaultTemperature ?? 0.7;
    const maxTokens = selection.sessionOverrides?.maxTokens ?? baseModel?.defaultMaxTokens ?? 4096;
    const topP = selection.sessionOverrides?.topP ?? baseModel?.defaultTopP ?? 0.9;
    const systemPrompt = selection.sessionOverrides?.systemPrompt ?? baseModel?.defaultSystemPrompt;

    // Build request payload with model parameters
    const requestPayload = {
      text: input,
      include_summary: true,
      progressive: true,
      // AI Model Parameters
      provider: selection.provider,
      model_id: baseModel?.modelId || selection.modelId,
      model_display_name: baseModel?.displayName || 'Unknown',
      // Model settings (with overrides if any)
      temperature,
      max_tokens: maxTokens,
      top_p: topP,
      system_prompt: systemPrompt,
    };

    console.log('📤 [FRONTEND] Sending fact-check request:', {
      provider: requestPayload.provider,
      model: requestPayload.model_display_name,
      temperature: requestPayload.temperature,
      max_tokens: requestPayload.max_tokens,
    });

    // Process fact-check request
    try {
      setLoadingPhase('Extracting claims...');
      setProgress(10);

      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `Server responded ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response stream');

      const setters: ProgressSetters = {
        setLoadingPhase,
        setProgress,
        setCurrentClaim,
        setFactResults,
        setSummary,
      };

      const allResults = await processStream(reader, controller, setters);
      setUpdated(new Date().toISOString());

      if (allResults.length > 0) {
        toast.success('Fact‑check complete');
      } else {
        setFactCheckError(
          'No verifiable claims found in the provided text. Please try with content that contains specific factual statements.'
        );
      }
    } catch (e: unknown) {
      const error = e as Error;
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
        toast.info('Fact-check cancelled');
        resetProgressState();
        return;
      }
      console.error('Fact-check error:', error);
      setFactCheckError(error.message ?? 'Server error occurred. Please try again.');
      resetProgressState();
    } finally {
      cleanup();
    }
  }, [input, resetState, handleValidationError, processStream, resetProgressState, cleanup]);

  const handleCancel = useCallback(() => {
    if (abortController) {
      console.log('Cancelling fact-check request...');
      abortController.abort();
      setAbortController(null);
    }
    setLoading(null);
    setLoadingPhase('');
    setProgress(0);
    setCurrentClaim(0);
    setShowResults(false);
    toast.info('Fact-check cancelled');
  }, [abortController]);

  const handleRetryInput = useCallback(() => {
    const resultsElement = document.querySelector('[data-results-view]');
    if (resultsElement) {
      resultsElement.classList.add('animate-out', 'slide-out-to-top', 'duration-300');
      setTimeout(() => {
        setShowResults(false);
        setFactCheckError('');
      }, 300);
    } else {
      setShowResults(false);
      setFactCheckError('');
    }
  }, []);

  const handleClear = useCallback(() => {
    const element =
      document.querySelector('[data-results-view]') || document.querySelector('[data-input-card]');
    if (element) {
      element.classList.add('animate-out', 'fade-out', 'duration-200');
      setTimeout(() => {
        setInput('');
        setFactResults([]);
        setSummary('');
        setUpdated('');
        setAIScore(null);
        setAIError(undefined);
        setShowResults(false);
        setFactCheckError('');
        setProgress(0);
        setCurrentClaim(0);
      }, 200);
    } else {
      setInput('');
      setFactResults([]);
      setSummary('');
      setUpdated('');
      setAIScore(null);
      setAIError(undefined);
      setShowResults(false);
      setFactCheckError('');
      setProgress(0);
      setCurrentClaim(0);
    }
  }, []);

  const handleAIDetection = useCallback((score: number | null, error?: string) => {
    setAIScore(score);
    setAIError(error);
  }, []);

  // Add setters for loading historical data
  const loadResults = useCallback(
    (results: FactCheckResult[], summaryText: string, updatedTime: string) => {
      setFactResults(results);
      setSummary(summaryText);
      setUpdated(updatedTime);
      setShowResults(true);
      setFactCheckError('');
    },
    []
  );

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
