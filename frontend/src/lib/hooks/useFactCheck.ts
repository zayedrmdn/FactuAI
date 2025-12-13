'use client';

import { useState, useCallback } from 'react';
import { toast } from 'sonner';
import { validateForFactCheck } from './useInputValidation';
import { FactCheckResult } from '@/types/dashboard/factcheck';
import { usePipelineModelsStore } from '@/stores/pipeline-models-store';
import type { PipelineTask } from '@/stores/pipeline-models-store';
import { getModelById } from '@/config/ai-models';

const API_URL = 'http://127.0.0.1:8000/api/analyze';

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

      // Update active task based on loading phase
      const phase = (data.message ?? '').toLowerCase();
      let activeTask: PipelineTask | null = null;

      if (
        phase.includes('intent') ||
        phase.includes('classifying') ||
        phase.includes('detecting')
      ) {
        activeTask = 'intent';
      } else if (phase.includes('extract') || phase.includes('claim')) {
        activeTask = 'extraction';
      } else if (
        phase.includes('verif') ||
        phase.includes('verifying') ||
        phase.includes('search') ||
        phase.includes('evidence') ||
        phase.includes('ranking')
      ) {
        activeTask = 'reasoning';
      }

      if (activeTask) {
        usePipelineModelsStore.getState().setActiveTask(activeTask);
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
      usePipelineModelsStore.getState().setActiveTask(null);
      break;
    case 'error':
      usePipelineModelsStore.getState().setActiveTask(null);
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

  // Compute average confidence (convert 0-1 scale to 0-100%)
  const avgConfidence =
    factResults.length > 0
      ? (factResults.reduce((sum, r) => {
        const conf = typeof r.confidence === 'number' ? r.confidence : 0;
        return sum + conf;
      }, 0) / factResults.length) * 100
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
    const { selection } = useAIStore.getState();
    const baseModel = getModelById(selection.modelId);

    // Get pipeline model configuration
    const pipelineModels = usePipelineModelsStore.getState();
    const intentModel = getModelById(pipelineModels.intent.modelId);
    const extractionModel = getModelById(pipelineModels.extraction.modelId);
    const summaryModel = getModelById(pipelineModels.summary.modelId);
    const reasoningModel = getModelById(pipelineModels.reasoning.modelId);

    // Get search providers configuration
    const { useSearchProvidersStore } = await import('@/stores/search-providers-store');
    const enabledSearchProviders = useSearchProvidersStore.getState().getEnabledProviders();

    // Get search limits configuration
    const { useSearchLimitsStore } = await import('@/stores/search-limits-store');
    const { numGoogle, numNews, numTavily } = useSearchLimitsStore.getState();

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
      // Default AI Model Parameters (for backward compatibility)
      provider: selection.provider,
      model_id: baseModel?.modelId || selection.modelId,
      model_display_name: baseModel?.displayName || 'Unknown',
      // Model settings (with overrides if any)
      temperature,
      max_tokens: maxTokens,
      top_p: topP,
      system_prompt: systemPrompt,
      // Search providers configuration
      enabled_search_providers: enabledSearchProviders,
      // Search result limits
      num_google: numGoogle,
      num_news: numNews,
      num_tavily: numTavily,
      // Pipeline-specific model configuration
      pipeline_models: {
        intent: {
          provider: pipelineModels.intent.provider,
          model_id: intentModel?.modelId,
          model_display_name: intentModel?.displayName,
        },
        extraction: {
          provider: pipelineModels.extraction.provider,
          model_id: extractionModel?.modelId,
          model_display_name: extractionModel?.displayName,
        },
        summary: {
          provider: pipelineModels.summary.provider,
          model_id: summaryModel?.modelId,
          model_display_name: summaryModel?.displayName,
        },
        reasoning: {
          provider: pipelineModels.reasoning.provider,
          model_id: reasoningModel?.modelId,
          model_display_name: reasoningModel?.displayName,
        },
      },
    };

    console.log('📤 [FRONTEND] Sending fact-check request:');
    console.log('   Provider:', requestPayload.provider);
    console.log('   Model:', requestPayload.model_display_name);
    console.log('   Temperature:', requestPayload.temperature);
    console.log('   Max Tokens:', requestPayload.max_tokens);
    console.log('   Search Providers:', enabledSearchProviders.join(', '));
    console.log('   Search Limits: Google =', numGoogle, '| NewsAPI =', numNews, '| Tavily =', numTavily);
    console.log('   Pipeline Models:');
    console.log(
      '     ⚡ Intent:',
      requestPayload.pipeline_models.intent.model_display_name,
      `(${requestPayload.pipeline_models.intent.provider})`
    );
    console.log(
      '     📝 Extraction:',
      requestPayload.pipeline_models.extraction.model_display_name,
      `(${requestPayload.pipeline_models.extraction.provider})`
    );
    console.log(
      '     📋 Summary:',
      requestPayload.pipeline_models.summary.model_display_name,
      `(${requestPayload.pipeline_models.summary.provider})`
    );
    console.log(
      '     🧠 Reasoning:',
      requestPayload.pipeline_models.reasoning.model_display_name,
      `(${requestPayload.pipeline_models.reasoning.provider})`
    );

    toast.message('Using models', {
      description: `Intent: ${requestPayload.pipeline_models.intent.model_display_name} | Extraction: ${requestPayload.pipeline_models.extraction.model_display_name} | Summary: ${requestPayload.pipeline_models.summary.model_display_name} | Reasoning: ${requestPayload.pipeline_models.reasoning.model_display_name}`,
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
    usePipelineModelsStore.getState().setActiveTask(null);
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
