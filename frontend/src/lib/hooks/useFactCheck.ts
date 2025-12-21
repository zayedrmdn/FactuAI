'use client';

import { useState, useCallback } from 'react';
import { toast } from 'sonner';
import { validateForFactCheck } from './useInputValidation';
import {
  FactCheckResult,
  FactCheckApiResult,
  mapApiResultToFactCheckResult,
} from '@/types/dashboard/factcheck';
import { usePipelineModelsStore, getModelById } from '@/features/ai-providers';
import { getApiUrl } from '@/lib/apiBase';

const API_ANALYZE_URL = getApiUrl('analyze');

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
        }, 0) /
          factResults.length) *
        100
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
    const { useAIStore } = await import('@/features/ai-providers');
    const { selection } = useAIStore.getState();
    const baseModel = getModelById(selection.modelId);

    // Get pipeline model configuration
    const pipelineModels = usePipelineModelsStore.getState();
    const intentModel = getModelById(pipelineModels.intent.modelId);
    const extractionModel = getModelById(pipelineModels.extraction.modelId);
    const summaryModel = getModelById(pipelineModels.summary.modelId);
    const reasoningModel = getModelById(pipelineModels.reasoning.modelId);

    // Get search providers configuration
    const { useSearchProvidersStore } = await import('@/features/search');
    const enabledSearchProviders = useSearchProvidersStore.getState().getEnabledProviders();

    // Get search limits configuration
    const { useSearchLimitsStore } = await import('@/features/search');
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
      // Analysis mode: "quick" or "deep"
      analysis_mode: pipelineModels.analysisMode,
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

    console.log('[FRONTEND] Sending fact-check request:');
    console.log('   Provider:', requestPayload.provider);
    console.log('   Model:', requestPayload.model_display_name);
    console.log('   Temperature:', requestPayload.temperature);
    console.log('   Max Tokens:', requestPayload.max_tokens);
    console.log('   Search Providers:', enabledSearchProviders.join(', '));
    console.log(
      '   Search Limits: Google =',
      numGoogle,
      '| NewsAPI =',
      numNews,
      '| Tavily =',
      numTavily
    );
    console.log('   Pipeline Models:');
    console.log(
      '     Intent:',
      requestPayload.pipeline_models.intent.model_display_name,
      `(${requestPayload.pipeline_models.intent.provider})`
    );
    console.log(
      '     Extraction:',
      requestPayload.pipeline_models.extraction.model_display_name,
      `(${requestPayload.pipeline_models.extraction.provider})`
    );
    console.log(
      '     Summary:',
      requestPayload.pipeline_models.summary.model_display_name,
      `(${requestPayload.pipeline_models.summary.provider})`
    );
    console.log(
      '     Reasoning:',
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

      const res = await fetch(API_ANALYZE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,
      });

      if (!res.ok) {
        const data = await res.json();
        const errorMessage = data.detail || data.error || `Server responded ${res.status}`;

        // Handle specific backend error messages with user-friendly responses
        if (errorMessage.toLowerCase().includes('no claims extracted')) {
          setFactCheckError(
            'No verifiable claims could be extracted from your input. Please try with content that contains specific factual statements (e.g., "The Eiffel Tower is 300 meters tall").'
          );
          return;
        }
        if (errorMessage.toLowerCase().includes('rate limit')) {
          setFactCheckError('Too many requests. Please wait a moment before trying again.');
          return;
        }
        if (errorMessage.toLowerCase().includes('service unavailable') || res.status === 503) {
          setFactCheckError(
            'The fact-checking service is temporarily unavailable. Please try again in a few moments.'
          );
          return;
        }

        throw new Error(errorMessage);
      }

      // Backend returns JSON AnalyzeResponse with claims array
      setLoadingPhase('Processing results...');
      setProgress(50);

      const response = (await res.json()) as {
        request_id: string;
        model_used: string;
        latency_ms: number;
        claims: FactCheckApiResult[];
      };

      // Map backend claims to frontend format
      const allResults: FactCheckResult[] = response.claims.map((claim) =>
        mapApiResultToFactCheckResult(claim)
      );

      setProgress(90);
      setFactResults(allResults);
      setUpdated(new Date().toISOString());
      setProgress(100);
      setLoadingPhase('Complete!');

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
  }, [input, resetState, handleValidationError, resetProgressState, cleanup]);

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
