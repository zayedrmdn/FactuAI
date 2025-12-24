'use client';

import { useCallback, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { useFactCheck } from './useFactCheck';
import { useHistory } from '@/features/history';
import { useInputType } from './useInputType';
import { HistoryItem } from '@/types/dashboard/factcheck';

export function useAppState() {
  // Compose existing hooks
  const {
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
    avgConfidence,
    handleFactCheck: baseHandleFactCheck,
    handleCancel,
    handleRetryInput,
    handleClear: baseClear,
    handleAIDetection: baseAIDetection,
    loadResults,
  } = useFactCheck();

  const {
    historyOpen,
    setHistoryOpen,
    history,
    pushHistory,
    deleteHistoryItem,
    clearAllHistory,
    saveImageToHistory,
    saveVideoToHistory,
  } = useHistory();

  const {
    currentInputType,
    currentImageData,
    currentVideoData,
    handleInputTypeChange: baseHandleInputTypeChange,
    clearInputTypeData,
  } = useInputType();

  // Track the last saved timestamp to prevent duplicate saves
  const lastSavedTimestamp = useRef<string | null>(null);

  // Enhanced input type change with history integration
  const handleInputTypeChange = useCallback(
    (
      type: 'text' | 'image' | 'video',
      data?: {
        imageData?: { url: string; aiScore: number | null; aiError?: string | undefined };
        videoData?: { filename: string; videoUrl?: string | undefined };
      }
    ) => {
      baseHandleInputTypeChange(type, data);

      // Auto-save to history if input exists
      if (data?.imageData && input.trim()) {
        saveImageToHistory(input, data.imageData.url, data.imageData.aiScore);
      }
      if (data?.videoData && input.trim()) {
        saveVideoToHistory(input, data.videoData.filename, data.videoData.videoUrl);
      }
    },
    [baseHandleInputTypeChange, input, saveImageToHistory, saveVideoToHistory]
  );

  // Enhanced fact-check (without history saving - moved to useEffect)
  const handleFactCheck = useCallback(async () => {
    await baseHandleFactCheck();
  }, [baseHandleFactCheck]);

  // Save to history when factResults updates (after successful fact-check)
  useEffect(() => {
    // Only save if we have results and they're fresh (not from loading history)
    // AND we haven't already saved this exact result
    if (
      factResults.length > 0 &&
      updated &&
      input.trim() &&
      lastSavedTimestamp.current !== updated
    ) {
      const historyData: Omit<HistoryItem, 'id' | 'timestamp'> = {
        input: input,
        summary,
        results: factResults,
        type: currentInputType,
        metadata: {},
      };

      if (currentInputType === 'image' && currentImageData) {
        historyData.metadata = {
          imageUrl: currentImageData.url,
          aiScore: currentImageData.aiScore ?? undefined,
        };
      }
      if (currentInputType === 'video' && currentVideoData) {
        historyData.metadata = {
          filename: currentVideoData.filename,
          videoUrl: currentVideoData.videoUrl,
        };
      }

      pushHistory(historyData);
      lastSavedTimestamp.current = updated;
    }
  }, [
    factResults,
    updated,
    input,
    summary,
    currentInputType,
    currentImageData,
    currentVideoData,
    pushHistory,
  ]);

  // Enhanced clear with input type clearing
  const handleClear = useCallback(() => {
    baseClear();
    clearInputTypeData();
  }, [baseClear, clearInputTypeData]);

  // AI detection with input type integration
  const handleAIDetection = useCallback(
    (score: number | null, error?: string) => {
      baseAIDetection(score, error);
      if (score !== null || error) {
        baseHandleInputTypeChange('image', {
          imageData: { url: '', aiScore: score, aiError: error ?? undefined },
        });
      }
    },
    [baseAIDetection, baseHandleInputTypeChange]
  );

  // Load history item
  const loadHistoryItem = useCallback(
    (item: HistoryItem) => {
      setInput(item.input);

      // Set input type and data
      const data: {
        imageData?: { url: string; aiScore: number | null; aiError?: string };
        videoData?: { filename: string; videoUrl?: string };
      } = {};
      if (item.type === 'image' && item.metadata?.imageUrl) {
        data.imageData = {
          url: item.metadata.imageUrl,
          aiScore: item.metadata.aiScore ?? null,
        };
      }
      if (item.type === 'video' && item.metadata?.videoUrl) {
        data.videoData = {
          filename: item.metadata.filename ?? '',
          videoUrl: item.metadata.videoUrl,
        };
      }
      baseHandleInputTypeChange(item.type || 'text', data);

      // Load results if they exist
      if (item.results && item.results.length > 0) {
        loadResults(item.results, item.summary, item.timestamp);
      }

      toast.info('Loaded from history');
    },
    [setInput, baseHandleInputTypeChange, loadResults]
  );

  return {
    // Input and results state
    input,
    setInput,
    showResults,
    factResults,
    summary,
    updated,
    avgConfidence,

    // Loading state
    loading,
    loadingPhase,
    progress,
    currentClaim,

    // Error state
    factCheckError,

    // History state
    historyOpen,
    setHistoryOpen,
    history,

    // Input type state
    currentInputType,
    currentImageData,
    currentVideoData,

    // Handlers
    handleFactCheck,
    handleCancel,
    handleRetryInput,
    handleClear,
    handleAIDetection,
    handleInputTypeChange,
    loadHistoryItem,
    deleteHistoryItem,
    clearAllHistory,
    saveImageToHistory,
    saveVideoToHistory,
  };
}
