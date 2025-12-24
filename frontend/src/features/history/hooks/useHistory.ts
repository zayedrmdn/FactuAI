'use client';

import { useState, useCallback, useEffect } from 'react';
import { toast } from 'sonner';
import { useHistoryStore } from '../stores/historyStore';
import { HistoryItem } from '@/types/dashboard/factcheck';

export function useHistory() {
  const [historyOpen, setHistoryOpen] = useState(false);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false);

  // Only subscribe to the history array - actions are accessed via getState()
  const history = useHistoryStore((state) => state.history);

  // Hydration check for Zustand persist
  useEffect(() => {
    setIsHistoryLoaded(true);
  }, []);

  // Use getState() for actions - these don't need to cause re-renders
  const pushHistory = useCallback((item: Omit<HistoryItem, 'id' | 'timestamp'>) => {
    useHistoryStore.getState().pushHistory(item);
  }, []);

  const deleteHistoryItem = useCallback((id: string) => {
    useHistoryStore.getState().deleteHistoryItem(id);
  }, []);

  const clearAllHistory = useCallback(() => {
    useHistoryStore.getState().clearAllHistory();
  }, []);

  const saveImageToHistory = useCallback(
    (text: string, imageUrl: string, aiScore: number | null) => {
      useHistoryStore.getState().saveImageToHistory(text, imageUrl, aiScore);
      toast.success('Image analysis saved to history');
    },
    []
  );

  const saveVideoToHistory = useCallback((text: string, filename: string, videoUrl?: string) => {
    useHistoryStore.getState().saveVideoToHistory(text, filename, videoUrl);
    toast.success('Video analysis saved to history');
  }, []);

  return {
    historyOpen,
    setHistoryOpen,
    history,
    isHistoryLoaded,
    pushHistory,
    deleteHistoryItem,
    clearAllHistory,
    saveImageToHistory,
    saveVideoToHistory,
  };
}
