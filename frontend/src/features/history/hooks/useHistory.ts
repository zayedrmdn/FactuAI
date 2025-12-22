'use client';

import { useState, useCallback, useEffect } from 'react';
import { toast } from 'sonner';
import { useHistoryStore } from '../stores/historyStore';
import { HistoryItem } from '@/types/dashboard/factcheck';

export function useHistory() {
  const [historyOpen, setHistoryOpen] = useState(false);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false);

  // Connect to Zustand store
  const history = useHistoryStore((state) => state.history);
  const pushHistoryStore = useHistoryStore((state) => state.pushHistory);
  const deleteHistoryItemStore = useHistoryStore((state) => state.deleteHistoryItem);
  const clearAllHistoryStore = useHistoryStore((state) => state.clearAllHistory);
  const saveImageToHistoryStore = useHistoryStore((state) => state.saveImageToHistory);
  const saveVideoToHistoryStore = useHistoryStore((state) => state.saveVideoToHistory);

  // Hydration check for Zustand persist
  useEffect(() => {
    setIsHistoryLoaded(true);
  }, []);

  const pushHistory = useCallback(
    (item: Omit<HistoryItem, 'id' | 'timestamp'>) => {
      pushHistoryStore(item);
    },
    [pushHistoryStore]
  );

  const deleteHistoryItem = useCallback(
    (id: string) => {
      deleteHistoryItemStore(id);
    },
    [deleteHistoryItemStore]
  );

  const clearAllHistory = useCallback(() => {
    clearAllHistoryStore();
  }, [clearAllHistoryStore]);

  const saveImageToHistory = useCallback(
    (text: string, imageUrl: string, aiScore: number | null) => {
      saveImageToHistoryStore(text, imageUrl, aiScore);
      toast.success('Image analysis saved to history');
    },
    [saveImageToHistoryStore]
  );

  const saveVideoToHistory = useCallback(
    (text: string, filename: string, videoUrl?: string) => {
      saveVideoToHistoryStore(text, filename, videoUrl);
      toast.success('Video analysis saved to history');
    },
    [saveVideoToHistoryStore]
  );

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
