"use client";

import { useState, useCallback, useEffect } from "react";
import { toast } from "sonner";
import { HistoryItem } from "../types/factcheck";

const MAX_HISTORY = 20;

export function useHistory() {
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false);

  // Load history from localStorage after component mounts
  useEffect(() => {
    const raw = localStorage.getItem("factuai_history");
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        // Validate and clean the history data
        const validatedHistory = Array.isArray(parsed) ? parsed.filter(item => {
          // Ensure each item has the required properties
          return item && 
                 typeof item.id === 'string' && 
                 typeof item.input === 'string' &&
                 typeof item.timestamp === 'string' &&
                 (item.type === 'text' || item.type === 'image' || item.type === 'video') &&
                 (Array.isArray(item.results) || item.results === undefined);
        }).map(item => ({
          ...item,
          results: Array.isArray(item.results) ? item.results : [],
          summary: typeof item.summary === 'string' ? item.summary : '',
          metadata: item.metadata || {}
        })) : [];
        
        setHistory(validatedHistory);
        
        // If we cleaned up any data, save the cleaned version
        if (validatedHistory.length !== parsed.length) {
          localStorage.setItem("factuai_history", JSON.stringify(validatedHistory));
        }
      } catch (e) {
        console.warn("Failed to parse history, starting fresh:", e);
        localStorage.removeItem("factuai_history");
        setHistory([]);
      }
    }
    setIsHistoryLoaded(true);
  }, []);

  // Save history to localStorage
  const saveHistory = useCallback((updatedHistory: HistoryItem[]) => {
    setHistory(updatedHistory);
    localStorage.setItem("factuai_history", JSON.stringify(updatedHistory));
  }, []);

  // Add item to history
  const pushHistory = useCallback((item: Omit<HistoryItem, "id" | "timestamp">) => {
    if (!isHistoryLoaded) return;
    
    const newItem: HistoryItem = {
      ...item,
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
    };
    const updatedHistory = [newItem, ...history].slice(0, MAX_HISTORY);
    saveHistory(updatedHistory);
  }, [history, isHistoryLoaded, saveHistory]);

  // Delete specific history item
  const deleteHistoryItem = useCallback((id: string) => {
    if (!isHistoryLoaded) return;
    
    const updatedHistory = history.filter(h => h.id !== id);
    saveHistory(updatedHistory);
  }, [history, isHistoryLoaded, saveHistory]);

  // Clear all history
  const clearAllHistory = useCallback(() => {
    if (!isHistoryLoaded) return;
    
    setHistory([]);
    localStorage.removeItem("factuai_history");
  }, [isHistoryLoaded]);

  // Save image processing to history
  const saveImageToHistory = useCallback((
    text: string,
    imageUrl: string,
    aiScore: number | null,
    aiError?: string
  ) => {
    if (!isHistoryLoaded) return;

    const historyData: Omit<HistoryItem, "id" | "timestamp"> = {
      input: text,
      summary: "",
      results: [],
      type: "image",
      metadata: {
        imageUrl: imageUrl,
        aiScore: aiScore ?? undefined
      }
    };

    pushHistory(historyData);
    toast.success("Image analysis saved to history");
  }, [isHistoryLoaded, pushHistory]);

  // Save video processing to history
  const saveVideoToHistory = useCallback((
    text: string,
    filename: string,
    videoUrl?: string
  ) => {
    if (!isHistoryLoaded) return;

    const historyData: Omit<HistoryItem, "id" | "timestamp"> = {
      input: text,
      summary: "",
      results: [],
      type: "video",
      metadata: {
        filename: filename,
        videoUrl: videoUrl
      }
    };

    pushHistory(historyData);
    toast.success("Video analysis saved to history");
  }, [isHistoryLoaded, pushHistory]);

  return {
    historyOpen,
    setHistoryOpen,
    history: history || [], // Ensure history is always an array
    isHistoryLoaded,
    pushHistory,
    deleteHistoryItem,
    clearAllHistory,
    saveImageToHistory,
    saveVideoToHistory,
  };
}
