import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { HistoryItem } from '@/types/dashboard/factcheck';

interface HistoryState {
  history: HistoryItem[];
  setHistory: (history: HistoryItem[]) => void;
  pushHistory: (item: Omit<HistoryItem, 'id' | 'timestamp'>) => void;
  deleteHistoryItem: (id: string) => void;
  clearAllHistory: () => void;
  saveImageToHistory: (text: string, imageUrl: string, aiScore: number | null) => void;
  saveVideoToHistory: (text: string, filename: string, videoUrl?: string) => void;
}

const MAX_HISTORY = 20;

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      history: [],

      setHistory: (history) => set({ history }),

      pushHistory: (item) => {
        const newItem: HistoryItem = {
          ...item,
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
        };

        set((state) => ({
          history: [newItem, ...state.history].slice(0, MAX_HISTORY),
        }));
      },

      deleteHistoryItem: (id) => {
        set((state) => ({
          history: state.history.filter((h) => h.id !== id),
        }));
      },

      clearAllHistory: () => {
        set({ history: [] });
      },

      saveImageToHistory: (text, imageUrl, aiScore) => {
        const item: Omit<HistoryItem, 'id' | 'timestamp'> = {
          input: text,
          summary: '',
          results: [],
          type: 'image',
          metadata: {
            imageUrl,
            aiScore: aiScore ?? undefined,
          },
        };
        get().pushHistory(item);
      },

      saveVideoToHistory: (text, filename, videoUrl) => {
        const item: Omit<HistoryItem, 'id' | 'timestamp'> = {
          input: text,
          summary: '',
          results: [],
          type: 'video',
          metadata: {
            filename,
            videoUrl,
          },
        };
        get().pushHistory(item);
      },
    }),
    {
      name: 'factuai_history', // unique name
      storage: createJSONStorage(() => localStorage),
      // Only simplify or migrate if structure changes, current standard is fine
    }
  )
);
