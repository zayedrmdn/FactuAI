"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { HistoryItem as HistoryItemType } from "../../types/factcheck";
import HistoryItem from "./HistoryItem";
import { TrashIcon, MagnifyingGlassIcon } from "@heroicons/react/24/outline";

interface HistoryPanelProps {
  open: boolean;
  toggle: () => void; // Kept for interface compatibility, but unused in new sticky layout
  history: HistoryItemType[];
  load: (item: HistoryItemType) => void;
  del: (id: string) => void;
  clearAll: () => void;
}

export default function HistoryPanel({
  history,
  load,
  del,
  clearAll
}: HistoryPanelProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState("all"); // 'all', 'text', 'image', 'video'

  const toggleExpanded = (id: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(id)) newExpanded.delete(id);
    else newExpanded.add(id);
    setExpandedItems(newExpanded);
  };

  const handleClearAll = () => {
    if (confirm("Clear your entire verification history? This cannot be undone.")) {
      clearAll();
    }
  };

  const filteredHistory = (history || []).filter(item => {
    if (filter === "all") return true;
    return item.type === filter;
  }).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  return (
    <div className="flex flex-col h-full bg-card">
      
      {/* 1. Header & Filters */}
      <div className="p-4 border-b shrink-0 space-y-3">
        {/* Search/Filter Bar */}
        <div className="flex items-center gap-2 bg-muted/50 p-1 rounded-lg">
           {['all', 'text', 'image', 'video'].map((type) => (
             <button
                key={type}
                onClick={() => setFilter(type)}
                className={`flex-1 text-[10px] uppercase font-bold py-1.5 rounded-md transition-all ${
                  filter === type 
                  ? 'bg-background text-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
                }`}
             >
               {type}
             </button>
           ))}
        </div>
      </div>

      {/* 2. Scrollable List Area */}
      <div className="flex-1 overflow-y-auto min-h-0 p-4 custom-scrollbar">
        {history.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="space-y-3">
            {filteredHistory.length === 0 ? (
               <p className="text-center text-sm text-muted-foreground py-8">No {filter} items found.</p>
            ) : (
              filteredHistory.map((item) => (
                <HistoryItem
                  key={item.id}
                  item={item}
                  isExpanded={expandedItems.has(item.id)}
                  onToggleExpanded={() => toggleExpanded(item.id)}
                  onLoad={() => load(item)}
                  onDelete={() => del(item.id)}
                />
              ))
            )}
          </div>
        )}
      </div>

      {/* 3. Footer (Stats & Clear) */}
      {history.length > 0 && (
        <div className="p-3 border-t bg-muted/20 shrink-0 flex items-center justify-between text-xs">
          <span className="text-muted-foreground font-medium">
            {history.length} items stored
          </span>
          <button 
            onClick={handleClearAll}
            className="flex items-center gap-1.5 text-destructive hover:text-destructive/80 transition-colors"
          >
            <TrashIcon className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-6 opacity-60 min-h-[300px]">
      <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
        <MagnifyingGlassIcon className="w-8 h-8 text-muted-foreground" />
      </div>
      <h3 className="text-sm font-semibold text-foreground">No History Yet</h3>
      <p className="text-xs text-muted-foreground mt-1 max-w-[180px]">
        Verifications you perform will appear here automatically.
      </p>
    </div>
  );
}