"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import { 
  ClockIcon, 
  ChevronDownIcon,
  ChevronUpIcon,
  TrashIcon
} from "@heroicons/react/24/outline";
import { useState } from "react";
import HistoryItem from "./HistoryItem";
import { HistoryItem as HistoryItemType } from "../../types/factcheck";

interface HistoryPanelProps {
  open: boolean;
  toggle: () => void;
  history: HistoryItemType[];
  load: (item: HistoryItemType) => void;
  del: (id: string) => void;
  clearAll: () => void;
  className?: string;
}

export default function HistoryPanel({
  open,
  toggle,
  history,
  load,
  del,
  clearAll,
  className = ""
}: HistoryPanelProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const toggleExpanded = (id: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedItems(newExpanded);
  };

  const handleClearAll = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to clear all ${(history || []).length} history items?`)) {
      clearAll();
      setExpandedItems(new Set()); // Clear expanded state too
    }
  };

  const sortedHistory = [...(history || [])].sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const historyLength = (history || []).length;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="w-full h-full"
    >
      <Card className={`${className} w-full h-full bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 shadow-lg hover:shadow-xl transition-all duration-300`}>
        <CardHeader
          onClick={toggle}
          className="flex flex-row items-center justify-between cursor-pointer hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 dark:hover:from-blue-900/20 dark:hover:to-indigo-900/20 transition-all duration-200 space-y-0 pb-6 rounded-t-lg p-6"
        >
          <CardTitle className="flex items-center gap-3 text-xl">
            <span className="text-2xl">🕘</span>
            <span className="font-bold text-gray-900 dark:text-white">Recent Activity</span>
            <motion.span 
              key={historyLength}
              initial={{ scale: 1.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="text-sm font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-3 py-1 rounded-full"
            >
              {historyLength}
            </motion.span>
          </CardTitle>
          
          <div className="flex items-center gap-3">
            {historyLength > 0 && (
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleClearAll}
                className="flex items-center gap-2 text-xs text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300 px-3 py-2 rounded-lg border border-red-300 hover:border-red-400 dark:border-red-600 dark:hover:border-red-500 transition-all duration-200 hover:bg-red-50 dark:hover:bg-red-900/20"
                title="Clear all history"
              >
                <TrashIcon className="w-4 h-4" />
                <span className="font-medium">Clear All</span>
              </motion.button>
            )}
            
            <motion.div
              animate={{ rotate: open ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="text-gray-400 dark:text-gray-500 p-1"
            >
              {open ? (
                <ChevronUpIcon className="w-5 h-5" />
              ) : (
                <ChevronDownIcon className="w-5 h-5" />
              )}
            </motion.div>
          </div>
        </CardHeader>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <CardContent className="space-y-4 max-h-[calc(100vh-300px)] overflow-y-auto px-6 pb-6">
                {historyLength === 0 ? (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center py-12"
                  >
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-full flex items-center justify-center mx-auto mb-6">
                      <span className="text-3xl">📝</span>
                    </div>
                    <p className="text-base text-gray-600 dark:text-gray-400 font-medium mb-2">No history yet.</p>
                    <p className="text-sm text-gray-500 dark:text-gray-500">
                      Your fact-checks will appear here
                    </p>
                  </motion.div>
                ) : (
                  <>
                    {/* Enhanced Summary Stats */}
                    <motion.div 
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 p-4 rounded-xl border border-blue-100 dark:border-blue-800/30"
                    >
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-medium leading-relaxed">
                        📊 <span className="font-semibold">{historyLength}</span> total •{' '}
                        📝 <span className="font-semibold">{(history || []).filter(h => h.type === 'text').length}</span> text •{' '}
                        🖼️ <span className="font-semibold">{(history || []).filter(h => h.type === 'image').length}</span> image •{' '}
                        🎥 <span className="font-semibold">{(history || []).filter(h => h.type === 'video').length}</span> video
                      </div>
                    </motion.div>

                    {/* History Items */}
                    <div className="space-y-4">
                      {sortedHistory.map((item, index) => (
                        <motion.div
                          key={item.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.2, delay: index * 0.05 }}
                          className="transform hover:scale-[1.02] transition-transform duration-200"
                        >
                          <HistoryItem
                            key={item.id}
                            item={item}
                            isExpanded={expandedItems.has(item.id)}
                            onToggleExpanded={() => toggleExpanded(item.id)}
                            onLoad={() => load(item)}
                            onDelete={() => del(item.id)}
                          />
                        </motion.div>
                      ))}
                    </div>

                    {/* Load More (if needed for pagination) */}
                    {historyLength > 10 && (
                      <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-center pt-6"
                      >
                        <button className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 hover:underline transition-colors font-medium">
                          Load more...
                        </button>
                      </motion.div>
                    )}
                  </>
                )}
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </motion.div>
  );
}
