"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import SettingsDialog from "./features/settings/SettingsDialog";
import InputCard from "./features/inputs/InputCard";
import ResultsView from "./features/results/ResultsView";
import HistoryPanel from "./features/history/HistoryPanel";
import { useSettings } from "./hooks/useSettings";
import { useAppState } from "./hooks/useAppState";

export default function DashboardPage() {
  const { prefs, savePrefs, isDark, toggleTheme, isLoaded: settingsLoaded } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [user, setUser] = useState<any>(null);
  
  const {
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
    
    // Save functions
    saveImageToHistory,
    saveVideoToHistory,
  } = useAppState();

  // Get user data
  useEffect(() => {
    const userData = localStorage.getItem("user");
    if (userData) {
      setUser(JSON.parse(userData));
    }
  }, []);

  // Sample demos for CTA buttons
  const loadSampleText = () => {
    setInput("Scientists have discovered that drinking 8 glasses of water daily can cure cancer. This breakthrough study was conducted at Harvard Medical School and published in Nature.");
  };

  const loadDemoClaim = () => {
    setInput("COVID-19 vaccines contain microchips that track your location and thoughts.");
  };

  // Don't render until settings are loaded to prevent hydration mismatch
  if (!settingsLoaded) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50/30 dark:from-gray-900 dark:to-gray-800"
    >
      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        prefs={prefs}
        savePrefs={savePrefs}
        toggleTheme={toggleTheme}
        isDark={isDark}
      />

      {/* Hero Section - Full Width Desktop Layout */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 shadow-sm"
      >
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 tracking-tight">
                Welcome back, {user?.username || user?.email?.split('@')[0] || 'User'} 👋
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">
                Start verifying news or exploring past fact-checks below.
              </p>
            </div>
            <div className="hidden md:flex items-center gap-4">
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{history.length}</div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Fact-checks</div>
              </div>
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-xl">🔍</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main Workspace - Proper Desktop Grid */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-12 gap-8 min-h-[calc(100vh-200px)]">
          {/* Left: Main Workspace (8 columns) */}
          <div className="col-span-12 lg:col-span-8">
            {showResults ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
                className="w-full"
              >
                <ResultsView
                  results={factResults}
                  summary={summary}
                  updated={updated}
                  loading={loading}
                  loadingPhase={loadingPhase}
                  progress={progress}
                  currentClaim={currentClaim}
                  prefs={prefs}
                  averageConfidence={avgConfidence}
                  onRetry={handleRetryInput}
                  onClear={handleClear}
                  onCancel={handleCancel}
                  openSettings={() => setSettingsOpen(true)}
                  error={factCheckError}
                />
              </motion.div>
            ) : (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
                className="space-y-6 w-full"
              >
                {/* Input Panel - Full Width Desktop */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-2xl p-8 border border-blue-100 dark:border-blue-800/30 shadow-sm transition-shadow hover:shadow-md focus-within:shadow-lg w-full min-w-[320px] sm:min-w-[480px] md:min-w-[600px] lg:min-w-[768px]">
                  <InputCard
                    input={input}
                    setInput={setInput}
                    loading={loading}
                    onFactCheck={handleFactCheck}
                    onClear={handleClear}
                    textSize={prefs.textSize}
                    openSettings={() => setSettingsOpen(true)}
                    onAIDetection={handleAIDetection}
                    onInputTypeChange={handleInputTypeChange}
                    saveImageToHistory={saveImageToHistory}
                    saveVideoToHistory={saveVideoToHistory}
                  />
                </div>

                {/* CTA Buttons - Left Aligned */}
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.4 }}
                  className="flex flex-wrap gap-4"
                >
                  <button
                    onClick={loadSampleText}
                    className="flex items-center gap-3 px-6 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:shadow-md transition-all duration-200 group min-w-[160px]"
                  >
                    <span className="text-xl group-hover:scale-110 transition-transform">🔄</span>
                    <span className="font-medium text-base">Load Sample</span>
                  </button>
                  <button
                    onClick={loadDemoClaim}
                    className="flex items-center gap-3 px-6 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:shadow-md transition-all duration-200 group min-w-[180px]"
                  >
                    <span className="text-xl group-hover:scale-110 transition-transform">🧾</span>
                    <span className="font-medium text-base">Try Demo Claim</span>
                  </button>
                </motion.div>
              </motion.div>
            )}
          </div>

          {/* Right: Sidebar (4 columns) */}
          <div className="col-span-12 lg:col-span-4">
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.3 }}
              className="w-full h-full"
            >
              <HistoryPanel
                open={historyOpen}
                toggle={() => setHistoryOpen(!historyOpen)}
                history={history}
                load={loadHistoryItem}
                del={deleteHistoryItem}
                clearAll={clearAllHistory}
              />
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
