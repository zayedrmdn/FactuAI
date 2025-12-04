// Path: frontend/src/app/dashboard/page.tsx
"use client";

import { useState } from "react";
import SettingsDialog from "./features/settings/SettingsDialog";
import InputCard from "./features/inputs/InputCard";
import ResultsView from "./features/results/ResultsView";
import HistoryPanel from "./features/history/HistoryPanel";
import { useSettings } from "./hooks/useSettings";
import { useAppState } from "./hooks/useAppState";
import { RotateCcw, FileText } from "lucide-react"; // Added icons for better UI

export default function DashboardPage() {
  const { prefs, savePrefs, isDark, toggleTheme, isLoaded: settingsLoaded } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);
  
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
      <div className="p-6 max-w-[1920px] mx-auto">
        <div className="animate-pulse grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8 h-[600px] bg-muted rounded-xl"></div>
          <div className="lg:col-span-4 h-[600px] bg-muted rounded-xl"></div>
        </div>
      </div>
    );
  }

  return (
    <>
      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        prefs={prefs}
        savePrefs={savePrefs}
        toggleTheme={toggleTheme}
        isDark={isDark}
      />

      {/* Main Content Grid - constrained max-width for large screens */}
      <div className="p-4 lg:p-8 max-w-[1920px] mx-auto flex flex-col h-full">
        {/* Context Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Investigation Console</h1>
          <p className="text-muted-foreground">
            Analyze text, images, or videos to verify claims against real-time evidence.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-12 items-start flex-1">
          
          {/* Left: Main Workspace (8 columns) */}
          <main className="lg:col-span-8 xl:col-span-9 flex flex-col gap-6 h-full">
            
            {showResults ? (
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
            ) : (
              /* Input Console - Wrapped to look like a Unified Tool */
              <div className="flex flex-col rounded-xl border bg-card shadow-sm overflow-hidden min-h-[600px] h-full transition-all">
                {/* The Input Card itself */}
                <div className="flex-1 p-1 h-full">
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

                {/* Integrated Footer for Actions (No longer orphan buttons) */}
                <div className="border-t bg-muted/30 p-4 flex flex-wrap items-center justify-between gap-4">
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    Quick Start
                  </span>
                  <div className="flex gap-3">
                    <button
                      onClick={loadSampleText}
                      className="inline-flex items-center gap-2 rounded-md border bg-background px-3 py-1.5 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Load Medical Sample
                    </button>
                    <button
                      onClick={loadDemoClaim}
                      className="inline-flex items-center gap-2 rounded-md border bg-background px-3 py-1.5 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground"
                    >
                      <FileText className="h-3.5 w-3.5" />
                      Load Conspiracy Sample
                    </button>
                  </div>
                </div>
              </div>
            )}
          </main>

          {/* Right: History Sidebar (4 columns) - Sticky & Constrained */}
          <aside className="hidden lg:block lg:col-span-4 xl:col-span-3 sticky top-6 h-[calc(100vh-3rem)] overflow-hidden flex flex-col">
            
            {/* The HistoryPanel is wrapped to force internal scrolling */}
            <div className="flex-1 overflow-hidden border rounded-xl bg-card shadow-sm">
                <HistoryPanel
                  history={history}
                  load={loadHistoryItem}
                  del={deleteHistoryItem}
                  clearAll={clearAllHistory}
                />
            </div>
          </aside>

        </div>
      </div>
    </>
  );
}