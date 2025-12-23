// Path: frontend/src/app/dashboard/page.tsx
'use client';

import { useState } from 'react';
import { SettingsDialog } from '@/features/dashboard-shell';
import { InputCard, ResultsView } from '@/features/analyze';
import { useSettings } from '@/lib/hooks/useSettings';
import { useAppState } from '@/lib/hooks/useAppState';
import { cn } from '@/lib/utils';
import { ButtonContainer } from './components/ButtonContainer';
import { DashboardHero } from './components/DashboardHero';

export default function DashboardPage() {
  const { prefs, savePrefs, isDark, toggleTheme, isLoaded: settingsLoaded } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const {
    input,
    setInput,
    showResults,
    factResults,
    summary,
    updated,
    avgConfidence,
    loading,
    loadingPhase,
    progress,
    currentClaim,
    factCheckError,
    handleFactCheck,
    handleCancel,
    handleRetryInput,
    handleClear,
    handleAIDetection,
    handleInputTypeChange,
    saveImageToHistory,
    saveVideoToHistory,
  } = useAppState();

  const loadSampleText = () => {
    setInput(
      'Scientists have discovered that drinking 8 glasses of water daily can cure cancer. This breakthrough study was conducted at Harvard Medical School and published in Nature.'
    );
  };

  const loadDemoClaim = () => {
    setInput('COVID-19 vaccines contain microchips that track your location and thoughts.');
  };

  if (!settingsLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
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

      {/* Main Dashboard Container */}
      <div className="relative h-full flex flex-col max-w-[1600px] mx-auto p-4 sm:p-6 lg:p-8">
        {/* Dynamic Layout Transition */}
        <div
          className={cn(
            'grid gap-8 transition-all duration-500 ease-in-out w-full flex-1',
            showResults
              ? 'grid-cols-1 lg:grid-cols-12 items-start'
              : 'grid-cols-1 place-content-center max-w-4xl mx-auto'
          )}
        >
          {/* Main Workspace Area */}
          <main
            className={cn(
              'flex flex-col gap-8 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] relative',
              showResults ? 'col-span-1 lg:col-span-12 w-full' : 'col-span-1 w-full'
            )}
          >
            {!showResults && (
              <div className="text-center space-y-6 mb-8 animate-in fade-in slide-in-from-bottom-4 duration-700 relative">
                {/* 3D Visual Anchor */}
                {/* Dynamic Visual Anchor */}
                <div className="relative mb-6">
                  <DashboardHero />
                </div>

                <div className="space-y-4">
                  <h1 className="text-4xl sm:text-5xl font-bold tracking-tight bg-gradient-to-br from-foreground to-foreground/60 bg-clip-text text-transparent pb-1">
                    FactuAI Console
                  </h1>
                  <p className="text-lg text-muted-foreground max-w-xl mx-auto">
                    The anti-misinformation engine. Verify claims with{' '}
                    <span className="text-foreground font-medium">real-time evidence</span> and AI
                    analysis.
                  </p>
                </div>
              </div>
            )}

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
              <div className="w-full">
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

                {/* Editorial Quick Starts */}
                <ButtonContainer onSample={loadSampleText} onDemo={loadDemoClaim} />
              </div>
            )}
          </main>
        </div>
      </div>
    </>
  );
}
